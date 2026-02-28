import json
import math
import os
import re
from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .model import BleachingMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")

DHW_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_dhw")
HS_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_hs")
MANIFEST_PATH = os.path.join(BASE_DIR, "data", "raw", "noaa_manifest.json")

REEF_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
LAT_COL = "lat"
LON_COL = "lon"

XR_ENGINE = os.getenv("XR_ENGINE", "h5netcdf")
REEF_KEY_DECIMALS = int(os.getenv("REEF_KEY_DECIMALS", "4"))
MAX_REEF_SNAP_CHECKS = int(os.getenv("REEF_SNAP_MAX_CHECKS", "300"))

DATE_FILE_RE_DHW = re.compile(r"^ct5km_dhw_v3\.1_(\d{8})\.nc$")
DATE_FILE_RE_HS = re.compile(r"^ct5km_hs_v3\.1_(\d{8})\.nc$")

app = FastAPI(title="Coral Bleaching Risk Estimator")

raw_origins = os.getenv(
    "CORS_ORIGINS",
    "https://aryasalem09.github.io,https://coral-bleaching-tracker.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
)
origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _must_exist(path: str, label: str):
    if not os.path.exists(path):
        raise RuntimeError(f"missing {label}: {path}")


def _ensure_dirs():
    os.makedirs(DHW_DIR, exist_ok=True)
    os.makedirs(HS_DIR, exist_ok=True)


def _xr_open(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine=XR_ENGINE)


def _file_non_empty(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _finite(x: float) -> bool:
    return np.isfinite(x)


def _finite_or_422(x: float, name: str):
    if not _finite(x):
        raise HTTPException(
            status_code=422,
            detail=f"{name} is not a finite number for this location/date (likely land/masked grid). try a nearby reef point.",
        )


def _bad_or_422(x: float, fill: float | None, name: str):
    if _is_bad_value(x, fill):
        raise HTTPException(
            status_code=422,
            detail=f"{name} is not a finite number for this location/date (likely land/masked grid). try a nearby reef point.",
        )


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _haversine_to_all_points(lat: float, lon: float) -> np.ndarray:
    r = 6371.0
    p1 = np.radians(lat)
    p2 = np.radians(REEF_POINTS[:, 0])
    dphi = np.radians(REEF_POINTS[:, 0] - lat)
    dl = np.radians(REEF_POINTS[:, 1] - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _coord_names(ds: xr.Dataset):
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    return lat_name, lon_name


def _pick_time0(da: xr.DataArray):
    return da.isel(time=0) if "time" in da.dims else da


def _fill_value(ds: xr.Dataset, var_name: str) -> float | None:
    v = ds[var_name]
    if "_FillValue" in v.attrs:
        return float(v.attrs["_FillValue"])
    if "_FillValue" in v.encoding:
        return float(v.encoding["_FillValue"])
    return None


def _is_bad_value(v: float, fill: float | None):
    if not np.isfinite(v):
        return True
    if fill is not None and np.isfinite(fill) and v == float(fill):
        return True
    if v in (-9999.0, -32768.0, 32767.0):
        return True
    if abs(v) > 1e6:
        return True
    return False


def _normalize_iso_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except Exception:
        return None


def _resolve_manifest_path(path_value: Any) -> str | None:
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None

    if os.path.isabs(path_text):
        return os.path.normpath(path_text)

    repo_path = os.path.normpath(os.path.join(REPO_DIR, path_text))
    if os.path.exists(repo_path):
        return repo_path

    backend_path = os.path.normpath(os.path.join(BASE_DIR, path_text))
    if os.path.exists(backend_path):
        return backend_path

    return repo_path


def _default_noaa_paths_for_iso_date(iso_date: str) -> dict[str, str]:
    ymd = iso_date.replace("-", "")
    return {
        "dhw": os.path.join(DHW_DIR, f"ct5km_dhw_v3.1_{ymd}.nc"),
        "hs": os.path.join(HS_DIR, f"ct5km_hs_v3.1_{ymd}.nc"),
    }


def _collect_local_files(directory: str, pattern: re.Pattern[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not os.path.isdir(directory):
        return out

    for name in os.listdir(directory):
        m = pattern.match(name)
        if not m:
            continue

        try:
            iso_date = datetime.strptime(m.group(1), "%Y%m%d").date().isoformat()
        except Exception:
            continue

        path = os.path.join(directory, name)
        if _file_non_empty(path):
            out[iso_date] = path

    return out


def _scan_local_dates_and_paths() -> tuple[list[str], dict[str, dict[str, str]]]:
    dhw_by_date = _collect_local_files(DHW_DIR, DATE_FILE_RE_DHW)
    hs_by_date = _collect_local_files(HS_DIR, DATE_FILE_RE_HS)

    dates = sorted(set(dhw_by_date).intersection(hs_by_date))
    by_date = {iso_date: {"dhw": dhw_by_date[iso_date], "hs": hs_by_date[iso_date]} for iso_date in dates}
    return dates, by_date


def _load_available_dates_from_manifest() -> tuple[list[str], dict[str, dict[str, str]], str]:
    if not os.path.exists(MANIFEST_PATH):
        dates, by_date = _scan_local_dates_and_paths()
        return dates, by_date, "scan"

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        dates, by_date = _scan_local_dates_and_paths()
        return dates, by_date, "scan"

    ok_dates_raw = manifest.get("ok_dates", []) if isinstance(manifest, dict) else []
    by_date_raw = manifest.get("by_date", {}) if isinstance(manifest, dict) else {}

    out_dates: list[str] = []
    out_by_date: dict[str, dict[str, str]] = {}
    seen: set[str] = set()

    for raw_date in ok_dates_raw if isinstance(ok_dates_raw, list) else []:
        iso_date = _normalize_iso_date(str(raw_date))
        if iso_date is None or iso_date in seen:
            continue

        defaults = _default_noaa_paths_for_iso_date(iso_date)
        entry = by_date_raw.get(iso_date, {}) if isinstance(by_date_raw, dict) else {}

        dhw_path = _resolve_manifest_path(entry.get("dhw")) if isinstance(entry, dict) else None
        hs_path = _resolve_manifest_path(entry.get("hs")) if isinstance(entry, dict) else None

        if not dhw_path:
            dhw_path = defaults["dhw"]
        if not hs_path:
            hs_path = defaults["hs"]

        if _file_non_empty(dhw_path) and _file_non_empty(hs_path):
            seen.add(iso_date)
            out_dates.append(iso_date)
            out_by_date[iso_date] = {"dhw": dhw_path, "hs": hs_path}

    if out_dates:
        out_dates.sort()
        return out_dates, out_by_date, "manifest"

    dates, by_date = _scan_local_dates_and_paths()
    return dates, by_date, "scan"


def _ensure_noaa_files(date_obj: pd.Timestamp):
    iso_date = date_obj.date().isoformat()
    paths = NOAA_FILES_BY_DATE.get(iso_date)

    if paths is None:
        paths = _default_noaa_paths_for_iso_date(iso_date)

    dhw_path = paths["dhw"]
    hs_path = paths["hs"]

    if not _file_non_empty(dhw_path) or not _file_non_empty(hs_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"no data for date {iso_date}. missing local NOAA files. "
                "run backend/src/download_noaa_daily.py and regenerate backend/data/raw/noaa_manifest.json"
            ),
        )

    return dhw_path, hs_path


@lru_cache(maxsize=1)
def _grid_lookup() -> dict[str, Any] | None:
    if not AVAILABLE_DATES:
        return None

    sample_date = AVAILABLE_DATES[-1]
    sample_paths = NOAA_FILES_BY_DATE.get(sample_date)
    if sample_paths is None:
        sample_paths = _default_noaa_paths_for_iso_date(sample_date)

    dhw_path = sample_paths["dhw"]
    if not _file_non_empty(dhw_path):
        return None

    ds = _xr_open(dhw_path)
    try:
        lat_name, lon_name = _coord_names(ds)
        if lat_name is None or lon_name is None:
            return None

        lat_values = np.asarray(ds.coords[lat_name].values, dtype=float)
        lon_values = np.asarray(ds.coords[lon_name].values, dtype=float)
        if lat_values.size == 0 or lon_values.size == 0:
            return None

        return {
            "lat_name": lat_name,
            "lon_name": lon_name,
            "lat_values": lat_values,
            "lon_values": lon_values,
            "lon_min": float(np.nanmin(lon_values)),
            "lon_max": float(np.nanmax(lon_values)),
        }
    finally:
        ds.close()


def _normalize_lon_for_grid(lon: float, lon_min: float, lon_max: float) -> float:
    if lon_min >= 0 and lon < 0:
        return lon % 360.0
    if lon_max <= 180 and lon > 180:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


@lru_cache(maxsize=65536)
def _nearest_grid_indices(lat_r: float, lon_r: float) -> tuple[int, int] | None:
    grid = _grid_lookup()
    if grid is None:
        return None

    lat_values = grid["lat_values"]
    lon_values = grid["lon_values"]

    lon_grid = _normalize_lon_for_grid(float(lon_r), grid["lon_min"], grid["lon_max"])

    lat_idx = int(np.abs(lat_values - float(lat_r)).argmin())
    lon_idx = int(np.abs(lon_values - lon_grid).argmin())
    return lat_idx, lon_idx


def _normalize_lon_for_dataset(lon: float, lon_coord: xr.DataArray) -> float:
    lon_values = np.asarray(lon_coord.values, dtype=float)
    if lon_values.size == 0:
        return lon

    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    return _normalize_lon_for_grid(lon, lon_min, lon_max)


def _read_point(ds: xr.Dataset, var_name: str, lat: float, lon: float) -> float:
    lat_name, lon_name = _coord_names(ds)
    if lat_name is None or lon_name is None:
        raise KeyError("missing lat/lon coords in dataset")

    da = _pick_time0(ds[var_name])

    nearest_idx = _nearest_grid_indices(round(float(lat), REEF_KEY_DECIMALS), round(float(lon), REEF_KEY_DECIMALS))
    if nearest_idx is not None and lat_name in da.dims and lon_name in da.dims:
        lat_idx, lon_idx = nearest_idx
        try:
            return float(da.isel({lat_name: lat_idx, lon_name: lon_idx}).values)
        except Exception:
            pass

    lon_for_sel = _normalize_lon_for_dataset(float(lon), ds[lon_name])
    selected = da.sel({lat_name: lat, lon_name: lon_for_sel}, method="nearest")
    return float(selected.values)


def _extract_noaa_values_or_none(
    ds_dhw: xr.Dataset,
    ds_hs: xr.Dataset,
    lat: float,
    lon: float,
    dhw_fill: float | None,
    hs_fill: float | None,
) -> tuple[float, float] | None:
    try:
        dhw = _read_point(ds_dhw, "degree_heating_week", lat, lon)
        hs = _read_point(ds_hs, "hotspot", lat, lon)
    except Exception:
        return None

    if _is_bad_value(dhw, dhw_fill):
        return None
    if _is_bad_value(hs, hs_fill):
        return None
    return float(dhw), float(hs)


@lru_cache(maxsize=8192)
def _reef_candidate_info(lat_r: float, lon_r: float) -> tuple[tuple[int, float], ...]:
    distances = _haversine_to_all_points(float(lat_r), float(lon_r))
    point_count = int(distances.shape[0])
    if point_count == 0:
        return tuple()

    limit = min(MAX_REEF_SNAP_CHECKS, point_count)
    if limit <= 0:
        return tuple()

    if limit == point_count:
        order = np.argsort(distances)
    else:
        partial = np.argpartition(distances, limit - 1)[:limit]
        order = partial[np.argsort(distances[partial])]

    return tuple((int(idx), float(distances[idx])) for idx in order)


def snap_to_nearest_valid_reef(
    lat: float,
    lon: float,
    iso_date: str,
    ds_dhw: xr.Dataset,
    ds_hs: xr.Dataset,
    dhw_fill: float | None,
    hs_fill: float | None,
):
    lat_r = round(float(lat), REEF_KEY_DECIMALS)
    lon_r = round(float(lon), REEF_KEY_DECIMALS)

    for idx, _ in _reef_candidate_info(lat_r, lon_r):
        rlat = float(REEF_POINTS[idx, 0])
        rlon = float(REEF_POINTS[idx, 1])
        values = _extract_noaa_values_or_none(ds_dhw, ds_hs, rlat, rlon, dhw_fill, hs_fill)
        if values is None:
            continue

        distance_km = haversine_km(float(lat), float(lon), rlat, rlon)
        return {
            "lat": rlat,
            "lon": rlon,
            "distance_km": float(distance_km),
            "dhw": float(values[0]),
            "hotspot": float(values[1]),
        }

    raise HTTPException(status_code=422, detail=f"no valid reef point found nearby for date {iso_date}.")


@lru_cache(maxsize=8192)
def _nearest_reef_cached(lat_r: float, lon_r: float) -> tuple[float, float, float]:
    candidates = _reef_candidate_info(lat_r, lon_r)
    if not candidates:
        raise HTTPException(status_code=422, detail="reef dataset is empty.")

    idx, cached_distance = candidates[0]
    best_lat = float(REEF_POINTS[idx, 0])
    best_lon = float(REEF_POINTS[idx, 1])
    return best_lat, best_lon, float(cached_distance)


def _available_date_bounds() -> tuple[str | None, str | None]:
    if not AVAILABLE_DATES:
        return None, None
    return AVAILABLE_DATES[0], AVAILABLE_DATES[-1]


@lru_cache(maxsize=8192)
def _available_dates_for_cached(_lat_r: float, _lon_r: float) -> tuple[str, ...]:
    return AVAILABLE_DATES_TUPLE


_must_exist(REEF_FEATURES_PATH, "reef_features.csv")
reef_df = pd.read_csv(REEF_FEATURES_PATH)
if LAT_COL not in reef_df.columns or LON_COL not in reef_df.columns:
    raise RuntimeError(f"reef_features.csv missing lat/lon. found: {reef_df.columns.tolist()}")

REEF_POINTS = np.column_stack(
    [reef_df[LAT_COL].astype(float).to_numpy(), reef_df[LON_COL].astype(float).to_numpy()]
)
REEF_LATS = REEF_POINTS[:, 0]
REEF_LONS = REEF_POINTS[:, 1]

device = torch.device("cpu")


def load_model():
    _must_exist(MODEL_PATH, "model weights (.pth)")
    m = BleachingMLP(4).to(device)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m


model = load_model()


def _model_prob(lat: float, lon: float, dhw: float, hotspot: float) -> float:
    x = torch.tensor([[lat, lon, dhw, hotspot]], dtype=torch.float32)
    with torch.no_grad():
        return float(model(x).cpu().numpy()[0][0])


_ensure_dirs()
AVAILABLE_DATES, NOAA_FILES_BY_DATE, DATES_SOURCE = _load_available_dates_from_manifest()
AVAILABLE_DATES_TUPLE = tuple(AVAILABLE_DATES)


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True, "service": "coral-bleaching-api"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    min_date, max_date = _available_date_bounds()
    return {
        "ok": True,
        "reef_points": int(REEF_POINTS.shape[0]),
        "available_dates": int(len(AVAILABLE_DATES)),
        "min_date": min_date,
        "max_date": max_date,
        "dates_source": DATES_SOURCE,
        "manifest_present": bool(os.path.exists(MANIFEST_PATH)),
        "cached_dhw_files": len([f for f in os.listdir(DHW_DIR) if f.endswith(".nc")]),
        "cached_hs_files": len([f for f in os.listdir(HS_DIR) if f.endswith(".nc")]),
        "xr_engine": XR_ENGINE,
    }


@app.get("/available-dates")
def available_dates():
    min_date, max_date = _available_date_bounds()
    return {
        "count": len(AVAILABLE_DATES),
        "min_date": min_date,
        "max_date": max_date,
        "source": DATES_SOURCE,
        "dates": AVAILABLE_DATES,
    }


@app.get("/available-dates-for")
def available_dates_for(lat: float = Query(...), lon: float = Query(...)):
    lat_r = round(float(lat), REEF_KEY_DECIMALS)
    lon_r = round(float(lon), REEF_KEY_DECIMALS)
    reef_key = f"{lat_r:.{REEF_KEY_DECIMALS}f}|{lon_r:.{REEF_KEY_DECIMALS}f}"
    dates = list(_available_dates_for_cached(lat_r, lon_r))

    min_date = dates[0] if dates else None
    max_date = dates[-1] if dates else None

    return {
        "reef_key": reef_key,
        "lat": lat_r,
        "lon": lon_r,
        "count": len(dates),
        "min_date": min_date,
        "max_date": max_date,
        "dates": dates,
    }


@app.get("/reef-points")
def reef_points(
    south: float = Query(...),
    west: float = Query(...),
    north: float = Query(...),
    east: float = Query(...),
    limit: int = Query(1800, ge=100, le=5000),
):
    south_f = float(min(south, north))
    north_f = float(max(south, north))
    west_f = float(west)
    east_f = float(east)

    lat_mask = (REEF_LATS >= south_f) & (REEF_LATS <= north_f)
    if east_f >= west_f:
        lon_mask = (REEF_LONS >= west_f) & (REEF_LONS <= east_f)
        lon_span = max(east_f - west_f, 0.0001)
    else:
        lon_mask = (REEF_LONS >= west_f) | (REEF_LONS <= east_f)
        lon_span = max((180.0 - west_f) + (east_f + 180.0), 0.0001)

    visible_indices = np.flatnonzero(lat_mask & lon_mask)
    total = int(visible_indices.size)

    if total == 0:
        return {"total": 0, "returned": 0, "points": []}

    if total > limit:
        grid_side = max(1, int(math.sqrt(limit)))
        lat_span = max(north_f - south_f, 0.0001)
        lat_step = max(lat_span / grid_side, 0.12)
        lon_step = max(lon_span / grid_side, 0.12)
        selected: list[int] = []
        seen_bins: set[tuple[int, int]] = set()

        for idx in visible_indices:
            point_lat = float(REEF_LATS[idx])
            point_lon = float(REEF_LONS[idx])
            normalized_lon = point_lon if east_f >= west_f or point_lon >= west_f else point_lon + 360.0
            west_origin = west_f if east_f >= west_f else west_f
            lat_bucket = int((point_lat - south_f) / lat_step)
            lon_bucket = int((normalized_lon - west_origin) / lon_step)
            bucket = (lat_bucket, lon_bucket)

            if bucket in seen_bins:
                continue

            seen_bins.add(bucket)
            selected.append(int(idx))

            if len(selected) >= limit:
                break

        visible_indices = np.asarray(selected, dtype=int)

    points = [
        {"lat": float(REEF_LATS[idx]), "lon": float(REEF_LONS[idx])}
        for idx in visible_indices
    ]
    return {
        "total": total,
        "returned": len(points),
        "points": points,
    }


class EstimateRequest(BaseModel):
    lat: float
    lon: float
    date: str


class FeatureEstimateRequest(BaseModel):
    lat: float
    lon: float
    dhw: float
    hotspot: float


class SensitivityRequest(BaseModel):
    lat: float
    lon: float
    dhw: float
    hotspot: float
    dhw_step: float = 1.0
    hotspot_step: float = 0.5


@app.get("/nearest-reef")
def nearest_reef(lat: float = Query(...), lon: float = Query(...)):
    lat_r = round(float(lat), REEF_KEY_DECIMALS)
    lon_r = round(float(lon), REEF_KEY_DECIMALS)
    best_lat, best_lon, _ = _nearest_reef_cached(lat_r, lon_r)
    best_distance = haversine_km(float(lat), float(lon), best_lat, best_lon)

    return {"lat": best_lat, "lon": best_lon, "distance_km": float(best_distance)}


@app.get("/estimate")
def estimate_risk_get_not_allowed():
    return JSONResponse(
        status_code=405,
        content={
            "error": "method_not_allowed",
            "message": "Use POST /estimate with JSON body: {\"lat\": number, \"lon\": number, \"date\": \"YYYY-MM-DD\"}.",
        },
    )


@app.post("/estimate")
def estimate_risk(req: EstimateRequest):
    try:
        date_obj = pd.to_datetime(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format (use YYYY-MM-DD).")

    requested_iso_date = date_obj.date().isoformat()
    date_obj = pd.to_datetime(requested_iso_date)

    input_lat = float(req.lat)
    input_lon = float(req.lon)

    dhw_path, hs_path = _ensure_noaa_files(date_obj)

    used_lat = input_lat
    used_lon = input_lon
    snapped = False
    snap_km = 0.0

    ds_dhw = _xr_open(dhw_path)
    ds_hs = _xr_open(hs_path)
    try:
        dhw_fill = _fill_value(ds_dhw, "degree_heating_week")
        hs_fill = _fill_value(ds_hs, "hotspot")

        values = _extract_noaa_values_or_none(ds_dhw, ds_hs, used_lat, used_lon, dhw_fill, hs_fill)
        if values is None:
            snap = snap_to_nearest_valid_reef(
                input_lat,
                input_lon,
                requested_iso_date,
                ds_dhw,
                ds_hs,
                dhw_fill,
                hs_fill,
            )
            used_lat = float(snap["lat"])
            used_lon = float(snap["lon"])
            snap_km = float(snap["distance_km"])
            snapped = True
            dhw = float(snap["dhw"])
            hotspot = float(snap["hotspot"])
        else:
            dhw, hotspot = values
            _bad_or_422(dhw, dhw_fill, "dhw")
            _bad_or_422(hotspot, hs_fill, "hotspot")
    finally:
        ds_dhw.close()
        ds_hs.close()

    risk_prob = _model_prob(used_lat, used_lon, dhw, hotspot)
    _finite_or_422(risk_prob, "risk_prob")

    return {
        "input_lat": input_lat,
        "input_lon": input_lon,
        "used_lat": used_lat,
        "used_lon": used_lon,
        "snapped": bool(snapped),
        "snap_km": float(snap_km),
        "date": requested_iso_date,
        "dhw": float(dhw),
        "hotspot": float(hotspot),
        "risk_prob": float(risk_prob),
        "risk_flag": int(risk_prob >= 0.6),
    }


@app.post("/estimate-from-features")
def estimate_from_features(req: FeatureEstimateRequest):
    risk_prob = _model_prob(req.lat, req.lon, req.dhw, req.hotspot)
    _finite_or_422(risk_prob, "risk_prob")
    return {"risk_prob": float(risk_prob), "risk_flag": int(risk_prob >= 0.6)}


@app.post("/sensitivity")
def sensitivity(req: SensitivityRequest):
    base = _model_prob(req.lat, req.lon, req.dhw, req.hotspot)
    p_dhw = _model_prob(req.lat, req.lon, req.dhw + req.dhw_step, req.hotspot)
    p_hot = _model_prob(req.lat, req.lon, req.dhw, req.hotspot + req.hotspot_step)

    _finite_or_422(base, "base")
    _finite_or_422(p_dhw, "p_dhw")
    _finite_or_422(p_hot, "p_hot")

    return {
        "base": float(base),
        "dhw_step": req.dhw_step,
        "hotspot_step": req.hotspot_step,
        "delta_dhw": float(p_dhw - base),
        "delta_hotspot": float(p_hot - base),
    }
