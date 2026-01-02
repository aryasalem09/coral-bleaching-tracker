import os
import math
import re
from datetime import datetime, date
from functools import lru_cache
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
import torch
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .model import BleachingMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")

DHW_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_dhw")
HS_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_hs")

REEF_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
LAT_COL = "lat"
LON_COL = "lon"

NOAA_DHW_BASE = os.getenv(
    "NOAA_DHW_BASE",
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/dhw",
)
NOAA_HS_BASE = os.getenv(
    "NOAA_HS_BASE",
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/hs",
)

# years scaped
NOAA_YEARS_BACK = int(os.getenv("NOAA_YEARS_BACK", "2"))

app = FastAPI(title="Coral Bleaching Risk Estimator")

raw_origins = os.getenv(
    "CORS_ORIGINS",
    "https://aryasalem09.github.io,http://localhost:5173,http://127.0.0.1:5173",
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
    # writable render disk
    os.makedirs(DHW_DIR, exist_ok=True)
    os.makedirs(HS_DIR, exist_ok=True)

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

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

def _read_point(ds: xr.Dataset, var_name: str, lat: float, lon: float) -> float:
    lat_name, lon_name = _coord_names(ds)
    if lat_name is None or lon_name is None:
        raise KeyError("missing lat/lon coords in dataset")
    da = ds[var_name].sel({lat_name: lat, lon_name: lon}, method="nearest")
    da = _pick_time0(da)
    return float(da.values)

_must_exist(REEF_FEATURES_PATH, "reef_features.csv")
reef_df = pd.read_csv(REEF_FEATURES_PATH)
if LAT_COL not in reef_df.columns or LON_COL not in reef_df.columns:
    raise RuntimeError(f"reef_features.csv missing lat/lon. found: {reef_df.columns.tolist()}")

REEF_POINTS = np.column_stack(
    [reef_df[LAT_COL].astype(float).to_numpy(), reef_df[LON_COL].astype(float).to_numpy()]
)

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

_DATE_RE = re.compile(r"(\d{8})")

def _http_get_text(url: str) -> str:
    # scraper
    req = Request(url, headers={"User-Agent": "coral-bleaching-tracker/1.0"})
    with urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", errors="ignore")

def _scrape_year_dates(base_url: str, year: int, prefix: str) -> set[str]:
    # YYYYMMDD
    url = f"{base_url}/{year}/"
    try:
        html = _http_get_text(url)
    except Exception:
        return set()

    #  ct5km_hs_v3.1_20250101.nc, ct5km_dhw_v3.1_20250101.nc, etc. :contentReference[oaicite:1]{index=1}
    pat = re.compile(rf"{re.escape(prefix)}_(\d{{8}})\.nc")
    out = set()
    for m in pat.finditer(html):
        ymd = m.group(1)
        try:
            out.add(datetime.strptime(ymd, "%Y%m%d").date().isoformat())
        except Exception:
            pass
    return out

def _refresh_available_dates() -> list[str]:
    # N years
    today = date.today()
    years = list(range(today.year - NOAA_YEARS_BACK, today.year + 1))

    dhw = set()
    hs = set()
    for y in years:
        dhw |= _scrape_year_dates(NOAA_DHW_BASE, y, "ct5km_dhw_v3.1")
        hs |= _scrape_year_dates(NOAA_HS_BASE, y, "ct5km_hs_v3.1")

    return sorted(dhw.intersection(hs))

def _download_file(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # cached
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    req = Request(url, headers={"User-Agent": "coral-bleaching-tracker/1.0"})
    try:
        with urlopen(req, timeout=60) as r:
            data = r.read()
    except HTTPError as e:
        raise HTTPException(status_code=404, detail=f"noaa file missing: {url} ({e.code})")
    except URLError:
        raise HTTPException(status_code=502, detail="failed to reach noaa host (network issue).")

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)

def _ensure_noaa_files(date_obj: pd.Timestamp):
    y = int(date_obj.strftime("%Y"))
    ymd = date_obj.strftime("%Y%m%d")

    dhw_fn = f"ct5km_dhw_v3.1_{ymd}.nc"
    hs_fn = f"ct5km_hs_v3.1_{ymd}.nc"

    dhw_path = os.path.join(DHW_DIR, dhw_fn)
    hs_path = os.path.join(HS_DIR, hs_fn)

    # noaa dir .../dhw/YYYY/file.nc :contentReference[oaicite:2]{index=2}
    dhw_url = f"{NOAA_DHW_BASE}/{y}/{dhw_fn}"
    hs_url = f"{NOAA_HS_BASE}/{y}/{hs_fn}"

    _download_file(dhw_url, dhw_path)
    _download_file(hs_url, hs_path)

    return dhw_path, hs_path

_ensure_dirs()
AVAILABLE_DATES = _refresh_available_dates()

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"ok": True, "service": "coral-bleaching-api"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "ok": True,
        "reef_points": int(REEF_POINTS.shape[0]),
        "available_dates": int(len(AVAILABLE_DATES)),
        "cached_dhw_files": len([f for f in os.listdir(DHW_DIR) if f.endswith(".nc")]),
        "cached_hs_files": len([f for f in os.listdir(HS_DIR) if f.endswith(".nc")]),
    }

@app.get("/available-dates")
def available_dates():
    # last 1000 so you don't blast the client
    return {"count": len(AVAILABLE_DATES), "dates": AVAILABLE_DATES[-1000:]}

def get_noaa_values(lat: float, lon: float, date_obj: pd.Timestamp):
    # pull files if needed, then read locally
    dhw_path, hs_path = _ensure_noaa_files(date_obj)

    ds_dhw = xr.open_dataset(dhw_path)
    ds_hs = xr.open_dataset(hs_path)
    try:
        dhw = _read_point(ds_dhw, "degree_heating_week", lat, lon)
        hs = _read_point(ds_hs, "hotspot", lat, lon)
        return float(dhw), float(hs)
    finally:
        ds_dhw.close()
        ds_hs.close()

def _is_valid_point_for_date(lat: float, lon: float, iso_date: str) -> bool:
    try:
        d = pd.to_datetime(iso_date)
    except Exception:
        return False

    try:
        dhw_path, hs_path = _ensure_noaa_files(d)
    except HTTPException:
        return False

    ds_dhw = xr.open_dataset(dhw_path)
    ds_hs = xr.open_dataset(hs_path)
    try:
        dhw_fill = _fill_value(ds_dhw, "degree_heating_week")
        hs_fill = _fill_value(ds_hs, "hotspot")

        dhw = _read_point(ds_dhw, "degree_heating_week", lat, lon)
        hs = _read_point(ds_hs, "hotspot", lat, lon)

        if _is_bad_value(dhw, dhw_fill):
            return False
        if _is_bad_value(hs, hs_fill):
            return False
        return True
    except Exception:
        return False
    finally:
        ds_dhw.close()
        ds_hs.close()

@lru_cache(maxsize=512)
def _available_dates_for_cached(lat_r: float, lon_r: float) -> tuple[str, ...]:
    good = []
    for d in AVAILABLE_DATES:
        if _is_valid_point_for_date(lat_r, lon_r, d):
            good.append(d)
    return tuple(good)

@app.get("/available-dates-for")
def available_dates_for(lat: float = Query(...), lon: float = Query(...)):
    lat_r = round(float(lat), 3)
    lon_r = round(float(lon), 3)
    ds = list(_available_dates_for_cached(lat_r, lon_r))
    return {"lat": lat_r, "lon": lon_r, "count": len(ds), "dates": ds}

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
    la = float(lat)
    lo = float(lon)

    r = 6371.0
    p1 = np.radians(la)
    p2 = np.radians(REEF_POINTS[:, 0])
    dphi = np.radians(REEF_POINTS[:, 0] - la)
    dl = np.radians(REEF_POINTS[:, 1] - lo)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(a))

    idx = int(np.argmin(d))
    best_lat = float(REEF_POINTS[idx, 0])
    best_lon = float(REEF_POINTS[idx, 1])
    best_d = float(d[idx])

    return {"lat": best_lat, "lon": best_lon, "distance_km": best_d}

@app.post("/estimate")
def estimate_risk(req: EstimateRequest):
    try:
        d = pd.to_datetime(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid date format (use yyyy-mm-dd).")

    dhw, hotspot = get_noaa_values(req.lat, req.lon, d)
    risk_prob = _model_prob(req.lat, req.lon, dhw, hotspot)
    return {
        "lat": req.lat,
        "lon": req.lon,
        "date": req.date,
        "dhw": dhw,
        "hotspot": hotspot,
        "risk_prob": risk_prob,
        "risk_flag": int(risk_prob >= 0.6),
    }

@app.post("/estimate-from-features")
def estimate_from_features(req: FeatureEstimateRequest):
    risk_prob = _model_prob(req.lat, req.lon, req.dhw, req.hotspot)
    return {"risk_prob": risk_prob, "risk_flag": int(risk_prob >= 0.6)}

@app.post("/sensitivity")
def sensitivity(req: SensitivityRequest):
    base = _model_prob(req.lat, req.lon, req.dhw, req.hotspot)
    p_dhw = _model_prob(req.lat, req.lon, req.dhw + req.dhw_step, req.hotspot)
    p_hot = _model_prob(req.lat, req.lon, req.dhw, req.hotspot + req.hotspot_step)
    return {
        "base": base,
        "dhw_step": req.dhw_step,
        "hotspot_step": req.hotspot_step,
        "delta_dhw": p_dhw - base,
        "delta_hotspot": p_hot - base,
    }
