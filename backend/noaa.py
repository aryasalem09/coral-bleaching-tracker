from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests
import xarray as xr

from backend.config import AUTO_DOWNLOAD_NOAA, NOAA_DHW_DIR, NOAA_HS_DIR, NOAA_MANIFEST_PATH, XR_ENGINE

DHW_BASE_URL = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
    "5km/v3.1_op/nc/v1.0/daily/dhw"
)
HS_BASE_URL = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
    "5km/v3.1_op/nc/v1.0/daily/hs"
)
DATE_FILE_RE_DHW = re.compile(r"^ct5km_dhw_v3\.1_(\d{8})\.nc$")
DATE_FILE_RE_HS = re.compile(r"^ct5km_hs_v3\.1_(\d{8})\.nc$")


def _file_non_empty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _default_paths_for_date(iso_date: str) -> dict[str, Path]:
    ymd = iso_date.replace("-", "")
    return {
        "dhw": NOAA_DHW_DIR / f"ct5km_dhw_v3.1_{ymd}.nc",
        "hs": NOAA_HS_DIR / f"ct5km_hs_v3.1_{ymd}.nc",
    }


def _normalize_iso_date(value: str) -> str | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _resolve_manifest_path(path_value: Any) -> Path | None:
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (NOAA_MANIFEST_PATH.parent.parent.parent / path).resolve()


def _scan_local_dates_and_paths() -> tuple[list[str], dict[str, dict[str, Path]]]:
    dhw_by_date: dict[str, Path] = {}
    hs_by_date: dict[str, Path] = {}

    for directory, pattern, target in [
        (NOAA_DHW_DIR, DATE_FILE_RE_DHW, dhw_by_date),
        (NOAA_HS_DIR, DATE_FILE_RE_HS, hs_by_date),
    ]:
        if not directory.exists():
            continue
        for file_path in directory.iterdir():
            match = pattern.match(file_path.name)
            if not match or not _file_non_empty(file_path):
                continue
            iso_date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
            target[iso_date] = file_path

    dates = sorted(set(dhw_by_date).intersection(hs_by_date))
    paths = {iso_date: {"dhw": dhw_by_date[iso_date], "hs": hs_by_date[iso_date]} for iso_date in dates}
    return dates, paths


@lru_cache(maxsize=1)
def available_noaa_dates() -> tuple[list[str], dict[str, dict[str, Path]], str]:
    if not NOAA_MANIFEST_PATH.exists():
        dates, paths = _scan_local_dates_and_paths()
        return dates, paths, "scan"

    try:
        manifest = json.loads(NOAA_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        dates, paths = _scan_local_dates_and_paths()
        return dates, paths, "scan"

    ok_dates = manifest.get("ok_dates", [])
    by_date = manifest.get("by_date", {})
    paths: dict[str, dict[str, Path]] = {}
    collected_dates: list[str] = []

    for raw_date in ok_dates:
        iso_date = _normalize_iso_date(str(raw_date))
        if not iso_date:
            continue
        defaults = _default_paths_for_date(iso_date)
        entry = by_date.get(iso_date, {}) if isinstance(by_date, dict) else {}
        dhw_path = _resolve_manifest_path(entry.get("dhw")) if isinstance(entry, dict) else None
        hs_path = _resolve_manifest_path(entry.get("hs")) if isinstance(entry, dict) else None
        dhw_file = dhw_path or defaults["dhw"]
        hs_file = hs_path or defaults["hs"]
        if _file_non_empty(dhw_file) and _file_non_empty(hs_file):
            collected_dates.append(iso_date)
            paths[iso_date] = {"dhw": dhw_file, "hs": hs_file}

    if collected_dates:
        return sorted(collected_dates), paths, "manifest"
    dates, scanned_paths = _scan_local_dates_and_paths()
    return dates, scanned_paths, "scan"


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    target_path.write_bytes(response.content)


def _download_for_date(iso_date: str) -> dict[str, Path]:
    day = datetime.strptime(iso_date, "%Y-%m-%d").date()
    defaults = _default_paths_for_date(iso_date)
    year = day.strftime("%Y")
    ymd = day.strftime("%Y%m%d")
    downloads = {
        "dhw": f"{DHW_BASE_URL}/{year}/ct5km_dhw_v3.1_{ymd}.nc",
        "hs": f"{HS_BASE_URL}/{year}/ct5km_hs_v3.1_{ymd}.nc",
    }
    for kind, url in downloads.items():
        target = defaults[kind]
        if not _file_non_empty(target):
            _download_file(url, target)
    available_noaa_dates.cache_clear()
    return defaults


def _ensure_paths_for_date(iso_date: str) -> dict[str, Path]:
    dates, paths, _ = available_noaa_dates()
    if iso_date in paths:
        return paths[iso_date]
    defaults = _default_paths_for_date(iso_date)
    if _file_non_empty(defaults["dhw"]) and _file_non_empty(defaults["hs"]):
        return defaults
    if AUTO_DOWNLOAD_NOAA:
        return _download_for_date(iso_date)
    raise FileNotFoundError(f"No local NOAA daily files are available for {iso_date}.")


def _xr_open(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine=XR_ENGINE)
    except Exception:
        return xr.open_dataset(path)


def _coord_names(ds: xr.Dataset) -> tuple[str | None, str | None]:
    lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    return lat_name, lon_name


def _pick_time0(da: xr.DataArray) -> xr.DataArray:
    return da.isel(time=0) if "time" in da.dims else da


def _normalize_lon(lon: float, lon_values: np.ndarray) -> float:
    lon_min = float(np.nanmin(lon_values))
    lon_max = float(np.nanmax(lon_values))
    if lon_min >= 0 and lon < 0:
        return lon % 360.0
    if lon_max <= 180 and lon > 180:
        return ((lon + 180.0) % 360.0) - 180.0
    return lon


def _is_bad_value(value: float) -> bool:
    if not np.isfinite(value):
        return True
    return value in {-9999.0, -32768.0, 32767.0} or abs(value) > 1e6


def _search_nearest_valid_point(ds_dhw: xr.Dataset, ds_hs: xr.Dataset, lat: float, lon: float) -> dict[str, float]:
    lat_name, lon_name = _coord_names(ds_dhw)
    if lat_name is None or lon_name is None:
        raise ValueError("NOAA dataset is missing latitude/longitude coordinates.")

    dhw = _pick_time0(ds_dhw["degree_heating_week"])
    hs = _pick_time0(ds_hs["hotspot"])
    lat_values = np.asarray(ds_dhw.coords[lat_name].values, dtype=float)
    lon_values = np.asarray(ds_dhw.coords[lon_name].values, dtype=float)
    lon_for_grid = _normalize_lon(float(lon), lon_values)

    lat_index = int(np.abs(lat_values - float(lat)).argmin())
    lon_index = int(np.abs(lon_values - lon_for_grid).argmin())

    best: dict[str, float] | None = None
    best_distance = float("inf")
    for radius in range(0, 4):
        lat_min = max(0, lat_index - radius)
        lat_max = min(len(lat_values) - 1, lat_index + radius)
        lon_min = max(0, lon_index - radius)
        lon_max = min(len(lon_values) - 1, lon_index + radius)
        for i in range(lat_min, lat_max + 1):
            for j in range(lon_min, lon_max + 1):
                dhw_value = float(dhw.isel({lat_name: i, lon_name: j}).values)
                hotspot_value = float(hs.isel({lat_name: i, lon_name: j}).values)
                if _is_bad_value(dhw_value) or _is_bad_value(hotspot_value):
                    continue
                used_lat = float(lat_values[i])
                used_lon = float(lon_values[j])
                distance = haversine_km(lat, lon, used_lat, used_lon)
                if distance < best_distance:
                    best_distance = distance
                    best = {
                        "dhw": dhw_value,
                        "hotspot": hotspot_value,
                        "used_lat": used_lat,
                        "used_lon": used_lon,
                        "snap_km": distance,
                        "snapped": bool(distance > 0.05),
                    }
        if best is not None:
            return best
    raise ValueError("No valid NOAA ocean grid cell was found near the selected site.")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def get_site_environmental_features(lat: float, lon: float, iso_date: str) -> dict[str, float | str | bool]:
    paths = _ensure_paths_for_date(iso_date)
    ds_dhw = _xr_open(paths["dhw"])
    ds_hs = _xr_open(paths["hs"])
    try:
        values = _search_nearest_valid_point(ds_dhw, ds_hs, lat=lat, lon=lon)
    finally:
        ds_dhw.close()
        ds_hs.close()

    return {
        **values,
        "date": iso_date,
        "mode": "noaa_live",
    }
