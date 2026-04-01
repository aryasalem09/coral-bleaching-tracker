from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr

from backend.config import XR_ENGINE
from backend.noaa_index import NoaaAvailabilityIndex, get_noaa_weekly_index
from backend.noaa_products import get_product_spec


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _extract_arrays(
    ds_dhw: xr.Dataset,
    ds_hs: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_name, lon_name = _coord_names(ds_dhw)
    if lat_name is None or lon_name is None:
        raise ValueError("NOAA dataset is missing latitude/longitude coordinates.")

    dhw_name = get_product_spec("dhw").variable_name
    hs_name = get_product_spec("hs").variable_name

    dhw_da = _pick_time0(ds_dhw[dhw_name]).transpose(lat_name, lon_name)
    hs_da = _pick_time0(ds_hs[hs_name]).transpose(lat_name, lon_name)

    lat_values = np.asarray(ds_dhw.coords[lat_name].values, dtype=float)
    lon_values = np.asarray(ds_dhw.coords[lon_name].values, dtype=float)
    dhw_values = np.asarray(dhw_da.values, dtype=float)
    hs_values = np.asarray(hs_da.values, dtype=float)
    return dhw_values, hs_values, lat_values, lon_values


def _sample_from_arrays(
    dhw_values: np.ndarray,
    hs_values: np.ndarray,
    lat_values: np.ndarray,
    lon_values: np.ndarray,
    *,
    lat: float,
    lon: float,
) -> dict[str, float]:
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
                dhw_value = float(dhw_values[i, j])
                hotspot_value = float(hs_values[i, j])
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


def sample_site_environmental_context(
    lat: float,
    lon: float,
    iso_date: str,
    *,
    index: NoaaAvailabilityIndex | None = None,
) -> dict[str, float | str | bool]:
    availability = index or get_noaa_weekly_index()
    paths = availability.get_paths_for_date(iso_date)
    if paths is None:
        raise FileNotFoundError(f"No paired weekly NOAA files are available for {iso_date}.")

    ds_dhw = _xr_open(paths["dhw"])
    ds_hs = _xr_open(paths["hs"])
    try:
        arrays = _extract_arrays(ds_dhw, ds_hs)
        values = _sample_from_arrays(*arrays, lat=float(lat), lon=float(lon))
    finally:
        ds_dhw.close()
        ds_hs.close()

    return {
        **values,
        "date": iso_date,
        "mode": "noaa_weekly_monday",
    }


def sample_points_for_date(
    iso_date: str,
    points: Iterable[dict[str, Any]],
    *,
    index: NoaaAvailabilityIndex | None = None,
) -> dict[Any, dict[str, float | str | bool]]:
    availability = index or get_noaa_weekly_index()
    paths = availability.get_paths_for_date(iso_date)
    if paths is None:
        raise FileNotFoundError(f"No paired weekly NOAA files are available for {iso_date}.")

    ds_dhw = _xr_open(paths["dhw"])
    ds_hs = _xr_open(paths["hs"])
    try:
        arrays = _extract_arrays(ds_dhw, ds_hs)
        sampled: dict[Any, dict[str, float | str | bool]] = {}
        for point in points:
            point_id = point["point_id"]
            try:
                sampled[point_id] = _sample_from_arrays(
                    *arrays,
                    lat=float(point["latitude"]),
                    lon=float(point["longitude"]),
                )
            except Exception as exc:
                sampled[point_id] = {
                    "error": str(exc),
                }
        return sampled
    finally:
        ds_dhw.close()
        ds_hs.close()
