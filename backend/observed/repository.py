from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from backend.config import OBSERVED_SITE_CATALOG_PATH, OBSERVED_SITE_DATE_PATH
from backend.ml.build_dataset import ensure_modeling_dataset
from backend.ml.label_standardization import ensure_standardized_observed_assets


def _parse_json_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    return [str(item) for item in decoded]


@lru_cache(maxsize=1)
def get_site_catalog() -> pd.DataFrame:
    if OBSERVED_SITE_CATALOG_PATH.exists():
        frame = pd.read_csv(
            OBSERVED_SITE_CATALOG_PATH,
            parse_dates=["first_observed_date", "latest_observed_date"],
        )
    else:
        _, frame = ensure_standardized_observed_assets()
    frame["site_id"] = frame["site_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def get_observed_site_dates() -> pd.DataFrame:
    if OBSERVED_SITE_DATE_PATH.exists():
        frame = pd.read_csv(OBSERVED_SITE_DATE_PATH, parse_dates=["date"])
    else:
        frame, _ = ensure_standardized_observed_assets()
    frame["site_id"] = frame["site_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def get_modeling_dataset() -> pd.DataFrame:
    frame = ensure_modeling_dataset()
    frame["site_id"] = frame["site_id"].astype(str)
    return frame


@lru_cache(maxsize=1)
def get_environmental_context_dataset() -> pd.DataFrame:
    frame = get_observed_site_dates().copy()
    frame["month"] = frame["date"].dt.to_period("M")
    grouped = (
        frame.groupby(["site_id", "month"], dropna=False)
        .agg(
            date=("date", "max"),
            hotspot_like=("hotspot_like", "mean"),
            dhw_like=("dhw_like", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            observed_record_count=("sample_row_count", "sum"),
            has_direct_observation=("is_direct_observation", "max"),
            has_derived_label_input=("is_derived_label", "max"),
        )
        .reset_index()
    )
    grouped["site_id"] = grouped["site_id"].astype(str)
    grouped["environmental_context_ready"] = grouped["hotspot_like"].notna() & grouped["dhw_like"].notna()
    return grouped


@lru_cache(maxsize=1)
def _site_lookup_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    catalog = get_site_catalog()
    site_ids = catalog["site_id"].astype(str).to_numpy()
    latitudes = catalog["latitude"].astype(float).to_numpy()
    longitudes = catalog["longitude"].astype(float).to_numpy()
    return site_ids, latitudes, longitudes


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def _haversine_to_all(lat: float, lon: float) -> np.ndarray:
    _, latitudes, longitudes = _site_lookup_arrays()
    radius_km = 6371.0
    phi1 = np.radians(lat)
    phi2 = np.radians(latitudes)
    d_phi = np.radians(latitudes - lat)
    d_lambda = np.radians(longitudes - lon)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def find_nearest_site(lat: float, lon: float) -> dict[str, Any]:
    site_ids, latitudes, longitudes = _site_lookup_arrays()
    distances = _haversine_to_all(lat, lon)
    index = int(np.argmin(distances))
    return {
        "site_id": str(site_ids[index]),
        "latitude": float(latitudes[index]),
        "longitude": float(longitudes[index]),
        "distance_km": float(distances[index]),
    }


def get_site_metadata(site_id: str) -> dict[str, Any]:
    catalog = get_site_catalog()
    match = catalog.loc[catalog["site_id"].astype(str) == str(site_id)]
    if match.empty:
        raise KeyError(f"Unknown site_id: {site_id}")

    row = match.iloc[0].to_dict()
    row["site_id"] = str(row["site_id"])
    row["provenance_sources"] = _parse_json_list(row.get("provenance_sources"))
    return row


def get_site_observations(site_id: str) -> pd.DataFrame:
    frame = get_observed_site_dates()
    site_frame = frame.loc[frame["site_id"].astype(str) == str(site_id)].copy()
    if site_frame.empty:
        raise KeyError(f"Unknown site_id: {site_id}")
    return site_frame.sort_values("date", ascending=False).reset_index(drop=True)


def get_site_month_records(site_id: str) -> pd.DataFrame:
    frame = get_modeling_dataset()
    site_frame = frame.loc[frame["site_id"].astype(str) == str(site_id)].copy()
    if site_frame.empty:
        raise KeyError(f"Unknown site_id: {site_id}")
    return site_frame.sort_values("date", ascending=False).reset_index(drop=True)


def get_site_environmental_month_records(site_id: str) -> pd.DataFrame:
    frame = get_environmental_context_dataset()
    site_frame = frame.loc[frame["site_id"].astype(str) == str(site_id)].copy()
    if site_frame.empty:
        raise KeyError(f"Unknown site_id: {site_id}")
    return site_frame.sort_values("date", ascending=False).reset_index(drop=True)


def recommended_observed_date(site_id: str) -> str | None:
    site_frame = get_site_observations(site_id)
    # Prefer the newest observation that is also usable by the analysis layers.
    # If none qualify, fall back to the newest direct observation instead of
    # leaving the UI on a date that cannot render anything meaningful.
    analysis_ready = site_frame.loc[site_frame["recommended_for_modeling"].fillna(False)]
    if not analysis_ready.empty:
        return pd.to_datetime(analysis_ready.iloc[0]["date"]).date().isoformat()

    direct_only = site_frame.loc[site_frame["is_direct_observation"].fillna(False)]
    if not direct_only.empty:
        return pd.to_datetime(direct_only.iloc[0]["date"]).date().isoformat()

    return None


def find_historical_context(
    site_id: str,
    requested_date: str | None = None,
    *,
    require_model_eligible: bool = True,
) -> dict[str, Any] | None:
    try:
        frame = get_site_month_records(site_id) if require_model_eligible else get_site_environmental_month_records(site_id)
    except KeyError:
        return None

    if frame.empty:
        return None

    if requested_date:
        requested = pd.to_datetime(requested_date, errors="coerce")
        if pd.notna(requested):
            frame = frame.loc[frame["date"] <= requested]
    valid = frame.loc[frame["hotspot_like"].notna() & frame["dhw_like"].notna()]
    if require_model_eligible:
        valid = valid.loc[valid["target_eligible"].fillna(False)]
    else:
        valid = valid.loc[valid["environmental_context_ready"].fillna(False)]
    if valid.empty:
        return None

    row = valid.iloc[0].to_dict()
    row["mode"] = "historical_forecast" if require_model_eligible else "historical_environmental"
    row["used_date"] = pd.to_datetime(row["date"]).date().isoformat()
    return row


def _sample_bbox(frame: pd.DataFrame, south: float, west: float, north: float, east: float, limit: int) -> pd.DataFrame:
    south_f = min(south, north)
    north_f = max(south, north)
    if east >= west:
        subset = frame.loc[
            (frame["latitude"] >= south_f)
            & (frame["latitude"] <= north_f)
            & (frame["longitude"] >= west)
            & (frame["longitude"] <= east)
        ].copy()
    else:
        subset = frame.loc[
            (frame["latitude"] >= south_f)
            & (frame["latitude"] <= north_f)
            & ((frame["longitude"] >= west) | (frame["longitude"] <= east))
        ].copy()

    if len(subset) <= limit:
        return subset

    lat_span = max(north_f - south_f, 0.0001)
    lon_span = max((east - west) if east >= west else (180 - west) + (east + 180), 0.0001)
    grid_side = max(1, int(math.sqrt(limit)))
    lat_step = max(lat_span / grid_side, 0.25)
    lon_step = max(lon_span / grid_side, 0.25)

    subset["lat_bucket"] = ((subset["latitude"] - south_f) / lat_step).astype(int)
    normalized_lon = subset["longitude"].where(subset["longitude"] >= west, subset["longitude"] + 360.0)
    subset["lon_bucket"] = ((normalized_lon - west) / lon_step).astype(int)
    subset["quality_rank"] = subset["observed_record_count"].fillna(0) + subset["observed_positive_count"].fillna(0)
    sampled = (
        subset.sort_values("quality_rank", ascending=False)
        .drop_duplicates(subset=["lat_bucket", "lon_bucket"])
        .head(limit)
        .drop(columns=["lat_bucket", "lon_bucket", "quality_rank"])
    )
    return sampled


def list_sites_in_bbox(
    south: float,
    west: float,
    north: float,
    east: float,
    limit: int = 1200,
) -> dict[str, Any]:
    catalog = get_site_catalog()
    sampled = _sample_bbox(catalog.copy(), south=south, west=west, north=north, east=east, limit=limit)
    sampled = sampled.sort_values("latest_observed_date", ascending=False)

    points = []
    for _, row in sampled.iterrows():
        points.append(
            {
                "site_id": str(row["site_id"]),
                "display_name": row["display_name"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "latest_observed_date": pd.to_datetime(row["latest_observed_date"]).date().isoformat()
                if pd.notna(row["latest_observed_date"])
                else None,
                "observed_record_count": int(row["observed_record_count"]),
                "observed_positive_count": int(row["observed_positive_count"]),
                "mean_label_quality_score": float(row["mean_label_quality_score"]),
            }
        )

    return {
        "total": int(len(catalog)),
        "returned": int(len(points)),
        "points": points,
    }
