from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
from pydantic import BaseModel

from backend.config import APP_VERSION, ensure_directories
from backend.ml.model_registry import get_model_runtime_status, load_model_info, load_model_metrics
from backend.ml.predict import predict_event_probability
from backend.noaa import (
    available_noaa_dates,
    get_site_environmental_features,
    get_site_weekly_feature_context,
    nearest_previous_noaa_monday,
    noaa_coverage_summary,
)
from backend.observed.repository import (
    find_historical_context,
    find_nearest_site,
    get_site_metadata,
    get_site_month_records,
    get_site_observations,
    list_sites_in_bbox,
    recommended_observed_date,
)
from backend.risk.explain import explain_risk
from backend.risk.scoring import score_environmental_risk
from backend.risk.thresholds import RISK_THRESHOLDS

ensure_directories()

app = FastAPI(title="Coral Bleaching Tracker API", version=APP_VERSION)
STARTED_AT = datetime.now(timezone.utc).isoformat()

app.add_middleware(GZipMiddleware, minimum_size=700)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SiteAnalysisRequest(BaseModel):
    site_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    date: Optional[str] = None
    prefer_live: bool = True


def _resolve_site(request: SiteAnalysisRequest | None = None, site_id: str | None = None) -> dict[str, Any]:
    resolved_site_id = site_id or (request.site_id if request else None)
    if resolved_site_id:
        try:
            return get_site_metadata(str(resolved_site_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if request and request.lat is not None and request.lon is not None:
        nearest = find_nearest_site(request.lat, request.lon)
        metadata = get_site_metadata(nearest["site_id"])
        metadata["distance_km_from_request"] = nearest["distance_km"]
        return metadata

    raise HTTPException(status_code=422, detail="Provide either site_id or lat/lon.")


def _resolve_dynamic_features(
    site: dict[str, Any],
    request: SiteAnalysisRequest,
    *,
    require_model_eligible: bool,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    requested_date = request.date

    if request.prefer_live:
        live_dates, _, _ = available_noaa_dates()
        if live_dates:
            candidate_date = request.date or live_dates[-1]
            try:
                used_live_date = nearest_previous_noaa_monday(candidate_date)
                if used_live_date is None:
                    raise FileNotFoundError("No weekly NOAA Monday date was available at or before the requested date.")
                current_live = get_site_environmental_features(
                    float(site["latitude"]),
                    float(site["longitude"]),
                    str(used_live_date),
                )
                if requested_date and used_live_date != requested_date:
                    warnings.append(
                        f"Requested date {requested_date} did not have a local weekly NOAA Monday file; "
                        f"used {used_live_date} instead."
                    )
                current_live["requested_date"] = requested_date
                return current_live, warnings
            except Exception as exc:
                if requested_date:
                    warnings.append(f"Fell back from live weekly NOAA data: {exc}")

    historical = find_historical_context(
        str(site["site_id"]),
        requested_date=requested_date,
        require_model_eligible=require_model_eligible,
    )
    if historical is not None:
        historical["requested_date"] = requested_date
        return historical, warnings

    raise HTTPException(
        status_code=404,
        detail=(
            "No valid environmental context was available for this site. "
            if not require_model_eligible
            else "No valid model-ready environmental context was available for this site. "
        )
        + "The requested live NOAA date was unavailable and no historical fallback row qualified.",
    )


def _prediction_features(site: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any]:
    used_date = dynamic.get("used_date") or dynamic.get("date")
    if not used_date:
        raise HTTPException(status_code=500, detail="Prediction context is missing a usable date.")

    month = datetime.fromisoformat(str(used_date)).month
    angle = (month - 1) * 2 * math.pi / 12.0
    return {
        "latitude": site.get("latitude"),
        "longitude": site.get("longitude"),
        "distance_to_shore_km": site.get("distance_to_shore_km"),
        "turbidity": site.get("turbidity"),
        "cyclone_frequency": site.get("cyclone_frequency"),
        "depth_mean_m": site.get("depth_mean_m"),
        "month_sin": math.sin(angle),
        "month_cos": math.cos(angle),
        "exposure": site.get("exposure"),
        **dynamic,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {"ok": True, "service": "coral-bleaching-api", "version": APP_VERSION}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "coral-bleaching-api",
        "started_at": STARTED_AT,
        "version": APP_VERSION,
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    live_dates, _, live_source = available_noaa_dates()
    weekly_coverage = noaa_coverage_summary()
    model_runtime = get_model_runtime_status()
    return {
        "service": "coral-bleaching-api",
        "version": APP_VERSION,
        "started_at": STARTED_AT,
        "model_ready": bool(model_runtime["ready"]),
        "model_status": model_runtime["status"],
        "model_status_message": model_runtime["message"],
        "live_noaa_dates_available": len(live_dates),
        "latest_live_noaa_date": live_dates[-1] if live_dates else None,
        "live_date_source": live_source,
        "live_noaa_schedule": "weekly_mondays",
        "live_noaa_first_date": weekly_coverage.get("paired_first_date"),
    }


@app.get("/api/sites")
def sites(
    south: float = Query(...),
    west: float = Query(...),
    north: float = Query(...),
    east: float = Query(...),
    limit: int = Query(1200, ge=100, le=4000),
) -> dict[str, Any]:
    return list_sites_in_bbox(south=south, west=west, north=north, east=east, limit=limit)


@app.get("/api/site/{site_id}")
def site_detail(site_id: str) -> dict[str, Any]:
    try:
        site = get_site_metadata(site_id)
        observations = get_site_observations(site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "site": {
            **site,
            "first_observed_date": site["first_observed_date"].date().isoformat()
            if site.get("first_observed_date") is not None
            else None,
            "latest_observed_date": site["latest_observed_date"].date().isoformat()
            if site.get("latest_observed_date") is not None
            else None,
        },
        "recommended_observed_date": recommended_observed_date(site_id),
        "observed_dates": [
            pd_timestamp.date().isoformat()
            for pd_timestamp in observations["date"].dropna().sort_values(ascending=False).tolist()
        ],
    }


@app.get("/api/site/{site_id}/observations")
def site_observations(site_id: str) -> dict[str, Any]:
    try:
        frame = get_site_observations(site_id)
        site = get_site_metadata(site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    records = []
    for _, row in frame.iterrows():
        records.append(
            {
                "date": row["date"].date().isoformat(),
                "observed_percent_bleaching": None if pd.isna(row["observed_percent_bleaching"]) else float(row["observed_percent_bleaching"]),
                "observed_severity_category": None
                if pd.isna(row.get("observed_severity_category"))
                else str(row.get("observed_severity_category")),
                "target_bleaching_event": None if pd.isna(row["target_bleaching_event"]) else int(row["target_bleaching_event"]),
                "label_quality_score": float(row["label_quality_score"]),
                "is_direct_observation": bool(row["is_direct_observation"]),
                "is_derived_label": bool(row["is_derived_label"]),
                "has_conflict_history": bool(row["has_conflict_history"]),
                "sample_row_count": int(row["sample_row_count"]),
                "source_count": int(row["source_count"]),
                "recommended_for_modeling": bool(row.get("recommended_for_modeling", False)),
                "provenance_sources": json.loads(row["provenance_sources"]) if row["provenance_sources"] else [],
            }
        )

    return {
        "site_id": site_id,
        "display_name": site["display_name"],
        "recommended_date": recommended_observed_date(site_id),
        "records": records,
    }


@app.get("/api/risk/info")
def risk_info() -> dict[str, Any]:
    return {
        "layer_name": "Environmental Stress Outlook",
        "definition": (
            "A transparent heat-stress score based on hotspot-like temperature stress and "
            "accumulated heat stress. It is not a confirmed bleaching observation."
        ),
        "thresholds": [threshold.__dict__ for threshold in RISK_THRESHOLDS],
        "fallback_behavior": (
            "The API uses local weekly Monday NOAA files when available and moves backward to the nearest valid Monday "
            "on or before the requested date. If that weekly context is unavailable, it falls back to the newest valid "
            "historical site-month environmental record."
        ),
    }


@app.post("/api/risk/score")
def risk_score(request: SiteAnalysisRequest) -> dict[str, Any]:
    site = _resolve_site(request=request)
    dynamic, warnings = _resolve_dynamic_features(site, request, require_model_eligible=False)
    hotspot = float(dynamic.get("hotspot", dynamic.get("hotspot_like")))
    dhw = float(dynamic.get("dhw", dynamic.get("dhw_like")))
    risk = score_environmental_risk(hotspot=hotspot, dhw=dhw)

    return {
        "site_id": site["site_id"],
        "display_name": site["display_name"],
        "requested_date": request.date,
        "used_date": dynamic.get("used_date") or dynamic.get("date"),
        "mode": dynamic.get("mode"),
        "hotspot": hotspot,
        "dhw": dhw,
        "category": risk.category,
        "score": risk.score,
        "color": risk.color,
        "explanation": explain_risk(risk),
        "used_latitude": dynamic.get("used_lat", site["latitude"]),
        "used_longitude": dynamic.get("used_lon", site["longitude"]),
        "snap_km": dynamic.get("snap_km", 0.0),
        "warnings": warnings,
    }


@app.get("/api/model/info")
def model_info() -> dict[str, Any]:
    return load_model_info()


@app.get("/api/model/metrics")
def model_metrics() -> dict[str, Any]:
    return load_model_metrics()


@app.get("/api/noaa/availability")
def noaa_availability() -> dict[str, Any]:
    return noaa_coverage_summary()


@app.post("/api/predict")
def predict(request: SiteAnalysisRequest) -> dict[str, Any]:
    model_runtime = get_model_runtime_status()
    if not model_runtime["ready"]:
        return {
            "available": False,
            "message": model_runtime["message"],
        }

    site = _resolve_site(request=request)
    warnings: list[str] = []
    try:
        dynamic = get_site_weekly_feature_context(
            lat=float(site["latitude"]),
            lon=float(site["longitude"]),
            requested_date=request.date,
        )
        if request.date and dynamic.get("used_date") != request.date:
            warnings.append(
                f"Requested date {request.date} did not have a local weekly NOAA Monday file; "
                f"used {dynamic.get('used_date')} instead."
            )
    except Exception as exc:
        return {
            "available": False,
            "message": (
                "Prediction is unavailable because the required weekly NOAA feature history could not be assembled "
                "for this site/date. The production model only supports full contiguous 12-week NOAA histories. "
                f"Loader detail: {exc}"
            ),
        }

    features = _prediction_features(site, dynamic)
    try:
        result = predict_event_probability(features)
    except Exception as exc:
        return {
            "available": False,
            "message": (
                "Prediction is unavailable because the model bundle could not be executed in this environment. "
                f"Retrain the bundle to restore predictions. Runtime error: {exc}"
            ),
        }

    return {
        "available": True,
        "site_id": site["site_id"],
        "display_name": site["display_name"],
        "requested_date": request.date,
        "used_date": dynamic.get("used_date") or dynamic.get("date"),
        "mode": dynamic.get("mode"),
        "predicted_event": result.predicted_event,
        "probability": result.probability,
        "threshold": result.threshold,
        "model_version": result.model_version,
        "target_definition": result.target_definition,
        "prediction_unit": result.prediction_unit,
        "input_feature_window": "Static site factors + weekly Monday NOAA current, lagged, rolling, and trend stress features",
        "data_quality_warning": None if not warnings else " ; ".join(warnings),
        "coverage_warning": (
            "Prediction quality is limited by cross-source label heterogeneity and by the local weekly NOAA coverage "
            "available for this site/date. This is a same-month site-event estimate, not a long-range forecast, and "
            "the published evaluation is time-held-out rather than fully site-independent."
        ),
        "features_used": {
            "distance_to_shore_km": features["distance_to_shore_km"],
            "turbidity": features["turbidity"],
            "cyclone_frequency": features["cyclone_frequency"],
            "depth_mean_m": features["depth_mean_m"],
            "hotspot_like": features["hotspot_like"],
            "dhw_like": features["dhw_like"],
            "hotspot_like_max_4w": features.get("hotspot_like_max_4w"),
            "dhw_like_max_12w": features.get("dhw_like_max_12w"),
            "weekly_history_weeks_available": features.get("weekly_history_weeks_available"),
            "weekly_missing_fraction_12w": features.get("weekly_missing_fraction_12w"),
            "exposure": features["exposure"],
        },
    }
