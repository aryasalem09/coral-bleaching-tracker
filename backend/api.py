from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import pandas as pd
from pydantic import BaseModel

from backend.config import APP_VERSION, ensure_directories
from backend.ml.model_registry import (
    get_model_runtime_status,
    load_model_info,
    load_model_metrics,
)
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
    get_site_observations,
    list_sites_in_bbox,
    recommended_observed_date,
)
from backend.risk.explain import explain_risk
from backend.risk.scoring import score_environmental_risk
from backend.risk.thresholds import RISK_THRESHOLDS

ensure_directories()

logger = logging.getLogger(__name__)

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


def _serialize_site(site: dict[str, Any]) -> dict[str, Any]:
    return {
        **site,
        "first_observed_date": site["first_observed_date"].date().isoformat()
        if site.get("first_observed_date") is not None
        else None,
        "latest_observed_date": site["latest_observed_date"].date().isoformat()
        if site.get("latest_observed_date") is not None
        else None,
    }


def _serialize_observation_row(row: pd.Series) -> dict[str, Any]:
    return {
        "date": row["date"].date().isoformat(),
        "observed_percent_bleaching": None
        if pd.isna(row["observed_percent_bleaching"])
        else float(row["observed_percent_bleaching"]),
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


def _serialize_observations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_serialize_observation_row(row) for _, row in frame.iterrows()]


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        text = str(value).strip()
        return text or None
    return parsed.date().isoformat()


def _resolve_selected_observed_date(site_id: str, records: list[dict[str, Any]], requested_date: str | None) -> str | None:
    if requested_date:
        return requested_date
    recommended = recommended_observed_date(site_id)
    if recommended:
        return recommended
    return records[0]["date"] if records else None


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
                logger.warning("Falling back from live NOAA context for site %s: %s", site["site_id"], exc)
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
        raise HTTPException(status_code=500, detail="Forecast context is missing a usable date.")

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


def _prediction_status_label(predicted_event: bool) -> str:
    return "forecast_above_threshold" if predicted_event else "forecast_below_threshold"


def _short_prediction_unavailable_message(model_loaded: bool, reason: str) -> str:
    if not model_loaded:
        return "Forecast unavailable because the backend could not load the trained model."
    if reason == "context_unavailable":
        return "Forecast unavailable for this site/date because the required 12-week heat-stress history could not be assembled."
    return "Forecast unavailable right now."


def _build_environmental_summary(site: dict[str, Any], selected_date: str | None, *, prefer_live: bool) -> dict[str, Any]:
    if not selected_date:
        return {
            "available": False,
            "message": "Select an observed survey date to view environmental context.",
        }

    try:
        dynamic, warnings = _resolve_dynamic_features(
            site,
            SiteAnalysisRequest(site_id=site["site_id"], date=selected_date, prefer_live=prefer_live),
            require_model_eligible=False,
        )
    except HTTPException as exc:
        return {
            "available": False,
            "message": str(exc.detail),
        }

    hotspot = float(dynamic.get("hotspot", dynamic.get("hotspot_like")))
    dhw = float(dynamic.get("dhw", dynamic.get("dhw_like")))
    risk = score_environmental_risk(hotspot=hotspot, dhw=dhw)
    return {
        "available": True,
        "requested_date": selected_date,
        "used_date": dynamic.get("used_date") or dynamic.get("date"),
        "mode": dynamic.get("mode"),
        "category": risk.category,
        "score": risk.score,
        "color": risk.color,
        "hotspot": hotspot,
        "dhw": dhw,
        "explanation": explain_risk(risk),
        "warnings": warnings,
    }


def _build_environmental_summary_from_weekly_history(
    selected_date: str | None,
    weekly_history: dict[str, Any],
) -> dict[str, Any]:
    if not selected_date or not weekly_history.get("available"):
        return {
            "available": False,
            "message": "Select an observed survey date to view environmental context.",
        }

    records = weekly_history.get("records", [])
    if not records:
        return {
            "available": False,
            "message": "Weekly NOAA history was assembled without any usable records.",
        }

    current = records[-1]
    hotspot = float(current["hotspot"])
    dhw = float(current["dhw"])
    risk = score_environmental_risk(hotspot=hotspot, dhw=dhw)
    return {
        "available": True,
        "requested_date": selected_date,
        "used_date": weekly_history.get("anchor_date"),
        "mode": "noaa_weekly_monday",
        "category": risk.category,
        "score": risk.score,
        "color": risk.color,
        "hotspot": hotspot,
        "dhw": dhw,
        "explanation": explain_risk(risk),
        "warnings": [],
    }


def _build_weekly_history_payload(site: dict[str, Any], selected_date: str | None) -> dict[str, Any]:
    if not selected_date:
        return {
            "available": False,
            "message": "Select an observed survey date to inspect weekly NOAA history.",
            "records": [],
        }

    try:
        weekly_context = get_site_weekly_feature_context(
            lat=float(site["latitude"]),
            lon=float(site["longitude"]),
            requested_date=selected_date,
        )
    except Exception as exc:
        logger.warning("Weekly NOAA history unavailable for site %s on %s: %s", site["site_id"], selected_date, exc)
        return {
            "available": False,
            "requested_date": selected_date,
            "message": (
                "Full weekly NOAA history could not be reconstructed for this site/date from the local or on-demand "
                "NOAA cache."
            ),
            "records": [],
        }

    records = weekly_context.get("history_records", [])
    hotspot_values = [float(record["hotspot"]) for record in records]
    dhw_values = [float(record["dhw"]) for record in records]
    return {
        "available": True,
        "requested_date": selected_date,
        "anchor_date": weekly_context.get("used_date"),
        "history_window_weeks": len(records),
        "records": records,
        "summary": {
            "weeks_returned": len(records),
            "max_hotspot": max(hotspot_values) if hotspot_values else None,
            "max_dhw": max(dhw_values) if dhw_values else None,
            "mean_hotspot": round(sum(hotspot_values) / len(hotspot_values), 4) if hotspot_values else None,
            "mean_dhw": round(sum(dhw_values) / len(dhw_values), 4) if dhw_values else None,
            "hotspot_positive_weeks": sum(value > 0 for value in hotspot_values),
            "dhw_alert_weeks": sum(value >= 4.0 for value in dhw_values),
            "source": weekly_context.get("mode"),
        },
        "message": None,
    }


def _build_prediction_payload(
    site: dict[str, Any],
    selected_date: str | None,
    *,
    prefer_live: bool,
) -> dict[str, Any]:
    model_runtime = get_model_runtime_status()
    base_payload = {
        "available": False,
        "status": "model_unavailable" if not model_runtime["model_loaded"] else "context_unavailable",
        "message": _short_prediction_unavailable_message(
            model_loaded=bool(model_runtime["model_loaded"]),
            reason="context_unavailable",
        ),
        "site_id": site["site_id"],
        "display_name": site["display_name"],
        "requested_date": selected_date,
        "model_loaded": bool(model_runtime["model_loaded"]),
        "model_version": model_runtime.get("model_version"),
        "prediction_unit": load_model_info().get("prediction_unit"),
        "target_definition": load_model_info().get("target_definition"),
        "forecast_horizon_days": load_model_info().get("forecast_horizon_days"),
        "forecast_horizon_weeks": load_model_info().get("forecast_horizon_weeks"),
        "probability_meaning": load_model_info().get("probability_meaning"),
        "ground_truth_definition": load_model_info().get("ground_truth_definition"),
    }

    if not selected_date:
        base_payload["message"] = "Select an observed survey date to request a 4-week forecast."
        return base_payload

    if not model_runtime["model_loaded"]:
        base_payload["message"] = _short_prediction_unavailable_message(
            model_loaded=False,
            reason="model_unavailable",
        )
        return base_payload

    context_source = "historical_forecast_row"
    context_notes: list[str] = [
        "This forecast uses only site details and NOAA heat history available on or before the forecast issue date."
    ]
    dynamic: dict[str, Any] | None = None

    if prefer_live:
        try:
            dynamic = get_site_weekly_feature_context(
                lat=float(site["latitude"]),
                lon=float(site["longitude"]),
                requested_date=selected_date,
            )
            context_source = "weekly_noaa_history"
        except Exception as exc:
            logger.warning(
                "Live weekly NOAA feature assembly failed for prediction site %s on %s: %s",
                site["site_id"],
                selected_date,
                exc,
            )
            context_notes.append(
                "Live NOAA history was unavailable for this request, so the backend fell back to the closest saved forecast-ready row when possible."
            )

    if dynamic is None:
        dynamic = find_historical_context(
            str(site["site_id"]),
            requested_date=selected_date,
            require_model_eligible=True,
        )
        if dynamic is None:
            base_payload["status"] = "context_unavailable"
            base_payload["message"] = _short_prediction_unavailable_message(
                model_loaded=True,
                reason="context_unavailable",
            )
            return base_payload

    feature_date_used = _iso_date(dynamic.get("used_date") or dynamic.get("date"))
    if feature_date_used and feature_date_used != selected_date:
        context_notes.append(
            f"The selected survey date {selected_date} maps to forecast issue date {feature_date_used} because the model forecasts from the nearest eligible Monday on or before that survey date."
        )
    if context_source == "historical_forecast_row":
        context_notes.append(
            "This forecast used archived feature rows that were built from past NOAA heat history and paired with future survey outcomes."
        )

    features = _prediction_features(site, dynamic)
    try:
        result = predict_event_probability(features)
    except Exception:
        logger.exception("Prediction execution failed for site %s on %s", site["site_id"], selected_date)
        return {
            **base_payload,
            "status": "execution_error",
            "message": "Forecast unavailable because the trained model could not execute in the current backend environment.",
        }

    weekly_history_weeks_available = features.get("weekly_history_weeks_available")
    weekly_missing_fraction = features.get("weekly_missing_fraction_12w")
    return {
        "available": True,
        "status": "available",
        "message": None,
        "requested_date": selected_date,
        "site_id": site["site_id"],
        "display_name": site["display_name"],
        "feature_date_used": feature_date_used,
        "used_date": feature_date_used,
        "forecast_issue_date": feature_date_used,
        "weekly_anchor_date": _iso_date(dynamic.get("weekly_anchor_date") or dynamic.get("used_date")),
        "context_source": context_source,
        "mode": dynamic.get("mode"),
        "model_loaded": True,
        "predicted_event": result.predicted_event,
        "predicted_class_label": _prediction_status_label(result.predicted_event),
        "probability": result.probability,
        "threshold": result.threshold,
        "model_version": result.model_version,
        "target_definition": result.target_definition,
        "prediction_unit": result.prediction_unit,
        "forecast_horizon_days": result.forecast_horizon_days,
        "forecast_horizon_weeks": result.forecast_horizon_weeks,
        "probability_meaning": result.probability_meaning,
        "ground_truth_definition": result.ground_truth_definition,
        "input_feature_window": "Static site factors plus 12 weeks of NOAA Monday heat-stress history ending on the forecast issue date.",
        "coverage_notes": context_notes,
        "data_quality_warning": None,
        "coverage_warning": " ".join(context_notes),
        "features_used": {
            "distance_to_shore_km": features["distance_to_shore_km"],
            "turbidity": features["turbidity"],
            "cyclone_frequency": features["cyclone_frequency"],
            "depth_mean_m": features["depth_mean_m"],
            "hotspot_like": features["hotspot_like"],
            "dhw_like": features["dhw_like"],
            "hotspot_like_max_4w": features.get("hotspot_like_max_4w"),
            "dhw_like_max_12w": features.get("dhw_like_max_12w"),
            "weekly_history_weeks_available": int(weekly_history_weeks_available)
            if weekly_history_weeks_available is not None
            else None,
            "weekly_missing_fraction_12w": float(weekly_missing_fraction)
            if weekly_missing_fraction is not None
            else None,
            "exposure": features["exposure"],
        },
    }


def _build_model_metadata() -> dict[str, Any]:
    runtime = get_model_runtime_status()
    info = load_model_info()
    return {
        "model_loaded": bool(runtime["model_loaded"]),
        "runtime_status": runtime["status"],
        "runtime_message": runtime["message"],
        "model_version": info.get("model_version") or runtime.get("model_version"),
        "prediction_unit": info.get("prediction_unit"),
        "target_definition": info.get("target_definition"),
        "trained_with_sklearn_version": runtime.get("trained_with_sklearn_version"),
        "sklearn_version": runtime.get("sklearn_version"),
        "artifact_path": runtime.get("artifact_path"),
        "decision_threshold": info.get("decision_threshold"),
        "feature_set": info.get("feature_set"),
        "model_family": info.get("model_family"),
        "forecast_horizon_days": info.get("forecast_horizon_days"),
        "forecast_horizon_weeks": info.get("forecast_horizon_weeks"),
        "feature_history_weeks": info.get("feature_history_weeks"),
        "probability_meaning": info.get("probability_meaning"),
        "ground_truth_definition": info.get("ground_truth_definition"),
        "threshold_selection_rule": info.get("threshold_selection_rule"),
        "input_feature_window": info.get("input_feature_window"),
    }


def _build_selected_site_payload(site: dict[str, Any], *, requested_date: str | None, prefer_live: bool) -> dict[str, Any]:
    observations_frame = get_site_observations(site["site_id"])
    observation_records = _serialize_observations(observations_frame)
    selected_date = _resolve_selected_observed_date(site["site_id"], observation_records, requested_date)
    observed_date_count = len({record["date"] for record in observation_records})
    weekly_history = _build_weekly_history_payload(site, selected_date)
    environmental_summary = (
        _build_environmental_summary_from_weekly_history(selected_date, weekly_history)
        if weekly_history["available"]
        else _build_environmental_summary(site, selected_date, prefer_live=prefer_live)
    )
    prediction = _build_prediction_payload(site, selected_date, prefer_live=prefer_live)
    model_metadata = _build_model_metadata()

    return {
        "site": _serialize_site(site),
        "selected_observed_date": selected_date,
        "observed_summary": {
            "record_count": int(site.get("observed_record_count", 0)),
            "unique_survey_dates": observed_date_count,
            "positive_observation_count": int(site.get("observed_positive_count", 0)),
            "mean_label_quality_score": float(site.get("mean_label_quality_score", 0.0)),
            "first_observed_date": site["first_observed_date"].date().isoformat()
            if site.get("first_observed_date") is not None
            else None,
            "latest_observed_date": site["latest_observed_date"].date().isoformat()
            if site.get("latest_observed_date") is not None
            else None,
            "observation_sparsity_note": (
                "Observed survey records are sparse and irregular. They are not the same thing as weekly NOAA environmental history."
            ),
            "single_survey_date_only": observed_date_count == 1,
        },
        "observed_timeline": {
            "recommended_date": recommended_observed_date(site["site_id"]),
            "records": observation_records,
        },
        "environmental_noaa": {
            "stress_outlook": environmental_summary,
            "weekly_history": weekly_history,
        },
        "prediction": prediction,
        "model_metadata": model_metadata,
        "data_availability": {
            "observed_timeline_available": bool(observation_records),
            "weekly_noaa_history_available": bool(weekly_history["available"]),
            "environmental_summary_available": bool(environmental_summary["available"]),
            "prediction_available": bool(prediction["available"]),
            "model_loaded": bool(model_metadata["model_loaded"]),
        },
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {"ok": True, "service": "coral-bleaching-api", "version": APP_VERSION}


@app.get("/health")
def health() -> dict[str, Any]:
    model_runtime = get_model_runtime_status()
    return {
        "ok": True,
        "service": "coral-bleaching-api",
        "started_at": STARTED_AT,
        "version": APP_VERSION,
        "model_loaded": model_runtime["model_loaded"],
        "model_version": model_runtime.get("model_version"),
        "artifact_path": model_runtime.get("artifact_path"),
        "sklearn_version": model_runtime.get("sklearn_version"),
        "trained_with_sklearn_version": model_runtime.get("trained_with_sklearn_version"),
        "loader_error": model_runtime.get("loader_error"),
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
        "model_loaded": bool(model_runtime["model_loaded"]),
        "model_status": model_runtime["status"],
        "model_status_message": model_runtime["message"],
        "model_version": model_runtime.get("model_version"),
        "sklearn_version": model_runtime.get("sklearn_version"),
        "trained_with_sklearn_version": model_runtime.get("trained_with_sklearn_version"),
        "live_noaa_dates_available": len(live_dates),
        "latest_live_noaa_date": live_dates[-1] if live_dates else None,
        "live_date_source": live_source,
        "live_noaa_schedule": "weekly_mondays",
        "live_noaa_first_date": weekly_coverage.get("paired_first_date"),
    }


@app.get("/api/model/status")
def model_status() -> dict[str, Any]:
    return get_model_runtime_status()


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
        "site": _serialize_site(site),
        "recommended_observed_date": recommended_observed_date(site_id),
        "observed_dates": [
            pd_timestamp.date().isoformat()
            for pd_timestamp in observations["date"].dropna().sort_values(ascending=False).tolist()
        ],
        "observed_date_count": int(observations["date"].nunique()),
    }


@app.get("/api/site/{site_id}/observations")
def site_observations(site_id: str) -> dict[str, Any]:
    try:
        frame = get_site_observations(site_id)
        site = get_site_metadata(site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "site_id": site_id,
        "display_name": site["display_name"],
        "recommended_date": recommended_observed_date(site_id),
        "records": _serialize_observations(frame),
    }


@app.get("/api/site/{site_id}/analysis")
def site_analysis(
    site_id: str,
    date: str | None = Query(default=None),
    prefer_live: bool = Query(default=True),
) -> dict[str, Any]:
    try:
        site = get_site_metadata(site_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _build_selected_site_payload(site, requested_date=date, prefer_live=prefer_live)


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
            "historical survey-backed environmental record."
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
    site = _resolve_site(request=request)
    return _build_prediction_payload(site, request.date, prefer_live=request.prefer_live)
