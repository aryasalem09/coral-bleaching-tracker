from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backend.config import NOAA_WEEKLY_FEATURE_AUDIT_PATH
from backend.noaa_index import get_noaa_weekly_index
from backend.noaa_sampling import sample_points_for_date

HISTORY_WEEKS = 12
LAG_OFFSETS = (1, 2, 4, 8)
ROLLING_WINDOWS = (4, 8, 12)


def _series_slope(values: list[float]) -> float | pd.NA:
    if len(values) < 2:
        return pd.NA
    x_values = np.arange(len(values), dtype=float)
    y_values = np.asarray(values, dtype=float)
    slope, _ = np.polyfit(x_values, y_values, 1)
    return float(slope)


def _window(values: list[float], size: int) -> list[float]:
    if not values:
        return []
    return values[-min(size, len(values)) :]


def build_weekly_feature_row(
    *,
    observation_date: pd.Timestamp,
    anchor_date: str,
    history_dates: list[str],
    sampled_history: list[dict[str, Any]],
) -> dict[str, Any]:
    hotspot_sequence = [float(item["hotspot"]) if item.get("hotspot") is not None else pd.NA for item in sampled_history]
    dhw_sequence = [float(item["dhw"]) if item.get("dhw") is not None else pd.NA for item in sampled_history]
    if not hotspot_sequence or pd.isna(hotspot_sequence[-1]) or not dhw_sequence or pd.isna(dhw_sequence[-1]):
        raise ValueError("The weekly NOAA history is missing current hotspot or DHW values.")

    anchor_dt = pd.to_datetime(anchor_date, errors="raise")
    history_span_days = (pd.to_datetime(history_dates[-1]) - pd.to_datetime(history_dates[0])).days if history_dates else 0
    expected_span_days = 7 * max(0, len(history_dates) - 1)
    missing_internal_weeks = max(0, int(round((history_span_days - expected_span_days) / 7.0)))

    feature_row: dict[str, Any] = {
        "hotspot_like": float(hotspot_sequence[-1]),
        "dhw_like": float(dhw_sequence[-1]),
        "weekly_anchor_date": anchor_date,
        "days_since_anchor_monday": int((observation_date - anchor_dt).days),
        "weekly_history_weeks_available": int(len(history_dates)),
        "weekly_expected_history_weeks": int(HISTORY_WEEKS),
        "weekly_missing_fraction_12w": round(1.0 - (len(history_dates) / HISTORY_WEEKS), 6),
        "weekly_history_span_days": int(history_span_days),
        "weekly_missing_internal_weeks": int(missing_internal_weeks),
    }

    for lag in LAG_OFFSETS:
        hotspot_name = f"hotspot_like_lag_{lag}w"
        dhw_name = f"dhw_like_lag_{lag}w"
        feature_row[hotspot_name] = hotspot_sequence[-(lag + 1)] if len(hotspot_sequence) > lag else pd.NA
        feature_row[dhw_name] = dhw_sequence[-(lag + 1)] if len(dhw_sequence) > lag else pd.NA

    for window in ROLLING_WINDOWS:
        hotspot_window = [value for value in _window(hotspot_sequence, window) if not pd.isna(value)]
        dhw_window = [value for value in _window(dhw_sequence, window) if not pd.isna(value)]
        feature_row[f"hotspot_like_mean_{window}w"] = float(np.mean(hotspot_window)) if hotspot_window else pd.NA
        feature_row[f"hotspot_like_max_{window}w"] = float(np.max(hotspot_window)) if hotspot_window else pd.NA
        feature_row[f"hotspot_like_min_{window}w"] = float(np.min(hotspot_window)) if hotspot_window else pd.NA
        feature_row[f"hotspot_like_trend_{window}w"] = _series_slope(hotspot_window)
        feature_row[f"hotspot_like_positive_weeks_{window}w"] = int(sum(value > 0 for value in hotspot_window)) if hotspot_window else 0

        feature_row[f"dhw_like_mean_{window}w"] = float(np.mean(dhw_window)) if dhw_window else pd.NA
        feature_row[f"dhw_like_max_{window}w"] = float(np.max(dhw_window)) if dhw_window else pd.NA
        feature_row[f"dhw_like_min_{window}w"] = float(np.min(dhw_window)) if dhw_window else pd.NA
        feature_row[f"dhw_like_trend_{window}w"] = _series_slope(dhw_window)
        feature_row[f"dhw_like_positive_weeks_{window}w"] = int(sum(value > 0 for value in dhw_window)) if dhw_window else 0
        feature_row[f"dhw_like_alert_weeks_{window}w"] = int(sum(value >= 4.0 for value in dhw_window)) if dhw_window else 0

    return feature_row


def add_weekly_noaa_history_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    availability = get_noaa_weekly_index()
    available_dates = availability.all_available_monday_dates()
    if not available_dates:
        raise FileNotFoundError(
            "No local weekly NOAA Monday files were found. Run `python3 -m backend.download_noaa_weekly_mondays` first."
        )

    working = dataset.copy()
    working["weekly_anchor_date"] = working["date"].map(
        lambda value: availability.nearest_previous_monday(pd.to_datetime(value, errors="coerce").date())
        if pd.notna(pd.to_datetime(value, errors="coerce"))
        else None
    )
    working["days_since_anchor_monday"] = working.apply(
        lambda row: (
            pd.to_datetime(row["date"], errors="coerce") - pd.to_datetime(row["weekly_anchor_date"], errors="coerce")
        ).days
        if pd.notna(row.get("date")) and pd.notna(row.get("weekly_anchor_date"))
        else pd.NA,
        axis=1,
    )
    working["weekly_history_dates"] = working["weekly_anchor_date"].map(
        lambda iso_date: availability.recent_history_dates(iso_date, weeks=HISTORY_WEEKS) if isinstance(iso_date, str) else []
    )
    working["weekly_history_weeks_available"] = working["weekly_history_dates"].map(len)

    point_requests_by_date: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in working.itertuples(index=False):
        if not isinstance(row.weekly_anchor_date, str) or not row.weekly_history_dates:
            continue
        site_id = str(row.site_id)
        for iso_date in row.weekly_history_dates:
            point_requests_by_date[iso_date][site_id] = {
                "point_id": site_id,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
            }

    sampled_by_date: dict[str, dict[str, dict[str, Any]]] = {}
    total_dates = len(point_requests_by_date)
    for index, iso_date in enumerate(sorted(point_requests_by_date.keys()), start=1):
        if index % 25 == 0 or index == total_dates:
            print(f"Sampling weekly NOAA history {index}/{total_dates} dates...", flush=True)
        sampled_by_date[iso_date] = sample_points_for_date(iso_date, point_requests_by_date[iso_date].values(), index=availability)

    feature_payloads: list[dict[str, Any]] = []
    excluded_missing_anchor = 0
    excluded_sampling_errors = 0
    sampling_errors: list[dict[str, Any]] = []

    for row in working.itertuples(index=False):
        if not isinstance(row.weekly_anchor_date, str) or not row.weekly_history_dates:
            excluded_missing_anchor += 1
            feature_payloads.append({"weekly_feature_error": "no_weekly_anchor"})
            continue

        if len(row.weekly_history_dates) < HISTORY_WEEKS:
            excluded_sampling_errors += 1
            feature_payloads.append(
                {
                    "weekly_feature_error": (
                        f"requires_full_{HISTORY_WEEKS}w_history_but_only_{len(row.weekly_history_dates)}_weeks_available"
                    )
                }
            )
            continue

        site_id = str(row.site_id)
        sampled_history: list[dict[str, Any]] = []
        missing_current = False
        error_message = ""
        sampling_failures: list[str] = []
        for iso_date in row.weekly_history_dates:
            sample = sampled_by_date.get(iso_date, {}).get(site_id)
            if not sample:
                sampling_failures.append(f"{iso_date}: missing sampled NOAA row")
                sampled_history.append({"hotspot": None, "dhw": None})
                continue
            if sample.get("error"):
                error_message = str(sample["error"])
                sampling_failures.append(f"{iso_date}: {error_message}")
                if iso_date == row.weekly_anchor_date:
                    missing_current = True
                sampled_history.append({"hotspot": None, "dhw": None})
                continue
            sampled_history.append(sample)

        if missing_current or not sampled_history or sampling_failures:
            excluded_sampling_errors += 1
            resolved_error = error_message or "; ".join(sampling_failures) or "missing current weekly NOAA sample"
            sampling_errors.append(
                {
                    "site_id": site_id,
                    "date": pd.to_datetime(row.date).date().isoformat(),
                    "weekly_anchor_date": row.weekly_anchor_date,
                    "error": resolved_error,
                }
            )
            feature_payloads.append({"weekly_feature_error": resolved_error})
            continue

        try:
            feature_row = build_weekly_feature_row(
                observation_date=pd.to_datetime(row.date, errors="raise"),
                anchor_date=row.weekly_anchor_date,
                history_dates=list(row.weekly_history_dates),
                sampled_history=sampled_history,
            )
            if int(feature_row["weekly_history_weeks_available"]) < HISTORY_WEEKS:
                raise ValueError(
                    f"requires_full_{HISTORY_WEEKS}w_history_but_only_{feature_row['weekly_history_weeks_available']}_weeks_available"
                )
            if int(feature_row["weekly_missing_internal_weeks"]) > 0:
                raise ValueError(
                    f"weekly_history_has_{feature_row['weekly_missing_internal_weeks']}_internal_gap_weeks"
                )
            feature_payloads.append(feature_row)
        except Exception as exc:
            excluded_sampling_errors += 1
            sampling_errors.append(
                {
                    "site_id": site_id,
                    "date": pd.to_datetime(row.date).date().isoformat(),
                    "weekly_anchor_date": row.weekly_anchor_date,
                    "error": str(exc),
                }
            )
            feature_payloads.append({"weekly_feature_error": str(exc)})

    feature_frame = pd.DataFrame(feature_payloads)
    working_for_output = working.drop(
        columns=[
            "weekly_anchor_date",
            "days_since_anchor_monday",
            "weekly_history_weeks_available",
        ],
        errors="ignore",
    )
    enriched = pd.concat([working_for_output.reset_index(drop=True), feature_frame.reset_index(drop=True)], axis=1)
    enriched["weekly_feature_ready"] = enriched["weekly_feature_error"].isna()

    audit = {
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "available_monday_count": len(available_dates),
        "first_available_monday": available_dates[0] if available_dates else None,
        "last_available_monday": available_dates[-1] if available_dates else None,
        "rows_requested": int(len(dataset)),
        "rows_ready": int(enriched["weekly_feature_ready"].fillna(False).sum()),
        "rows_excluded_no_weekly_anchor": int(excluded_missing_anchor),
        "rows_excluded_sampling_error": int(excluded_sampling_errors),
        "sampling_error_examples": sampling_errors[:25],
        "history_weeks_target": HISTORY_WEEKS,
        "lag_offsets_weeks": list(LAG_OFFSETS),
        "rolling_windows_weeks": list(ROLLING_WINDOWS),
        "temporal_alignment": (
            "Each legacy site-month feature row uses the nearest available Monday on or before the observed site-date "
            "as the current weekly anchor. Lagged and rolling NOAA features use only Mondays at or before that anchor date."
        ),
    }
    NOAA_WEEKLY_FEATURE_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOAA_WEEKLY_FEATURE_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return enriched, audit
