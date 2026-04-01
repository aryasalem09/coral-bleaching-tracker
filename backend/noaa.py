from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from backend.ml.noaa_weekly_features import HISTORY_WEEKS, build_weekly_feature_row
from backend.noaa_index import NoaaAvailabilityIndex, get_noaa_weekly_index
from backend.noaa_sampling import haversine_km, sample_site_environmental_context


def available_noaa_dates() -> tuple[list[str], dict[str, dict[str, Path]], str]:
    index = get_noaa_weekly_index()
    return index.all_available_monday_dates(), dict(index.paired_paths), index.source


def noaa_coverage_summary() -> dict[str, Any]:
    return get_noaa_weekly_index().coverage_summary()


def nearest_previous_noaa_monday(iso_date: str | date | None) -> str | None:
    index = get_noaa_weekly_index()
    if iso_date is None:
        dates = index.all_available_monday_dates()
        return dates[-1] if dates else None
    return index.nearest_previous_monday(iso_date)


def recent_noaa_history_dates(iso_date: str | date, *, weeks: int) -> list[str]:
    return get_noaa_weekly_index().recent_history_dates(iso_date, weeks=weeks)


def get_site_environmental_features(
    lat: float,
    lon: float,
    iso_date: str,
    *,
    index: NoaaAvailabilityIndex | None = None,
) -> dict[str, float | str | bool]:
    return sample_site_environmental_context(lat=float(lat), lon=float(lon), iso_date=iso_date, index=index)


def get_site_weekly_feature_context(
    *,
    lat: float,
    lon: float,
    requested_date: str | None,
    index: NoaaAvailabilityIndex | None = None,
) -> dict[str, Any]:
    availability = index or get_noaa_weekly_index()
    if requested_date:
        anchor_date = availability.nearest_previous_monday(requested_date)
    else:
        dates = availability.all_available_monday_dates()
        anchor_date = dates[-1] if dates else None
    if anchor_date is None:
        raise FileNotFoundError("No weekly NOAA Monday files are available at or before the requested date.")

    history_dates = availability.recent_history_dates(anchor_date, weeks=HISTORY_WEEKS)
    if not history_dates:
        raise FileNotFoundError("No weekly NOAA history was available for the requested site-date.")
    if len(history_dates) < HISTORY_WEEKS:
        raise FileNotFoundError(
            f"A full contiguous {HISTORY_WEEKS}-week weekly NOAA history is required for prediction, "
            f"but only {len(history_dates)} weeks were available."
        )

    sampled_history: list[dict[str, Any]] = []
    sampling_failures: list[str] = []
    for iso_date in history_dates:
        try:
            sampled_history.append(
                sample_site_environmental_context(
                    lat=float(lat),
                    lon=float(lon),
                    iso_date=iso_date,
                    index=availability,
                )
            )
        except Exception as exc:
            sampling_failures.append(f"{iso_date}: {exc}")
            sampled_history.append({"hotspot": None, "dhw": None, "error": str(exc)})

    if sampling_failures:
        raise FileNotFoundError(
            "The required weekly NOAA history could not be sampled cleanly for this site/date: "
            + "; ".join(sampling_failures)
        )

    feature_row = build_weekly_feature_row(
        observation_date=pd.to_datetime(requested_date or anchor_date, errors="raise"),
        anchor_date=anchor_date,
        history_dates=history_dates,
        sampled_history=sampled_history,
    )
    if int(feature_row["weekly_history_weeks_available"]) < HISTORY_WEEKS:
        raise FileNotFoundError(
            f"A full contiguous {HISTORY_WEEKS}-week weekly NOAA history is required for prediction, "
            f"but only {feature_row['weekly_history_weeks_available']} weeks were assembled."
        )
    if int(feature_row["weekly_missing_internal_weeks"]) > 0:
        raise FileNotFoundError(
            "Prediction is unavailable because the weekly NOAA history has internal gaps and no longer matches "
            "the production training coverage."
        )
    return {
        **feature_row,
        "requested_date": requested_date,
        "used_date": anchor_date,
        "history_dates": history_dates,
        "mode": "noaa_weekly_monday",
    }
