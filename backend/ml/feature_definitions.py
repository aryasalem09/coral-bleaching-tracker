from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

TARGET_COLUMN = "target_bleaching_event"
PERCENT_TARGET_COLUMN = "observed_percent_bleaching"
SEVERITY_COLUMN = "observed_severity_category"
PRODUCTION_TARGET_NAME = "binary_bleaching_event"
PREDICTION_UNIT = "site-month"

STATIC_NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "distance_to_shore_km",
    "turbidity",
    "cyclone_frequency",
    "depth_mean_m",
]
STATIC_CATEGORICAL_FEATURES = ["exposure"]
LEGACY_DYNAMIC_NUMERIC_FEATURES = [
    "observed_same_month_hotspot_like",
    "observed_same_month_dhw_like",
]
WEEKLY_DYNAMIC_NUMERIC_FEATURES = [
    "hotspot_like",
    "dhw_like",
    "hotspot_like_lag_1w",
    "dhw_like_lag_1w",
    "hotspot_like_lag_2w",
    "dhw_like_lag_2w",
    "hotspot_like_lag_4w",
    "dhw_like_lag_4w",
    "hotspot_like_lag_8w",
    "dhw_like_lag_8w",
    "hotspot_like_mean_4w",
    "hotspot_like_max_4w",
    "hotspot_like_min_4w",
    "hotspot_like_trend_4w",
    "hotspot_like_positive_weeks_4w",
    "dhw_like_mean_4w",
    "dhw_like_max_4w",
    "dhw_like_min_4w",
    "dhw_like_trend_4w",
    "dhw_like_positive_weeks_4w",
    "dhw_like_alert_weeks_4w",
    "hotspot_like_mean_8w",
    "hotspot_like_max_8w",
    "hotspot_like_min_8w",
    "hotspot_like_trend_8w",
    "hotspot_like_positive_weeks_8w",
    "dhw_like_mean_8w",
    "dhw_like_max_8w",
    "dhw_like_min_8w",
    "dhw_like_trend_8w",
    "dhw_like_positive_weeks_8w",
    "dhw_like_alert_weeks_8w",
    "hotspot_like_mean_12w",
    "hotspot_like_max_12w",
    "hotspot_like_min_12w",
    "hotspot_like_trend_12w",
    "hotspot_like_positive_weeks_12w",
    "dhw_like_mean_12w",
    "dhw_like_max_12w",
    "dhw_like_min_12w",
    "dhw_like_trend_12w",
    "dhw_like_positive_weeks_12w",
    "dhw_like_alert_weeks_12w",
    "days_since_anchor_monday",
    "weekly_history_weeks_available",
    "weekly_missing_fraction_12w",
    "weekly_history_span_days",
    "weekly_missing_internal_weeks",
]
DYNAMIC_NUMERIC_FEATURES = WEEKLY_DYNAMIC_NUMERIC_FEATURES
TEMPORAL_FEATURES = ["month_sin", "month_cos"]

NUMERIC_FEATURES = STATIC_NUMERIC_FEATURES + DYNAMIC_NUMERIC_FEATURES + TEMPORAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES
LEGACY_NUMERIC_FEATURES = STATIC_NUMERIC_FEATURES + LEGACY_DYNAMIC_NUMERIC_FEATURES + TEMPORAL_FEATURES
LEGACY_FEATURES = LEGACY_NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES


def add_temporal_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    out = df.copy()
    month = pd.to_datetime(out[date_column], errors="coerce").dt.month.fillna(1).astype(int)
    angle = 2 * math.pi * (month - 1) / 12.0
    out["month_sin"] = angle.map(math.sin)
    out["month_cos"] = angle.map(math.cos)
    return out


def select_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[ALL_FEATURES].copy()


def ensure_feature_columns(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    expected = list(columns or ALL_FEATURES)
    out = df.copy()
    for column in expected:
        if column not in out.columns:
            out[column] = pd.NA
    return out[expected]
