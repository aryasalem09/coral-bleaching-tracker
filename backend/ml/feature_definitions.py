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
DYNAMIC_NUMERIC_FEATURES = ["hotspot_like", "dhw_like"]
TEMPORAL_FEATURES = ["month_sin", "month_cos"]

NUMERIC_FEATURES = STATIC_NUMERIC_FEATURES + DYNAMIC_NUMERIC_FEATURES + TEMPORAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES + STATIC_CATEGORICAL_FEATURES


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

