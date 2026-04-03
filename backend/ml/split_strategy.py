from __future__ import annotations

import pandas as pd

from backend.ml.feature_definitions import FORECAST_HORIZON_DAYS

TRAIN_END = "2012-12-31"
VALIDATION_END = "2016-12-31"


def assign_time_split(
    df: pd.DataFrame,
    date_column: str = "date",
    *,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_column], errors="coerce")
    purge_window = pd.to_timedelta(int(horizon_days), unit="D")
    train_cutoff = pd.Timestamp(TRAIN_END) - purge_window
    validation_cutoff = pd.Timestamp(VALIDATION_END) - purge_window

    split = pd.Series("excluded", index=out.index, dtype="string")
    split.loc[dates <= train_cutoff] = "train"
    split.loc[(dates > pd.Timestamp(TRAIN_END)) & (dates <= validation_cutoff)] = "validation"
    split.loc[dates > pd.Timestamp(VALIDATION_END)] = "test"
    out["split"] = split
    return out


def split_frame(df: pd.DataFrame, date_column: str = "date") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = assign_time_split(df, date_column=date_column)
    return (
        split_df.loc[split_df["split"] == "train"].copy(),
        split_df.loc[split_df["split"] == "validation"].copy(),
        split_df.loc[split_df["split"] == "test"].copy(),
    )
