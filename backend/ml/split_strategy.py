from __future__ import annotations

import pandas as pd

TRAIN_END = "2012-12-31"
VALIDATION_END = "2016-12-31"


def assign_time_split(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[date_column], errors="coerce")
    split = pd.Series("train", index=out.index, dtype="string")
    split.loc[dates > TRAIN_END] = "validation"
    split.loc[dates > VALIDATION_END] = "test"
    out["split"] = split
    return out


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = assign_time_split(df)
    return (
        split_df.loc[split_df["split"] == "train"].copy(),
        split_df.loc[split_df["split"] == "validation"].copy(),
        split_df.loc[split_df["split"] == "test"].copy(),
    )
