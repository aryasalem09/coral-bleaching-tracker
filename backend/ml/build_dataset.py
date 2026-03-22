from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import (
    OBSERVED_EXCLUSION_LOG_PATH,
    OBSERVED_SITE_CATALOG_PATH,
    OBSERVED_SITE_DATE_PATH,
    OBSERVED_SITE_MONTH_PATH,
    SPLIT_SUMMARY_PATH,
)
from backend.ml.feature_definitions import (
    PERCENT_TARGET_COLUMN,
    SEVERITY_COLUMN,
    TARGET_COLUMN,
    add_temporal_features,
)
from backend.ml.label_standardization import ensure_standardized_observed_assets
from backend.ml.split_strategy import assign_time_split


def build_modeling_dataset() -> pd.DataFrame:
    standardized, catalog = ensure_standardized_observed_assets()

    month_df = standardized.copy()
    month_df["month"] = month_df["date"].dt.to_period("M")
    dataset = (
        month_df.groupby(["site_id", "month"], dropna=False)
        .agg(
            date=("date", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            distance_to_shore_km=("distance_to_shore_km", "mean"),
            exposure=("exposure", lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else pd.NA),
            turbidity=("turbidity", "mean"),
            cyclone_frequency=("cyclone_frequency", "mean"),
            depth_mean_m=("depth_mean_m", "mean"),
            hotspot_like=("hotspot_like", "mean"),
            dhw_like=("dhw_like", "mean"),
            observed_percent_bleaching=("observed_percent_bleaching", "mean"),
            label_quality_score=("label_quality_score", "mean"),
            source_count=("source_count", "max"),
            has_conflict_history=("has_conflict_history", "max"),
            has_derived_label_input=("is_derived_label", "max"),
            observed_record_count=("sample_row_count", "sum"),
            observed_rows_in_month=("site_id", "size"),
            observed_dates_in_month=("date", "nunique"),
            country_name=("country_name", lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else pd.NA),
            ecoregion_name=("ecoregion_name", lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else pd.NA),
        )
        .reset_index()
    )

    dataset["observed_percent_bleaching"] = pd.to_numeric(dataset["observed_percent_bleaching"], errors="coerce")
    dataset[TARGET_COLUMN] = pd.Series(pd.NA, index=dataset.index, dtype="Int64")
    observed_mask = dataset["observed_percent_bleaching"].notna()
    dataset.loc[observed_mask, TARGET_COLUMN] = (
        dataset.loc[observed_mask, "observed_percent_bleaching"] > 0
    ).astype("Int64")
    dataset[PERCENT_TARGET_COLUMN] = dataset["observed_percent_bleaching"]
    dataset[SEVERITY_COLUMN] = pd.cut(
        dataset["observed_percent_bleaching"],
        bins=[-0.01, 0, 10, 30, 100],
        labels=["none", "mild", "moderate", "severe"],
    ).astype("string")
    dataset["target_is_direct_observation"] = dataset["observed_percent_bleaching"].notna() & ~dataset["has_derived_label_input"].fillna(False)
    dataset["target_is_binary_derivation"] = dataset["has_derived_label_input"].fillna(False)

    exclusion_reasons: list[pd.DataFrame] = []
    excluded = dataset.loc[dataset["observed_percent_bleaching"].isna(), ["site_id", "date"]].copy()
    if not excluded.empty:
        excluded["reason"] = "missing_observed_percent_bleaching"
        exclusion_reasons.append(excluded)

    excluded = dataset.loc[dataset["date"].isna(), ["site_id", "date"]].copy()
    if not excluded.empty:
        excluded["reason"] = "missing_date"
        exclusion_reasons.append(excluded)

    for column in ["latitude", "longitude", "hotspot_like", "dhw_like"]:
        excluded = dataset.loc[dataset[column].isna(), ["site_id", "date"]].copy()
        if not excluded.empty:
            excluded["reason"] = f"missing_{column}"
            exclusion_reasons.append(excluded)

    excluded = dataset.loc[dataset["has_derived_label_input"].fillna(False), ["site_id", "date"]].copy()
    if not excluded.empty:
        excluded["reason"] = "comment_derived_percent_bleaching"
        exclusion_reasons.append(excluded)

    eligible_mask = (
        dataset["date"].notna()
        & dataset["observed_percent_bleaching"].notna()
        & dataset["latitude"].notna()
        & dataset["longitude"].notna()
        & dataset["hotspot_like"].notna()
        & dataset["dhw_like"].notna()
        & ~dataset["has_derived_label_input"].fillna(False)
        & dataset["label_quality_score"].fillna(0).ge(0.45)
    )
    dataset["target_eligible"] = eligible_mask
    dataset["coverage_warning"] = ""
    dataset.loc[dataset["has_conflict_history"].fillna(False), "coverage_warning"] = (
        "Multiple raw rows for the same site-date disagreed and were averaged."
    )
    dataset.loc[dataset["has_derived_label_input"].fillna(False), "coverage_warning"] = (
        "Observed percent bleaching appears to have been backfilled from coded comments."
    )

    dataset = add_temporal_features(dataset, date_column="date")
    dataset = assign_time_split(dataset, date_column="date")

    eligible = dataset.loc[dataset["target_eligible"]].copy()
    OBSERVED_SITE_MONTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(OBSERVED_SITE_MONTH_PATH, index=False)

    if exclusion_reasons:
        pd.concat(exclusion_reasons, ignore_index=True).drop_duplicates().to_csv(OBSERVED_EXCLUSION_LOG_PATH, index=False)
    else:
        pd.DataFrame(columns=["site_id", "date", "reason"]).to_csv(OBSERVED_EXCLUSION_LOG_PATH, index=False)

    split_summary = (
        eligible.groupby("split")[TARGET_COLUMN]
        .agg(count="size", positive_rate="mean")
        .reset_index()
        .to_dict(orient="records")
    )
    SPLIT_SUMMARY_PATH.write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    return eligible


def ensure_modeling_dataset() -> pd.DataFrame:
    if OBSERVED_SITE_MONTH_PATH.exists():
        frame = pd.read_csv(OBSERVED_SITE_MONTH_PATH, parse_dates=["date"])
        required_columns = {
            "has_derived_label_input",
            "target_is_direct_observation",
            "target_is_binary_derivation",
            "target_eligible",
            "split",
        }
        if required_columns.issubset(set(frame.columns)):
            return frame
    return build_modeling_dataset()


if __name__ == "__main__":
    frame = build_modeling_dataset()
    print(frame.head(3).to_string())
    print(f"\nBuilt {len(frame):,} eligible site-month rows.")
