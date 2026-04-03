from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import (
    FORECAST_DATASET_PATH,
    OBSERVED_EXCLUSION_LOG_PATH,
    OBSERVED_SITE_MONTH_PATH,
    SPLIT_SUMMARY_PATH,
)
from backend.ml.feature_definitions import (
    FEATURE_HISTORY_WEEKS,
    FORECAST_HORIZON_DAYS,
    FORECAST_HORIZON_WEEKS,
    OBSERVED_PERCENT_COLUMN,
    OBSERVED_SEVERITY_COLUMN,
    TARGET_COLUMN,
    add_temporal_features,
)
from backend.ml.label_standardization import ensure_standardized_observed_assets
from backend.ml.noaa_weekly_features import add_weekly_noaa_history_features
from backend.ml.split_strategy import assign_time_split

LEGACY_SOURCE_REQUIRED_COLUMNS = {
    "date",
    "weekly_anchor_date",
    "hotspot_like_lag_1w",
    "dhw_like_mean_12w",
    "target_eligible",
}
FORECAST_REQUIRED_COLUMNS = {
    "date",
    "reference_observed_date",
    "future_window_end_date",
    "future_observation_count_4w",
    "future_positive_observation_count_4w",
    "target_eligible",
    TARGET_COLUMN,
    "split",
}


def _mode_or_na(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    mode = clean.mode()
    return mode.iat[0] if not mode.empty else clean.iat[0]


def _build_legacy_feature_source_dataset() -> pd.DataFrame:
    standardized, _ = ensure_standardized_observed_assets()

    month_df = standardized.copy()
    month_df["month"] = month_df["date"].dt.to_period("M")
    dataset = (
        month_df.groupby(["site_id", "month"], dropna=False)
        .agg(
            date=("date", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            distance_to_shore_km=("distance_to_shore_km", "mean"),
            exposure=("exposure", _mode_or_na),
            turbidity=("turbidity", "mean"),
            cyclone_frequency=("cyclone_frequency", "mean"),
            depth_mean_m=("depth_mean_m", "mean"),
            observed_same_month_hotspot_like=("hotspot_like", "mean"),
            observed_same_month_dhw_like=("dhw_like", "mean"),
            observed_percent_bleaching=("observed_percent_bleaching", "mean"),
            label_quality_score=("label_quality_score", "mean"),
            source_count=("source_count", "max"),
            has_conflict_history=("has_conflict_history", "max"),
            has_derived_label_input=("is_derived_label", "max"),
            observed_record_count=("sample_row_count", "sum"),
            observed_rows_in_month=("site_id", "size"),
            observed_dates_in_month=("date", "nunique"),
            country_name=("country_name", _mode_or_na),
            ecoregion_name=("ecoregion_name", _mode_or_na),
        )
        .reset_index()
    )

    dataset["observed_percent_bleaching"] = pd.to_numeric(dataset["observed_percent_bleaching"], errors="coerce")
    dataset["target_bleaching_event"] = pd.Series(pd.NA, index=dataset.index, dtype="Int64")
    observed_mask = dataset["observed_percent_bleaching"].notna()
    dataset.loc[observed_mask, "target_bleaching_event"] = (
        dataset.loc[observed_mask, "observed_percent_bleaching"] > 0
    ).astype("Int64")
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

    for column in ["latitude", "longitude"]:
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
    dataset, _ = add_weekly_noaa_history_features(dataset)

    weekly_excluded = dataset.loc[~dataset["weekly_feature_ready"].fillna(False), ["site_id", "date", "weekly_feature_error"]].copy()
    if not weekly_excluded.empty:
        weekly_excluded["reason"] = weekly_excluded["weekly_feature_error"].fillna("missing_weekly_noaa_context")
        exclusion_reasons.append(weekly_excluded[["site_id", "date", "reason"]])

    dataset["target_eligible"] = dataset["target_eligible"].fillna(False) & dataset["weekly_feature_ready"].fillna(False)
    eligible = dataset.loc[dataset["target_eligible"]].copy()
    eligible.drop(columns=["weekly_feature_error"], inplace=True, errors="ignore")
    OBSERVED_SITE_MONTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(OBSERVED_SITE_MONTH_PATH, index=False)

    if exclusion_reasons:
        pd.concat(exclusion_reasons, ignore_index=True).drop_duplicates().to_csv(OBSERVED_EXCLUSION_LOG_PATH, index=False)
    else:
        pd.DataFrame(columns=["site_id", "date", "reason"]).to_csv(OBSERVED_EXCLUSION_LOG_PATH, index=False)

    return eligible


def ensure_feature_source_dataset() -> pd.DataFrame:
    if OBSERVED_SITE_MONTH_PATH.exists():
        frame = pd.read_csv(
            OBSERVED_SITE_MONTH_PATH,
            parse_dates=["date", "weekly_anchor_date"],
        )
        if LEGACY_SOURCE_REQUIRED_COLUMNS.issubset(set(frame.columns)):
            return frame
    return _build_legacy_feature_source_dataset()


def attach_future_bleaching_targets(
    feature_rows: pd.DataFrame,
    future_observations: pd.DataFrame,
    *,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:
    working = feature_rows.copy()
    working["site_id"] = working["site_id"].astype("string")
    working["date"] = pd.to_datetime(working["date"], errors="coerce")

    observed = future_observations.copy()
    observed["site_id"] = observed["site_id"].astype("string")
    observed["date"] = pd.to_datetime(observed["date"], errors="coerce")
    observed = observed.loc[
        observed["site_id"].notna()
        & observed["date"].notna()
        & observed[OBSERVED_PERCENT_COLUMN].notna()
    ].sort_values(["site_id", "date"])

    working["future_window_end_date"] = working["date"] + pd.to_timedelta(int(horizon_days), unit="D")
    working["future_observation_count_4w"] = 0
    working["future_positive_observation_count_4w"] = 0
    working["first_future_observation_date"] = pd.NaT
    working["first_future_positive_date"] = pd.NaT
    working["days_to_first_future_observation"] = pd.Series(pd.NA, index=working.index, dtype="Int64")
    working["days_to_first_future_positive"] = pd.Series(pd.NA, index=working.index, dtype="Int64")

    for site_id, anchor_index in working.groupby("site_id", sort=False).groups.items():
        site_rows = working.loc[anchor_index].sort_values("date")
        site_observed = observed.loc[observed["site_id"] == site_id].copy()
        if site_observed.empty:
            continue

        observed_dates = site_observed["date"].to_numpy(dtype="datetime64[ns]")
        positive_mask = (
            site_observed[OBSERVED_PERCENT_COLUMN]
            .fillna(0)
            .gt(0)
            .to_numpy(dtype=bool)
        )
        positive_cumsum = np.concatenate([[0], positive_mask.astype(int).cumsum()])

        counts: list[int] = []
        positive_counts: list[int] = []
        first_future_dates: list[pd.Timestamp | pd.NaT] = []
        first_positive_dates: list[pd.Timestamp | pd.NaT] = []
        days_to_first: list[int | None] = []
        days_to_positive: list[int | None] = []

        for anchor_date in site_rows["date"]:
            if pd.isna(anchor_date):
                counts.append(0)
                positive_counts.append(0)
                first_future_dates.append(pd.NaT)
                first_positive_dates.append(pd.NaT)
                days_to_first.append(None)
                days_to_positive.append(None)
                continue

            anchor_np = np.datetime64(anchor_date.to_datetime64())
            end_np = anchor_np + np.timedelta64(int(horizon_days), "D")
            start_index = int(np.searchsorted(observed_dates, anchor_np, side="right"))
            end_index = int(np.searchsorted(observed_dates, end_np, side="right"))
            window_count = max(0, end_index - start_index)
            window_positive_count = int(positive_cumsum[end_index] - positive_cumsum[start_index])

            counts.append(window_count)
            positive_counts.append(window_positive_count)

            if window_count <= 0:
                first_future_dates.append(pd.NaT)
                first_positive_dates.append(pd.NaT)
                days_to_first.append(None)
                days_to_positive.append(None)
                continue

            first_future = pd.Timestamp(observed_dates[start_index])
            first_future_dates.append(first_future)
            days_to_first.append(int((first_future - anchor_date).days))

            if window_positive_count <= 0:
                first_positive_dates.append(pd.NaT)
                days_to_positive.append(None)
                continue

            positive_window = positive_mask[start_index:end_index]
            positive_offset = int(np.argmax(positive_window))
            first_positive = pd.Timestamp(observed_dates[start_index + positive_offset])
            first_positive_dates.append(first_positive)
            days_to_positive.append(int((first_positive - anchor_date).days))

        ordered_index = site_rows.index
        working.loc[ordered_index, "future_observation_count_4w"] = counts
        working.loc[ordered_index, "future_positive_observation_count_4w"] = positive_counts
        working.loc[ordered_index, "first_future_observation_date"] = first_future_dates
        working.loc[ordered_index, "first_future_positive_date"] = first_positive_dates
        working.loc[ordered_index, "days_to_first_future_observation"] = pd.Series(
            days_to_first,
            index=ordered_index,
            dtype="Int64",
        )
        working.loc[ordered_index, "days_to_first_future_positive"] = pd.Series(
            days_to_positive,
            index=ordered_index,
            dtype="Int64",
        )

    working["target_eligible"] = working["future_observation_count_4w"].gt(0)
    working[TARGET_COLUMN] = pd.Series(pd.NA, index=working.index, dtype="Int64")
    eligible_mask = working["target_eligible"].fillna(False)
    working.loc[eligible_mask, TARGET_COLUMN] = (
        working.loc[eligible_mask, "future_positive_observation_count_4w"].gt(0)
    ).astype("Int64")
    working["target_label_window"] = f"next_{FORECAST_HORIZON_WEEKS}w"
    working["target_label_source"] = "future_direct_observation"
    return working


def build_modeling_dataset() -> pd.DataFrame:
    source = ensure_feature_source_dataset().copy()
    source["site_id"] = source["site_id"].astype("string")
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["weekly_anchor_date"] = pd.to_datetime(source["weekly_anchor_date"], errors="coerce")

    # One forecast row per site + Monday issue date keeps training weights stable.
    source = (
        source.sort_values(["site_id", "weekly_anchor_date", "date"])
        .drop_duplicates(["site_id", "weekly_anchor_date"], keep="last")
        .reset_index(drop=True)
    )
    source["reference_observed_date"] = source["date"]
    source["date"] = source["weekly_anchor_date"]
    source["days_since_anchor_monday"] = 0
    source = add_temporal_features(source, date_column="date")

    standardized, _ = ensure_standardized_observed_assets()
    future_observations = standardized.loc[
        standardized["is_direct_observation"].fillna(False),
        ["site_id", "date", OBSERVED_PERCENT_COLUMN, OBSERVED_SEVERITY_COLUMN, "label_quality_score"],
    ].copy()
    dataset = attach_future_bleaching_targets(source, future_observations, horizon_days=FORECAST_HORIZON_DAYS)
    dataset = assign_time_split(dataset, date_column="date", horizon_days=FORECAST_HORIZON_DAYS)
    dataset["split_eligible"] = dataset["split"].isin(["train", "validation", "test"])
    dataset["target_eligible"] = dataset["target_eligible"].fillna(False) & dataset["split_eligible"].fillna(False)

    dataset["coverage_warning"] = ""
    dataset.loc[~dataset["target_eligible"].fillna(False), "coverage_warning"] = (
        "No direct survey observation was available in the next 4 weeks, so the future label is unknown."
    )
    dataset.loc[
        dataset["future_observation_count_4w"].gt(0) & dataset["future_positive_observation_count_4w"].eq(0),
        "coverage_warning",
    ] = "Future label is negative because direct surveys in the next 4 weeks reported no bleaching event."
    dataset.loc[
        dataset["future_positive_observation_count_4w"].gt(0),
        "coverage_warning",
    ] = "Future label is positive because a direct survey in the next 4 weeks reported bleaching."

    eligible = dataset.loc[dataset["target_eligible"]].copy()
    eligible = eligible.sort_values(["date", "site_id"]).reset_index(drop=True)

    FORECAST_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(FORECAST_DATASET_PATH, index=False)

    split_summary = (
        eligible.groupby("split")[TARGET_COLUMN]
        .agg(count="size", positive_rate="mean")
        .reset_index()
        .to_dict(orient="records")
    )
    SPLIT_SUMMARY_PATH.write_text(
        json.dumps(
            {
                "dataset": "observed_site_forecast_4w_dataset",
                "feature_history_weeks": FEATURE_HISTORY_WEEKS,
                "forecast_horizon_days": FORECAST_HORIZON_DAYS,
                "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
                "rows_total": int(len(eligible)),
                "split_summary": split_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return eligible


def ensure_modeling_dataset() -> pd.DataFrame:
    if FORECAST_DATASET_PATH.exists():
        frame = pd.read_csv(
            FORECAST_DATASET_PATH,
            parse_dates=[
                "date",
                "reference_observed_date",
                "weekly_anchor_date",
                "future_window_end_date",
                "first_future_observation_date",
                "first_future_positive_date",
            ],
        )
        if FORECAST_REQUIRED_COLUMNS.issubset(set(frame.columns)):
            return frame
    return build_modeling_dataset()


if __name__ == "__main__":
    frame = build_modeling_dataset()
    print(frame.head(3).to_string())
    print(f"\nBuilt {len(frame):,} eligible forecast rows.")
