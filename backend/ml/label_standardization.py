from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.config import (
    OBSERVED_CONFLICT_LOG_PATH,
    OBSERVED_RAW_PATH,
    OBSERVED_SITE_CATALOG_PATH,
    OBSERVED_SITE_DATE_PATH,
    ensure_directories,
)

STATIC_CATEGORICAL_COLUMNS = [
    "data_source",
    "ocean_name",
    "realm_name",
    "ecoregion_name",
    "country_name",
    "state_name",
    "city_town_name",
    "site_name",
    "reef_id",
    "exposure",
]

STATIC_NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "distance_to_shore_km",
    "turbidity",
    "cyclone_frequency",
]

DYNAMIC_NUMERIC_COLUMNS = [
    "depth_m",
    "percent_cover",
    "percent_bleaching",
    "clim_sst",
    "temperature_mean",
    "temperature_minimum",
    "temperature_maximum",
    "windspeed",
    "ssta",
    "ssta_dhw",
    "tsa",
    "tsa_dhw",
]


def _normalize_text(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nd", "nan", "none"}:
        return pd.NA
    return text


def _mode_or_na(series: pd.Series) -> str | pd.NA:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    mode = clean.mode()
    return _normalize_text(mode.iloc[0]) if not mode.empty else _normalize_text(clean.iloc[0])


def _mean_or_na(series: pd.Series) -> float | pd.NA:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return pd.NA
    return float(clean.mean())


def _scalar_or_na(value: object) -> float | pd.NA:
    scalar = pd.to_numeric(pd.Series([value]), errors="coerce").dropna()
    if scalar.empty:
        return pd.NA
    return float(scalar.iloc[0])


def _quality_score(row: pd.Series) -> float:
    score = 0.0
    if row["is_direct_observation"]:
        score += 0.45
    if row["is_derived_label"]:
        score -= 0.15
    if row["has_precise_date"]:
        score += 0.15
    if row["has_precise_location"]:
        score += 0.15
    if not row["has_conflict_history"]:
        score += 0.1
    missing_penalty = min(float(row["missing_metadata_level"]) * 0.05, 0.15)
    source_bonus = min(float(row["source_count"]) * 0.02, 0.1)
    score = score - missing_penalty + source_bonus
    return round(max(0.0, min(score, 1.0)), 3)


def load_raw_observations(path: Path | None = None) -> pd.DataFrame:
    source_path = Path(path or OBSERVED_RAW_PATH)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Observed bleaching source file is missing: {source_path}. "
            "Download the BCO-DMO dataset before building the model."
        )

    df = pd.read_csv(source_path, low_memory=False)
    rename_map = {
        "Site_ID": "site_id",
        "Sample_ID": "sample_id",
        "Data_Source": "data_source",
        "Latitude_Degrees": "latitude",
        "Longitude_Degrees": "longitude",
        "Ocean_Name": "ocean_name",
        "Reef_ID": "reef_id",
        "Realm_Name": "realm_name",
        "Ecoregion_Name": "ecoregion_name",
        "Country_Name": "country_name",
        "State_Island_Province_Name": "state_name",
        "City_Town_Name": "city_town_name",
        "Site_Name": "site_name",
        "Distance_to_Shore": "distance_to_shore_km",
        "Exposure": "exposure",
        "Turbidity": "turbidity",
        "Cyclone_Frequency": "cyclone_frequency",
        "Depth_m": "depth_m",
        "Substrate_Name": "substrate_name",
        "Percent_Cover": "percent_cover",
        "Bleaching_Level": "bleaching_level",
        "Percent_Bleaching": "percent_bleaching",
        "ClimSST": "clim_sst",
        "Temperature_Mean": "temperature_mean",
        "Temperature_Minimum": "temperature_minimum",
        "Temperature_Maximum": "temperature_maximum",
        "Windspeed": "windspeed",
        "SSTA": "ssta",
        "SSTA_DHW": "ssta_dhw",
        "TSA": "tsa",
        "TSA_DHW": "tsa_dhw",
        "Date": "date",
        "Site_Comments": "site_comments",
        "Sample_Comments": "sample_comments",
        "Bleaching_Comments": "bleaching_comments",
    }
    df = df.rename(columns=rename_map)

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].map(_normalize_text)

    numeric_columns = STATIC_NUMERIC_COLUMNS + DYNAMIC_NUMERIC_COLUMNS
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["site_id"] = df["site_id"].astype("string")
    df["sample_id"] = df["sample_id"].astype("string")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def standardize_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = df.copy()
    comment_fields = (
        working["bleaching_comments"].fillna("").astype(str)
        + " "
        + working["sample_comments"].fillna("").astype(str)
        + " "
        + working["site_comments"].fillna("").astype(str)
    ).str.lower()
    working["label_is_comment_derived"] = (
        comment_fields.str.contains("averaged from code", na=False)
        | comment_fields.str.contains("bleaching index", na=False)
        | comment_fields.str.contains("same bleaching severity", na=False)
    )
    metadata_columns = ["distance_to_shore_km", "exposure", "turbidity", "cyclone_frequency", "depth_m"]
    working["metadata_missing_count"] = working[metadata_columns].isna().sum(axis=1)
    working["metadata_completeness_rank"] = working[metadata_columns].notna().sum(axis=1)
    working = working.sort_values(
        by=["site_id", "date", "metadata_completeness_rank", "sample_id"],
        ascending=[True, True, False, True],
        na_position="last",
    )

    grouped = working.groupby(["site_id", "date"], dropna=False, sort=False)
    first_rows = grouped.nth(0).reset_index()
    standardized = grouped.agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        distance_to_shore_km=("distance_to_shore_km", "mean"),
        turbidity=("turbidity", "mean"),
        cyclone_frequency=("cyclone_frequency", "mean"),
        depth_mean_m=("depth_m", "mean"),
        observed_percent_bleaching=("percent_bleaching", "mean"),
        observed_percent_bleaching_min=("percent_bleaching", "min"),
        observed_percent_bleaching_max=("percent_bleaching", "max"),
        observed_percent_bleaching_std=("percent_bleaching", "std"),
        hotspot_like=("tsa", "mean"),
        dhw_like=("tsa_dhw", "mean"),
        sample_row_count=("sample_id", "size"),
        unique_percent_bleaching_values=("percent_bleaching", "nunique"),
        source_count=("data_source", "nunique"),
        label_is_comment_derived=("label_is_comment_derived", "max"),
        missing_metadata_level=("metadata_missing_count", "min"),
        provenance_sources=("data_source", lambda s: json.dumps(sorted({str(value) for value in s.dropna().unique()}))),
        provenance_sample_ids=("sample_id", lambda s: json.dumps(sorted({str(value) for value in s.dropna().unique()})[:50])),
    ).reset_index()

    categorical_columns = [
        "ocean_name",
        "realm_name",
        "ecoregion_name",
        "country_name",
        "state_name",
        "city_town_name",
        "site_name",
        "reef_id",
        "exposure",
    ]
    standardized = standardized.merge(
        first_rows[["site_id", "date", *categorical_columns]],
        on=["site_id", "date"],
        how="left",
    )

    standardized["observed_percent_bleaching"] = pd.to_numeric(
        standardized["observed_percent_bleaching"],
        errors="coerce",
    )
    standardized["target_bleaching_event"] = pd.Series(pd.NA, index=standardized.index, dtype="Int64")
    observed_mask = standardized["observed_percent_bleaching"].notna()
    standardized.loc[observed_mask, "target_bleaching_event"] = (
        standardized.loc[observed_mask, "observed_percent_bleaching"] > 0
    ).astype("Int64")
    standardized["observed_severity_category"] = pd.cut(
        standardized["observed_percent_bleaching"],
        bins=[-0.01, 0, 10, 30, 100],
        labels=["none", "mild", "moderate", "severe"],
    ).astype("string")

    standardized["is_direct_observation"] = standardized["observed_percent_bleaching"].notna() & ~standardized["label_is_comment_derived"]
    standardized["is_derived_label"] = standardized["observed_percent_bleaching"].notna() & standardized["label_is_comment_derived"]
    standardized["has_precise_date"] = standardized["date"].notna()
    standardized["has_precise_location"] = standardized["latitude"].notna() & standardized["longitude"].notna()
    standardized["has_conflict_history"] = standardized["unique_percent_bleaching_values"].fillna(0).astype(int).gt(1)

    missing_metadata_cols = [
        "distance_to_shore_km",
        "exposure",
        "turbidity",
        "cyclone_frequency",
        "depth_mean_m",
        "hotspot_like",
        "dhw_like",
    ]
    standardized["missing_metadata_level"] = standardized[missing_metadata_cols].isna().sum(axis=1)
    standardized["label_quality_score"] = standardized.apply(_quality_score, axis=1)
    standardized["recommended_for_modeling"] = (
        standardized["is_direct_observation"]
        & ~standardized["is_derived_label"]
        & standardized["has_precise_date"]
        & standardized["has_precise_location"]
        & standardized["hotspot_like"].notna()
        & standardized["dhw_like"].notna()
        & standardized["label_quality_score"].ge(0.45)
    )

    conflicts = standardized.loc[
        standardized["has_conflict_history"],
        [
            "site_id",
            "date",
            "sample_row_count",
            "unique_percent_bleaching_values",
            "observed_percent_bleaching_min",
            "observed_percent_bleaching_max",
            "label_is_comment_derived",
            "provenance_sources",
            "provenance_sample_ids",
        ],
    ].copy()

    return standardized.sort_values(["date", "site_id"]).reset_index(drop=True), conflicts


def build_site_catalog(standardized: pd.DataFrame) -> pd.DataFrame:
    grouped = standardized.groupby("site_id", dropna=False)
    catalog = grouped.apply(_aggregate_site, include_groups=False).reset_index()
    catalog["display_name"] = catalog.apply(_site_display_name, axis=1)
    return catalog.sort_values("display_name").reset_index(drop=True)


def _aggregate_site(group: pd.DataFrame) -> pd.Series:
    source_names: set[str] = set()
    for value in group["provenance_sources"].dropna():
        try:
            decoded = json.loads(str(value))
        except json.JSONDecodeError:
            decoded = [str(value)]
        for item in decoded:
            source_names.add(str(item))
    return pd.Series(
        {
            "latitude": _mean_or_na(group["latitude"]),
            "longitude": _mean_or_na(group["longitude"]),
            "ocean_name": _mode_or_na(group["ocean_name"]),
            "realm_name": _mode_or_na(group["realm_name"]),
            "ecoregion_name": _mode_or_na(group["ecoregion_name"]),
            "country_name": _mode_or_na(group["country_name"]),
            "state_name": _mode_or_na(group["state_name"]),
            "city_town_name": _mode_or_na(group["city_town_name"]),
            "site_name": _mode_or_na(group["site_name"]),
            "reef_id": _mode_or_na(group["reef_id"]),
            "distance_to_shore_km": _mean_or_na(group["distance_to_shore_km"]),
            "exposure": _mode_or_na(group["exposure"]),
            "turbidity": _mean_or_na(group["turbidity"]),
            "cyclone_frequency": _mean_or_na(group["cyclone_frequency"]),
            "depth_mean_m": _mean_or_na(group["depth_mean_m"]),
            "observed_record_count": int(len(group)),
            "observed_positive_count": int(group["target_bleaching_event"].fillna(0).astype(int).sum()),
            "latest_observed_date": group["date"].max(),
            "first_observed_date": group["date"].min(),
            "provenance_source_count": int(len(source_names)),
            "provenance_sources": json.dumps(sorted(source_names)),
            "mean_label_quality_score": round(float(group["label_quality_score"].mean()), 3),
        }
    )


def _site_display_name(row: pd.Series) -> str:
    for column in ["site_name", "city_town_name", "ecoregion_name", "country_name"]:
        value = row.get(column)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"Site {row['site_id']}"


def ensure_standardized_observed_assets() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_directories()
    if OBSERVED_SITE_DATE_PATH.exists() and OBSERVED_SITE_CATALOG_PATH.exists():
        standardized = pd.read_csv(OBSERVED_SITE_DATE_PATH, parse_dates=["date"])
        catalog = pd.read_csv(
            OBSERVED_SITE_CATALOG_PATH,
            parse_dates=["first_observed_date", "latest_observed_date"],
        )
        required_standardized_columns = {
            "label_is_comment_derived",
            "is_direct_observation",
            "is_derived_label",
            "recommended_for_modeling",
        }
        has_missing_target_bug = bool(
            (
                standardized["observed_percent_bleaching"].isna()
                & standardized["target_bleaching_event"].notna()
            ).any()
        ) if {"observed_percent_bleaching", "target_bleaching_event"}.issubset(set(standardized.columns)) else True
        if required_standardized_columns.issubset(set(standardized.columns)) and not has_missing_target_bug:
            return standardized, catalog

    raw = load_raw_observations()
    standardized, conflicts = standardize_labels(raw)
    catalog = build_site_catalog(standardized)

    standardized.to_csv(OBSERVED_SITE_DATE_PATH, index=False)
    catalog.to_csv(OBSERVED_SITE_CATALOG_PATH, index=False)
    conflicts.to_csv(OBSERVED_CONFLICT_LOG_PATH, index=False)
    return standardized, catalog
