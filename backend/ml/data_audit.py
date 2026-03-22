from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class RawObservedAudit:
    row_count: int
    site_count: int
    date_min: str | None
    date_max: str | None
    missing_percent_bleaching: int
    positive_percent_bleaching: int
    zero_percent_bleaching: int
    duplicated_sample_ids: int
    duplicated_site_dates: int


def summarize_raw_observed_data(df: pd.DataFrame) -> dict[str, object]:
    date_series = pd.to_datetime(df["Date"], errors="coerce")
    percent = pd.to_numeric(df["Percent_Bleaching"], errors="coerce")
    audit = RawObservedAudit(
        row_count=int(len(df)),
        site_count=int(df["Site_ID"].nunique(dropna=True)),
        date_min=date_series.min().date().isoformat() if date_series.notna().any() else None,
        date_max=date_series.max().date().isoformat() if date_series.notna().any() else None,
        missing_percent_bleaching=int(percent.isna().sum()),
        positive_percent_bleaching=int((percent > 0).sum()),
        zero_percent_bleaching=int((percent == 0).sum()),
        duplicated_sample_ids=int(df.duplicated(subset=["Sample_ID"]).sum()),
        duplicated_site_dates=int(df.duplicated(subset=["Site_ID", "Date"]).sum()),
    )
    return asdict(audit)


def summarize_standardized_labels(df: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": int(len(df)),
        "site_count": int(df["site_id"].nunique(dropna=True)),
        "date_min": df["date"].min().date().isoformat() if not df.empty else None,
        "date_max": df["date"].max().date().isoformat() if not df.empty else None,
        "target_positive_rate": round(float(df["target_bleaching_event"].mean()), 4) if not df.empty else None,
        "conflict_history_rows": int(df["has_conflict_history"].fillna(False).sum()),
        "direct_observation_rows": int(df["is_direct_observation"].fillna(False).sum()),
    }
