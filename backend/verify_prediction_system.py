from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.api import app
from backend.config import FORECAST_DATASET_PATH

REPO_DIR = Path(__file__).resolve().parents[1]
VERIFICATION_PATH = REPO_DIR / "docs" / "prediction_verification.md"


def _sample_sites(limit: int = 5) -> list[dict[str, str]]:
    catalog = pd.read_csv(REPO_DIR / "backend" / "data" / "processed" / "observed_site_catalog.csv")
    catalog["site_id"] = catalog["site_id"].astype(str)

    forecast = pd.read_csv(FORECAST_DATASET_PATH, parse_dates=["date", "reference_observed_date"])
    forecast["site_id"] = forecast["site_id"].astype(str)

    latest = forecast.sort_values("date", ascending=False).drop_duplicates("site_id")
    latest = latest.merge(
        catalog[["site_id", "display_name", "country_name"]],
        on="site_id",
        how="left",
        suffixes=("", "_catalog"),
    ).sort_values(["reference_observed_date", "date"], ascending=[False, False])

    seen_names: set[str] = set()
    sampled: list[dict[str, str]] = []
    for _, row in latest.iterrows():
        display_name = str(row["display_name"])
        if display_name in seen_names:
            continue
        seen_names.add(display_name)
        sampled.append(
            {
                "site_id": str(row["site_id"]),
                "display_name": display_name,
                "country_name": str(row["country_name_catalog"]),
                "survey_date": pd.to_datetime(row["reference_observed_date"]).date().isoformat(),
            }
        )
        if len(sampled) >= limit:
            break
    return sampled


def main() -> None:
    client = TestClient(app)
    model_status = client.get("/api/model/status").json()
    model_info = client.get("/api/model/info").json()
    sample_sites = _sample_sites(limit=5)

    prediction_rows: list[dict[str, str]] = []
    for site in sample_sites:
        response = client.post(
            "/api/predict",
            json={"site_id": site["site_id"], "date": site["survey_date"], "prefer_live": False},
        )
        payload = response.json()
        prediction_rows.append(
            {
                "site_id": site["site_id"],
                "site_name": site["display_name"],
                "country": site["country_name"],
                "survey_date": site["survey_date"],
                "forecast_worked": "yes" if payload.get("available") else "no",
                "probability": f"{payload.get('probability', 0.0):.4f}" if payload.get("available") else "n/a",
                "forecast_issue_date": str(payload.get("forecast_issue_date") or payload.get("feature_date_used") or "n/a"),
                "context_source": str(payload.get("context_source") or "n/a"),
                "notes": (
                    "Archived forecast row succeeded."
                    if payload.get("available")
                    else str(payload.get("message") or "Forecast unavailable.")
                ),
            }
        )

    analysis_site = sample_sites[0]
    analysis_payload = client.get(
        f"/api/site/{analysis_site['site_id']}/analysis",
        params={"date": analysis_site["survey_date"], "prefer_live": True},
    ).json()
    weekly_history = analysis_payload["environmental_noaa"]["weekly_history"]
    prediction_payload = analysis_payload["prediction"]

    lines = [
        "# Prediction Verification",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Model Status",
        "",
        f"- model_loaded: `{model_status['model_loaded']}`",
        f"- model_version: `{model_status.get('model_version')}`",
        f"- sklearn_version: `{model_status.get('sklearn_version')}`",
        f"- trained_with_sklearn_version: `{model_status.get('trained_with_sklearn_version')}`",
        f"- artifact_path: `{model_status.get('artifact_path')}`",
        "",
        "## Forecast Definition",
        "",
        f"- target_definition: `{model_info.get('target_definition')}`",
        f"- prediction_unit: `{model_info.get('prediction_unit')}`",
        f"- forecast_horizon_weeks: `{model_info.get('forecast_horizon_weeks')}`",
        f"- probability_meaning: `{model_info.get('probability_meaning')}`",
        f"- ground_truth_definition: `{model_info.get('ground_truth_definition')}`",
        "",
        "## Sample Forecast Checks",
        "",
        "| Site ID | Site | Country | Survey date | Forecast worked | Probability | Forecast issue date | Context source | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in prediction_rows:
        lines.append(
            f"| {row['site_id']} | {row['site_name']} | {row['country']} | {row['survey_date']} | {row['forecast_worked']} | {row['probability']} | {row['forecast_issue_date']} | {row['context_source']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Selected-Site Payload Check",
            "",
            f"- Site checked: `{analysis_site['site_id']} - {analysis_site['display_name']}` on `{analysis_site['survey_date']}`",
            f"- selected_observed_date: `{analysis_payload['selected_observed_date']}`",
            f"- observed timeline records: `{len(analysis_payload['observed_timeline']['records'])}`",
            f"- observed timeline wording note: `{analysis_payload['observed_summary']['observation_sparsity_note']}`",
            f"- weekly NOAA history available: `{weekly_history['available']}`",
            f"- weekly NOAA history records: `{len(weekly_history['records'])}`",
            f"- forecast available inside payload: `{prediction_payload['available']}`",
            f"- forecast issue date: `{prediction_payload.get('forecast_issue_date')}`",
            f"- probability meaning: `{prediction_payload.get('probability_meaning')}`",
            "",
            "## Edge Notes",
            "",
            "- Most sites still have sparse survey timelines. The forecast dataset only uses issue dates with at least one direct survey in the next 4 weeks, so missing surveys are not forced into negative labels.",
            "- Full weekly NOAA history depends on reconstructing Monday NOAA files. The backend can still fall back to saved forecast rows for historical survey dates when live NOAA history is unavailable.",
            "- Verification uses `prefer_live=false` for sample API calls so it checks the archived forecast path that supports historical survey dates without waiting on NOAA downloads.",
        ]
    )

    VERIFICATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote verification report to {VERIFICATION_PATH}")


if __name__ == "__main__":
    main()
