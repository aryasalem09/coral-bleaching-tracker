from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.api import app

REPO_DIR = Path(__file__).resolve().parents[1]
VERIFICATION_PATH = REPO_DIR / "docs" / "prediction_verification.md"


def _sample_sites(limit: int = 5) -> list[dict[str, str]]:
    catalog = pd.read_csv(REPO_DIR / "backend" / "data" / "processed" / "observed_site_catalog.csv")
    catalog["site_id"] = catalog["site_id"].astype(str)

    model = pd.read_csv(REPO_DIR / "backend" / "data" / "processed" / "observed_site_month_dataset.csv", parse_dates=["date"])
    model["site_id"] = model["site_id"].astype(str)

    observed = pd.read_csv(REPO_DIR / "backend" / "data" / "processed" / "observed_site_date_clean.csv", parse_dates=["date"])
    observed["site_id"] = observed["site_id"].astype(str)
    observed_counts = observed.groupby("site_id")["date"].nunique().rename("obs_dates")

    latest = model.sort_values("date", ascending=False).drop_duplicates("site_id")
    latest = latest.merge(observed_counts, left_on="site_id", right_index=True, how="left")
    latest = latest.merge(
        catalog[["site_id", "display_name", "country_name"]],
        on="site_id",
        how="left",
        suffixes=("", "_catalog"),
    ).sort_values(["obs_dates", "date"], ascending=[False, False])

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
                "date": row["date"].date().isoformat(),
            }
        )
        if len(sampled) >= limit:
            break
    return sampled


def main() -> None:
    client = TestClient(app)
    model_status = client.get("/api/model/status").json()
    sample_sites = _sample_sites(limit=5)

    prediction_rows: list[dict[str, str]] = []
    for site in sample_sites:
        response = client.post(
            "/api/predict",
            json={"site_id": site["site_id"], "date": site["date"], "prefer_live": False},
        )
        payload = response.json()
        prediction_rows.append(
            {
                "site_id": site["site_id"],
                "site_name": site["display_name"],
                "country": site["country_name"],
                "date_tested": site["date"],
                "prediction_worked": "yes" if payload.get("available") else "no",
                "probability": f"{payload.get('probability', 0.0):.4f}" if payload.get("available") else "n/a",
                "feature_date_used": str(payload.get("feature_date_used") or payload.get("used_date") or "n/a"),
                "context_source": str(payload.get("context_source") or "n/a"),
                "notes": (
                    "Archived site-month prediction succeeded."
                    if payload.get("available")
                    else str(payload.get("message") or "Prediction unavailable.")
                ),
            }
        )

    analysis_site = sample_sites[0]
    analysis_payload = client.get(
        f"/api/site/{analysis_site['site_id']}/analysis",
        params={"date": analysis_site["date"], "prefer_live": True},
    ).json()
    weekly_history = analysis_payload["environmental_noaa"]["weekly_history"]

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
        "## Sample Prediction Checks",
        "",
        "| Site ID | Site | Country | Date tested | Prediction worked | Probability | Feature date used | Context source | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in prediction_rows:
        lines.append(
            f"| {row['site_id']} | {row['site_name']} | {row['country']} | {row['date_tested']} | {row['prediction_worked']} | {row['probability']} | {row['feature_date_used']} | {row['context_source']} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Selected-Site Payload Check",
            "",
            f"- Site checked: `{analysis_site['site_id']} - {analysis_site['display_name']}` on `{analysis_site['date']}`",
            f"- selected_observed_date: `{analysis_payload['selected_observed_date']}`",
            f"- observed timeline records: `{len(analysis_payload['observed_timeline']['records'])}`",
            f"- observed timeline wording note: `{analysis_payload['observed_summary']['observation_sparsity_note']}`",
            f"- weekly NOAA history available: `{weekly_history['available']}`",
            f"- weekly NOAA history records: `{len(weekly_history['records'])}`",
            f"- prediction available inside payload: `{analysis_payload['prediction']['available']}`",
            "",
            "## Edge Notes",
            "",
            "- Most sites in the cleaned observed dataset still have only one survey-backed date; that is source sparsity, not a missing weekly NOAA timeline.",
            "- Full weekly NOAA history depends on reconstructing Monday NOAA files. The backend now attempts on-demand cache fills, so the first weekly-history request for a date window can be slower than archived prediction lookups.",
            "- Prediction checks intentionally use `prefer_live=false` so they verify the archived model-ready site-month path that powers historical observed dates without waiting on NOAA downloads.",
        ]
    )

    VERIFICATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote verification report to {VERIFICATION_PATH}")


if __name__ == "__main__":
    main()
