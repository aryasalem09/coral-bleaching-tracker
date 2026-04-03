# Prediction Verification

Generated: `2026-04-02T08:31:34.562542+00:00`

## Model Status

- model_loaded: `True`
- model_version: `2026.04.01`
- sklearn_version: `1.6.1`
- trained_with_sklearn_version: `1.6.1`
- artifact_path: `C:\Users\aryas\PycharmProjects\CoralBleachingTracker\backend\ml\artifacts\bleaching_event_model.joblib`

## Forecast Definition

- target_definition: `observed_bleaching_event_in_next_4_weeks`
- prediction_unit: `site-anchor-date`
- forecast_horizon_weeks: `4`
- probability_meaning: `Probability that at least one direct observed bleaching event will be recorded for this site in the next 4 weeks.`
- ground_truth_definition: `Ground truth comes from direct observed bleaching records. NOAA heat data are predictors, not labels.`

## Sample Forecast Checks

| Site ID | Site | Country | Survey date | Forecast worked | Probability | Forecast issue date | Context source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 15445 | Siaba Besar (Sector 1) | Indonesia | 2020-08-15 | yes | 0.1374 | 2020-08-10 | historical_forecast_row | Archived forecast row succeeded. |
| 15444 | Mawan (Sector 1) | Indonesia | 2020-08-15 | yes | 0.2326 | 2020-08-10 | historical_forecast_row | Archived forecast row succeeded. |
| 3591 | Bora Bora | French Polynesia | 2020-05-30 | yes | 0.5430 | 2020-05-25 | historical_forecast_row | Archived forecast row succeeded. |
| 15443 | Pengah Kecil (Sector 1) SW Side | Indonesia | 2020-03-17 | yes | 0.4144 | 2020-03-16 | historical_forecast_row | Archived forecast row succeeded. |
| 4215 | North Lombok Regency | Indonesia | 2020-03-17 | yes | 0.6973 | 2020-03-16 | historical_forecast_row | Archived forecast row succeeded. |

## Selected-Site Payload Check

- Site checked: `15445 - Siaba Besar (Sector 1)` on `2020-08-15`
- selected_observed_date: `2020-08-15`
- observed timeline records: `3`
- observed timeline wording note: `Observed survey records are sparse and irregular. They are not the same thing as weekly NOAA environmental history.`
- weekly NOAA history available: `True`
- weekly NOAA history records: `12`
- forecast available inside payload: `True`
- forecast issue date: `2020-08-10`
- probability meaning: `Probability that at least one direct observed bleaching event will be recorded for this site in the next 4 weeks.`

## Edge Notes

- Most sites still have sparse survey timelines. The forecast dataset only uses issue dates with at least one direct survey in the next 4 weeks, so missing surveys are not forced into negative labels.
- Full weekly NOAA history depends on reconstructing Monday NOAA files. The backend can still fall back to saved forecast rows for historical survey dates when live NOAA history is unavailable.
- Verification uses `prefer_live=false` for sample API calls so it checks the archived forecast path that supports historical survey dates without waiting on NOAA downloads.
