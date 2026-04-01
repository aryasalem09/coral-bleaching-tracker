# Prediction Verification

Generated: `2026-04-01T03:11:31.866159+00:00`

## Model Status

- model_loaded: `True`
- model_version: `2026.03.31`
- sklearn_version: `1.6.1`
- trained_with_sklearn_version: `1.6.1`
- artifact_path: `C:\Users\aryas\PycharmProjects\CoralBleachingTracker\backend\ml\artifacts\bleaching_event_model.joblib`

## Sample Prediction Checks

| Site ID | Site | Country | Date tested | Prediction worked | Probability | Feature date used | Context source | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3579 | Bora Bora | French Polynesia | 2020-03-12 | yes | 0.3632 | 2020-03-12 | historical_model_row | Archived site-month prediction succeeded. |
| 3031 | Dahab | Egypt | 2019-12-27 | yes | 0.0495 | 2019-12-27 | historical_model_row | Archived site-month prediction succeeded. |
| 1318 | Alice Town | Bahamas | 2019-08-24 | yes | 0.9814 | 2019-08-24 | historical_model_row | Archived site-month prediction succeeded. |
| 6703 | Curacao | Netherlands Antilles | 2012-04-10 | yes | 0.1083 | 2012-04-10 | historical_model_row | Archived site-month prediction succeeded. |
| 6505 | Aventuras | Mexico | 2003-09-18 | yes | 0.5808 | 2003-09-18 | historical_model_row | Archived site-month prediction succeeded. |

## Selected-Site Payload Check

- Site checked: `3579 - Bora Bora` on `2020-03-12`
- selected_observed_date: `2020-03-12`
- observed timeline records: `31`
- observed timeline wording note: `Observed survey records are sparse and irregular. They are not the same thing as weekly NOAA environmental history.`
- weekly NOAA history available: `True`
- weekly NOAA history records: `12`
- prediction available inside payload: `True`

## Edge Notes

- Most sites in the cleaned observed dataset still have only one survey-backed date; that is source sparsity, not a missing weekly NOAA timeline.
- Full weekly NOAA history depends on reconstructing Monday NOAA files. The backend now attempts on-demand cache fills, so the first weekly-history request for a date window can be slower than archived prediction lookups.
- Prediction checks intentionally use `prefer_live=false` so they verify the archived model-ready site-month path that powers historical observed dates without waiting on NOAA downloads.
