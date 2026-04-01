# Coral Bleaching Tracker

Coral bleaching explorer with three explicitly separated layers:

- Observed Bleaching: cleaned historical site observations
- Environmental Stress Outlook: transparent NOAA heat-stress context
- Model Prediction: supervised `site-month` bleaching-event probability

The app is designed to stay honest about what is observation, what is environmental stress, and what is supervised ML output.

## Current Architecture

- Frontend: Vite + React + TypeScript
- Backend API: `backend/api.py`
- Observed data repository: `backend/observed/repository.py`
- Weekly NOAA availability and sampling:
  - `backend/noaa_products.py`
  - `backend/noaa_index.py`
  - `backend/noaa_sampling.py`
  - `backend/noaa.py`
- Weekly NOAA downloader:
  - `backend/download_noaa_weekly_mondays.py`
- Supervised ML pipeline:
  - `backend/ml/build_dataset.py`
  - `backend/ml/noaa_weekly_features.py`
  - `backend/ml/train_model.py`

## Data Layout

Processed observed/modeling assets:

- `backend/data/processed/observed_site_date_clean.csv`
- `backend/data/processed/observed_site_catalog.csv`
- `backend/data/processed/observed_site_month_dataset.csv`
- `backend/data/processed/noaa_weekly_feature_audit.json`

Raw NOAA weekly-Monday cache:

- `backend/data/raw/noaa_dhw/`
- `backend/data/raw/noaa_hs/`
- `backend/data/raw/noaa_manifest_weekly_mondays.json`

Model artifacts:

- `backend/ml/artifacts/bleaching_event_model.joblib`
- `backend/ml/artifacts/model_info.json`
- `backend/ml/artifacts/metrics.json`
- `backend/ml/artifacts/training_report.md`
- `backend/ml/artifacts/feature_importance.csv`

The raw NOAA archive is intentionally ignored by Git because the complete weekly cache is large.

## NOAA Weekly Pipeline

The project now downloads NOAA Coral Reef Watch daily NetCDF files for the Monday of every week across the paired product range.

Features of the downloader:

- remote year/date discovery from NOAA directory listings
- atomic file writes
- retry logic
- skip-valid-file behavior
- resumable reruns
- missing-date logging
- manifest output with per-date and per-product status

Canonical command:

```bash
python3 -m backend.download_noaa_weekly_mondays
```

Examples:

```bash
python3 -m backend.download_noaa_weekly_mondays --workers 20
python3 -m backend.download_noaa_weekly_mondays --start 2000-01-01 --end 2019-12-31
```

More detail: `docs/noaa_weekly_pipeline.md`

## Modeling Decision

The production target remains a binary bleaching event at the `site-month` level.

Why it stays that way:

- the observed labels are not a trustworthy weekly supervision stream
- source programs use heterogeneous percent-bleaching conventions
- a weekly target would overclaim temporal precision

What changed instead:

- the model now uses weekly Monday NOAA history aligned to the nearest prior Monday on or before each observed site-date
- lagged, rolling, and trend features are constructed without future leakage
- the training loop compares the legacy same-month formulation against the new weekly-history formulation
- production prediction only runs when a full contiguous 12-week NOAA history can be assembled, because that is the support the trained model actually saw

Current selected production model:

- `weekly_history_hist_gradient_boosting`
- held-out test AUROC: `0.666`
- held-out test PR-AUC: `0.516`
- weekly-history versus legacy same-month HGB PR-AUC gain: `+0.048`

More detail: `docs/modeling_decision.md`

## Local Run

### Backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python3 -m uvicorn backend.api:app --reload
```

The backend serves at `http://127.0.0.1:8000`.

### Frontend

From `frontend`:

```bash
npm install
npm run dev
```

Optional environment variable:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

If `VITE_API_BASE_URL` is unset in local development, the frontend defaults to `http://127.0.0.1:8000`.

## Retraining

1. Download the weekly NOAA Monday archive.
2. Rebuild the modeling dataset.
3. Retrain the supervised model.

Commands:

```bash
python3 -m backend.download_noaa_weekly_mondays
python3 -m backend.ml.build_dataset
python3 -m backend.ml.train_model
```

## Useful API Routes

- `GET /health`
- `GET /api/summary`
- `GET /api/model/status`
- `GET /api/noaa/availability`
- `GET /api/sites?south=...&west=...&north=...&east=...&limit=...`
- `GET /api/site/{site_id}`
- `GET /api/site/{site_id}/observations`
- `GET /api/site/{site_id}/analysis?date=...&prefer_live=...`
- `GET /api/risk/info`
- `POST /api/risk/score`
- `GET /api/model/info`
- `GET /api/model/metrics`
- `POST /api/predict`

## Prediction Honesty

- Observed bleaching is not model output.
- Environmental stress is not a confirmed bleaching observation.
- Prediction is only returned by `POST /api/predict`.
- Prediction remains a same-month `site-month` event estimate, not a long-range forecast.
- The backend does not substitute a threshold heuristic and call it ML prediction.
- If the required contiguous 12-week weekly NOAA history is missing for a requested site/date, the API returns prediction unavailable.

## Documentation

- `docs/noaa_weekly_pipeline.md`
- `docs/modeling_decision.md`
- `docs/deployment_notes.md`
- `docs/data_label_audit.md`
