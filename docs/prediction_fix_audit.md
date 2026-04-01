# Prediction Fix Audit

## Current Architecture

- Backend API entrypoint: `backend/api.py`
- Observed data repository and site lookup: `backend/observed/repository.py`
- Weekly NOAA availability, cache, and sampling:
  - `backend/noaa.py`
  - `backend/noaa_index.py`
  - `backend/noaa_sampling.py`
  - `backend/noaa_cache.py`
- Model training/export:
  - `backend/ml/build_dataset.py`
  - `backend/ml/noaa_weekly_features.py`
  - `backend/ml/train_model.py`
- Model loading/inference:
  - `backend/ml/model_registry.py`
  - `backend/ml/predict.py`
- Frontend selected-site rendering:
  - `frontend/src/components/map/MapEstimateLeaflet.tsx`
  - `frontend/src/components/help/LayerExplainer.tsx`
  - `frontend/src/lib/api.ts`

## Where Observed Data Comes From

- Cleaned observed survey rows live in `backend/data/processed/observed_site_date_clean.csv`.
- Site-level summary metadata lives in `backend/data/processed/observed_site_catalog.csv`.
- The observed timeline in the UI should be interpreted as sparse survey-backed dates only.
- Many sites legitimately have a single observed survey date after cleaning and deduplication.

## Where NOAA Weekly Data Comes From

- The full weekly Monday NOAA source is the CRW daily NetCDF archive sampled at Monday dates.
- Local weekly NOAA files are stored under:
  - `backend/data/raw/noaa_dhw/`
  - `backend/data/raw/noaa_hs/`
- The repo does not commit that full raw cache.
- For model training, weekly Monday NOAA history was already transformed into site-month features and written into `backend/data/processed/observed_site_month_dataset.csv`.
- For UI weekly-history display, the backend now reconstructs Monday history from the local cache when present and can fill missing Monday files on demand.

## Where Model Artifacts Are Loaded

- Primary model bundle: `backend/ml/artifacts/bleaching_event_model.joblib`
- Runtime metadata: `backend/ml/artifacts/model_info.json`
- Evaluation metrics: `backend/ml/artifacts/metrics.json`
- Runtime health and deserialization are handled in `backend/ml/model_registry.py`.

## What Was Broken

- The backend dependency spec left `scikit-learn` unpinned.
- The committed joblib artifact had been serialized with `scikit-learn 1.6.1`.
- Render/local installs pulled newer sklearn builds, which reproduced:
  - `Can't get attribute '_RemainderColsList' on module sklearn.compose._column_transformer`
- The existing `/api/predict` route tried to assemble live weekly NOAA history from raw Monday NetCDF files every time.
- Fresh repo/deploy environments did not include that raw NOAA cache, so prediction often returned unavailable even when the archived model-ready site-month features already existed.
- The frontend also conflated:
  - sparse observed survey dates
  - denser weekly NOAA environmental history
  - model prediction output
- When prediction was unavailable, the UI could still imply a threshold interpretation instead of explicitly saying the model was unavailable.

## What Changed

- Pinned backend dependencies, including `scikit-learn==1.6.1`, in `backend/requirements.txt`.
- Added `backend/requirements-dev.txt` for API verification tooling.
- Re-trained and re-exported the model under the pinned sklearn version, and recorded `trained_with_sklearn_version` in the artifact metadata.
- Extended runtime model health reporting:
  - `/health`
  - `/api/model/status`
  - `/api/summary`
- Added explicit model runtime fields:
  - `model_loaded`
  - `model_version`
  - `artifact_path`
  - `sklearn_version`
  - `trained_with_sklearn_version`
  - `loader_error`
- Hardened prediction inference so historical observed dates can use the archived model-ready site-month row instead of requiring live NOAA reconstruction for every request.
- Added `backend/noaa_cache.py` so weekly Monday NOAA files can be fetched on demand when the raw cache is missing.
- Added explicit selected-site payload design with separate concepts in `/api/site/{site_id}/analysis`:
  - `observed_summary`
  - `observed_timeline`
  - `environmental_noaa.stress_outlook`
  - `environmental_noaa.weekly_history`
  - `prediction`
  - `model_metadata`
  - `data_availability`
- Updated the frontend so:
  - observed timeline language stays survey-only
  - environmental tab presents NOAA weekly history separately
  - prediction tab shows model-only language
  - unavailable prediction states say the model is unavailable instead of implying “below threshold”
- Added backend API tests in `backend/tests/test_prediction_api.py`.
- Added end-to-end verification output in `docs/prediction_verification.md`.
