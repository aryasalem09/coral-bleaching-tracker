# Coral Bleaching Tracker

Coral bleaching explorer with three explicitly separate layers:

- Observed Bleaching: cleaned historical site observations from the BCO-DMO-derived table.
- Environmental Stress Outlook: transparent hotspot-like and DHW-like risk scoring.
- Model Prediction: supervised `site-month` probability for a binary bleaching event.

The app is designed to stay honest about what is observation, what is heuristic stress scoring, and what is supervised ML.

## Current Architecture

- Frontend: Vite + React 19 + TypeScript + `react-leaflet`
- Backend: FastAPI in `backend/api.py`
- Observed data repository: `backend/observed/repository.py`
- NOAA live-feature loader: `backend/noaa.py`
- Risk scoring: `backend/risk`
- Supervised ML pipeline: `backend/ml`
- Deprecated legacy wrappers: `backend/src`

## Data And Model Files

- Raw observed source: `backend/data/raw/global_coral_bleaching_bco_dmo.csv`
- Processed observed assets:
  - `backend/data/processed/observed_site_date_clean.csv`
  - `backend/data/processed/observed_site_catalog.csv`
  - `backend/data/processed/observed_site_month_dataset.csv`
- Model artifacts:
  - `backend/ml/artifacts/bleaching_event_model.joblib`
  - `backend/ml/artifacts/model_info.json`
  - `backend/ml/artifacts/metrics.json`
  - `backend/ml/artifacts/training_report.md`

Optional NOAA daily files can be stored under:

- `backend/data/raw/noaa_dhw`
- `backend/data/raw/noaa_hs`

If those NOAA files are absent, the website still runs. Risk and prediction fall back to historical site-month context when possible.

## Local Run

### Backend

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python -m uvicorn backend.api:app --reload
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

## Local Validation Commands

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

Backend:

```bash
python -m uvicorn backend.api:app --reload
```

Useful routes:

- `GET /health`
- `GET /api/summary`
- `GET /api/sites?south=...&west=...&north=...&east=...&limit=...`
- `GET /api/site/{site_id}`
- `GET /api/site/{site_id}/observations`
- `GET /api/risk/info`
- `POST /api/risk/score`
- `GET /api/model/info`
- `GET /api/model/metrics`
- `POST /api/predict`

## Prediction Honesty

- Observed bleaching is not model output.
- Environmental stress is not confirmed bleaching.
- Prediction is only returned by `POST /api/predict`.
- Prediction is a same-month `site-month` event estimate, not a long-range forecast.
- Published evaluation is time-held-out, not fully site-independent.
- If the model bundle is missing or invalid, the backend returns an explicit unavailable message instead of silently substituting heuristic risk output.

## Rebuild The Model

If the prediction bundle needs to be regenerated:

```bash
python -m backend.ml.build_dataset
python -m backend.ml.train_model
```

This rewrites the bundle and the metadata files in `backend/ml/artifacts`.

## Deployment Notes

- Local frontend builds now use `/` as the base path.
- GitHub Pages builds use `/coral-bleaching-tracker/` automatically when built in GitHub Actions.
- The old `backend/src` scripts are legacy compatibility paths only and should not be used for new development.

## Docs

- `docs/merge_readiness_audit.md`
- `docs/system_audit.md`
- `docs/data_label_audit.md`
- `docs/dataset_construction_rules.md`
- `docs/modeling_decision.md`
- `docs/deployment_notes.md`
- `docs/refactor_audit.md`
