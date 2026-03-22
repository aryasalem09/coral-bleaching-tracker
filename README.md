# Coral Bleaching Tracker

Production-oriented coral bleaching explorer with three explicitly separated layers:

- Observed Bleaching: recorded site outcomes from the cleaned BCO-DMO observation table.
- Environmental Stress Outlook: transparent heat-stress scoring from hotspot-like stress and accumulated heat stress.
- Model Prediction: supervised `site-month` probability for a binary bleaching event.

## What This Refactor Actually Does

- Replaces the old synthetic or weak-label workflow with the audited pipeline in [`backend/ml`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml).
- Keeps comment-derived bleaching rows visible in the observed layer, but excludes them from supervised training.
- Separates risk scoring from prediction so the risk layer can still work when model-eligible labels are weak or missing.
- Loads visible reef sites by viewport instead of shipping the full site detail graph at startup.
- Adds backend warmup handling, a lightweight `/health` route, gzip, lazy model loading, and device-adaptive rendering.

## Architecture

- Frontend: Vite + React 19 + TypeScript + `react-leaflet`
- Backend: FastAPI
- ML: scikit-learn tabular classification pipeline
- Raw observed source: `backend/data/raw/global_bleaching_environmental.csv`
- Processed outputs:
  - `backend/data/processed/observed_site_date_clean.csv`
  - `backend/data/processed/observed_site_catalog.csv`
  - `backend/data/processed/observed_site_month_dataset.csv`
- Model artifacts:
  - `backend/ml/artifacts/model_info.json`
  - `backend/ml/artifacts/metrics.json`
  - `backend/ml/artifacts/bleaching_event_model.joblib`

## Layer Meanings

### Observed Bleaching

Observed historical bleaching after deterministic site-date aggregation, duplicate handling, provenance tracking, and conflict logging.

### Environmental Stress Outlook

Transparent thermal-stress scoring from hotspot-like stress and DHW-like accumulated heat. This is not a confirmed bleaching outcome and it is not the supervised model.

### Model Prediction

Supervised probability for `binary_bleaching_event` at the `site-month` level. It is a same-month estimate, not a long-range forecast. The published metrics are time-held-out, not fully site-independent.

## Local Setup

### Backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/src/requirements.txt
uvicorn backend.api:app --reload
```

The backend defaults to `http://127.0.0.1:8000`.

### Frontend

From [`frontend`](/Users/nandy/Downloads/coral-bleaching-tracker/frontend):

```bash
npm install
npm run dev
```

Optional environment variable:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In local development the frontend falls back to `http://127.0.0.1:8000` automatically if `VITE_API_BASE_URL` is unset.

## Training Pipeline

```bash
python3 -m backend.ml.build_dataset
python3 -m backend.ml.train_model
```

Artifacts are written to [`backend/ml/artifacts`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml/artifacts).

## API Overview

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

## Frontend Behavior

### Adaptive Rendering

The frontend infers `low`, `medium`, or `high` capability tiers from reduced-motion preference, viewport size, `deviceMemory`, and `hardwareConcurrency`. Those tiers change point density, ambient effects, tile-label rendering, and transition richness.

### Reef Click Date Behavior

When a reef is clicked, the UI starts on the newest observed date that is either analysis-ready or at least has an observed bleaching value. If the active risk or prediction layer resolves to an older backend-validated date, the timeline can realign to that older usable date.

### Timeline Scope

The timeline is reef-specific, not global. Risk and prediction requests are cancellation-aware so stale responses from rapid reef switching do not overwrite the current site.

### Cold Start UX

The frontend polls `/health`, shows a warmup banner while the backend wakes up, and defers detail fetches until the user requests them.

## Scientific Honesty Notes

- Observed bleaching is not model output.
- Environmental stress is not confirmed bleaching.
- Prediction is only used for the supervised model endpoint.
- Comment-derived bleaching percentages are excluded from supervised training even if they remain visible in the observed layer.
- The production target is binary because the continuous and severity-style labels remain heterogeneous across sources.

## Project Docs

- [System audit](/Users/nandy/Downloads/coral-bleaching-tracker/docs/system_audit.md)
- [Data and label audit](/Users/nandy/Downloads/coral-bleaching-tracker/docs/data_label_audit.md)
- [Dataset construction rules](/Users/nandy/Downloads/coral-bleaching-tracker/docs/dataset_construction_rules.md)
- [Modeling decision](/Users/nandy/Downloads/coral-bleaching-tracker/docs/modeling_decision.md)
- [Deployment notes](/Users/nandy/Downloads/coral-bleaching-tracker/docs/deployment_notes.md)
- [Refactor audit](/Users/nandy/Downloads/coral-bleaching-tracker/docs/refactor_audit.md)
