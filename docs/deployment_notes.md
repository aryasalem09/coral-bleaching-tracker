# Deployment Notes

## Local Runtime

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python3 -m uvicorn backend.api:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

If `npm` is unavailable in the environment, frontend lint/build validation has to be performed on a machine with Node.js installed.

## Deployment Shape

- The frontend is configured for GitHub Pages via [`.github/workflows/pages.yml`](/Users/nandy/Downloads/coral-bleaching-tracker/.github/workflows/pages.yml).
- The production frontend environment points at the Render-hosted backend URL in [`frontend/.env.production`](/Users/nandy/Downloads/coral-bleaching-tracker/frontend/.env.production).
- If you deploy the backend somewhere other than Render, update `VITE_API_BASE_URL` before publishing the frontend.

## NOAA Data Expectations

The app can still boot without the raw NOAA weekly cache.

- observed bleaching endpoints still work
- historical environmental fallback still works where processed context exists
- prediction returns unavailable if the trained artifact or the required contiguous 12-week NOAA inputs are missing

The raw NOAA weekly archive is intentionally local-only because of size.

## Model Artifact Compatibility

Scikit-learn model artifacts are environment-sensitive. If the runtime reports the bundle as invalid, retrain in the current environment:

```bash
python3 -m backend.ml.train_model
```

That regenerates:

- `backend/ml/artifacts/bleaching_event_model.joblib`
- `backend/ml/artifacts/model_info.json`
- `backend/ml/artifacts/metrics.json`
- `backend/ml/artifacts/training_report.md`

## Startup Weight

The backend does not scan or open all NOAA files at import time.

- NOAA availability is cached lazily
- site sampling opens only the specific weekly files needed for the request
- prediction builds the weekly feature window on demand

## Production Honesty

- Observed Bleaching is not model output.
- Environmental Stress Outlook is not a confirmed bleaching observation.
- Model Prediction is supervised model output only.
- No heuristic threshold fallback is presented as ML prediction.
