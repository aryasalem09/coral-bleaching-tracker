# Deployment Prediction Notes

## Runtime Baseline

- Python runtime: `backend/runtime.txt` -> `python-3.11.9`
- Backend dependencies are now pinned in `backend/requirements.txt`
- Critical compatibility pin:
  - `scikit-learn==1.6.1`

## Why sklearn Is Pinned

- The production model bundle is a joblib-serialized sklearn pipeline.
- That payload includes sklearn `ColumnTransformer` internals.
- Loading the artifact under newer sklearn releases produced the `_RemainderColsList` deserialization failure.
- The repo now records the training sklearn version in `model_info.json` and in the model bundle itself.

## NOAA Weekly History Behavior

- The repo does not commit the full raw Monday NOAA cache.
- Prediction for historical observed dates no longer depends on that raw cache because the processed site-month dataset already contains model-ready weekly-derived features.
- Weekly NOAA history display now works from:
  1. local Monday NOAA cache when present
  2. on-demand Monday NOAA fetches when files are missing
- First-time weekly-history requests for a date window can therefore be slower than prediction requests.

## Relevant Environment Variables

- `AUTO_DOWNLOAD_NOAA`
  - default is now effectively enabled in code
  - set to `false` to disable on-demand NOAA cache fills
- `NOAA_DOWNLOAD_TIMEOUT_SECONDS`
  - default `60`
- `NOAA_DOWNLOAD_RETRIES`
  - default `2`
- `NOAA_DOWNLOAD_WORKERS`
  - default `4`
- `CBT_MODEL_VERSION`
  - optional override for the model version label

## Health Checks

- `GET /health`
- `GET /api/model/status`
- `GET /api/summary`

Recommended deploy smoke checks:

1. `GET /api/model/status`
   - expect `model_loaded: true`
   - expect `trained_with_sklearn_version: 1.6.1`
2. `POST /api/predict`
   - use a known observed site/date with `prefer_live=false`
   - expect `available: true`
3. `GET /api/site/{site_id}/analysis`
   - verify `observed_timeline` and `environmental_noaa.weekly_history` are separate objects

## Deploy Recommendation

- Keep Render on Python `3.11.9` to match `runtime.txt`.
- Install exactly from `backend/requirements.txt`.
- Do not remove the sklearn pin unless the model is retrained and re-exported under the new version first.
- If deploy latency matters, consider warming the NOAA cache for a few frequently demonstrated sample dates after deploy, but do not reintroduce misleading UI fallbacks if weekly history is unavailable.
