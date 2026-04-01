# Precommit Weekly NOAA Audit

Date: `2026-03-31`

## Scope

Strict pre-commit audit of the NOAA weekly Monday downloader, weekly availability/index layer, weekly-feature ML pipeline, retrained model artifacts, backend integration, frontend integration, generated outputs, and local/deployment docs.

## Repository Map

Frontend entrypoints:

- `frontend/src/App.tsx`
- `frontend/src/components/map/MapEstimateLeaflet.tsx`
- `frontend/src/lib/api.ts`

Backend entrypoints:

- `backend/api.py`
- `backend/observed/repository.py`
- `backend/noaa.py`

NOAA weekly pipeline:

- `backend/download_noaa_weekly_mondays.py`
- `backend/noaa_products.py`
- `backend/noaa_index.py`
- `backend/noaa_sampling.py`
- `backend/src/download_noaa_weekly_mondays.py`

ML pipeline:

- `backend/ml/noaa_weekly_features.py`
- `backend/ml/feature_definitions.py`
- `backend/ml/build_dataset.py`
- `backend/ml/train_model.py`
- `backend/ml/model_registry.py`
- `backend/ml/predict.py`

Generated outputs inspected:

- `backend/data/raw/noaa_manifest_weekly_mondays.json`
- `backend/data/processed/observed_site_month_dataset.csv`
- `backend/data/processed/noaa_weekly_feature_audit.json`
- `backend/data/processed/observed_exclusions.csv`
- `backend/ml/artifacts/bleaching_event_model.joblib`
- `backend/ml/artifacts/model_info.json`
- `backend/ml/artifacts/metrics.json`
- `backend/ml/artifacts/training_report.md`
- `backend/ml/artifacts/feature_importance.csv`
- `backend/ml/artifacts/confusion_matrix.png`
- `backend/ml/artifacts/precision_recall_curve.png`
- `backend/ml/artifacts/roc_curve.png`
- `backend/ml/artifacts/calibration_curve.png`

Docs inspected:

- `README.md`
- `docs/noaa_weekly_pipeline.md`
- `docs/modeling_decision.md`
- `docs/deployment_notes.md`
- `docs/data_label_audit.md`

## Executed Validation

Repository/index validation:

- Indexed the repo with `rg --files`.
- Inspected every NOAA-weekly-related changed file plus dependent runtime files.
- Ran `python3 -m py_compile $(rg --files backend -g'*.py')`.

NOAA downloader / manifest validation:

- Cross-checked manifest counts against files on disk.
- Verified `requested_dates=2141`, `ok_dates=2141`, `failed_dates=0`.
- Verified both product directories contain exactly `2141` non-empty Monday files.
- Verified no requested Mondays are missing on disk and no extra Mondays are silently present.
- Verified every manifest `ok` path exists and its `size_bytes` matches the real file size.
- Verified `paired_first_date=1985-03-25` and `paired_last_date=2026-03-30`.

Weekly feature / dataset validation:

- Rebuilt the dataset with `python3 -m backend.ml.build_dataset`.
- Verified the rebuilt dataset still has `14517` eligible rows.
- Verified split counts remain `train=9710`, `validation=3141`, `test=1666`.
- Verified every eligible row still has `weekly_history_weeks_available=12` and `weekly_missing_internal_weeks=0`.
- Verified `28` rows are excluded for `no_weekly_anchor` and `5` for invalid NOAA grid sampling.
- Verified `weekly_anchor_date <= date` for all eligible rows.

Model / artifact validation:

- Reloaded the artifact with `joblib`.
- Retrained with `python3 -m backend.ml.train_model`.
- Verified artifact, `model_info.json`, `metrics.json`, `training_report.md`, and `feature_importance.csv` are internally consistent.
- Verified selected model remains `weekly_history_hist_gradient_boosting`.
- Verified decision threshold remains `0.25`.
- Verified held-out test metrics remain:
  - `AUROC=0.6656`
  - `PR-AUC=0.5160`
  - `F1=0.5506`
  - `Precision=0.4328`
  - `Recall=0.7564`
  - `Brier=0.2242`
- Verified weekly-history vs legacy same-month HGB PR-AUC gain remains `+0.0477`.
- Verified new-site test PR-AUC remains `0.5500`.

Backend route validation:

- Enumerated FastAPI routes and verified all required endpoints exist:
  - `GET /health`
  - `GET /api/summary`
  - `GET /api/sites`
  - `GET /api/site/{site_id}`
  - `GET /api/site/{site_id}/observations`
  - `GET /api/risk/info`
  - `POST /api/risk/score`
  - `GET /api/model/info`
  - `GET /api/model/metrics`
  - `GET /api/noaa/availability`
  - `POST /api/predict`
- Exercised those routes with `fastapi.testclient.TestClient`.
- Verified `/api/summary` reports weekly NOAA coverage honestly.
- Verified `/api/noaa/availability` returns the real paired Monday coverage.
- Verified `/api/risk/score` works for live weekly NOAA context and for historical environmental fallback.
- Verified `/api/predict` returns a real prediction when a full weekly history exists and an honest unavailable response when it does not.

Frontend/static validation:

- Audited `frontend/src/lib/api.ts` against actual backend response shapes.
- Audited `frontend/src/App.tsx`, `frontend/src/components/map/MapEstimateLeaflet.tsx`, and `frontend/src/components/help/LayerExplainer.tsx` for runtime mismatches and misleading copy.
- Installed frontend dependencies with `npm ci`.
- Ran `npm run lint`.
- Ran `npm run build`.
- Ran `npm audit --omit=dev --json` and verified there are `0` production dependency vulnerabilities.

## Issues Found And Fixed

1. Prediction coverage mismatch.

- Problem: live prediction accepted short or gapped NOAA histories even though the rebuilt training dataset contains only full contiguous 12-week histories.
- Fix: enforced full contiguous 12-week history in `backend/noaa.py` and hardened dataset feature generation in `backend/ml/noaa_weekly_features.py`.
- Result: prediction now returns honest unavailable for out-of-support weekly histories instead of extrapolating beyond training coverage.

2. Risk fallback was unintentionally model-eligibility-gated.

- Problem: `get_environmental_context_dataset()` short-circuited to the model-ready dataset, which could hide valid historical environmental context from the risk layer.
- Fix: decoupled historical environmental fallback in `backend/observed/repository.py` so it always derives from observed site-date environmental records instead of model-eligible rows only.
- Result: historical risk fallback now stays separate from model eligibility, which matches the UI’s scientific separation.

3. Frontend mislabeled live weekly NOAA risk context.

- Problem: the map UI treated live weekly NOAA responses as historical context because it only looked for a nonexistent `noaa_live` mode.
- Fix: added explicit mode labeling for `noaa_weekly_monday`, `historical_environmental`, and `historical_observed` in `frontend/src/components/map/MapEstimateLeaflet.tsx`.

4. Frontend local-coverage footer could underreport NOAA availability.

- Problem: the footer depended on `summary` alone and could say there was no local NOAA coverage even if `/api/noaa/availability` had already loaded.
- Fix: made the footer fall back to the availability payload in `frontend/src/components/map/MapEstimateLeaflet.tsx`.

5. Runtime instruction mismatch.

- Problem: the missing-model runtime message still told users to run `python -m ...`.
- Fix: updated `backend/ml/model_registry.py` to use `python3 -m backend.ml.train_model`.

6. Deployment/docs clarity gaps.

- Problem: docs did not clearly state the GitHub Pages + Render assumption and did not clearly state the full 12-week inference requirement.
- Fix: updated `README.md`, `docs/noaa_weekly_pipeline.md`, `docs/modeling_decision.md`, and `docs/deployment_notes.md`.

## Remaining Risk

- `npm ci` still reports `6` dev-only vulnerabilities when dev dependencies are included. Production dependencies audited clean with `npm audit --omit=dev`.
- This Codex shell did not automatically inherit the user's `nvm` PATH, so frontend commands were executed with an explicit Node PATH prefix during the audit.

## Audit Verdict

Status: `READY TO COMMIT`

Reason:

- No remaining backend/data/model/frontend integration issue was found after the fixes above.
- The frontend lint/build gate now passes, the backend/API smoke tests pass, and the NOAA weekly pipeline plus retrained model artifacts are internally consistent.

What is ready:

- NOAA weekly Monday downloader and manifest/index coherence
- weekly feature dataset and exclusion audit
- retrained model bundle and metrics/report consistency
- backend routes and prediction/risk behavior
- docs and runtime instructions

Non-blocking follow-up:

- If you want to clean up the remaining dev-only audit warnings, run `cd frontend && npm audit` and update or pin the affected tooling packages.
