# Refactor Audit

## Scope

This audit was written skeptically against the refactor itself. Claims were checked against code paths, processed data, artifacts, and backend smoke tests run locally on 2026-03-17.

## Claim-By-Claim Verification

| Claim | Status | Evidence | Audit Result |
| --- | --- | --- | --- |
| 1. The app is now a three-layer system | VERIFIED | `frontend/src/components/map/MapEstimateLeaflet.tsx` layer toggle and per-layer cards, `frontend/src/components/help/LayerExplainer.tsx`, `backend/api.py` endpoints `/api/site/*`, `/api/risk/*`, `/api/model/*`, `/api/predict` | Observed, risk, and prediction are now distinct in both API shape and UI structure. |
| 2. The old weak-label path was replaced by a real observed-label workflow | VERIFIED | `backend/ml/label_standardization.py::standardize_labels`, `backend/ml/build_dataset.py::build_modeling_dataset`, `backend/observed/repository.py::get_observed_site_dates` | The production data flow now comes from cleaned observed rows, not synthetic labels. |
| 3. The production target is binary site-month bleaching event prediction | VERIFIED | `backend/ml/feature_definitions.py::PRODUCTION_TARGET_NAME`, `backend/ml/feature_definitions.py::PREDICTION_UNIT`, `backend/ml/build_dataset.py` | The deployed target is `binary_bleaching_event` on `site-month` rows. |
| 4. Binary was chosen over multiclass/regression for defensible reasons | VERIFIED | `docs/modeling_decision.md`, `docs/data_label_audit.md`, `docs/dataset_construction_rules.md`, `backend/ml/label_standardization.py` | After excluding comment-derived labels and reviewing class support, binary is the most defensible production target. |
| 5. Histogram gradient boosting is the deployed model | VERIFIED | `backend/ml/train_model.py::train_and_evaluate`, `backend/ml/artifacts/model_info.json`, `backend/ml/predict.py::predict_event_probability` | The saved bundle and model info both point to `hist_gradient_boosting`. |
| 6. The reported metrics are from a proper held-out evaluation | PARTIALLY VERIFIED | `backend/ml/split_strategy.py::assign_time_split`, `backend/ml/train_model.py::train_and_evaluate`, `backend/ml/artifacts/metrics.json` | Metrics come from a time-held-out test split, but the split is not site-independent. This was under-described before the audit and is now documented explicitly. |
| 7. Frontend load is lighter because sites load by viewport | PARTIALLY VERIFIED | `frontend/src/components/map/MapEstimateLeaflet.tsx::fetchViewportSites`, `frontend/src/lib/api.ts::getSites`, `backend/api.py::sites`, `backend/observed/repository.py::list_sites_in_bbox` | The code is viewport-based and summary-only on first load. Browser build/runtime was not verified here because `node` is unavailable in this environment. |
| 8. Reef detail, risk, and prediction load on demand | VERIFIED | `frontend/src/components/map/MapEstimateLeaflet.tsx::handleSiteSelect`, analysis `useEffect`, `backend/api.py` reef, risk, and prediction endpoints | These requests are deferred until reef selection or layer use. |
| 9. Capability tiers actually affect rendering | PARTIALLY VERIFIED | `frontend/src/hooks/useCapabilityTier.ts`, `frontend/src/components/map/MapEstimateLeaflet.tsx::viewportLimitForTier`, label tile conditional, `frontend/src/styles/layout.css` capability selectors | Tiers now change point density, label tiles, backdrop blur, ambient effects, and transition behavior. Static code inspection confirms this; browser runtime was not verified here. |
| 10. Newest-valid-date fallback really works | PARTIALLY VERIFIED | `backend/observed/repository.py::recommended_observed_date`, `backend/api.py::_resolve_dynamic_features`, `frontend/src/lib/dateUtils.ts::pickNewestUsableObservationDate`, `frontend/src/components/map/MapEstimateLeaflet.tsx` active-layer date alignment | Reef click now starts on the newest observed usable date, and risk or prediction can realign to backend-returned older usable dates. This is stronger than before, but still not browser-tested end-to-end in this environment. |
| 11. Timeline is reef-specific | VERIFIED | `frontend/src/components/map/MapEstimateLeaflet.tsx`, `backend/api.py::site_detail`, `backend/api.py::site_observations` | Timeline values come from the selected reef’s observation list, not a global timeline. |
| 12. Observed, risk, and prediction are separated in wording and UI | VERIFIED | `frontend/src/App.tsx`, `frontend/src/components/map/MapEstimateLeaflet.tsx`, `frontend/src/components/help/LayerExplainer.tsx`, `backend/api.py::risk_info`, `backend/api.py::predict` | Naming and help text now separate recorded outcomes, environmental outlook, and supervised prediction. |
| 13. The tutorial or explainer is accurate | PARTIALLY VERIFIED | `frontend/src/components/help/LayerExplainer.tsx`, `frontend/src/components/ui/{TutorialModal.tsx,HelpModal.tsx,HeroIntro.tsx}` | The active explainer is accurate after audit edits. Some older tutorial components are still unused code paths, so they were cleaned but not runtime-verified. |
| 14. Legacy misleading scripts now fail loudly | PARTIALLY VERIFIED | `backend/src/_deprecated_pipeline.py`, deprecated wrappers in `backend/src`, compatibility shims `backend/src/train.py` and `backend/src/api_model_predict.py` | Most misleading generator and analysis scripts exit loudly. Two legacy entrypoints remain as compatibility wrappers instead of fail-loud stubs. |

## ML And Data Science Validity Audit

### A. Target Validity

- Status: improved to acceptable, but still cautious
- Evidence:
  - `backend/ml/label_standardization.py` now flags `label_is_comment_derived`
  - `is_direct_observation` excludes comment-derived percent values
  - `backend/ml/build_dataset.py` excludes `has_derived_label_input`
- Result:
  - The production target is now built from observed bleaching percentages rather than environmental thresholds.
  - The earlier implementation was too generous toward comment-derived numeric rows. That was patched during this audit.

### B. Leakage

- Status: no obvious future-feature leakage found
- Evidence:
  - `backend/ml/feature_definitions.py` uses static site features, same-row stress features, and month seasonality only
  - `backend/ml/build_dataset.py` uses same-row covariates already paired with the observation
  - `backend/ml/split_strategy.py` is purely time-based
  - `backend/api.py::_resolve_dynamic_features` falls back only to historical rows with `date <= requested_date`
- Result:
  - I did not find direct future leakage.
  - The pipeline is still limited by source alignment assumptions because the observed table already bundles labels and environmental covariates together.

### C. Splits

- Status: time-aware but not site-independent
- Evidence:
  - `backend/ml/split_strategy.py`
  - `backend/ml/train_model.py` writes `split_overlap_summary`
  - `backend/ml/artifacts/metrics.json`
- Result:
  - This is not classic leakage, but it is a meaningful generalization caveat.
  - The evaluation should be read as future-time performance with some repeated-site exposure, not strict new-reef performance.

### D. Label Processing

- Status: deterministic and auditable after fixes
- Evidence:
  - `backend/ml/label_standardization.py::standardize_labels`
  - `backend/data/processed/observed_conflicts.csv`
  - `backend/data/processed/observed_exclusions.csv`
- Result:
  - Duplicate handling is deterministic.
  - Conflict rows are logged.
  - Provenance is preserved.
  - Missing percent-bleaching rows were incorrectly coerced to event `0` before this audit; that bug is now fixed.

### E. Model Evaluation

- Status: held-out and honest, but still limited
- Evidence:
  - `backend/ml/train_model.py`
  - `backend/ml/evaluate_model.py`
  - `backend/ml/artifacts/metrics.json`
- Result:
  - Metrics are computed on held-out test rows.
  - Threshold selection happens on validation only, then transfers to test.
  - Calibration diagnostics exist, but the app should still avoid over-interpreting probability as certainty.

### F. Production Honesty

- Status: acceptable after fixes
- Evidence:
  - `backend/api.py::predict`
  - `backend/ml/predict.py`
- Result:
  - `/api/predict` returns actual model output from the saved model bundle.
  - It does not silently fall back to the heuristic risk score.
  - `/api/risk/score` is now explicitly separate and can succeed where prediction does not.

### Prediction-System Verdict

PLAUSIBLE BUT NEEDS FIXES

Reason:

- The target is now direct enough to defend.
- I did not find obvious future leakage.
- Held-out evaluation is real.
- But the current published evaluation is still not site-independent, and label heterogeneity remains a meaningful limitation.

## Frontend Audit

### Verified Strengths

- Request cancellation and request IDs guard against stale site or map responses in `frontend/src/components/map/MapEstimateLeaflet.tsx`.
- Viewport loading is real and summary-only on first load.
- Risk and prediction availability are now handled separately instead of one request masking the other.
- Risk and prediction now fetch only when their layer is active instead of eagerly on every date change.
- Low capability mode now removes label tiles and heavier visual effects.

### Issues Found And Fixed

- Warmup polling could restart unnecessarily because `ensureBackendReady` captured stale status from React state.
- Risk and model metadata could fail once during cold start and never retry.
- Risk was incorrectly hidden whenever model-ready dates were absent.
- The model PR-AUC card was hardcoded to `hist_gradient_boosting` instead of the selected model name.
- Several unmounted help/tutorial components contained stale wording.

### Still Unverified

- No frontend build or browser runtime verification was possible because `node` is not installed in this environment.

## Backend And API Audit

### Verified Strengths

- `/health` is lightweight and does not load the model bundle.
- Model loading is lazy through `backend/ml/model_registry.py`.
- Summary and viewport-site routes are lighter than reef-detail routes.
- Prediction and risk endpoints are separate.

### Issues Found And Fixed

- Risk historical fallback was previously tied to model eligibility and could 404 unnecessarily on risk-only reefs.
- Missing percent-bleaching rows were exposed as event `0` in observed API payloads.
- Prediction coverage wording overstated the evaluation before the audit; it now mentions the time-held-out but not site-independent limitation.

## Documentation Consistency Audit

Updated during this audit:

- `README.md`
- `docs/data_label_audit.md`
- `docs/dataset_construction_rules.md`
- `docs/modeling_decision.md`
- `docs/system_audit.md`
- `docs/deployment_notes.md`

Main corrections:

- removed stale counts from the stricter derived-label exclusion
- updated the deployed threshold to `0.25`
- updated held-out metrics to match current artifacts
- clarified that prediction is same-month and not a long-range forecast
- clarified that the published evaluation is not site-independent
- clarified that risk fallback is broader than prediction fallback

## Patches Applied During This Audit

- Excluded comment-derived percent rows more strictly from supervised eligibility.
- Added cache invalidation checks so stale processed files rebuild after schema or semantics fixes.
- Fixed missing observed labels being coerced to event `0`.
- Separated risk historical fallback from model historical fallback.
- Made the frontend analysis layer robust to risk-only vs prediction-only availability.
- Stopped eager risk and prediction fetching on the observed layer.
- Reduced low-tier frontend workload further.
- Cleaned stale wording in active and inactive explainer components.

## Remaining Unresolved Concerns

- Frontend build and browser runtime remain unverified in this environment because `node` is missing.
- The model evaluation is still time-held-out rather than site-independent.
- The model remains a same-month event estimator, not a true forward forecast.
- Live NOAA behavior here depends on local NOAA daily files, which were not present during the smoke tests.
