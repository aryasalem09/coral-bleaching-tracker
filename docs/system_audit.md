# System Audit

## Current Architecture

### Frontend

- Stack: Vite + React 19 + TypeScript + `react-leaflet`
- Routing: single map-first dashboard, no dedicated router
- State management: local React state with request IDs, abort controllers, deferred marker rendering, and memoized derived values
- Map stack: Leaflet tiles plus `CircleMarker` site points
- Explanation layer: integrated `LayerExplainer` plus status and footnote cards

### Backend

- Stack: FastAPI
- Main entrypoint: [`backend/api.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/api.py)
- Data services:
  - [`backend/observed/repository.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/observed/repository.py) for cached catalogs, observation timelines, and historical context lookup
  - [`backend/noaa.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/noaa.py) for live NOAA daily-file lookup and nearest-cell extraction
  - [`backend/risk`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/risk) for transparent stress scoring
  - [`backend/ml`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml) for label cleanup, dataset construction, training, evaluation, and prediction

## Main Findings

### Reef Point Loading

- Frontend requests only visible points through `GET /api/sites`.
- Backend samples the site catalog inside the requested bbox and caps density by viewport limit.

### Timeline And Reef Detail Loading

- Reef metadata comes from `GET /api/site/{site_id}`.
- Reef observation rows come from `GET /api/site/{site_id}/observations`.
- Timeline choices are reef-specific, not global.

### Bleaching Data Source

- Observed bleaching now comes from the cleaned BCO-DMO global bleaching table processed into site-date and site-month assets.
- Comment-derived rows are preserved for transparency in the observed layer but excluded from supervised training.

### Environmental Data Source

- Training uses same-row environmental covariates already present in the observed table.
- Inference first attempts local NOAA daily files.
- Historical fallback is now split correctly:
  - risk fallback uses the newest site-month row with valid environmental context
  - prediction fallback uses the newest site-month row that is both environmentally valid and model-eligible

### Prediction Logic

- The current production prediction path is real supervised ML.
- The old repo contained synthetic or weakly grounded prediction scripts; most of those now fail loudly, while a small number of legacy API and training entrypoints remain as compatibility shims.

### Performance Bottlenecks Addressed

- Initial load no longer requests full-detail reef data.
- Heavy aggregation happens in preprocessing and cached backend tables rather than repeated frontend transforms.
- Map density is reduced by viewport sampling and capability-tier limits.
- Request cancellation and monotonically increasing request IDs reduce stale overwrite bugs during rapid interaction.

### Remaining Scientific Honesty Caveat

- The deployed model is a same-month site-event estimate, not a long-range forecast.
- The published held-out metrics are time-aware, but not site-independent.

## Main Problems Found In The Original Repo

### Scientific Honesty Problems

- Synthetic and heuristic prediction paths coexisted with real observed data.
- The old vocabulary blurred NOAA stress, observed bleaching, and “prediction.”
- Weak or undocumented label assumptions were too easy to mistake for ground truth.

### Frontend Problems

- Too much data risked being loaded too early.
- Reef interaction logic was not clearly reef-specific.
- Cold-start backend behavior was not treated as a first-class UX flow.

### Backend Problems

- There was no dedicated lightweight `/health` route.
- Cached summaries and deferred detail endpoints were not clearly separated.
- Old scripts could still pull contributors toward deprecated data-generation paths.

## What Changed

### Data And Modeling

- Added audited label cleanup and dataset construction in:
  - [`backend/ml/label_standardization.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml/label_standardization.py)
  - [`backend/ml/build_dataset.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml/build_dataset.py)
  - [`backend/ml/train_model.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml/train_model.py)
- Built auditable processed assets:
  - `observed_site_date_clean.csv`
  - `observed_site_catalog.csv`
  - `observed_site_month_dataset.csv`
- Replaced the old fake prediction path with a supervised binary `site-month` model selected by validation PR-AUC.

### API And Backend

- Added `GET /health` that does not load heavy data or the model.
- Added gzip middleware for larger responses.
- Kept model loading lazy through [`backend/ml/model_registry.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/ml/model_registry.py).
- Separated risk fallback from model eligibility in [`backend/observed/repository.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/observed/repository.py) and [`backend/api.py`](/Users/nandy/Downloads/coral-bleaching-tracker/backend/api.py).
- Added clear endpoints for observed timelines, risk info and scoring, and model info, metrics, and prediction.

### Frontend

- Initial load now fetches only viewport-limited site summaries.
- Reef detail, observations, risk, and prediction load on demand.
- The analysis panel now handles risk and prediction availability separately instead of treating them as one success or failure state.
- Device capability tiers reduce actual work by lowering point density, removing label tiles on low tier, and disabling heavier ambient effects.
- A warmup banner communicates backend cold starts.

### Reef Date Logic

- Reef click starts on the newest observed date with either analysis-ready QA or at least an observed bleaching value.
- When the active risk or prediction layer resolves to an older backend-valid date, the timeline can realign to that older date.
- Request cancellation and request IDs reduce race-condition overwrites during rapid reef switching.

## Why The New Architecture Is Better

- It is more honest: observed outcomes, risk scoring, and supervised prediction are explicitly separate.
- It is lighter: viewport-limited fetches and cached tables reduce first-load cost.
- It is safer: `/health` is cheap, model load is lazy, and the API does not silently turn risk into prediction.
- It is easier to audit: the observed-label, risk, and prediction paths now live in explicit modules instead of scattered legacy scripts.
