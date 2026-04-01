# Model Training Report

Selected model: `weekly_history_hist_gradient_boosting`
Model version: `2026.03.31`
Feature set: `weekly_history`
Model family: `hist_gradient_boosting`
Climatology baseline probability: `0.571`
Decision threshold: `0.25`

## Modeling decision

- Production target remains binary site-month bleaching event prediction.
- Weekly NOAA Monday files are aligned to the nearest Monday on or before each observed site-date.
- Lagged, rolling, and trend heat-stress features use only current-or-earlier Mondays, so the model does not look into the future.

## Test metrics

- AUROC: 0.666
- PR-AUC: 0.516
- F1: 0.551
- Precision: 0.433
- Recall: 0.756
- Brier score: 0.224

## Formulation comparison

- Legacy same-month HGB test PR-AUC: 0.468
- Weekly-history HGB test PR-AUC: 0.516
- Weekly minus legacy PR-AUC: +0.048
- Weekly minus legacy AUROC: +0.028

## Climatology baseline

- Test PR-AUC: 0.350
- Test AUROC: 0.500

## Split overlap audit

- Train/validation overlapping sites: 379
- Train/test overlapping sites: 248
- Validation/test overlapping sites: 456
- Test-only new sites: 391