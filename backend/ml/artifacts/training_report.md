# Model Training Report

Selected model: `hist_gradient_boosting`
Model version: `2026.03.17`
Climatology baseline probability: `0.571`
Decision threshold: `0.25`

## Test metrics

- AUROC: 0.642
- PR-AUC: 0.471
- F1: 0.527
- Precision: 0.427
- Recall: 0.688
- Brier score: 0.234

## Climatology baseline

- Test PR-AUC: 0.351
- Test AUROC: 0.500

## Split overlap audit

- Train/validation overlapping sites: 379
- Train/test overlapping sites: 248
- Validation/test overlapping sites: 456
- Test-only new sites: 383