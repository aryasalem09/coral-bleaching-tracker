# Forecast Model Training Report

Selected model: `forecast_4w_hist_gradient_boosting`
Model version: `2026.04.01`
Model family: `hist_gradient_boosting`
Forecast horizon: `4 weeks`
Feature history: `12 weeks`
Decision threshold: `0.25`

## What the model does

- Ground truth comes from direct observed bleaching records, not NOAA itself.
- The model predicts whether bleaching will be observed in the next 4 weeks after the forecast issue date.
- NOAA HotSpot and DHW values are predictors, not labels.
- This is a probabilistic forecast, not a confirmed observation.

## Split strategy

- Train rows use the earliest forecast issue dates.
- Validation rows use later dates.
- Test rows use the latest dates.
- Rows whose 4-week label window would cross a split boundary are excluded from training and validation.

## Validation metrics

- PR-AUC: 0.871
- AUROC: 0.820
- F1: 0.782
- Precision: 0.697
- Recall: 0.889
- Brier score: 0.177

## Test metrics

- PR-AUC: 0.523
- AUROC: 0.671
- F1: 0.541
- Precision: 0.417
- Recall: 0.770
- Brier score: 0.221
- Confusion matrix counts: TN 419, FP 558, FN 119, TP 399

## Baseline comparison

- Climatology test PR-AUC: 0.346
- Climatology test AUROC: 0.500

## Class balance

- Train positive rate: 0.567
- Validation positive rate: 0.566
- Test positive rate: 0.346