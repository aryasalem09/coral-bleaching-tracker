# Forecast Modeling Decision

## Old Model

The old production model was not a true forecast.

It used a `site-month` row and predicted whether that same site-month had any observed bleaching. Even though the weekly NOAA features were anchored to the nearest earlier Monday, the target still came from the same observed period as the row.

## New Model

The new production model is a forward-looking forecast.

- Forecast issue date: Monday anchor date `t`
- Feature window: 12 weeks of NOAA Monday HotSpot and DHW history ending at `t`, plus static site factors
- Forecast horizon: next 4 weeks
- Prediction target: whether a direct observed bleaching event will be recorded for that site during `(t, t + 4 weeks]`

This is a probabilistic forecast, not a confirmed observation.

## Ground Truth

- Ground truth comes from direct observed bleaching records, not NOAA itself.
- NOAA heat data are predictors, not labels.
- Label `1` means at least one direct survey in the next 4 weeks reported bleaching above 0%.
- Label `0` means there was at least one direct survey in the next 4 weeks and none reported bleaching.
- Rows without any direct survey in the next 4 weeks are excluded from training and evaluation so missing surveys are not treated as negative labels.

## Leakage Controls

- Features use only information available on or before the Monday forecast issue date.
- The label window starts strictly after the anchor date.
- No future NOAA values are used in the predictive path.
- No observation-derived features from the future window are used as predictors.
- Train and validation splits use a 4-week purge window around the split boundaries so labels do not spill into later periods.

## Split Strategy

- Train: earliest forecast issue dates through `2012-12-03`
- Validation: `2013-01-01` through `2016-12-03`
- Test: after `2016-12-31`
- Dates near the train and validation cutoffs are excluded because their 4-week label windows would overlap the next split.

## Model Selection

Candidates compared on the forecast-safe dataset:

- Logistic regression
- HistGradientBoosting
- Climatology baseline

Selection metric: validation PR-AUC.

Threshold rule: best validation F1.

## Selected Model

- Model: `forecast_4w_hist_gradient_boosting`
- Validation PR-AUC: `0.871`
- Validation AUROC: `0.820`
- Test PR-AUC: `0.523`
- Test AUROC: `0.671`
- Test F1: `0.541`
- Test precision: `0.417`
- Test recall: `0.770`
- Test Brier score: `0.221`
- Climatology test PR-AUC: `0.346`

The forecast model beats the trivial climatology baseline on held-out test PR-AUC and AUROC, but the scores are still moderate. It should be treated as a rough risk forecast, not a definitive event detector.

## Limitations

- Survey coverage is sparse and irregular, so the forecast dataset only includes issue dates with direct survey coverage in the next 4 weeks.
- Many sites appear in multiple time splits. That is acceptable for time-ordered forecasting, but it is not a fully site-independent evaluation.
- The live forecast path still requires a full contiguous 12-week NOAA history for the requested issue date.
- The production path currently serves the 4-week horizon only. The code is structured so a longer horizon can be added later, but it is not exposed as a user-facing prediction yet.
