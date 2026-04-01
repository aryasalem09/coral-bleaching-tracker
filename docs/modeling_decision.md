# Modeling Decision

## Production Target

The production target remains a binary bleaching event at the `site-month` level.

This project did **not** switch to a weekly target because the observed bleaching labels are not a reliable weekly supervision stream. The observed data are sparse site-dates drawn from heterogeneous source programs, so a weekly target would overstate label precision.

## Why Weekly NOAA Inputs Still Help

Weekly NOAA history materially improves the input formulation without changing the target definition.

The model now uses:

- current weekly Monday HotSpot and DHW
- lagged weekly Monday HotSpot and DHW
- rolling weekly means, maxima, minima, and trend slopes
- weekly coverage indicators
- static reef/site covariates
- seasonal month encoding

This preserves a defensible target while giving the model much richer thermal-stress context than a single same-month snapshot.

## Temporal Alignment

For each observed `site-month` row:

1. find the nearest available Monday on or before the observed site-date
2. treat that Monday as the current weekly anchor
3. build lagged and rolling features using only that Monday and earlier Mondays

This prevents future leakage.

## Leakage Controls

- Weekly NOAA features are never pulled from after the observed site-date.
- Model selection still uses held-out future years.
- Metrics are published with an explicit note that the primary split is time-held-out, not fully site-independent.
- A separate new-site subset is retained as an additional stress test.

## Comparison Strategy

Training compares:

- legacy same-month feature candidates
- weekly-history feature candidates
- climatology baseline

The selected production model is whichever candidate wins on validation PR-AUC, with test and new-site metrics saved in `backend/ml/artifacts/metrics.json`.

## Current Result

Latest training selected:

- `weekly_history_hist_gradient_boosting`
- test AUROC: `0.666`
- test PR-AUC: `0.516`
- test F1: `0.551`
- new-site test PR-AUC: `0.550`

Against the legacy same-month HGB baseline, the weekly-history formulation improved:

- PR-AUC by `+0.048`
- AUROC by `+0.028`
- Brier score by `-0.011`

## Runtime Decision

Prediction uses the current trained artifact only.

The backend does **not** replace the model with a heuristic threshold system. If the weekly NOAA feature window required by the model cannot be assembled for a requested site/date, the API returns prediction unavailable.

Because every eligible training row had a complete contiguous 12-week NOAA history, the runtime now requires that same 12-week support at inference time instead of extrapolating from shorter or gapped windows.
