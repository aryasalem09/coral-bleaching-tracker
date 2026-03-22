# Modeling Decision

## Ground Truth Available In The Repo

The repo has real observed bleaching outcomes from the BCO-DMO global bleaching table, but they are not all equally trustworthy for supervised learning.

What exists:

- direct numeric percent bleaching
- site coordinates and region metadata
- same-row environmental covariates
- enough site-month rows to support supervised learning

What is excluded from the production target:

- comment-derived or coded bleaching percentages
- free-text severity descriptions
- coarse `Bleaching_Level` categories

## Target Candidates Considered

### Option A: Multiclass Severity Classification

Candidate classes:

- none
- mild
- moderate
- severe

Reasons rejected as the main production target:

- the bins are analyst-imposed thresholds rather than a globally harmonized survey protocol
- moderate and severe support are much thinner after removing comment-derived rows
- the class boundary meaning is less stable across sources than the simple event boundary

### Option B: Regression On Percent Bleaching

Reasons rejected as the main production target:

- percent bleaching is real, but cross-source measurement heterogeneity remains substantial
- 1,409 site-date rows needed numeric conflict averaging
- 2,559 aggregated site-date rows were flagged as comment-derived and excluded
- a regression output would overstate precision relative to the cleaned signal quality

### Option C: Binary Bleaching Event Classification

Definition:

- `binary_bleaching_event = observed_percent_bleaching > 0`

Reasons chosen:

- the target is built from observed bleaching outcomes, not environmental thresholds
- it survives label cleanup with the strongest support
- it is easier to explain honestly than fragile severity classes or noisy regression values

## Prediction Unit Decision

Candidate units considered:

- site-date
- site-week
- site-month
- region-month

Chosen unit:

- `site-month`

Why:

- the raw source often behaves like month-level timing even when stored as a day
- `site-month` avoids false day-level precision
- it preserves site specificity better than `region-month`

## Production Model Choice

Candidate models trained:

- logistic regression
- histogram-based gradient boosting
- climatology baseline

Selection rule:

- highest validation PR-AUC

Selected model:

- `hist_gradient_boosting`

Decision threshold:

- `0.25`, chosen from the validation set by F1 search and then applied unchanged to the held-out test split

## Current Production Metrics

### Selected Model Test Metrics

- AUROC: 0.642
- PR-AUC: 0.471
- F1: 0.527
- Precision: 0.427
- Recall: 0.688
- Brier score: 0.234

### Climatology Baseline Test Metrics

- AUROC: 0.500
- PR-AUC: 0.351
- F1: 0.519

### Additional New-Site-Only Test Slice

- AUROC: 0.539
- PR-AUC: 0.522
- F1: 0.550

Interpretation:

- the selected model beats climatology on ranking metrics
- performance is still moderate and should be framed as cautious probabilistic support
- the extra new-site slice is informative, but it is not a full spatial holdout design

## Split Logic Caveat

The published split is time-aware, but not site-independent.

Current overlap summary from `backend/ml/artifacts/metrics.json`:

- train/validation overlapping sites: 379
- train/test overlapping sites: 248
- validation/test overlapping sites: 456
- test-only new sites: 383

That means the held-out evaluation is useful for future-time generalization, but it should not be marketed as a strict new-reef benchmark.

## What Was Rejected And Why

- Multiclass as the production target: too dependent on unstable threshold bins after label cleanup
- Regression as the production target: too noisy and too easy to overclaim
- Any synthetic or heuristic prediction workflow: scientifically unacceptable

## Remaining Limitations

- Cross-source heterogeneity still limits confidence in the label system.
- The production evaluation is time-held-out rather than site-independent.
- The deployed endpoint is a same-month site-event estimate, not a long-range forecast.
- Live NOAA scoring in this workspace still depends on whether local daily NOAA files are present.
- The model is decision-support tooling, not a definitive ecological statement.
