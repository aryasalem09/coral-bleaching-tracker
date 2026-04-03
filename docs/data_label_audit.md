# Data Label Audit

## Observed Label Source

The supervised target is built from the cleaned processed observation assets derived from the global coral bleaching table already used by the project.

The standardized observation pipeline preserves:

- direct observation flags
- comment-derived label flags
- conflict history for duplicated site-date rows
- source provenance counts
- label quality scores

## Why The Target Stays Binary

Percent bleaching is scientifically useful, but the source table mixes heterogeneous measurement protocols and source programs. A binary event target remains the most defensible production target in this repository.

The training pipeline excludes:

- missing observed percent bleaching
- comment-derived label rows
- low-quality label rows
- rows without usable weekly NOAA alignment

## Weekly NOAA Upgrade Does Not Relabel The Data

The weekly NOAA expansion improves the **features**, not the labels.

That distinction matters:

- observed bleaching remains observation-derived
- weekly NOAA inputs are environmental covariates
- the production model now predicts a binary event probability for the next 4 weeks after the forecast issue date

## Residual Limitations

- label definitions still vary across contributing source programs
- the held-out evaluation is primarily time-based, not fully site-independent
- early observations before NOAA weekly coverage begins cannot be aligned to the weekly feature pipeline
- a small number of candidate rows are still lost when no valid NOAA ocean grid cell can be sampled near the recorded site coordinate

Current legacy feature-store audit:

- candidate site-month rows examined: `19,676`
- rows aligned successfully to weekly NOAA history: `19,643`
- rows excluded because no prior NOAA Monday existed: `28`
- rows excluded because no valid NOAA ocean grid cell could be sampled: `5`

Those counts describe the archived weekly feature store that the forecast dataset is built from. The production target is now forward-looking even though the feature-source rows were originally assembled at the site-month level.
