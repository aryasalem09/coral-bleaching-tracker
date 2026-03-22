# Dataset Construction Rules

## Production Unit

- One production modeling row = one `site-month`
- Raw observations are standardized at `site-date`
- Site-date rows are then aggregated into `site-month` rows

## Label Standardization Rules

### Column Normalization

- Raw BCO-DMO columns are renamed into a consistent snake_case schema.
- Object fields normalize empty strings, `nd`, `nan`, and `none` to missing.
- Numeric columns are coerced with `errors="coerce"`.
- Dates are parsed to pandas datetimes.

### Identifier And Region Handling

- `site_id` is the primary production key.
- Region and site text fields are normalized through trimming and null-standardization.
- The older v1 extract is not merged into production because it lacks the same `site_id` / `sample_id` structure.

### Severity Standardization

- Direct numeric label: `observed_percent_bleaching`
- Descriptive bins used only for UI labeling:
  - `none`: 0
  - `mild`: 0 to 10
  - `moderate`: 10 to 30
  - `severe`: 30 to 100

These bins are descriptive only. The production supervised target is binary.

### Missing And Unknown Handling

- Missing numeric values remain missing.
- Missing bleaching values are not coerced into `0` event labels.
- Missing text values become `pd.NA`.
- Derived labels are never silently substituted for missing direct numeric labels.

## Deduplication Rules

### Site-Date Grouping

- Raw rows are sorted by:
  - `site_id`
  - `date`
  - metadata completeness descending
  - `sample_id`
- Group key: `site_id + date`

### Aggregation Behavior

- Coordinates and numeric site or environmental fields: mean
- Provenance sources: unique sorted list
- Provenance sample IDs: unique sorted list, truncated for logging
- Metadata text fields: taken from the highest-completeness row after deterministic sorting

## Conflict Resolution Rules

- Numeric conflicts are detected when a site-date group contains more than one percent bleaching value.
- Numeric bleaching conflict rule: average the available numeric values.
- Metadata conflict rule: keep the highest-completeness row after deterministic sorting.
- Conflict history is preserved:
  - `has_conflict_history = true`
  - conflict rows are written to `backend/data/processed/observed_conflicts.csv`

## Comment-Derived Label Rules

- `label_is_comment_derived = true` when comment text suggests coded or backfilled bleaching values.
- Site-date rows with `label_is_comment_derived = true` become:
  - `is_direct_observation = false`
  - `is_derived_label = true`
- These rows remain in the observed timeline for transparency, but they are excluded from supervised target eligibility.
- At the site-month stage, `has_derived_label_input = true` excludes the row from `target_eligible`.

## Label QA Flags

Each site-date row includes:

- `is_direct_observation`
- `is_derived_label`
- `has_precise_date`
- `has_precise_location`
- `has_conflict_history`
- `missing_metadata_level`
- `source_count`
- `label_quality_score`
- `recommended_for_modeling`

Each site-month row additionally includes:

- `has_derived_label_input`
- `target_is_direct_observation`
- `target_is_binary_derivation`
- `target_eligible`

## Label Quality Score

The score is bounded to `[0, 1]`.

It rewards:

- direct observation
- precise date
- precise location
- no conflict history
- multiple contributing sources when present

It penalizes:

- missing metadata
- comment-derived label status

## Inclusion And Exclusion Rules

### Site-Date Recommendation Rule

`recommended_for_modeling = true` requires:

- direct observed percent bleaching
- valid date
- valid latitude and longitude
- non-missing hotspot-like stress
- non-missing DHW-like stress
- `label_quality_score >= 0.45`

### Final Site-Month Eligibility Rule

`target_eligible = true` requires:

- non-missing date
- non-missing observed percent bleaching
- non-missing latitude and longitude
- non-missing hotspot-like stress
- non-missing DHW-like stress
- `has_derived_label_input = false`
- `label_quality_score >= 0.45`

### Exclusion Logging

Excluded rows are logged to `backend/data/processed/observed_exclusions.csv`.

Reason counts are non-exclusive because one row can fail more than one rule.

## Temporal Alignment Rules

### Training Dataset

- No future data are joined into the label row.
- Environmental features used for modeling are the same-row, same-month covariates already stored with the observed record.
- Seasonality is encoded through `month_sin` and `month_cos`.

### Inference

- Risk and prediction first try local NOAA daily files when available.
- Risk fallback then searches the newest historical site-month row with valid environmental context, even if that row is not model-eligible.
- Prediction fallback searches the newest historical site-month row that is both environmentally valid and model-eligible.
- Neither fallback uses future rows relative to the requested date.

## Spatial Alignment Rules

### Training Dataset

- Site coordinates come from the observed dataset itself.
- Duplicate site-date rows average coordinates within the grouped row.

### Live NOAA Inference

- Site coordinates are mapped to the nearest valid NOAA ocean grid cell.
- Search expands outward across a small local radius until a non-bad cell is found.
- If no valid grid cell exists nearby, the request fails rather than inventing a value.

## Why This Construction Is Defensible

- It preserves provenance.
- It logs conflicts instead of hiding them.
- It keeps comment-derived labels visible without letting them become supervised ground truth.
- It avoids future leakage.
- It aligns the prediction unit with the real temporal precision of the source data.

## Target Support Snapshot

Final eligible site-month rows: 14,475

Binary support:

- positive event rows: 7,907
- non-event rows: 6,568

Descriptive severity support on the same eligible rows:

- none: 6,568
- mild: 4,718
- moderate: 1,649
- severe: 1,540
