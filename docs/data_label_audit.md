# Data And Label Audit

## Candidate Label Sources Found

| Source | Location | Label Type | Direct Or Derived | Trust Decision |
| --- | --- | --- | --- | --- |
| BCO-DMO enriched global bleaching table | `backend/data/raw/global_bleaching_environmental.csv` | Numeric `Percent_Bleaching` plus environmental covariates | Mixed: some rows are direct, some are comment-derived | Primary production source after cleanup |
| Older BCO-DMO observed extract | `backend/data/raw/bcodmo_observed_bleaching_v1.csv` | Numeric `Average_Bleaching` | More direct, but lower metadata richness | Audited for overlap only, not merged into production |
| `Bleaching_Comments` / `Sample_Comments` / `Site_Comments` in the enriched table | same file | Textual provenance and coded severity hints | Derived | Excluded from supervised target construction |
| `Bleaching_Level` in the enriched table | same file | Coarse categorical label | Derived / low-definition | Not used as production target |

## Why The Enriched Global Table Was Chosen

- It contains 41,361 raw rows with stable `Site_ID`, `Sample_ID`, coordinates, region metadata, and environmental covariates in one table.
- The older v1 extract has much less metadata and would be hard to merge without duplicate inflation.
- The enriched table is the only source that supports one auditable pipeline from raw observation to site catalog, observed timeline, modeling dataset, and API payload.

## Coverage Summary

### Primary Raw Source

- Raw rows: 41,361
- Unique sites: 12,702
- Date range: 1980-06-15 to 2020-08-15

### Comment-Derived Label Audit

- Raw rows flagged from comments as coded or backfilled: 2,816
- Site-date rows flagged as comment-derived after aggregation: 2,559
- Those rows remain visible in the observed layer with provenance, but they are excluded from supervised training.

### After Site-Date Standardization

- Site-date rows: 20,034
- Direct observations retained: 14,809
- Comment-derived observations retained for observed-only display: 2,559
- Conflict-history rows: 1,409
- Rows recommended for modeling: 14,760

### Final Eligible Site-Month Modeling Dataset

- Rows: 14,475
- Countries represented: 84
- Ecoregions represented: 96
- Split counts:
  - train: 9,709
  - validation: 3,140
  - test: 1,626

## Label Granularity

- The raw source behaves mostly like site-date data, but many dates appear to represent month-level timing encoded as mid-month dates.
- That is why the supervised production unit is `site-month`, not `site-day` or `site-week`.

## Direct Vs Derived Labels

### Direct Labels Trusted For Supervised Learning

- Numeric `Percent_Bleaching` rows that are not flagged as comment-derived

### Derived Or Weak Labels Excluded From The Production Target

- Comment fields containing phrases such as:
  - `averaged from code`
  - `bleaching index`
  - `same bleaching severity`
- `Bleaching_Level`
- Any workflow that reconstructs bleaching outcomes directly from environmental thresholds

## Missingness Issues

- Raw rows missing `Percent_Bleaching`: 6,846
- Site-date rows missing hotspot-like stress: 65
- Site-date rows missing DHW-like stress: 65
- Site-date rows missing coordinates: 0

Final site-month exclusion log counts are non-exclusive because one row can fail for more than one reason:

- `missing_observed_percent_bleaching`: 2,596
- `comment_derived_percent_bleaching`: 2,559
- `missing_hotspot_like`: 62
- `missing_dhw_like`: 62

## Duplication Issues

- Duplicate raw `Sample_ID` rows: 14,356
- Duplicate raw `Site_ID + Date` rows: 21,327
- Production deduplication aggregates to `site_id + date` and preserves provenance instead of silently dropping duplicates.

## Conflict Issues

- Site-date rows with conflicting numeric bleaching values: 1,409
- These rows are flagged with `has_conflict_history = true`
- Conflict details are written to `backend/data/processed/observed_conflicts.csv`

## Timestamp Alignment Issues

- Source timestamps are not uniformly day-precise
- Many values appear to be month-level observations represented on the 15th day
- This is a key reason the production model uses `site-month`

## Spatial Alignment Issues

- The enriched global table includes site coordinates directly, which makes site aggregation and nearest-site lookup tractable.
- NOAA live scoring still requires nearest valid ocean-grid snapping because reef coordinates do not always land exactly on a valid NOAA cell.

## Class Balance Summary

### Binary Production Target

- Definition: `observed_percent_bleaching > 0`
- Positive rate on the final eligible site-month dataset: 0.5463

### Descriptive Severity Support On The Same Eligible Rows

- none: 6,568
- mild: 4,718
- moderate: 1,649
- severe: 1,540

Those severity bins are still useful for descriptive UI labeling, but they are not the production supervised target.

## Trust Decisions

### Trusted For Production Modeling

- Direct numeric `Percent_Bleaching` rows from the enriched table
- Same-row environmental covariates already aligned in that source table

### Used With Caution

- Severity bins derived from percent bleaching for descriptive UI only
- Comment-derived rows for observed-only display and provenance
- Older v1 extract for overlap and audit context only

### Not Trusted As Production Targets

- Free-text bleaching comments
- `Bleaching_Level`
- Any label reconstructed from thermal-stress thresholds alone
