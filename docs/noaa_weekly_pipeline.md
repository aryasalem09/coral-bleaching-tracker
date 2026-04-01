# NOAA Weekly Monday Pipeline

## Purpose

The project now treats NOAA Coral Reef Watch 5 km daily NetCDF files as a weekly-Monday archive for production modeling and live site analysis.

The downloader:

- discovers the remote DHW and HotSpot year directories directly from NOAA
- finds the remote first and last available dates per product
- enumerates every Monday in the paired product date range
- downloads the corresponding Monday DHW and HotSpot files
- skips already valid local files
- uses atomic temporary files plus `os.replace`
- retries transient failures
- records unavailable dates without crashing

## Products

- `dhw`: NOAA CRW Degree Heating Week daily NetCDF
- `hs`: NOAA CRW HotSpot daily NetCDF

Even though the NOAA source is daily, this project stores Monday snapshots only for the production weekly-history workflow.

## Local Storage

- `backend/data/raw/noaa_dhw/`
- `backend/data/raw/noaa_hs/`
- `backend/data/raw/noaa_manifest_weekly_mondays.json`

The raw directory remains ignored by Git because the full weekly archive is large.

## Manifest Structure

The weekly manifest records:

- `requested_dates`
- `ok_dates`
- `failed_dates`
- `product_coverage`
- `first_available_dates`
- `last_available_dates`
- `date_status`

Each `date_status` entry records per-product:

- `ok`
- `status_code`
- `size_bytes`
- `path`
- `error`
- `skipped`
- `remote_listed`

Current full manifest snapshot in this repo workspace:

- requested Mondays: `2,141`
- succeeded Mondays: `2,141`
- failed Mondays: `0`
- paired range: `1985-03-25` through `2026-03-30`

## Resumability

The downloader is safe to rerun.

- valid non-empty files are skipped
- zero-byte partial files are removed before retry
- manifest writes are atomic
- a rerun continues filling missing Mondays instead of starting over

## Canonical Command

```bash
python3 -m backend.download_noaa_weekly_mondays
```

Useful options:

```bash
python3 -m backend.download_noaa_weekly_mondays --workers 20
python3 -m backend.download_noaa_weekly_mondays --start 2000-01-01 --end 2019-12-31
```

## Availability Layer

`backend/noaa_index.py` is the reusable availability/index module.

It provides:

- all paired local Monday dates
- per-product local Monday dates
- nearest valid previous Monday lookup
- date coverage summaries

`backend/noaa.py` exposes the higher-level helpers used by the API.

## Runtime Behavior

- Risk uses the nearest local weekly Monday on or before the requested date.
- Prediction uses the nearest local weekly Monday plus lagged/rolling weekly history only when a full contiguous 12-week window can be assembled.
- If the required weekly history is missing or gapped, prediction returns an honest unavailable response instead of fabricating a fallback.
