# Deployment Notes

## Why Cold Starts Matter Here

This repo has a static-frontend-plus-Python-API deployment shape:

- frontend has `gh-pages` scripts
- backend includes `runtime.txt`

That pattern often means the frontend stays awake while the Python service sleeps on lower-cost hosting. The first user request can then wait while the backend process restarts.

## Lightweight Health Route

The backend exposes:

```json
GET /health
{
  "ok": true,
  "service": "coral-bleaching-api",
  "started_at": "...",
  "version": "..."
}
```

Design goals:

- no heavy data load
- no model load
- safe for uptime monitors and warm pings

## How The Frontend Handles Cold Starts

- The frontend polls `/health` before assuming the API is ready.
- A warmup banner is shown while the backend is waking.
- The shell UI can render before reef detail, risk, or prediction data arrive.
- Site detail and analysis endpoints are requested only after user interaction.

## Current Backend Responsiveness Choices

- lazy model loading through `backend/ml/model_registry.py`
- cached site catalogs, observation rows, and historical context tables with `lru_cache`
- lightweight summary endpoints separate from reef-detail endpoints
- viewport-limited site fetches instead of all-site eager payloads
- gzip middleware for larger JSON responses

## How To Reduce User Pain Further In Production

- Keep a small uptime monitor pinging `/health`.
- Prefer an always-on backend tier if budget allows.
- Cache NOAA daily files locally if same-day live scoring matters.
- Keep model artifacts on local disk or fast attached storage.
- Avoid frontend startup fetches that depend on full reef detail.

## Hosting Caveats

- If the backend sleeps, the first request after idle time can still be slow.
- If local NOAA daily files are missing, both risk and prediction endpoints fall back to historical context.
- Risk fallback is broader than prediction fallback:
  - risk can use historical environmental context without a model-eligible label row
  - prediction requires a historical row that passes model QA rules
