# Frontend

Vite + React 19 + TypeScript client for the Coral Bleaching Tracker.

## Local Run

```bash
npm install
npm run dev
```

The dev client expects the FastAPI backend at `http://127.0.0.1:8000` by default.

Override with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## Checks

```bash
npm run lint
npm run build
```

## Runtime Notes

- The map loads viewport-limited site summaries first, then fetches site detail on click.
- The warmup banner polls `/health` so the UI can handle backend cold starts.
- Observed, risk, and prediction are separate UI layers and separate API calls.
- Production builds use `/` locally and switch to `/coral-bleaching-tracker/` automatically in GitHub Actions for Pages deployment.
