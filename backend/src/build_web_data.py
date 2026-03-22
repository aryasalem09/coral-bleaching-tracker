"""Deprecated legacy export script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/build_web_data.py",
        "Use `/api/sites`, `/api/site/{site_id}`, and `/api/site/{site_id}/observations` for lightweight map data instead.",
    )
