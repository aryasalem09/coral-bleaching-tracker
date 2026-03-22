"""Deprecated legacy exploratory script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/eda.py",
        "Inspect the cleaned outputs in `backend/data/processed/` and the reports in `docs/` instead of the old heuristic feature file.",
    )
