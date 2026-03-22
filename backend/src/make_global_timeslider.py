"""Deprecated legacy visualization script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/make_global_timeslider.py",
        "The old `reef_with_preds.csv` export is no longer produced because it mixed weak labels with model output.",
    )
