"""Deprecated legacy comparison script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/reef_with_preds.py",
        "The old `reef_with_preds.csv` comparison path has been removed because it was tied to the deprecated synthetic-label workflow.",
    )
