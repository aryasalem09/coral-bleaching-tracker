"""Deprecated legacy NOAA preprocessing script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/noaa_prep.py",
        "Use `backend/ml/label_standardization.py`, `backend/ml/build_dataset.py`, and `backend/noaa.py` for the audited data path.",
    )
