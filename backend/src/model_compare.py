"""Deprecated legacy model-comparison script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/model_compare.py",
        "Candidate comparison is now part of `backend/ml/train_model.py` and is based on the cleaned observed-label dataset.",
    )
