"""Deprecated legacy risk-map script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/risk_map.py",
        "Use `POST /api/risk/score` for the transparent environmental stress layer instead of the old model-colored map.",
    )
