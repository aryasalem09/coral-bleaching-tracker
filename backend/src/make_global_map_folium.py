"""Deprecated legacy folium map script."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.src._deprecated_pipeline import deprecated_main


if __name__ == "__main__":
    deprecated_main(
        "backend/src/make_global_map_folium.py",
        "The production app now serves observed sites and analysis layers on demand through `backend/api.py`.",
    )
