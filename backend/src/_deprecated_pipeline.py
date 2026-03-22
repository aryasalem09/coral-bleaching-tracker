"""Helpers for legacy scripts kept only as explicit deprecation stubs."""

from __future__ import annotations


def deprecated_main(script_name: str, replacement: str) -> None:
    raise SystemExit(
        f"{script_name} is deprecated. This repository now uses the audited observed-label pipeline under "
        f"`backend/ml` and the production API in `backend/api.py`. {replacement}"
    )
