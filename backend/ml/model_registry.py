from __future__ import annotations

import json
from functools import lru_cache

import joblib

from backend.config import MODEL_BUNDLE_PATH, MODEL_INFO_PATH, MODEL_METRICS_PATH


def model_artifact_exists() -> bool:
    return MODEL_BUNDLE_PATH.exists()


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, object]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError("Model artifact is missing. Run backend/ml/train_model.py first.")
    return joblib.load(MODEL_BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_model_info() -> dict[str, object]:
    if MODEL_INFO_PATH.exists():
        return json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
    return {"available": False}


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, object]:
    if MODEL_METRICS_PATH.exists():
        return json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    return {"available": False}
