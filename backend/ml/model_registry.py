from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib

from backend.config import MODEL_BUNDLE_PATH, MODEL_INFO_PATH, MODEL_METRICS_PATH


def _artifact_missing_message() -> str:
    return "Model artifact is missing. Run `python -m backend.ml.train_model` to rebuild it."


@lru_cache(maxsize=1)
def get_model_runtime_status() -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.exists():
        return {
            "status": "missing",
            "ready": False,
            "message": _artifact_missing_message(),
        }

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        return {
            "status": "invalid",
            "ready": False,
            "message": (
                "Model artifact is present but could not be loaded with the current environment. "
                f"Retrain the model bundle to restore prediction support. Loader error: {exc}"
            ),
        }

    required_keys = {"estimator", "feature_columns", "decision_threshold"}
    if not required_keys.issubset(bundle):
        missing = ", ".join(sorted(required_keys - set(bundle)))
        return {
            "status": "invalid",
            "ready": False,
            "message": f"Model artifact is missing required bundle keys: {missing}. Retrain the model bundle.",
        }

    return {
        "status": "ready",
        "ready": True,
        "message": "Model bundle loaded successfully.",
    }


def model_artifact_exists() -> bool:
    return bool(get_model_runtime_status()["ready"])


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, object]:
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(_artifact_missing_message())
    return joblib.load(MODEL_BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_model_info() -> dict[str, object]:
    runtime = get_model_runtime_status()
    if MODEL_INFO_PATH.exists():
        payload = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        payload["runtime_status"] = runtime["status"]
        payload["runtime_ready"] = runtime["ready"]
        payload["runtime_message"] = runtime["message"]
        return payload
    return {
        "available": False,
        "runtime_status": runtime["status"],
        "runtime_ready": runtime["ready"],
        "runtime_message": runtime["message"],
    }


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, object]:
    runtime = get_model_runtime_status()
    if MODEL_METRICS_PATH.exists():
        payload = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        payload["runtime_status"] = runtime["status"]
        payload["runtime_ready"] = runtime["ready"]
        payload["runtime_message"] = runtime["message"]
        return payload
    return {
        "available": False,
        "runtime_status": runtime["status"],
        "runtime_ready": runtime["ready"],
        "runtime_message": runtime["message"],
    }
