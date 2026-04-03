from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import joblib
import sklearn

from backend.config import MODEL_BUNDLE_PATH, MODEL_INFO_PATH, MODEL_METRICS_PATH

logger = logging.getLogger(__name__)


def _artifact_missing_message() -> str:
    return "Forecast model artifact is missing. Run `python3 -m backend.ml.train_model` to rebuild it."


def clear_model_registry_cache() -> None:
    get_model_runtime_status.cache_clear()
    load_model_bundle.cache_clear()
    load_model_info.cache_clear()
    load_model_metrics.cache_clear()


@lru_cache(maxsize=1)
def get_model_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "missing",
        "ready": False,
        "model_loaded": False,
        "message": _artifact_missing_message(),
        "artifact_path": str(MODEL_BUNDLE_PATH),
        "model_version": None,
        "sklearn_version": sklearn.__version__,
        "trained_with_sklearn_version": None,
        "loader_error": None,
    }

    info_payload: dict[str, Any] | None = None
    if MODEL_INFO_PATH.exists():
        try:
            info_payload = json.loads(MODEL_INFO_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            info_payload = None
    if info_payload:
        status["model_version"] = info_payload.get("model_version")
        status["trained_with_sklearn_version"] = info_payload.get("trained_with_sklearn_version")

    if not MODEL_BUNDLE_PATH.exists():
        return status

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        logger.exception("Failed to load model artifact from %s", MODEL_BUNDLE_PATH)
        status.update(
            {
                "status": "invalid",
                "ready": False,
                "model_loaded": False,
                "message": "Forecast model unavailable in the current backend environment.",
                "loader_error": str(exc),
            }
        )
        return status

    required_keys = {"estimator", "feature_columns", "decision_threshold"}
    if not required_keys.issubset(bundle):
        missing = ", ".join(sorted(required_keys - set(bundle)))
        status.update(
            {
                "status": "invalid",
                "ready": False,
                "model_loaded": False,
                "message": "Forecast model artifact is incomplete and must be rebuilt.",
                "loader_error": f"Missing required bundle keys: {missing}",
            }
        )
        return status

    status.update(
        {
            "status": "ready",
            "ready": True,
            "model_loaded": True,
            "message": "Model bundle loaded successfully.",
            "model_version": bundle.get("model_version", status.get("model_version")),
            "trained_with_sklearn_version": bundle.get(
                "trained_with_sklearn_version",
                status.get("trained_with_sklearn_version"),
            ),
        }
    )
    return status


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
        payload["sklearn_version"] = runtime["sklearn_version"]
        payload["trained_with_sklearn_version"] = runtime["trained_with_sklearn_version"]
        payload["artifact_path"] = runtime["artifact_path"]
        return payload
    return {
        "available": False,
        "runtime_status": runtime["status"],
        "runtime_ready": runtime["ready"],
        "runtime_message": runtime["message"],
        "sklearn_version": runtime["sklearn_version"],
        "trained_with_sklearn_version": runtime["trained_with_sklearn_version"],
        "artifact_path": runtime["artifact_path"],
    }


@lru_cache(maxsize=1)
def load_model_metrics() -> dict[str, object]:
    runtime = get_model_runtime_status()
    if MODEL_METRICS_PATH.exists():
        payload = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
        payload["runtime_status"] = runtime["status"]
        payload["runtime_ready"] = runtime["ready"]
        payload["runtime_message"] = runtime["message"]
        payload["sklearn_version"] = runtime["sklearn_version"]
        payload["trained_with_sklearn_version"] = runtime["trained_with_sklearn_version"]
        return payload
    return {
        "available": False,
        "runtime_status": runtime["status"],
        "runtime_ready": runtime["ready"],
        "runtime_message": runtime["message"],
        "sklearn_version": runtime["sklearn_version"],
        "trained_with_sklearn_version": runtime["trained_with_sklearn_version"],
    }
