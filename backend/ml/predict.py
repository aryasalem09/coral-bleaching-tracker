from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.config import MODEL_DECISION_THRESHOLD
from backend.ml.feature_definitions import ALL_FEATURES, PRODUCTION_TARGET_NAME, PREDICTION_UNIT, ensure_feature_columns
from backend.ml.model_registry import load_model_bundle


@dataclass
class PredictionResult:
    probability: float
    predicted_event: bool
    threshold: float
    model_version: str
    target_definition: str
    prediction_unit: str


def predict_event_probability(feature_row: dict[str, object]) -> PredictionResult:
    bundle = load_model_bundle()
    estimator = bundle["estimator"]
    feature_frame = ensure_feature_columns(pd.DataFrame([feature_row]), bundle.get("feature_columns", ALL_FEATURES))
    probability = float(estimator.predict_proba(feature_frame)[0, 1])
    threshold = float(bundle.get("decision_threshold", MODEL_DECISION_THRESHOLD))
    return PredictionResult(
        probability=probability,
        predicted_event=bool(probability >= threshold),
        threshold=threshold,
        model_version=str(bundle.get("model_version", "unknown")),
        target_definition=str(bundle.get("target_definition", PRODUCTION_TARGET_NAME)),
        prediction_unit=str(bundle.get("prediction_unit", PREDICTION_UNIT)),
    )
