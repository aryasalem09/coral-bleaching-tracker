from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import (
    FEATURE_IMPORTANCE_PATH,
    MODEL_BUNDLE_PATH,
    MODEL_INFO_PATH,
    MODEL_METRICS_PATH,
    MODEL_VERSION,
    TRAINING_REPORT_PATH,
    ensure_directories,
)
from backend.ml.build_dataset import ensure_modeling_dataset
from backend.ml.evaluate_model import binary_metrics, save_binary_diagnostics
from backend.ml.feature_definitions import (
    ALL_FEATURES,
    FEATURE_HISTORY_WEEKS,
    FORECAST_GROUND_TRUTH_SUMMARY,
    FORECAST_HORIZON_DAYS,
    FORECAST_HORIZON_WEEKS,
    FORECAST_PROBABILITY_MEANING,
    NUMERIC_FEATURES,
    OBSERVED_PERCENT_COLUMN,
    PREDICTION_UNIT,
    PRODUCTION_TARGET_NAME,
    STATIC_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    THRESHOLD_SELECTION_RULE,
)


@dataclass(frozen=True)
class CandidateModelSpec:
    name: str
    estimator: Pipeline
    model_family: str


def _best_threshold_for_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        score = binary_metrics(y_true, probabilities, threshold=threshold)["f1"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                STATIC_CATEGORICAL_FEATURES,
            ),
        ]
    )


def _candidate_models() -> dict[str, CandidateModelSpec]:
    scale_preprocessor = _make_preprocessor(scale_numeric=True)
    tree_preprocessor = _make_preprocessor(scale_numeric=False)
    candidates = [
        CandidateModelSpec(
            name="forecast_4w_logistic_regression",
            estimator=Pipeline(
                [
                    ("preprocessor", scale_preprocessor),
                    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            model_family="logistic_regression",
        ),
        CandidateModelSpec(
            name="forecast_4w_hist_gradient_boosting",
            estimator=Pipeline(
                [
                    ("preprocessor", tree_preprocessor),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            max_depth=5,
                            learning_rate=0.06,
                            max_iter=300,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            model_family="hist_gradient_boosting",
        ),
    ]
    return {candidate.name: candidate for candidate in candidates}


def _class_balance(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(frame)),
        "positive_rate": float(frame[TARGET_COLUMN].astype(int).mean()) if not frame.empty else 0.0,
        "positive_rows": int(frame[TARGET_COLUMN].astype(int).sum()) if not frame.empty else 0,
        "negative_rows": int(len(frame) - frame[TARGET_COLUMN].astype(int).sum()) if not frame.empty else 0,
    }


def train_and_evaluate() -> dict[str, object]:
    ensure_directories()
    dataset = ensure_modeling_dataset()
    train_df = dataset.loc[dataset["split"] == "train"].copy()
    validation_df = dataset.loc[dataset["split"] == "validation"].copy()
    test_df = dataset.loc[dataset["split"] == "test"].copy()

    y_train = train_df[TARGET_COLUMN].astype(int)
    y_validation = validation_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)
    train_sites = set(train_df["site_id"].astype(str))
    validation_sites = set(validation_df["site_id"].astype(str))
    test_sites = set(test_df["site_id"].astype(str))

    results: dict[str, object] = {}
    best_name = ""
    best_spec: CandidateModelSpec | None = None
    best_model: Pipeline | None = None
    best_score = -1.0
    climatology_probability = float(y_train.mean())

    for name, spec in _candidate_models().items():
        model = spec.estimator
        model.fit(train_df[ALL_FEATURES], y_train)

        validation_probabilities = model.predict_proba(validation_df[ALL_FEATURES])[:, 1]
        decision_threshold = _best_threshold_for_f1(y_validation.to_numpy(), validation_probabilities)
        validation_metrics = binary_metrics(y_validation.to_numpy(), validation_probabilities, threshold=decision_threshold)
        test_probabilities = model.predict_proba(test_df[ALL_FEATURES])[:, 1]
        test_metrics = binary_metrics(y_test.to_numpy(), test_probabilities, threshold=decision_threshold)

        results[name] = {
            "model_family": spec.model_family,
            "feature_columns": ALL_FEATURES,
            "decision_threshold": decision_threshold,
            "validation": validation_metrics,
            "test": test_metrics,
        }
        if validation_metrics["pr_auc"] > best_score:
            best_score = validation_metrics["pr_auc"]
            best_name = name
            best_spec = spec
            best_model = model

    climatology_validation = np.full(shape=len(y_validation), fill_value=climatology_probability, dtype=float)
    climatology_test = np.full(shape=len(y_test), fill_value=climatology_probability, dtype=float)
    results["climatology_baseline"] = {
        "model_family": "climatology",
        "feature_columns": [],
        "validation": binary_metrics(y_validation.to_numpy(), climatology_validation),
        "test": binary_metrics(y_test.to_numpy(), climatology_test),
    }

    if best_model is None or best_spec is None:
        raise RuntimeError("No model candidates were trained.")

    best_threshold = float(results[best_name]["decision_threshold"])
    best_test_probabilities = best_model.predict_proba(test_df[ALL_FEATURES])[:, 1]
    save_binary_diagnostics(y_test.to_numpy(), best_test_probabilities, threshold=best_threshold)

    new_site_mask = ~test_df["site_id"].astype(str).isin(train_sites | validation_sites)
    new_site_metrics = None
    if new_site_mask.any():
        new_site_probabilities = best_model.predict_proba(test_df.loc[new_site_mask, ALL_FEATURES])[:, 1]
        new_site_metrics = binary_metrics(
            test_df.loc[new_site_mask, TARGET_COLUMN].astype(int).to_numpy(),
            new_site_probabilities,
            threshold=best_threshold,
        )

    bundle = {
        "estimator": best_model,
        "feature_columns": ALL_FEATURES,
        "decision_threshold": best_threshold,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "trained_with_sklearn_version": sklearn.__version__,
        "target_definition": PRODUCTION_TARGET_NAME,
        "prediction_unit": PREDICTION_UNIT,
        "model_family": best_spec.model_family,
        "feature_history_weeks": FEATURE_HISTORY_WEEKS,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
        "probability_meaning": FORECAST_PROBABILITY_MEANING,
        "ground_truth_definition": FORECAST_GROUND_TRUTH_SUMMARY,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "input_feature_window": (
            "Static site factors plus 12 weeks of NOAA Monday heat-stress history ending on the forecast issue date."
        ),
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    test_metrics = results[best_name]["test"]
    split_class_balance = {
        "train": _class_balance(train_df),
        "validation": _class_balance(validation_df),
        "test": _class_balance(test_df),
    }
    training_data_summary = {
        "rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "date_min": pd.to_datetime(dataset["date"]).min().date().isoformat(),
        "date_max": pd.to_datetime(dataset["date"]).max().date().isoformat(),
        "reference_observed_date_min": pd.to_datetime(dataset["reference_observed_date"]).min().date().isoformat(),
        "reference_observed_date_max": pd.to_datetime(dataset["reference_observed_date"]).max().date().isoformat(),
        "positive_rate": float(dataset[TARGET_COLUMN].astype(int).mean()),
        "split_class_balance": split_class_balance,
        "future_observation_window_mean": float(dataset["future_observation_count_4w"].mean()),
        "future_positive_window_mean": float(dataset["future_positive_observation_count_4w"].mean()),
    }
    info = {
        "available": True,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "trained_with_sklearn_version": sklearn.__version__,
        "target_definition": PRODUCTION_TARGET_NAME,
        "prediction_unit": PREDICTION_UNIT,
        "feature_columns": ALL_FEATURES,
        "validation_metric_used_for_selection": "pr_auc",
        "decision_threshold": best_threshold,
        "model_family": best_spec.model_family,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
        "feature_history_weeks": FEATURE_HISTORY_WEEKS,
        "probability_meaning": FORECAST_PROBABILITY_MEANING,
        "ground_truth_definition": FORECAST_GROUND_TRUTH_SUMMARY,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "input_feature_window": bundle["input_feature_window"],
        "training_data_summary": training_data_summary,
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    MODEL_METRICS_PATH.write_text(
        json.dumps(
            {
                "available": True,
                "selected_model": best_name,
                "selected_model_summary": {
                    "model_family": best_spec.model_family,
                    "decision_threshold": best_threshold,
                    "validation_metric_used_for_selection": "pr_auc",
                    "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
                },
                "candidate_results": results,
                "split_overlap_summary": {
                    "train_validation_overlap_sites": len(train_sites & validation_sites),
                    "train_test_overlap_sites": len(train_sites & test_sites),
                    "validation_test_overlap_sites": len(validation_sites & test_sites),
                    "test_only_new_sites": len(test_sites - train_sites - validation_sites),
                },
                "training_data_summary": training_data_summary,
                "selected_model_additional_evaluation": {
                    "new_site_test": new_site_metrics,
                },
                "forecast_definition": {
                    "target_definition": PRODUCTION_TARGET_NAME,
                    "prediction_unit": PREDICTION_UNIT,
                    "forecast_horizon_days": FORECAST_HORIZON_DAYS,
                    "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
                    "feature_history_weeks": FEATURE_HISTORY_WEEKS,
                    "ground_truth_definition": FORECAST_GROUND_TRUTH_SUMMARY,
                    "probability_meaning": FORECAST_PROBABILITY_MEANING,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    importance = permutation_importance(
        best_model,
        validation_df[ALL_FEATURES],
        y_validation,
        n_repeats=10,
        random_state=42,
        scoring="average_precision",
    )
    feature_importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_proxy": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_proxy", ascending=False)
    FEATURE_IMPORTANCE_PATH.write_text(feature_importance.to_csv(index=False), encoding="utf-8")

    training_report_lines = [
        "# Forecast Model Training Report",
        "",
        f"Selected model: `{best_name}`",
        f"Model version: `{MODEL_VERSION}`",
        f"Model family: `{best_spec.model_family}`",
        f"Forecast horizon: `{FORECAST_HORIZON_WEEKS} weeks`",
        f"Feature history: `{FEATURE_HISTORY_WEEKS} weeks`",
        f"Decision threshold: `{best_threshold:.2f}`",
        "",
        "## What the model does",
        "",
        "- Ground truth comes from direct observed bleaching records, not NOAA itself.",
        "- The model predicts whether bleaching will be observed in the next 4 weeks after the forecast issue date.",
        "- NOAA HotSpot and DHW values are predictors, not labels.",
        "- This is a probabilistic forecast, not a confirmed observation.",
        "",
        "## Split strategy",
        "",
        "- Train rows use the earliest forecast issue dates.",
        "- Validation rows use later dates.",
        "- Test rows use the latest dates.",
        "- Rows whose 4-week label window would cross a split boundary are excluded from training and validation.",
        "",
        "## Validation metrics",
        "",
        f"- PR-AUC: {results[best_name]['validation']['pr_auc']:.3f}",
        f"- AUROC: {results[best_name]['validation']['auroc']:.3f}",
        f"- F1: {results[best_name]['validation']['f1']:.3f}",
        f"- Precision: {results[best_name]['validation']['precision']:.3f}",
        f"- Recall: {results[best_name]['validation']['recall']:.3f}",
        f"- Brier score: {results[best_name]['validation']['brier_score']:.3f}",
        "",
        "## Test metrics",
        "",
        f"- PR-AUC: {test_metrics['pr_auc']:.3f}",
        f"- AUROC: {test_metrics['auroc']:.3f}",
        f"- F1: {test_metrics['f1']:.3f}",
        f"- Precision: {test_metrics['precision']:.3f}",
        f"- Recall: {test_metrics['recall']:.3f}",
        f"- Brier score: {test_metrics['brier_score']:.3f}",
        f"- Confusion matrix counts: TN {test_metrics['true_negative']}, FP {test_metrics['false_positive']}, FN {test_metrics['false_negative']}, TP {test_metrics['true_positive']}",
        "",
        "## Baseline comparison",
        "",
        f"- Climatology test PR-AUC: {results['climatology_baseline']['test']['pr_auc']:.3f}",
        f"- Climatology test AUROC: {results['climatology_baseline']['test']['auroc']:.3f}",
        "",
        "## Class balance",
        "",
        f"- Train positive rate: {split_class_balance['train']['positive_rate']:.3f}",
        f"- Validation positive rate: {split_class_balance['validation']['positive_rate']:.3f}",
        f"- Test positive rate: {split_class_balance['test']['positive_rate']:.3f}",
    ]
    TRAINING_REPORT_PATH.write_text("\n".join(training_report_lines), encoding="utf-8")

    return {
        "selected_model": best_name,
        "candidate_results": results,
    }


if __name__ == "__main__":
    train_and_evaluate()
