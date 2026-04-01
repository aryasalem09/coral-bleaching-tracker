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
    LEGACY_FEATURES,
    LEGACY_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    PREDICTION_UNIT,
    PRODUCTION_TARGET_NAME,
    STATIC_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


@dataclass(frozen=True)
class CandidateModelSpec:
    name: str
    feature_columns: list[str]
    numeric_features: list[str]
    estimator: Pipeline
    feature_set: str
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


def _make_preprocessor(scale_numeric: bool, numeric_features: list[str]) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
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
    feature_sets = {
        "legacy_same_month": {
            "feature_columns": LEGACY_FEATURES,
            "numeric_features": LEGACY_NUMERIC_FEATURES,
        },
        "weekly_history": {
            "feature_columns": ALL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
        },
    }
    candidates: list[CandidateModelSpec] = []
    for feature_set_name, feature_config in feature_sets.items():
        scale_preprocessor = _make_preprocessor(
            scale_numeric=True,
            numeric_features=list(feature_config["numeric_features"]),
        )
        tree_preprocessor = _make_preprocessor(
            scale_numeric=False,
            numeric_features=list(feature_config["numeric_features"]),
        )
        candidates.append(
            CandidateModelSpec(
                name=f"{feature_set_name}_logistic_regression",
                feature_columns=list(feature_config["feature_columns"]),
                numeric_features=list(feature_config["numeric_features"]),
                estimator=Pipeline(
                    [
                        ("preprocessor", scale_preprocessor),
                        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
                    ]
                ),
                feature_set=feature_set_name,
                model_family="logistic_regression",
            )
        )
        candidates.append(
            CandidateModelSpec(
                name=f"{feature_set_name}_hist_gradient_boosting",
                feature_columns=list(feature_config["feature_columns"]),
                numeric_features=list(feature_config["numeric_features"]),
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
                feature_set=feature_set_name,
                model_family="hist_gradient_boosting",
            )
        )
    return {candidate.name: candidate for candidate in candidates}


def _load_prior_metrics_summary() -> dict[str, object] | None:
    if not MODEL_METRICS_PATH.exists():
        return None
    try:
        payload = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    selected_model = payload.get("selected_model")
    selected_test = (
        payload.get("candidate_results", {}).get(selected_model, {}).get("test")
        if isinstance(payload.get("candidate_results"), dict)
        else None
    )
    return {
        "selected_model": selected_model,
        "test_metrics": selected_test,
    }


def train_and_evaluate() -> dict[str, object]:
    ensure_directories()
    prior_metrics_summary = _load_prior_metrics_summary()
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
        x_train = train_df[spec.feature_columns]
        x_validation = validation_df[spec.feature_columns]
        x_test = test_df[spec.feature_columns]

        model = spec.estimator
        model.fit(x_train, y_train)

        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        decision_threshold = _best_threshold_for_f1(y_validation.to_numpy(), validation_probabilities)
        validation_metrics = binary_metrics(y_validation.to_numpy(), validation_probabilities, threshold=decision_threshold)
        test_probabilities = model.predict_proba(x_test)[:, 1]
        test_metrics = binary_metrics(y_test.to_numpy(), test_probabilities, threshold=decision_threshold)

        results[name] = {
            "feature_set": spec.feature_set,
            "model_family": spec.model_family,
            "feature_columns": spec.feature_columns,
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
        "feature_set": "none",
        "model_family": "climatology",
        "feature_columns": [],
        "validation": binary_metrics(y_validation.to_numpy(), climatology_validation),
        "test": binary_metrics(y_test.to_numpy(), climatology_test),
    }

    if best_model is None or best_spec is None:
        raise RuntimeError("No model candidates were trained.")

    best_threshold = float(results[best_name]["decision_threshold"])
    best_feature_columns = best_spec.feature_columns
    best_x_validation = validation_df[best_feature_columns]
    best_x_test = test_df[best_feature_columns]
    best_test_probabilities = best_model.predict_proba(best_x_test)[:, 1]
    save_binary_diagnostics(y_test.to_numpy(), best_test_probabilities, threshold=best_threshold)

    new_site_mask = ~test_df["site_id"].astype(str).isin(train_sites | validation_sites)
    new_site_metrics = None
    if new_site_mask.any():
        new_site_probabilities = best_model.predict_proba(test_df.loc[new_site_mask, best_feature_columns])[:, 1]
        new_site_metrics = binary_metrics(
            test_df.loc[new_site_mask, TARGET_COLUMN].astype(int).to_numpy(),
            new_site_probabilities,
            threshold=best_threshold,
        )

    weekly_test = results.get("weekly_history_hist_gradient_boosting", {}).get("test")
    legacy_test = results.get("legacy_same_month_hist_gradient_boosting", {}).get("test")
    formulation_comparison = None
    if isinstance(weekly_test, dict) and isinstance(legacy_test, dict):
        formulation_comparison = {
            metric: float(weekly_test[metric] - legacy_test[metric])
            for metric in ["auroc", "pr_auc", "f1", "precision", "recall", "brier_score"]
            if metric in weekly_test and metric in legacy_test
        }

    bundle = {
        "estimator": best_model,
        "feature_columns": best_feature_columns,
        "decision_threshold": best_threshold,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        # sklearn pipelines are not forward-compatible across arbitrary releases
        # because ColumnTransformer internals are part of the pickle payload.
        # We therefore record and pin the exact training sklearn version.
        "trained_with_sklearn_version": sklearn.__version__,
        "target_definition": PRODUCTION_TARGET_NAME,
        "prediction_unit": PREDICTION_UNIT,
        "feature_set": best_spec.feature_set,
        "model_family": best_spec.model_family,
        "input_feature_window": "Static site factors plus weekly Monday NOAA current, lagged, rolling, and trend heat-stress features",
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    test_metrics = results[best_name]["test"]
    training_data_summary = {
        "rows": int(len(dataset)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "date_min": pd.to_datetime(dataset["date"]).min().date().isoformat(),
        "date_max": pd.to_datetime(dataset["date"]).max().date().isoformat(),
        "positive_rate": float(dataset[TARGET_COLUMN].astype(int).mean()),
    }
    info = {
        "available": True,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "trained_with_sklearn_version": sklearn.__version__,
        "target_definition": PRODUCTION_TARGET_NAME,
        "prediction_unit": PREDICTION_UNIT,
        "feature_columns": best_feature_columns,
        "validation_metric_used_for_selection": "pr_auc",
        "decision_threshold": best_threshold,
        "feature_set": best_spec.feature_set,
        "model_family": best_spec.model_family,
        "input_feature_window": "Weekly Monday NOAA history aligned to the nearest prior Monday for each site-month row",
        "training_data_summary": training_data_summary,
    }
    MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    MODEL_METRICS_PATH.write_text(
        json.dumps(
            {
                "available": True,
                "selected_model": best_name,
                "selected_model_summary": {
                    "feature_set": best_spec.feature_set,
                    "model_family": best_spec.model_family,
                    "decision_threshold": best_threshold,
                },
                "candidate_results": results,
                "split_overlap_summary": {
                    "train_validation_overlap_sites": len(train_sites & validation_sites),
                    "train_test_overlap_sites": len(train_sites & test_sites),
                    "validation_test_overlap_sites": len(validation_sites & test_sites),
                    "test_only_new_sites": len(test_sites - train_sites - validation_sites),
                },
                "training_data_summary": training_data_summary,
                "formulation_comparison": formulation_comparison,
                "selected_model_additional_evaluation": {
                    "new_site_test": new_site_metrics,
                },
                "prior_artifact_summary": prior_metrics_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    importance = permutation_importance(
        best_model,
        best_x_validation,
        y_validation,
        n_repeats=10,
        random_state=42,
        scoring="average_precision",
    )
    feature_importance = pd.DataFrame(
        {
            "feature": best_feature_columns,
            "importance_proxy": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_proxy", ascending=False)
    FEATURE_IMPORTANCE_PATH.write_text(feature_importance.to_csv(index=False), encoding="utf-8")

    legacy_pr_auc = (
        results.get("legacy_same_month_hist_gradient_boosting", {}).get("test", {}).get("pr_auc")
        if isinstance(results.get("legacy_same_month_hist_gradient_boosting"), dict)
        else None
    )
    weekly_pr_auc = (
        results.get("weekly_history_hist_gradient_boosting", {}).get("test", {}).get("pr_auc")
        if isinstance(results.get("weekly_history_hist_gradient_boosting"), dict)
        else None
    )
    training_report_lines = [
        "# Model Training Report",
        "",
        f"Selected model: `{best_name}`",
        f"Model version: `{MODEL_VERSION}`",
        f"Feature set: `{best_spec.feature_set}`",
        f"Model family: `{best_spec.model_family}`",
        f"Climatology baseline probability: `{climatology_probability:.3f}`",
        f"Decision threshold: `{best_threshold:.2f}`",
        "",
        "## Modeling decision",
        "",
        "- Production target remains binary site-month bleaching event prediction.",
        "- Weekly NOAA Monday files are aligned to the nearest Monday on or before each observed site-date.",
        "- Lagged, rolling, and trend heat-stress features use only current-or-earlier Mondays, so the model does not look into the future.",
        "",
        "## Test metrics",
        "",
        f"- AUROC: {test_metrics['auroc']:.3f}",
        f"- PR-AUC: {test_metrics['pr_auc']:.3f}",
        f"- F1: {test_metrics['f1']:.3f}",
        f"- Precision: {test_metrics['precision']:.3f}",
        f"- Recall: {test_metrics['recall']:.3f}",
        f"- Brier score: {test_metrics['brier_score']:.3f}",
        "",
        "## Formulation comparison",
        "",
        f"- Legacy same-month HGB test PR-AUC: {legacy_pr_auc:.3f}" if isinstance(legacy_pr_auc, float) else "- Legacy same-month HGB test PR-AUC: n/a",
        f"- Weekly-history HGB test PR-AUC: {weekly_pr_auc:.3f}" if isinstance(weekly_pr_auc, float) else "- Weekly-history HGB test PR-AUC: n/a",
    ]
    if formulation_comparison:
        training_report_lines.extend(
            [
                f"- Weekly minus legacy PR-AUC: {formulation_comparison['pr_auc']:+.3f}",
                f"- Weekly minus legacy AUROC: {formulation_comparison['auroc']:+.3f}",
            ]
        )
    training_report_lines.extend(
        [
            "",
            "## Climatology baseline",
            "",
            f"- Test PR-AUC: {results['climatology_baseline']['test']['pr_auc']:.3f}",
            f"- Test AUROC: {results['climatology_baseline']['test']['auroc']:.3f}",
            "",
            "## Split overlap audit",
            "",
            f"- Train/validation overlapping sites: {len(train_sites & validation_sites)}",
            f"- Train/test overlapping sites: {len(train_sites & test_sites)}",
            f"- Validation/test overlapping sites: {len(validation_sites & test_sites)}",
            f"- Test-only new sites: {len(test_sites - train_sites - validation_sites)}",
        ]
    )
    TRAINING_REPORT_PATH.write_text("\n".join(training_report_lines), encoding="utf-8")

    return {
        "selected_model": best_name,
        "candidate_results": results,
    }


if __name__ == "__main__":
    train_and_evaluate()
