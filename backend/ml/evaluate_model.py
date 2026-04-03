from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from backend.config import CALIBRATION_PATH, CONFUSION_MATRIX_PATH, PRECISION_RECALL_PATH, ROC_PATH


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(predictions)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_binary_diagnostics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> None:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title("4-Week Bleaching Forecast Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=180)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision, color="#0f8b6d", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(PRECISION_RECALL_PATH, dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, color="#c34b32", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=180)
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    frame = pd.DataFrame({"y_true": y_true, "probability": probabilities})
    frame["bin"] = pd.cut(frame["probability"], bins=bins, include_lowest=True)
    calibration = frame.groupby("bin", observed=False).agg(
        observed_rate=("y_true", "mean"),
        predicted_mean=("probability", "mean"),
        count=("y_true", "size"),
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    ax.plot(calibration["predicted_mean"], calibration["observed_rate"], marker="o", color="#2563eb")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration Curve")
    fig.tight_layout()
    fig.savefig(CALIBRATION_PATH, dpi=180)
    plt.close(fig)
