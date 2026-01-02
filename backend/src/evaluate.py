import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from dataset import ReefDataset
from model import BleachingMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def evaluate_model(df_path=DATA_PATH, label_col="bleaching_present"):
    print("Loading dataset from:", df_path)
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[label_col])

    feature_cols = [
        c for c in df.columns
        if c not in ["baa", "turbidity", label_col, "date"]
    ]


    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = df[label_col].values.astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BleachingMLP(input_dim=len(feature_cols)).to(device)
    print("model from:", MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(X.to(device)).cpu().numpy().flatten()

    # 0.5 baseline
    print("\n threshold 0.5")
    preds_05 = (probs > 0.5).astype(int)
    print(classification_report(y, preds_05))
    try:
        print("ROC AUC:", roc_auc_score(y, probs))
    except ValueError:
        print("no ROC AUC")

    # sweep
    print("\n class 1 sweep")
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        preds = (probs > t).astype(int)
        print(f"\n threshold = {t:.2f} ---")
        print(classification_report(y, preds, digits=3))

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)

    os.makedirs(FIG_DIR, exist_ok=True)
    roc_path = os.path.join(FIG_DIR, "roc_curve.png")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Pos:")
    plt.ylabel("True Pos:")

    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)

    plt.close()

    print("\nROC curve:", roc_path)


if __name__ == "__main__":
    evaluate_model()
