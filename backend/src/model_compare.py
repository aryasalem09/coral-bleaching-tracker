import os
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from model import BleachingMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")


def compare_models(df_path=DATA_PATH, label_col="bleaching_present"):
    print("data:", df_path)
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[label_col])

    feature_cols = [c for c in df.columns if c != label_col]

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # logistic reg
    print("\nlogistic reg")
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    lr_probs = log_reg.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs > 0.5).astype(int)

    print(classification_report(y_test, lr_preds))
    try:
        print("ROC AUC:", roc_auc_score(y_test, lr_probs))
    except ValueError:
        print("no ROC AUC")

    # pyt model
    print("\n NN Bleaching MLP")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BleachingMLP(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        nn_probs = model(X_test_torch).cpu().numpy().flatten()

    nn_preds = (nn_probs > 0.5).astype(int)

    print(classification_report(y_test, nn_preds))
    try:
        print("ROC AUC:", roc_auc_score(y_test, nn_probs))
    except ValueError:
        print("no ROC AUC.")


if __name__ == "__main__":
    compare_models()
