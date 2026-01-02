import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

from dataset import ReefDataset
from model import BleachingMLP

# Absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
MOD_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")


def train_model(
    df_path=DATA_PATH,
    label_col="bleaching_present",
    epochs=50,
    batch_size=256,
    lr=1e-3,
):
    print("dataset:", df_path)
    df = pd.read_csv(df_path)
    df = df.dropna(subset=[label_col])

    # turbidity is useless and baa has a leakage
    feature_cols = [
        c for c in df.columns
        if c not in ["baa", "turbidity", label_col, "date"]
    ]


    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 3 way split with stratification
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df[label_col],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[label_col],
    )

    print("Train size:", len(train_df), "Val size:", len(val_df), "Test size:", len(test_df))

    # datasets / loaders
    train_ds = ReefDataset(train_df, feature_cols, label_col)
    val_ds = ReefDataset(val_df, feature_cols, label_col)
    test_ds = ReefDataset(test_df, feature_cols, label_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # class weights (0/1)
    y_all = df[label_col].values
    classes = np.array([0, 1])
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_all)
    w0, w1 = cw[0], cw[1]

    print("class weights; 0:", w0, "1:", w1)

    w0 = torch.tensor(w0, dtype=torch.float32, device=device)
    w1 = torch.tensor(w1, dtype=torch.float32, device=device)

    model = BleachingMLP(input_dim=len(feature_cols)).to(device)

    # BCE lps, pos > neg weightage
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    os.makedirs(os.path.dirname(MOD_PATH), exist_ok=True)

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X)  # 0/1 cuz of the sigmoid in the model

            # BCE loss per sample
            loss_raw = criterion(preds, y)

            # class weights
            sample_weights = torch.where(y == 1.0, w1, w0)
            loss = (loss_raw * sample_weights).mean()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss_raw = criterion(preds, y)
                sample_weights = torch.where(y == 1.0, w1, w0)
                loss = (loss_raw * sample_weights).mean()
                val_loss += loss.item()

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # stopping easrly if model loss plateaus
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MOD_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("ES")
                break

    print("training DONE")
    print(f"val loss BEST: {best_val_loss:.4f}")
    print(f"best model: {MOD_PATH}")

    # best model testing
    print("\nbest mode:")
    model.load_state_dict(torch.load(MOD_PATH, map_location=device))
    model.eval()

    probs = []
    labs = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            probs = model(X).cpu().numpy().flatten()
            probs.extend(probs)
            labs.extend(y.cpu().numpy().flatten())

    probs = np.array(probs)
    labs = np.array(labs)
    prediction_labs = (probs > 0.5).astype(int)

    print("\nweighted training performance")
    print(classification_report(labs, prediction_labs))
    try:
        print("test ROC AUC:", roc_auc_score(labs, probs))
    except ValueError:
        print("ERROR only one class")


if __name__ == "__main__":
    train_model()
