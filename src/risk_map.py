import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from model import BleachingMLP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def build_risk_map():
    print("data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    feature_cols = [c for c in df.columns if c != "bleaching_present"]

    # scale
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = BleachingMLP(input_dim=len(feature_cols)).to(device)
    print("Loading model from:", MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        probs = model(X.to(device)).cpu().numpy().flatten()

    df["bleaching_risk"] = probs

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "bleaching_risk_world_map.png")


    proj = ccrs.PlateCarree()

    plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)

    # land borders, coastlines
    ax.add_feature(cfeature.LAND, alpha=0.6)
    ax.add_feature(cfeature.OCEAN, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_global()
    ax.set_title("Global Coral Bleaching Risk from Model Preds")

    # reef points
    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["bleaching_risk"],
        s=5,
        alpha=0.7,
        transform=proj,
    )

    # risk
    cb = plt.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Predicted Bleaching Risk")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("risk map:", out_path)


if __name__ == "__main__":
    build_risk_map()
