import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from model import BleachingMLP


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bleaching_mlp_OLD.pth")
OUT_CSV = os.path.join(BASE_DIR, "data", "processed", "reef_with_preds.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
OUT_FIG = os.path.join(FIG_DIR, "global_bleaching_map.png")


def main(threshold=0.6):
    print("Loading dataset from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    label_col = "bleaching_present"
    df = df.dropna(subset=[label_col])


    feature_cols = [
        c for c in df.columns
        if c not in ["baa", "turbidity", label_col, "date"]
    ]

    print("Using feature columns:", feature_cols)


    # raw lat/lon
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    model = BleachingMLP(input_dim=len(feature_cols)).to(device)
    print("Loading model from:", MODEL_PATH)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # prediction
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(X_tensor).cpu().numpy().flatten()


    df["bleaching_prob"] = probs
    df["bleaching_pred"] = (df["bleaching_prob"] > threshold).astype(int)

    # prediction csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print("predictions:", OUT_CSV)

    # map
    os.makedirs(FIG_DIR, exist_ok=True)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())


    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.coastlines(linewidth=0.7)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


    sc = ax.scatter(
        df["lon"],
        df["lat"],
        c=df["bleaching_prob"],
        s=1,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        transform=ccrs.PlateCarree(),  # lat/lon for cartopy
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Bleaching Probability")

    ax.set_title("Global Coral Bleaching Probability Map")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print("map saved:", OUT_FIG)


if __name__ == "__main__":
    main()
