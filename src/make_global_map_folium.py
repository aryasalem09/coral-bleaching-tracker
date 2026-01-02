import os
import pandas as pd
import folium
from folium.plugins import HeatMap


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "data", "processed", "reef_with_preds.csv")
OUT_HTML = os.path.join(BASE_DIR, "reports", "figures", "global_bleaching_map_interactive.html")


def main(max_points=50000, min_prob=0.1):
    print("preds:", PRED_CSV)
    df = pd.read_csv(PRED_CSV)


    df = df.dropna(subset=["lat", "lon", "bleaching_prob"])
    df = df[df["bleaching_prob"] >= min_prob]

    # performancemaxxing
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
        print(f"downsampled to {len(df)} points")

    # 0,0
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="CartoDB positron"  # basemap
    )


    heat_data = [
        [row["lat"], row["lon"], row["bleaching_prob"]]
        for _, row in df.iterrows()
    ]

    HeatMap(
        heat_data,
        radius=6,
        blur=8,
        max_zoom=4,
        min_opacity=0.3,
    ).add_to(m)


    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    m.save(OUT_HTML)
    print("map saved:", OUT_HTML)


if __name__ == "__main__":
    main()
