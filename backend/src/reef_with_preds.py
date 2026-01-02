import os
import pandas as pd
import folium
from folium.plugins import MarkerCluster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "data", "processed", "reef_with_preds.csv")
OUT_HTML = os.path.join(BASE_DIR, "reports", "figures", "ml_vs_noaa_comparison.html")


def main(
    threshold: float = 0.6,
    max_points: int = 30000,
    min_prob: float = 0.1,
):
    print("preds from:", PRED_CSV)
    df = pd.read_csv(PRED_CSV)


    required_cols = ["lat", "lon", "bleaching_prob", "bleaching_present"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"{col} not found in reef_with_preds.csv. "
                "run files."
            )


    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.NaT


    df = df.dropna(subset=["lat", "lon", "bleaching_prob", "bleaching_present"])
    df = df[df["bleaching_prob"] >= min_prob]

    # NOAA
    df["noaa_label"] = df["bleaching_present"].astype(int)

    # ML
    df["ml_label"] = (df["bleaching_prob"] >= threshold).astype(int)

    # agree/disagree
    def categorize(row):
        n = row["noaa_label"]
        m = row["ml_label"]
        if n == 0 and m == 0:
            return "Agree: No bleaching"
        elif n == 1 and m == 1:
            return "Agree: Bleaching"
        elif n == 0 and m == 1:
            return "ML-only bleaching"
        else:  # n == 1 and m == 0
            return "NOAA-only bleaching"

    df["category"] = df.apply(categorize, axis=1)


    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
        print(f"downsampled to {len(df)} points for the map.")

    # Color mapping for categories
    category_colors = {
        "Agree: No bleaching": "#4ade80",      # green
        "Agree: Bleaching": "#ef4444",         # red
        "ML-only bleaching": "#f97316",        # orange
        "NOAA-only bleaching": "#3b82f6",      # blue
    }

    # base
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="CartoDB positron",
    )

    # one marker cluster
    clusters = {}
    for cat in category_colors.keys():
        fg = folium.FeatureGroup(name=cat, show=(cat != "Agree: No bleaching"))
        cluster = MarkerCluster(name=cat).add_to(fg)
        fg.add_to(m)
        clusters[cat] = cluster

    # Add points
    for _, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        prob = float(row["bleaching_prob"])
        noaa_label = int(row["noaa_label"])
        ml_label = int(row["ml_label"])
        cat = row["category"]
        dt = row["date"]

        color = category_colors.get(cat, "gray")

        noaa_text = "Bleaching" if noaa_label == 1 else "No bleaching"
        ml_text = "Bleaching" if ml_label == 1 else "No bleaching"
        date_str = dt.date().isoformat() if pd.notnull(dt) else "N/A"

        popup_html = f"""
        <div style="
            font-family: Arial;
            font-size: 13px;
            padding: 8px;
            line-height: 1.4;
        ">
            <div style="font-size: 15px; font-weight: bold; margin-bottom: 6px;">
                ML vs NOAA 
            </div>

            <div><b>Latitude:</b> {lat:.2f}</div>
            <div><b>Longitude:</b> {lon:.2f}</div>
            <div style="margin-top: 6px;">
                <b>Date:</b> {date_str}
            </div>

            <div style="margin-top: 6px;">
                <b>NOAA (BAA-based):</b> {noaa_text}
            </div>
            <div>
                <b>ML Model Prediction:</b> {ml_text}
            </div>
            <div>
                <b>ML Probability:</b> {prob:.2f}
            </div>

            <div style="margin-top: 6px;">
                <b>Category:</b> {cat}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(clusters[cat])

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
    legend_html = """
    <div style="
        position: fixed;
        bottom: 50px;
        left: 50px;
        z-index: 9999;
        background-color: white;
        padding: 10px;
        border: 2px solid #444;
        border-radius: 5px;
        font-size: 12px;
    ">
      <b>ML vs NOAA Comparison</b><br>
      <span style="color:#ef4444;">&#9679;</span> Agree: Bleaching<br>
      <span style="color:#4ade80;">&#9679;</span> Agree: No bleaching<br>
      <span style="color:#f97316;">&#9679;</span> ML-only bleaching<br>
      <span style="color:#3b82f6;">&#9679;</span> NOAA-only bleaching
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    m.save(OUT_HTML)
    print("saved to:", OUT_HTML)


if __name__ == "__main__":
    main()
