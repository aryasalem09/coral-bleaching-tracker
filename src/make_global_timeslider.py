import os
import pandas as pd
import folium
from folium.plugins import HeatMapWithTime, MarkerCluster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "data", "processed", "reef_with_preds.csv")
OUT_HTML = os.path.join(BASE_DIR, "reports", "figures", "global_bleaching_timeslider.html")


def main(
    max_points_per_month=6000,
    max_markers=1500,
    min_prob=0.1,
):
    print("Loading predictions from:", PRED_CSV)
    df = pd.read_csv(PRED_CSV)

    # error handling
    if "date" not in df.columns:
        raise ValueError(
            "chekc pred reefs csv for the dates column "
            "noaa_prep.py needs to add it and rerun the pipeline"
        )

    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    # cleaning again
    df = df.dropna(subset=["lat", "lon", "bleaching_prob"])
    df = df[df["bleaching_prob"] >= min_prob]


    df = df.sort_values("date")

    # month slices
    months = sorted(df["year_month"].unique())
    print("Number of months:", len(months))
    print("First month:", months[0], "Last month:", months[-1])

    time_slices = []
    time_labels = []

    for m in months:
        df_m = df[df["year_month"] == m]

        # performancemaxxing so im downsizing here
        if len(df_m) > max_points_per_month:
            df_m = df_m.sample(n=max_points_per_month, random_state=42)

        heat_data = [
            [row["lat"], row["lon"], row["bleaching_prob"]]
            for _, row in df_m.iterrows()
        ]

        time_slices.append(heat_data)
        time_labels.append(m)

    # basic map
    m = folium.Map(
        location=[0, 0],
        zoom_start=2,
        tiles="CartoDB positron"
    )

    # time slider heatmap
    HeatMapWithTime(
        data=time_slices,
        index=time_labels,
        radius=6,
        auto_play=False,
        max_opacity=0.9,
        min_opacity=0.3,
        use_local_extrema=False,
    ).add_to(m)

    # sampled markers
    if len(df) > max_markers:
        df_markers = df.sample(n=max_markers, random_state=123)
    else:
        df_markers = df

    marker_cluster = MarkerCluster(name="Bleaching Points").add_to(m)

    for _, row in df_markers.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        prob = float(row["bleaching_prob"])
        dt = row["date"]

        # Color mapping
        if prob >= 0.8:
            color = "red"
        elif prob >= 0.5:
            color = "orange"
        elif prob >= 0.2:
            color = "yellow"
        else:
            color = "green"

        # Emoji label
        risk_display = {
            "red":    "🟥 Very High",
            "orange": "🟧 High",
            "yellow": "🟨 Moderate",
            "green":  "🟩 Low",
        }[color]

        popup_html = f"""
        <div style="
            font-family: Arial;
            font-size: 13px;
            padding: 8px;
            line-height: 1.4;
        ">
            <div style="font-size: 15px; font-weight: bold; margin-bottom: 6px;">
                Coral Bleaching Information
            </div>

            <div><b>Latitude:</b> {lat:.2f}</div>
            <div><b>Longitude:</b> {lon:.2f}</div>

            <div style="margin-top: 6px;">
                <b>Date:</b> {dt.date()}
            </div>

            <div style="margin-top: 6px;">
                <b>Bleaching Probability:</b> {prob:.2f}
            </div>

            <div style="margin-top: 6px;">
                <b>Risk Level:</b> {risk_display}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(marker_cluster)

    folium.LayerControl().add_to(m)

    # legend
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
    <b>Bleaching Risk Legend</b><br>
    🟥 Very High (≥ 0.8)<br>
    🟧 High (0.5–0.8)<br>
    🟨 Moderate (0.2–0.5)<br>
    🟩 Low (≥ 0.1)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # save
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    m.save(OUT_HTML)
    print("map saved:", OUT_HTML)


if __name__ == "__main__":
    main()
