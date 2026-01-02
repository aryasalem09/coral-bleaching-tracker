import os
import json
import calendar
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_CSV = os.path.join(BASE_DIR, "data", "processed", "reef_with_preds.csv")
OUT_DIR = os.path.join(BASE_DIR, "web_data")
OUT_JSON = os.path.join(OUT_DIR, "bleaching_by_month.json")


def main(
    min_prob=0.1,
    max_points_per_month=6000,
):
    print("Loading predictions from:", PRED_CSV)
    df = pd.read_csv(PRED_CSV)

    if "date" not in df.columns:
        raise ValueError(
            "chekc pred reefs csv for the dates column "
            "noaa_prep.py needs to add it and rerun the pipeline"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["lat", "lon", "bleaching_prob"])
    df = df[df["bleaching_prob"] >= min_prob]
    df = df.sort_values("date")

    # Y-M Key
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    months = sorted(df["year_month"].unique())
    print("Months:", months[0], "→", months[-1], f"({len(months)} total)")

    data = {
        "timeKeys": months,  # YYYY_MM
        "labels": [],        # prettification
        "slices": {},        # list of lat lon prob
    }

    for ym in months:
        df_m = df[df["year_month"] == ym]

        if len(df_m) > max_points_per_month:
            df_m = df_m.sample(n=max_points_per_month, random_state=42)

        # Convert to [lat, lon, prob]
        points = df_m[["lat", "lon", "bleaching_prob"]].values.astype(float)
        data["slices"][ym] = points.tolist()

        year_str, month_str = ym.split("-")
        year = int(year_str)
        month = int(month_str)
        label = f"{calendar.month_abbr[month]} {year}"
        data["labels"].append(label)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(data, f)

    print("json:", OUT_JSON)


if __name__ == "__main__":
    main()
