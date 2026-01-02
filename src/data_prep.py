import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "global_coral_bleaching_bco_dmo.csv")
OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")


def build_processed_dataset():
    print("rawd:", RAW_PATH)

    df = pd.read_csv(RAW_PATH, low_memory=False)

    numeric_cols = [
        "Latitude_Degrees",
        "Longitude_Degrees",
        "Percent_Bleaching",
        "SSTA",
        "SSTA_DHWMean",
        "Turbidity",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise ValueError(f"exp column '{c}' not found in ds")

    df = df.rename(
        columns={
            "Latitude_Degrees": "lat",
            "Longitude_Degrees": "lon",
            "SSTA": "sst_anom",
            "SSTA_DHWMean": "dhw",
            "Turbidity": "turbidity",
        }
    )

    df["bleaching_present"] = (df["Percent_Bleaching"] > 0).astype(int)

    cols_to_keep = ["lat", "lon", "sst_anom", "dhw", "turbidity", "bleaching_present"]
    df_model = df[cols_to_keep].dropna()

    print("pdataset shape:", df_model.shape)
    print("cb: 0 no 1 bleach")
    print(df_model["bleaching_present"].value_counts())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_model.to_csv(OUT_PATH, index=False)
    print("ds:", OUT_PATH)


if __name__ == "__main__":
    build_processed_dataset()
