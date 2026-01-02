import os
import numpy as np
import pandas as pd


def generate_dataset(n=200):


    lat = np.random.uniform(-20, -15, n)
    lon = np.random.uniform(146, 152, n)


    sst_anom = np.random.normal(loc=0.6, scale=0.4, size=n)
    dhw = np.random.normal(loc=5.0, scale=3.0, size=n)
    turbidity = np.random.normal(loc=0.2, scale=0.1, size=n)


    bleaching_prob = (
        0.3 * (sst_anom > 0.7).astype(float) +
        0.5 * (dhw > 6).astype(float) +
        0.2 * np.random.rand(n)
    )

    bleaching = (bleaching_prob > 0.35).astype(int)

    df = pd.DataFrame({
        "lat": lat,
        "lon": lon,
        "sst_anom": sst_anom,
        "dhw": dhw,
        "turbidity": turbidity,
        "bleaching_present": bleaching
    })

    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")

    df = generate_dataset(200)
    df.to_csv(OUT_PATH, index=False)

    print(f"synth dataset with {len(df)} rows.")
    print(f"saved: {OUT_PATH}")
