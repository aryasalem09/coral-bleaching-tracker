import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## BAA, DHW, HS
BAA_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa")
DHW_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_dhw")
HS_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_hs")

OUT_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")

N_SAMPLES_PER_FILE = 3000


def build_from_noaa():
    baa_files = sorted(glob.glob(os.path.join(BAA_DIR, "ct5km_baa5-max-7d_v3.1_*.nc")))
    if not baa_files:
        print("cant find the file", BAA_DIR)
        return

    print("Found", len(baa_files), "BAA NetCDF files.")

    lat_name = "lat"
    lon_name = "lon"
    time_name = "time"


    baa_var_name = "bleaching_alert_area"    # BAA / alert level
    dhw_var_name = "degree_heating_week"     # ct5km_dhw_v3.1_(y).nc
    hs_var_name  = "hotspot"                 # ct5km_hs_v3.1_(y).nc

    dfs = []

    for idx, baa_path in enumerate(baa_files, start=1):
        fname = os.path.basename(baa_path)
        print(f"[{idx}/{len(baa_files)}] process {fname}")

        # ct5km_baa5-max-7d_v3.1_(whatever date YYYYMMDD).nc
        ymd = fname.split("_")[-1].split(".")[0]
        date = pd.to_datetime(ymd, format="%Y%m%d")

        dhw_path = os.path.join(DHW_DIR, f"ct5km_dhw_v3.1_{ymd}.nc")
        hs_path = os.path.join(HS_DIR,  f"ct5km_hs_v3.1_{ymd}.nc")

        # error handling
        if not (os.path.exists(dhw_path) and os.path.exists(hs_path)):
            print("no dhw or hotsopt found for thiis data ERROR", ymd, "- skipping.")
            continue

        ds_baa = xr.open_dataset(baa_path)
        ds_dhw = xr.open_dataset(dhw_path)
        ds_hs = xr.open_dataset(hs_path)


        if time_name in ds_baa.dims:
            ds_baa = ds_baa.isel({time_name: -1})
            ds_dhw = ds_dhw.isel({time_name: -1})
            ds_hs = ds_hs.isel({time_name: -1})

        try:
            lats = ds_baa[lat_name].values
            lons = ds_baa[lon_name].values
            baa = ds_baa[baa_var_name].values
            dhw = ds_dhw[dhw_var_name].values
            hs = ds_hs[hs_var_name].values
        except KeyError as e:
            print("no var", e)
            ds_baa.close()
            ds_dhw.close()
            ds_hs.close()
            continue

        # lat/lon grid
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        lat_flat = lat_grid.ravel()
        lon_flat = lon_grid.ravel()
        baa_flat = baa.ravel()
        dhw_flat = dhw.ravel()
        hs_flat = hs.ravel()

        n_points = baa_flat.shape[0]
        n_take = min(N_SAMPLES_PER_FILE, n_points)
        idxs = np.random.choice(n_points, size=n_take, replace=False)

        df_file = pd.DataFrame({
            "lat": lat_flat[idxs],
            "lon": lon_flat[idxs],
            "baa": baa_flat[idxs],
            "dhw": dhw_flat[idxs],
            "sst_anom": hs_flat[idxs],
            "turbidity": np.zeros(n_take),
            "date": date,
        })

        #  1 IF BAA > 0
        df_file["bleaching_present"] = (df_file["baa"] > 0).astype(int)

        df_file.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_file.dropna(subset=["lat", "lon", "baa", "dhw", "sst_anom"], inplace=True)

        # lat band
        df_file = df_file[(df_file["lat"] >= -30) & (df_file["lat"] <= 30)]

        if not df_file.empty:
            dfs.append(df_file)

        ds_baa.close()
        ds_dhw.close()
        ds_hs.close()

    # erorr handling
    if not dfs:
        print("no data frames found ERROR ")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print("merged dataset shape:", df_all.shape)
    print("cb 0/1 no/bleach:")
    print(df_all["bleaching_present"].value_counts())

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_all.to_csv(OUT_PATH, index=False)
    print("saved to:", OUT_PATH)


if __name__ == "__main__":
    build_from_noaa()
