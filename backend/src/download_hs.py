import os
import pathlib
import requests

# Base URL for HotSpot (HS) daily data
BASE_URL_HS = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/"
    "5km/v3.1_op/nc/v1.0/daily/hs"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "raw", "noaa_hs")
os.makedirs(OUT_DIR, exist_ok=True)


# ct5km_hs_v3.1_20240101.nc HOTSPOT
FILE_TEMPLATE_HS = "ct5km_hs_v3.1_{ymd}.nc"

START_YEAR = 1985
END_YEAR = 2025


def download_file(url, out_path):
    if pathlib.Path(out_path).exists():
        print("skip:", out_path)
        return

    print("dw:", url)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("save:", out_path)
    else:
        print("fail:", url, "stat:", r.status_code)


def main():
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            ymd = f"{year}{month:02d}01"
            filename = FILE_TEMPLATE_HS.format(ymd=ymd)
            url = f"{BASE_URL_HS}/{year}/{filename}"
            out_path = os.path.join(OUT_DIR, filename)
            download_file(url, out_path)


if __name__ == "__main__":
    main()
