"""Legacy wrapper for the canonical weekly NOAA downloader."""

from backend.download_noaa_weekly_mondays import download_weekly_mondays


if __name__ == "__main__":
    download_weekly_mondays()
