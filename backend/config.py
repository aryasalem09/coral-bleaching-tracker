from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = PACKAGE_DIR / "ml"
ML_ARTIFACTS_DIR = ML_DIR / "artifacts"

OBSERVED_RAW_CANDIDATES = [
    RAW_DIR / "global_coral_bleaching_bco_dmo.csv",
    RAW_DIR / "global_bleaching_environmental.csv",
]
OBSERVED_RAW_PATH = next((path for path in OBSERVED_RAW_CANDIDATES if path.exists()), OBSERVED_RAW_CANDIDATES[0])
OBSERVED_SITE_DATE_PATH = PROCESSED_DIR / "observed_site_date_clean.csv"
OBSERVED_SITE_MONTH_PATH = PROCESSED_DIR / "observed_site_month_dataset.csv"
OBSERVED_SITE_CATALOG_PATH = PROCESSED_DIR / "observed_site_catalog.csv"
OBSERVED_CONFLICT_LOG_PATH = PROCESSED_DIR / "observed_conflicts.csv"
OBSERVED_EXCLUSION_LOG_PATH = PROCESSED_DIR / "observed_exclusions.csv"

MODEL_BUNDLE_PATH = ML_ARTIFACTS_DIR / "bleaching_event_model.joblib"
MODEL_METRICS_PATH = ML_ARTIFACTS_DIR / "metrics.json"
MODEL_INFO_PATH = ML_ARTIFACTS_DIR / "model_info.json"
FEATURE_IMPORTANCE_PATH = ML_ARTIFACTS_DIR / "feature_importance.csv"
SPLIT_SUMMARY_PATH = ML_ARTIFACTS_DIR / "split_summary.json"
TRAINING_REPORT_PATH = ML_ARTIFACTS_DIR / "training_report.md"

CONFUSION_MATRIX_PATH = ML_ARTIFACTS_DIR / "confusion_matrix.png"
PRECISION_RECALL_PATH = ML_ARTIFACTS_DIR / "precision_recall_curve.png"
ROC_PATH = ML_ARTIFACTS_DIR / "roc_curve.png"
CALIBRATION_PATH = ML_ARTIFACTS_DIR / "calibration_curve.png"

NOAA_DHW_DIR = RAW_DIR / "noaa_dhw"
NOAA_HS_DIR = RAW_DIR / "noaa_hs"
NOAA_MANIFEST_PATH = RAW_DIR / "noaa_manifest.json"
NOAA_WEEKLY_MANIFEST_PATH = RAW_DIR / "noaa_manifest_weekly_mondays.json"
NOAA_WEEKLY_FEATURE_AUDIT_PATH = PROCESSED_DIR / "noaa_weekly_feature_audit.json"

APP_VERSION = os.getenv("CBT_VERSION", "2.0.0")
MODEL_VERSION = os.getenv("CBT_MODEL_VERSION", "2026.03.31")
XR_ENGINE = os.getenv("XR_ENGINE", "h5netcdf")
AUTO_DOWNLOAD_NOAA = os.getenv("AUTO_DOWNLOAD_NOAA", "false").lower() in {"1", "true", "yes"}
REEF_KEY_DECIMALS = int(os.getenv("REEF_KEY_DECIMALS", "4"))
MAX_VIEWPORT_POINTS = int(os.getenv("MAX_VIEWPORT_POINTS", "2500"))
MODEL_DECISION_THRESHOLD = float(os.getenv("MODEL_DECISION_THRESHOLD", "0.5"))


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, ML_ARTIFACTS_DIR, NOAA_DHW_DIR, NOAA_HS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
