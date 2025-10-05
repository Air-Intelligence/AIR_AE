"""
Configuration file for Bay Area air quality prediction pipeline
"""

from pathlib import Path
from datetime import datetime, timedelta

# ============================================================================
# SPATIAL SCOPE
# ============================================================================
BBOX = {
    "west": -122.75,
    "south": 36.9,
    "east": -121.6,
    "north": 38.3,
    "name": "Bay Area",
}

# Alternative: LA Basin (use if Bay Area data insufficient)
BBOX_LA = {
    "west": -118.7,
    "south": 33.6,
    "east": -117.6,
    "north": 34.3,
    "name": "LA Basin",
}

# ============================================================================
# TEMPORAL SCOPE
# ============================================================================
DATE_START = datetime(2023, 9, 1)
WEEKS = 6  # 복구: 1TB HDD로 6주 전체 사용
DATE_END = DATE_START + timedelta(weeks=WEEKS)

# Alternative: 4 weeks if download too slow
# DATE_END = datetime(2024, 8, 29)
# WEEKS = 4

# ============================================================================
# VARIABLE MAPPING
# ============================================================================
# TEMPO NetCDF variable names to standard lowercase names
TEMPO_VAR_MAPPING = {
    "vertical_column_troposphere": "no2",  # NO2 tropospheric vertical column
    "vertical_column_total": "o3",  # O3 total vertical column
}

# ============================================================================
# FEATURIZATION & STORAGE
# ============================================================================

# ROI (Region of Interest) for spatial subsetting - tighter than BBOX
ROI = {
    "west": -122.5,
    "south": 37.2,
    "east": -121.5,
    "north": 38.0,
    "name": "SF Bay Core",
}

# ============================================================================
# DATA SOURCES
# ============================================================================

# TEMPO L3 Collections (ASDC)
TEMPO_COLLECTIONS = {
    "NO2": {
        "short_name": "TEMPO_NO2_L3",
        "version": "V03",
    },  # V03 학습용 (2023-08-02 ~ 2025-09-16)
    "O3": {"short_name": "TEMPO_O3TOT_L3", "version": "V03"},  # V03 학습용
    "CLDO4": {"short_name": "TEMPO_CLDO4_L3", "version": "V04"},
    "NO2_NRT": "TEMPO_NO2_L3",  # V04 NRT용 (2025-09-17 이후, 버전은 코드에서 V04 사용)
    "O3_NRT": "TEMPO_O3TOT_L3",  # V04 NRT용
}

# MERRA-2 Collection
MERRA2_COLLECTION = "M2T1NXSLV"  # Single-Level Diagnostics, hourly

# MERRA-2 Variables (only essential ones to minimize download)
MERRA2_VARS = ["PBLH", "U10M", "V10M"]

# OpenAQ Parameters
OPENAQ_API_KEY = "0c8fb4e80f195ef18259f12508f0ce74b3cddc2ab82cc388b9bd3660ac9d27cd"
OPENAQ_CITIES = ["San Francisco", "Oakland", "San Jose", "Berkeley", "Fremont"]
OPENAQ_PARAMETER = "pm25"
OPENAQ_LIMIT = 10000  # Max results per request

# AirNow API Parameters
AIRNOW_API_KEY = (
    "03225A7B-2723-4155-8DE4-F17CC41F6C63"  # Get from https://docs.airnowapi.org
)
AIRNOW_CITIES = {
    "San Francisco": {"lat": 37.7749, "lon": -122.4194},
    "Oakland": {"lat": 37.8044, "lon": -122.2712},
    "San Jose": {"lat": 37.3382, "lon": -121.8863},
    "Berkeley": {"lat": 37.8715, "lon": -122.2730},
    "Fremont": {"lat": 37.5485, "lon": -121.9886},
}
AIRNOW_DISTANCE = 25  # Search radius in miles

# ============================================================================
# DIRECTORIES
# ============================================================================
BASE_DIR = Path(__file__).parent

# 1TB HDD 사용 (/mnt/data)
DATA_ROOT = Path("/mnt/data")
RAW_DIR = DATA_ROOT / "raw"
PROC_DIR = DATA_ROOT / "processed"  # Processed intermediate data
DATA_DIR = DATA_ROOT / "data"  # Final training data
TABLES_DIR = DATA_ROOT / "tables"
MODELS_DIR = DATA_ROOT / "models"
PLOTS_DIR = DATA_ROOT / "plots"

# Raw data subdirectories
RAW_OPENAQ = RAW_DIR / "OpenAQ"
RAW_TEMPO_NO2 = RAW_DIR / "TEMPO_NO2"
RAW_TEMPO_O3 = RAW_DIR / "TEMPO_O3"
RAW_TEMPO_CLDO4 = RAW_DIR / "TEMPO_CLDO4"
RAW_MERRA2 = RAW_DIR / "MERRA2"

# Create all directories
for directory in [
    RAW_OPENAQ,
    RAW_TEMPO_NO2,
    RAW_TEMPO_O3,
    RAW_TEMPO_CLDO4,
    RAW_MERRA2,
    PROC_DIR,
    DATA_DIR,
    TABLES_DIR,
    MODELS_DIR,
    PLOTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Feature directories (1TB HDD 사용)
FEATURES_DIR = DATA_ROOT / "features"
FEATURES_TEMPO_TRAIN = FEATURES_DIR / "tempo" / "train_6w"  # 6주로 변경
FEATURES_TEMPO_NRT = FEATURES_DIR / "tempo" / "nrt_roll3d"
FEATURES_MERRA2 = FEATURES_DIR / "merra2" / "train_6w"  # 6주로 변경
FEATURES_OPENAQ = FEATURES_DIR / "openaq"

# Create feature directories
for directory in [
    FEATURES_TEMPO_TRAIN,
    FEATURES_TEMPO_NRT,
    FEATURES_MERRA2,
    FEATURES_OPENAQ,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Storage policy (1TB HDD - 용량 제한 완화)
AUTO_DELETE_RAW_AFTER_FEATURIZE = False  # 1TB 여유로 원본 보존
DISK_USAGE_LIMIT_GB = 900  # 디스크 사용량 상한 (GB) - 1TB의 90%
DISK_CHECK_ENABLED = True  # 디스크 체크 활성화

# ============================================================================
# PREPROCESSING
# ============================================================================

# Time resampling (optional)
RESAMPLE_FREQ = "1H"  # 1-hour intervals for short-term forecasting, set None to skip

# QC Thresholds
QC_THRESHOLDS = {
    "no2_min": 0,  # Molecules/cm²
    "no2_max": 1e17,
    "o3_min": 0,
    "o3_max": 1e19,
    "pblh_min": 0,  # meters
    "pblh_max": 5000,
    "wind_max": 50,  # m/s
    "pm25_min": 0,  # µg/m³
    "pm25_max": 500,
}

# Winsorizing percentiles
WINSORIZE_LOWER = 0.001  # 0.1%
WINSORIZE_UPPER = 0.999  # 99.9%

# Label joining configuration
LABEL_JOIN = {
    "time_tolerance": "30min",  # Time matching tolerance
    "spatial_method": "kdtree",  # Spatial matching method: "kdtree" or "radius"
    "radius_km": 10.0,  # Used when spatial_method == "radius"
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Lag hours (최소화: 1, 3시간만)
LAG_HOURS = [1, 3]

# Moving average windows (hours) - 제거 (실시간 구현 간소화)
MA_WINDOWS = []

# ============================================================================
# MODELING
# ============================================================================

# Train/validation split
TRAIN_WEEKS = 4  # First 4 weeks for training
VAL_WEEKS = 2  # Last 2 weeks for validation

# Model algorithms (flags for 07_train_residual.py)
USE_LIGHTGBM = True
USE_XGBOOST = False  # Set True if LightGBM performance insufficient
USE_RANDOMFOREST = False  # Set True for ensemble
USE_ENSEMBLE = False  # Average predictions from multiple models

# LightGBM Hyperparameters (default, tuning optional)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

# XGBoost Hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
}

# Random Forest Hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
}

# ============================================================================
# OUTPUT FILES
# ============================================================================

# Intermediate data files
OPENAQ_CSV = TABLES_DIR / "openaq_pm25_6weeks.csv"
MERGED_PARQUET = TABLES_DIR / "bay_area_6w_merged.parquet"
FEATURES_PARQUET = TABLES_DIR / "features_engineered.parquet"

# Model files
BASELINE_MODEL = MODELS_DIR / "baseline_persistence.pkl"
LGBM_MODEL = MODELS_DIR / "residual_lgbm.pkl"
XGB_MODEL = MODELS_DIR / "residual_xgb.pkl"
RF_MODEL = MODELS_DIR / "residual_rf.pkl"
ENSEMBLE_MODEL = MODELS_DIR / "ensemble.pkl"

# Evaluation outputs
METRICS_JSON = MODELS_DIR / "evaluation_metrics.json"
PREDICTIONS_CSV = TABLES_DIR / "predictions.csv"

# Plot files
PLOT_TIMESERIES = PLOTS_DIR / "timeseries_pred_vs_obs.png"
PLOT_SCATTER = PLOTS_DIR / "scatter_pred_vs_obs.png"
PLOT_RESIDUALS = PLOTS_DIR / "residuals_histogram.png"
PLOT_FEATURE_IMP = PLOTS_DIR / "feature_importance.png"

# ============================================================================
# DOWNLOAD SETTINGS
# ============================================================================

# aria2 settings for parallel download
ARIA2_PARAMS = {
    "max_concurrent_downloads": 12,
    "max_connection_per_server": 4,
    "min_split_size": "10M",
    "continue": True,
    "retry_wait": 5,
    "max_tries": 10,
}

# Earthdata credentials (will be read from ~/.netrc)
# User should set up .netrc with:
# machine urs.earthdata.nasa.gov
#     login YOUR_USERNAME
#     password YOUR_PASSWORD

# ============================================================================
# PHASE 2 OPTIONS (if Phase 1 performance insufficient)
# ============================================================================

ENABLE_CLDO4 = False  # Set True to download cloud data
CLDO4_WEEKS = 4  # Reduce to 4 weeks if enabled (large file size)

EXTEND_DATE_RANGE = False  # Set True to extend by 2 more weeks
EXTENDED_DATE_END = datetime(2024, 9, 29)  # 8 weeks total

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = BASE_DIR / "pipeline.log"

# ============================================================================
# NRT (Near Real-Time) SETTINGS for V04
# ============================================================================

NRT_RECENT_DAYS = 3  # 최근 며칠치 데이터 다운로드 (실시간 웹 표시용, 72시간)

# NRT V04 데이터 저장 경로 (V03 학습 데이터와 분리)
RAW_TEMPO_NO2_NRT = RAW_DIR / "tempo_v04" / "no2"
RAW_TEMPO_O3_NRT = RAW_DIR / "tempo_v04" / "o3"

# NRT 디렉터리 생성
for directory in [RAW_TEMPO_NO2_NRT, RAW_TEMPO_O3_NRT]:
    directory.mkdir(parents=True, exist_ok=True)

print(
    f"✓ Config loaded: {BBOX['name']}, {DATE_START.date()} to {DATE_END.date()} ({WEEKS} weeks)"
)
print(f"✓ Directories: {RAW_DIR}, {TABLES_DIR}, {MODELS_DIR}, {PLOTS_DIR}")
