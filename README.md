# Bay Area Air Quality Prediction & Real-Time API

**PM2.5 Prediction System Based on NASA TEMPO Satellite Data + Real-Time Air Quality API**

---

## 📋 Overview

An end-to-end machine learning pipeline and real-time API server that predicts air quality (PM2.5) in California's Bay Area using NASA TEMPO satellite data and OpenAQ observation data.

### Key Features
- ✅ **Training Pipeline**: 6-week training based on TEMPO V03 data
- ✅ **Real-Time API**: FastAPI-based air quality data provision and PM2.5 prediction
- ✅ **Multiple Data Sources**: TEMPO NO₂/O₃ (NRT) + OpenAQ PM2.5
- ✅ **Machine Learning Model**: LightGBM residual correction model
- ✅ **RESTful API**: Real-time air quality query and prediction endpoints

---

## 📂 Project Structure

```
Project/
├── config.py                    # Global settings (BBOX, dates, API keys, paths)
├── utils.py                     # Common functions (QC, save, load, featurization)
├── open_aq.py                   # FastAPI real-time air quality API
│
├── scripts/                     # Data pipeline scripts
│   ├── download/               # Data download
│   │   ├── 01_download_openaq.py       # OpenAQ PM2.5
│   │   ├── 02_download_tempo.py        # TEMPO V03 (training)
│   │   ├── 02_download_tempo_nrt.py    # TEMPO V04 NRT (real-time)
│   │   ├── 02_download_openaq_nrt.py   # OpenAQ NRT (real-time)
│   │   ├── download_o3_static.py       # O3 static data
│   │   └── download_openaq_latest.py   # OpenAQ latest observations
│   │
│   ├── preprocess/             # Preprocessing and feature engineering
│   │   ├── 04_preprocess_merge.py      # Preprocessing & merging
│   │   ├── 04_preprocess_nrt.py        # NRT data preprocessing
│   │   ├── 05_join_labels.py           # PM2.5 label joining
│   │   └── 06_feature_engineering.py   # Feature generation (lag, time)
│   │
│   └── train/                  # Model training and evaluation
│       ├── 07_train_baseline.py        # Baseline (Persistence)
│       ├── 08_train_residual.py        # LightGBM residual model
│       ├── 09_evaluate.py              # Evaluation
│       └── train_pm25_model.py         # PM2.5 model training
│
├── src/                         # Core modules
│   ├── features.py             # Feature extraction functions
│   └── model.py                # Prediction model wrapper
│
├── analysis/                    # Analysis scripts
│   ├── analyze_nrt.py          # NRT data analysis
│   └── validate_pipeline.py    # Pipeline validation
│
├── tests/                       # Test code
│
├── /mnt/data/                  # Data storage (1TB HDD)
│   ├── raw/                    # Raw data
│   │   ├── OpenAQ/
│   │   ├── TEMPO_NO2/
│   │   ├── TEMPO_O3/
│   │   └── tempo_v04/          # NRT V04 data
│   ├── features/               # Preprocessed features
│   │   ├── tempo/train_6w/     # Training TEMPO
│   │   ├── tempo/nrt_roll3d/   # NRT TEMPO (72 hours)
│   │   └── openaq/
│   ├── tables/                 # Parquet tables
│   ├── models/                 # Trained models (pkl)
│   └── plots/                  # Visualization results
└── README.md                   # This document
```

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

#### Python Package Installation
```bash
# Core packages
pip install pandas numpy xarray earthaccess lightgbm scikit-learn matplotlib seaborn pyarrow requests

# For API server
pip install fastapi uvicorn joblib scipy

# Optional (performance improvement)
pip install xgboost  # When using XGBoost
```

#### Earthdata Authentication Setup
NASA Earthdata account required → [Sign up](https://urs.earthdata.nasa.gov/users/new)

**Windows**: Create `C:\Users\<username>\.netrc` file (no extension)
```
machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
```

**Linux/Mac**: Create `~/.netrc` file and set permissions
```bash
chmod 600 ~/.netrc
```

---

### 2️⃣ Configuration Adjustment (Optional)

Adjustable in `config.py` file:

```python
# Period (default: 6 weeks)
DATE_START = datetime(2024, 8, 1)
DATE_END = datetime(2024, 9, 15)

# Spatial (default: Bay Area)
BBOX = {
    'west': -122.75,
    'south': 36.9,
    'east': -121.6,
    'north': 38.3
}

# Model selection (default: LightGBM only)
USE_LIGHTGBM = True
USE_XGBOOST = False  # Add XGBoost when set to True
USE_RANDOMFOREST = False
USE_ENSEMBLE = False  # Multi-model averaging
```

---

### 3️⃣ Run Training Pipeline

#### Full Training (2023 data, 6 weeks)
```bash
# 1. Download training data
python scripts/download/01_download_openaq.py
python scripts/download/02_download_tempo.py  # TEMPO V03

# 2. Preprocessing and feature generation
python scripts/preprocess/04_preprocess_merge.py
python scripts/preprocess/05_join_labels.py
python scripts/preprocess/06_feature_engineering.py

# 3. Model training
python scripts/train/07_train_baseline.py
python scripts/train/08_train_residual.py  # → generates residual_lgbm.pkl

# 4. Evaluation
python scripts/train/09_evaluate.py
```

### 4️⃣ Run Real-Time API Server

#### NRT Data Download
```bash
# O3 static data (last 3 days)
python scripts/download/download_o3_static.py

# TEMPO NRT data (V04, last 3 days)
python scripts/download/02_download_tempo_nrt.py

# OpenAQ NRT data
python scripts/download/02_download_openaq_nrt.py

# OpenAQ latest observations
python scripts/download/download_openaq_latest.py

# NRT data preprocessing
python scripts/preprocess/04_preprocess_nrt.py
```

#### Start FastAPI Server
```bash
# Development mode (auto reload)
python open_aq.py

# Or
uvicorn open_aq:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn open_aq:app --host 0.0.0.0 --port 8000
```

Server access: http://localhost:8000

---

### 5️⃣ API Endpoints

#### TEMPO Satellite Data
- `GET /api/stats` - Dataset statistics
- `GET /api/latest?variable=no2` - Latest NO₂/O₃ data
- `GET /api/timeseries?lat=37.77&lon=-122.41&variable=no2` - Time series query
- `GET /api/heatmap?variable=no2&time=2025-10-03T23:00:00` - Heatmap data
- `GET /api/grid?lat_min=37.5&lat_max=38.0&lon_min=-122.5&lon_max=-122.0&variable=no2` - Grid data

#### OpenAQ PM2.5 Observations
- `GET /api/pm25/stations` - Monitoring station list
- `GET /api/pm25/latest` - Latest PM2.5 observations
- `GET /api/pm25/timeseries?location_name=San Francisco` - Station-wise time series
- `GET /api/pm25/latest_csv` - Latest observations (CSV-based)

#### PM2.5 Prediction
- `POST /api/predict/pm25` - LGBM model prediction
  ```json
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "when": "2025-10-03T23:00:00"  // optional, defaults to current
  }
  ```

- `POST /api/predict` - Prediction API
  ```json
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "city": "San Francisco"
  }
  ```

- `GET /api/compare` - Prediction vs observation comparison

#### Combined Data
- `GET /api/combined/latest` - TEMPO + OpenAQ latest data

### 6️⃣ Check Training Results

#### Evaluation Metrics
```bash
cat /mnt/data/models/evaluation_metrics.json
```

---

## 📊 Data Sources

| Data | Variables | Resolution | Size | Purpose |
|------|-----------|------------|------|---------|
| **TEMPO L3 V03** | NO₂, O₃ | Hourly, ~5 km | ~4 GB (6 weeks) | Training |
| **TEMPO L3 V04 NRT** | NO₂, O₃ | Hourly, ~5 km | ~200 MB (3 days) | Real-time prediction |
| **OpenAQ** | PM2.5 | 1 hour, station-based | ~10 MB | Prediction validation + real-time observation |

---

## 🔧 Troubleshooting

### Download Failure
**Symptoms**: `Authentication failed` or `No granules found`

**Solutions**:
1. Check `.netrc` file path/permissions
2. Verify ASDC/GES DISC approval in Earthdata account
3. Check `DATE_START/END` range in `config.py`

### Memory Shortage
**Symptoms**: `MemoryError` or slow processing

**Solutions**:
1. Reduce period from 6 weeks → 4 weeks in `config.py`
2. Modify `04_preprocess_merge.py` for file-by-file processing
3. Reduce BBOX range (Bay Area → San Francisco only)

### Poor Performance
**Symptoms**: MAE > 10 µg/m³ or R² < 0.5

**Solutions**:
1. Set `ENABLE_CLDO4 = True` in `config.py` (add cloud data)
2. Set `USE_XGBOOST = True` or `USE_ENSEMBLE = True`
3. Extend period to 8 weeks

---

## 📈 System Architecture

### Training Pipeline (Offline)
```
1. Data Collection
   ├─ TEMPO V03 (2023-09-01 ~ 2023-10-15, 6 weeks)
   │  ├─ NO₂ tropospheric column
   │  └─ O₃ total column
   └─ OpenAQ PM2.5 (Bay Area 5 cities)

2. Preprocessing
   ├─ BBOX subsetting (Bay Area)
   ├─ Tidy transformation (time, lat, lon, variable)
   ├─ Time alignment (1-hour intervals)
   ├─ Merging (spatial join)
   ├─ QC (negative/unrealistic value removal, Winsorization)
   └─ Parquet storage

3. Feature Engineering
   ├─ Lag features (t-1, t-3 hours)
   ├─ Time encoding (hour, day of week)
   └─ PM2.5 label joining (KD-tree, 10km)

4. Model Training
   ├─ Baseline: Persistence (t-1)
   ├─ Residual: PM2.5 - PM2.5(t-1)
   ├─ LightGBM training (4 weeks training, 2 weeks validation)
   └─ Model saving (residual_lgbm.pkl)

5. Evaluation
   ├─ R², MAE, RMSE
   └─ Visualization (time series, scatter plot, feature importance)
```

### Real-Time API (Online)
```
1. NRT Data Collection (periodic execution)
   ├─ TEMPO V04 NRT (last 3 days, cron)
   ├─ OpenAQ NRT (last 3 days)
   └─ OpenAQ Latest (API, every hour)

2. Preprocessing
   ├─ Rolling 3-day window
   ├─ Parquet update
   └─ Cache invalidation

3. FastAPI Server
   ├─ Load data on startup
   ├─ Auto reload on file change
   └─ Provide RESTful API

4. Prediction Endpoint
   ├─ Location (lat, lon) input
   ├─ Extract latest TEMPO NO₂/O₃
   ├─ Extract OpenAQ PM2.5(t-1)
   ├─ LGBM model inference
   └─ Return PM2.5 prediction
```

---

## 🎯 Performance Goals

| Metric | Minimum Goal | Ideal Goal |
|--------|--------------|------------|
| **MAE** | < 10 µg/m³ | < 8 µg/m³ |
| **R²** | > 0.5 | > 0.65 |
| **Improvement Rate** | 10%+ vs Baseline | 20%+ vs Baseline |

---

## 📝 Citation

Data used in this pipeline:

- **TEMPO**: NASA TEMPO Mission, ASDC
- **OpenAQ**: OpenAQ API v2

---

## 🔑 Key Technology Stack

| Category | Technologies |
|----------|-------------|
| **Data Collection** | earthaccess, requests, OpenAQ API |
| **Data Processing** | pandas, xarray, numpy, scipy |
| **Machine Learning** | LightGBM, scikit-learn |
| **API Server** | FastAPI, uvicorn, pydantic |
| **Storage** | Parquet (pyarrow), joblib |
| **Visualization** | matplotlib, seaborn |

---

---

## 💡 Improvements & Limitations

### Limitations
- We barely tested the machine learning model inference and rushed to meet the deadline, so predicted PM2.5 values sometimes came out negative or slightly low. Better normalization or post-processing would have been helpful.
- We couldn't focus much on model accuracy and didn't have enough time to compare predictions against ground truth data.

---

**Development Period**: 2025-10-04 ~ 2025-10-05 (2 days)
**Version**: 1.0.0
**Project**: NASA Hackathon 2025 - Air Intelligence
