from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import project config
sys.path.append(str(Path(__file__).parent))
import config

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger("open_aq")

# ============================================================================
# PATHS (from config.py)
# ============================================================================
MODEL_PATH = config.MODELS_DIR / "residual_lgbm.pkl"
MODEL_DIR = config.MODELS_DIR

TEMPO_PARQUET_PATH = config.FEATURES_TEMPO_NRT / "nrt_merged.parquet"
OPENAQ_PARQUET_PATH = config.FEATURES_OPENAQ / "openaq_nrt.parquet"
O3_STATIC_PARQUET_PATH = config.FEATURES_DIR / "tempo" / "o3_static.parquet"
OPENAQ_LATEST_CSV_PATH = config.RAW_OPENAQ / "latest_observations.csv"

POLLUTANT_SCALING: Dict[str, float] = {"no2": 1e15, "o3": 1e15}
POLLUTANT_UNITS: Dict[str, str] = {
    "no2": "x10^15 molecules/cm^2",
    "o3": "x10^15 molecules/cm^2",
    "pm25": "ug/m³",
}

DataFrameLoader = Callable[[Path], pd.DataFrame]
PostProcessor = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class DataSetCache:
    path: Path
    reader: DataFrameLoader
    postprocess: Optional[PostProcessor] = None
    name: str = ""

    _cache: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _mtime: Optional[float] = field(default=None, init=False, repr=False)

    def load(self, *, force: bool = False) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.name or self.path.name} not found at {self.path}"
            )

        current_mtime = self.path.stat().st_mtime
        if force or self._cache is None or self._mtime != current_mtime:
            df = self.reader(self.path)
            if self.postprocess:
                df = self.postprocess(df)
            self._cache = df
            self._mtime = current_mtime
            logger.info(
                "Loaded %s (%s rows) from %s",
                self.name or self.path.name,
                len(df),
                self.path,
            )

        if self._cache is None:
            raise RuntimeError("Cache load failed")
        return self._cache


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _with_parsed_time(column: str = "time", *, errors: str = "raise") -> PostProcessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[column] = pd.to_datetime(df[column], errors=errors)
        return df

    return _inner


DATASETS: Dict[str, DataSetCache] = {
    "tempo": DataSetCache(
        path=TEMPO_PARQUET_PATH,
        reader=_read_parquet,
        postprocess=_with_parsed_time(),
        name="TEMPO NO2 NRT data",
    ),
    "openaq": DataSetCache(
        path=OPENAQ_PARQUET_PATH,
        reader=_read_parquet,
        postprocess=_with_parsed_time(),
        name="OpenAQ NRT data",
    ),
    "o3_static": DataSetCache(
        path=O3_STATIC_PARQUET_PATH,
        reader=_read_parquet,
        postprocess=_with_parsed_time(errors="coerce"),
        name="TEMPO O3 static data",
    ),
    "openaq_latest": DataSetCache(
        path=OPENAQ_LATEST_CSV_PATH,
        reader=_read_csv,
        postprocess=_with_parsed_time(errors="coerce"),
        name="OpenAQ latest observations",
    ),
}


def load_tempo_data(force: bool = False) -> pd.DataFrame:
    return DATASETS["tempo"].load(force=force)


def load_openaq_data(force: bool = False) -> pd.DataFrame:
    return DATASETS["openaq"].load(force=force)


def load_o3_static_data(force: bool = False) -> pd.DataFrame:
    return DATASETS["o3_static"].load(force=force)


def load_openaq_latest_data(force: bool = False) -> pd.DataFrame:
    return DATASETS["openaq_latest"].load(force=force)


@lru_cache(maxsize=1)
def _get_model() -> joblib.BaseEstimator:
    """
    Load LightGBM model (cached with lru_cache for singleton pattern)
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500, detail=f"Model not found: {MODEL_PATH}"
        )
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded LGBM model from %s", MODEL_PATH)
    return model


class PredictReq(BaseModel):
    lat: float
    lon: float
    when: Optional[str] = Field(None, description="ISO8601 UTC; omit = now")


def _parse_request_time(raw: Optional[str]) -> pd.Timestamp:
    if raw is None:
        return pd.Timestamp.utcnow()

    try:
        ts = pd.to_datetime(raw, utc=True)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid datetime format: {raw}"
        ) from exc

    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]

    timestamp = pd.Timestamp(ts.to_pydatetime().replace(tzinfo=None))
    return timestamp


def _snapshot(df: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    subset = df[df["time"] <= when]
    if subset.empty:
        latest_time = df["time"].max()
        return df[df["time"] == latest_time]

    latest_time = subset["time"].max()
    return df[df["time"] == latest_time]


def _nearest_or_mean(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    value_col: str,
    radius: float = 0.05,
    return_distance: bool = False,
) -> float | tuple[float, float]:
    """
    Get nearest or mean value within radius.

    Args:
        df: DataFrame with lat, lon, and value columns
        lat: Target latitude
        lon: Target longitude
        value_col: Column name to extract value from
        radius: Search radius in degrees
        return_distance: If True, also return distance in km

    Returns:
        Value (float) or (value, distance_km) tuple if return_distance=True
    """
    nearby = df[
        df["lat"].between(lat - radius, lat + radius)
        & df["lon"].between(lon - radius, lon + radius)
    ]

    if not nearby.empty:
        value = float(nearby[value_col].mean())
        if return_distance:
            # Calculate mean distance using haversine
            import numpy as np
            nearby_lats = np.deg2rad(nearby["lat"].values)
            nearby_lons = np.deg2rad(nearby["lon"].values)
            lat_rad = np.deg2rad(lat)
            lon_rad = np.deg2rad(lon)

            dlat = nearby_lats - lat_rad
            dlon = nearby_lons - lon_rad
            a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(nearby_lats) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            dist_km = 6371.0088 * c  # Earth radius in km
            mean_dist = float(np.mean(dist_km))
            return value, mean_dist
        return value

    # Nearest point fallback
    idx = ((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2).idxmin()
    value = float(df.loc[idx, value_col])

    if return_distance:
        # Calculate haversine distance to nearest point
        import numpy as np
        nearest_lat = np.deg2rad(df.loc[idx, "lat"])
        nearest_lon = np.deg2rad(df.loc[idx, "lon"])
        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        dlat = nearest_lat - lat_rad
        dlon = nearest_lon - lon_rad
        a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(nearest_lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_km = float(6371.0088 * c)
        return value, dist_km

    return value


def _ensure_variable(df: pd.DataFrame, variable: str) -> None:
    if variable not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found. Available columns: {available}",
        )


def _apply_pollutant_scaling(df: pd.DataFrame, variable: str) -> Optional[str]:
    factor = POLLUTANT_SCALING.get(variable)
    if factor:
        df[variable] = df[variable] / factor
    return POLLUTANT_UNITS.get(variable)


def _latest_timeframe(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    latest_time = df["time"].max()
    return latest_time, df[df["time"] == latest_time].copy()


DOWNLOAD_HINTS = {
    "o3_static": "python scripts/download/download_o3_static.py",
    "openaq_latest": "python scripts/download/download_openaq_latest.py",
}

WARMUP_KEYS = ("tempo", "openaq", "o3_static", "openaq_latest")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for key in WARMUP_KEYS:
        try:
            DATASETS[key].load()
        except FileNotFoundError as exc:
            hint = DOWNLOAD_HINTS.get(key)
            if hint:
                logger.warning("%s (hint: %s)", exc, hint)
            else:
                logger.warning("%s", exc)
        except Exception as exc:  # 👈 이 두 줄 추가!
            logger.warning("Warmup skip (%s): %s", key, exc)

    logger.info("TEMPO NRT API server started")
    yield


app = FastAPI(
    title="TEMPO NRT API",
    description="NASA TEMPO near real-time air quality API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/predict/pm25")
def predict_pm25_lgbm(req: PredictReq):
    import time
    import numpy as np
    start_time = time.time()

    when = _parse_request_time(req.when)
    logger.info(f"PM2.5 prediction request: lat={req.lat}, lon={req.lon}, when={when}")

    try:
        df_no2 = load_tempo_data()
        df_o3 = load_o3_static_data()
        df_obs = load_openaq_latest_data()

        snapshot_no2 = _snapshot(df_no2, when)
        snapshot_o3 = _snapshot(df_o3, when)
        obs_time, snapshot_obs = _latest_timeframe(df_obs)

        f_no2 = _nearest_or_mean(snapshot_no2, req.lat, req.lon, "no2")
        f_o3 = _nearest_or_mean(snapshot_o3, req.lat, req.lon, "o3")
        # Get PM2.5 lag1 and distance to nearest observation point
        f_pm25_lag1, dist_km = _nearest_or_mean(snapshot_obs, req.lat, req.lon, "pm25", return_distance=True)

        # Time encoding (matching training features)
        hour_sin = np.sin(2 * np.pi * when.hour / 24)
        hour_cos = np.cos(2 * np.pi * when.hour / 24)
        dow_sin = np.sin(2 * np.pi * when.dayofweek / 7)
        dow_cos = np.cos(2 * np.pi * when.dayofweek / 7)

        # Build features matching training (9 features total)
        # Feature order MUST match: ['no2', 'o3', 'dist_km', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'pm25_lag1', 'pm25_lag3']
        features = pd.DataFrame(
            [
                {
                    "no2": f_no2,
                    "o3": f_o3,
                    "dist_km": dist_km,
                    "hour_sin": hour_sin,
                    "hour_cos": hour_cos,
                    "dow_sin": dow_sin,
                    "dow_cos": dow_cos,
                    "pm25_lag1": f_pm25_lag1,
                    "pm25_lag3": f_pm25_lag1,  # Use lag1 as approximation for lag3
                }
            ]
        ).fillna(0.0)

        model = _get_model()
        y = float(model.predict(features)[0])

        elapsed = time.time() - start_time
        logger.info(f"PM2.5 prediction successful: {y:.2f} µg/m³ (took {elapsed:.3f}s)")

        return {
            "when": when.isoformat(),
            "pred_pm25": y,
            "features": {
                "lat": req.lat,
                "lon": req.lon,
                "no2": f_no2,
                "o3": f_o3,
                "pm25_lag1": f_pm25_lag1,
                "dist_km": dist_km,
                "hour": when.hour,
                "dow": when.dayofweek,
                "obs_time": obs_time.isoformat()
            },
            "model": MODEL_PATH.name,
        }

    except Exception as exc:
        elapsed = time.time() - start_time
        logger.error(f"PM2.5 prediction failed after {elapsed:.3f}s: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.get("/")
def root():
    return {
        "message": "TEMPO + OpenAQ NRT API",
        "version": "1.0.0",
        "endpoints": {
            "tempo": [
                "/api/stats",
                "/api/latest",
                "/api/timeseries",
                "/api/heatmap",
                "/api/grid",
            ],
            "openaq": [
                "/api/pm25/stations",
                "/api/pm25/latest",
                "/api/pm25/timeseries",
                "/api/pm25/latest_csv",
            ],
            "combined": [
                "/api/combined/latest",
            ],
            "forecast": [
                "/api/predict/pm25",
            ],
        },
    }


@app.get("/api/stats")
def get_stats():
    df = load_tempo_data()

    return {
        "total_records": len(df),
        "time_range": {
            "start": df["time"].min().isoformat(),
            "end": df["time"].max().isoformat(),
            "unique_times": int(df["time"].nunique()),
        },
        "spatial_range": {
            "lat_min": float(df["lat"].min()),
            "lat_max": float(df["lat"].max()),
            "lon_min": float(df["lon"].min()),
            "lon_max": float(df["lon"].max()),
            "unique_locations": int(len(df[["lat", "lon"]].drop_duplicates())),
        },
        "variables": {
            "no2": {
                "min": float(df["no2"].min()),
                "max": float(df["no2"].max()),
                "mean": float(df["no2"].mean()),
            },
            "o3": {
                "min": float(df["o3"].min()),
                "max": float(df["o3"].max()),
                "mean": float(df["o3"].mean()),
            },
        },
    }


@app.get("/api/latest")
def get_latest(
    variable: str = Query(
        "no2", description="Variable name (e.g. no2, o3, uv_aerosol_index)"
    ),
):
    df = load_tempo_data()
    _ensure_variable(df, variable)

    latest_time, frame = _latest_timeframe(df)
    frame = frame.dropna(subset=[variable])
    unit = _apply_pollutant_scaling(frame, variable)

    return {
        "time": latest_time.isoformat(),
        "variable": variable,
        "unit": unit,
        "count": len(frame),
        "data": frame[["lat", "lon", variable]].to_dict(orient="records"),
    }


@app.get("/api/timeseries")
def get_timeseries(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    variable: str = Query("no2", description="Variable name"),
    radius: float = Query(0.05, description="Search radius (degrees)"),
):
    df = load_tempo_data()
    _ensure_variable(df, variable)

    nearby = df[
        (df["lat"] >= lat - radius)
        & (df["lat"] <= lat + radius)
        & (df["lon"] >= lon - radius)
        & (df["lon"] <= lon + radius)
    ].copy()

    if nearby.empty:
        raise HTTPException(
            status_code=404, detail=f"No data found near ({lat}, {lon})"
        )

    ts = nearby.groupby("time")[variable].mean().reset_index().sort_values("time")

    unit = None
    factor = POLLUTANT_SCALING.get(variable)
    if factor:
        ts[variable] = ts[variable] / factor
        unit = POLLUTANT_UNITS.get(variable)

    # Convert to list of dicts (optimized, avoiding iterrows)
    data = ts[["time", variable]].to_dict(orient="records")
    for item in data:
        item["time"] = item["time"].isoformat()
        item["value"] = float(item.pop(variable))

    return {
        "location": {"lat": lat, "lon": lon},
        "variable": variable,
        "unit": unit,
        "count": len(data),
        "data": data,
    }


@app.get("/api/heatmap")
def get_heatmap(
    time: Optional[str] = Query(
        None, description="ISO timestamp (e.g. 2025-10-03T23:00:00)"
    ),
    variable: str = Query("no2", description="Variable name"),
):
    df = load_tempo_data()
    _ensure_variable(df, variable)

    if time:
        try:
            target_time = pd.to_datetime(time)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid timestamp: {time}"
            ) from exc

        time_diff = (df["time"] - target_time).abs()
        closest_time = df.loc[time_diff.idxmin(), "time"]
    else:
        closest_time = df["time"].max()

    frame = df[df["time"] == closest_time].copy().dropna(subset=[variable])
    unit = _apply_pollutant_scaling(frame, variable)

    return {
        "time": closest_time.isoformat(),
        "variable": variable,
        "unit": unit,
        "count": len(frame),
        "bounds": {
            "lat_min": float(frame["lat"].min()),
            "lat_max": float(frame["lat"].max()),
            "lon_min": float(frame["lon"].min()),
            "lon_max": float(frame["lon"].max()),
        },
        "data": frame[["lat", "lon", variable]].to_dict(orient="records"),
    }


@app.get("/api/grid")
def get_grid(
    lat_min: float = Query(..., description="Minimum latitude"),
    lat_max: float = Query(..., description="Maximum latitude"),
    lon_min: float = Query(..., description="Minimum longitude"),
    lon_max: float = Query(..., description="Maximum longitude"),
    variable: str = Query("no2", description="Variable name"),
    time_start: Optional[str] = Query(None, description="Start time (inclusive)"),
    time_end: Optional[str] = Query(None, description="End time (inclusive)"),
):
    df = load_tempo_data()
    _ensure_variable(df, variable)

    filtered = df[
        (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
    ].copy()

    if time_start:
        filtered = filtered[filtered["time"] >= pd.to_datetime(time_start)]
    if time_end:
        filtered = filtered[filtered["time"] <= pd.to_datetime(time_end)]

    filtered = filtered.dropna(subset=[variable])
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found in specified region")

    unit = _apply_pollutant_scaling(filtered, variable)

    return {
        "bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
        },
        "time_range": {
            "start": filtered["time"].min().isoformat(),
            "end": filtered["time"].max().isoformat(),
        },
        "variable": variable,
        "unit": unit,
        "count": len(filtered),
        "data": filtered[["time", "lat", "lon", variable]].to_dict(orient="records"),
    }


@app.get("/api/pm25/latest_csv")
def get_pm25_latest_csv(
    force: bool = Query(False, description="Ignore cache and reload file"),
):
    df = load_openaq_latest_data(force=force)

    latest_time, frame = _latest_timeframe(df)

    return {
        "time": latest_time.isoformat(),
        "count": len(frame),
        "data": frame[["lat", "lon", "pm25", "location_name", "location_id"]].to_dict(
            orient="records"
        ),
    }


@app.get("/api/pm25/stations")
def get_pm25_stations():
    df = load_openaq_data()

    latest = (
        df.sort_values("time")
        .groupby(["lat", "lon", "location_name"], as_index=False)
        .tail(1)
    )

    # Convert to list of dicts (optimized, avoiding iterrows)
    stations = latest[["lat", "lon", "location_name", "pm25", "time"]].to_dict(orient="records")
    for station in stations:
        station["name"] = station.pop("location_name")
        station["pm25"] = float(station["pm25"])
        station["time"] = station["time"].isoformat()

    return {
        "count": len(stations),
        "stations": stations,
    }


@app.get("/api/pm25/latest")
def get_pm25_latest():
    df = load_openaq_data()

    latest_time, frame = _latest_timeframe(df)

    return {
        "time": latest_time.isoformat(),
        "count": len(frame),
        "data": frame[["lat", "lon", "pm25", "location_name"]].to_dict(
            orient="records"
        ),
    }


@app.get("/api/pm25/timeseries")
def get_pm25_timeseries(
    location_name: Optional[str] = Query(None, description="Station name"),
):
    df = load_openaq_data()

    if location_name:
        df_filtered = df[df["location_name"] == location_name].copy()
        if df_filtered.empty:
            raise HTTPException(
                status_code=404, detail=f"Station '{location_name}' not found"
            )
        ts = df_filtered.groupby("time")["pm25"].mean().reset_index()
        label = location_name
    else:
        ts = df.groupby("time")["pm25"].mean().reset_index()
        label = "All Stations (Average)"

    ts = ts.sort_values("time")

    # Convert to list of dicts (optimized, avoiding iterrows)
    data = ts[["time", "pm25"]].to_dict(orient="records")
    for item in data:
        item["time"] = item["time"].isoformat()
        item["pm25"] = float(item["pm25"])

    return {
        "location": label,
        "count": len(data),
        "data": data,
    }


@app.get("/api/combined/latest")
def get_combined_latest():
    df_tempo = load_tempo_data()
    tempo_time, df_tempo_latest = _latest_timeframe(df_tempo)

    df_tempo_latest = df_tempo_latest.dropna(subset=["no2", "o3"]).copy()
    df_tempo_latest["no2"] = df_tempo_latest["no2"] / POLLUTANT_SCALING["no2"]
    df_tempo_latest["o3"] = df_tempo_latest["o3"] / POLLUTANT_SCALING["o3"]

    df_openaq = load_openaq_data()
    openaq_time, df_openaq_latest = _latest_timeframe(df_openaq)

    return {
        "tempo": {
            "time": tempo_time.isoformat(),
            "count": len(df_tempo_latest),
            "unit": POLLUTANT_UNITS["no2"],
            "data": df_tempo_latest[["lat", "lon", "no2", "o3"]].to_dict(
                orient="records"
            ),
        },
        "openaq": {
            "time": openaq_time.isoformat(),
            "count": len(df_openaq_latest),
            "unit": POLLUTANT_UNITS["pm25"],
            "data": df_openaq_latest[["lat", "lon", "pm25", "location_name"]].to_dict(
                orient="records"
            ),
        },
    }


# ============================================================================
# 아래 엔드포인트들은 누락된 모델 파일 (pm25_lgbm.pkl, feature_scaler.pkl,
# feature_info.json)에 의존하므로 비활성화됨
# /api/predict/pm25 엔드포인트가 동일한 PM2.5 예측 기능을 제공함
# ============================================================================

# @app.post("/api/predict")
# async def predict_pm25(
#     lat: float = Query(..., description="Latitude"),
#     lon: float = Query(..., description="Longitude"),
#     city: str = Query("San Francisco", description="City"),
# ):
#     from src.features import extract_near
#     from src.model import get_predictor
#
#     try:
#         df_tempo = load_tempo_data()
#         tempo_times = sorted(df_tempo["time"].unique())
#         if len(tempo_times) < 2:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Insufficient TEMPO data (need at least 2 time points)",
#             )
#
#         time_t = tempo_times[-1]
#         time_t1 = tempo_times[-2]
#
#         no2_t = extract_near(df_tempo, lat, lon, time=time_t, value_col="no2")
#         no2_lag1 = extract_near(df_tempo, lat, lon, time=time_t1, value_col="no2")
#
#         if no2_t is None or no2_lag1 is None:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No TEMPO NO2 data near ({lat}, {lon})",
#             )
#
#         df_o3 = load_o3_static_data()
#         o3_times = sorted(df_o3["time"].unique())
#         if len(o3_times) < 2:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Insufficient O3 static data",
#             )
#
#         o3_time_t = o3_times[-1]
#         o3_time_t1 = o3_times[-2]
#
#         o3_t = extract_near(df_o3, lat, lon, time=o3_time_t, value_col="o3")
#         o3_lag1 = extract_near(df_o3, lat, lon, time=o3_time_t1, value_col="o3")
#
#         if o3_t is None or o3_lag1 is None:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No O3 data near ({lat}, {lon})",
#             )
#
#         hour = time_t.hour
#         dow = time_t.dayofweek
#
#         predictor = get_predictor(model_dir=str(MODEL_DIR))
#         result = predictor.predict(no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)
#
#         return {
#             "predicted_pm25": result["pm25_pred"],
#             "confidence_lower": result["confidence_lower"],
#             "confidence_upper": result["confidence_upper"],
#             "prediction_time": time_t.isoformat(),
#             "location": {
#                 "lat": lat,
#                 "lon": lon,
#                 "city": city,
#             },
#             "inputs": {
#                 "no2_current": no2_t,
#                 "no2_lag1": no2_lag1,
#                 "o3_current": o3_t,
#                 "o3_lag1": o3_lag1,
#                 "hour": hour,
#                 "dow": dow,
#             },
#         }
#
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(
#             status_code=500, detail=f"Prediction failed: {exc}"
#         ) from exc


# @app.get("/api/compare")
# async def compare_predictions():
#     from src.features import extract_near
#     from src.model import get_predictor
#     import numpy as np
#
#     try:
#         df_ground_truth = load_openaq_latest_data()
#
#         df_tempo = load_tempo_data()
#         df_o3 = load_o3_static_data()
#
#         tempo_times = sorted(df_tempo["time"].unique())
#         o3_times = sorted(df_o3["time"].unique())
#
#         if len(tempo_times) < 2 or len(o3_times) < 2:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Insufficient satellite data for prediction",
#             )
#
#         time_t = tempo_times[-1]
#         time_t1 = tempo_times[-2]
#         o3_time_t = o3_times[-1]
#         o3_time_t1 = o3_times[-2]
#
#         predictor = get_predictor(model_dir=str(MODEL_DIR))
#         results = []
#
#         for _, station_row in df_ground_truth.iterrows():
#             lat = station_row["lat"]
#             lon = station_row["lon"]
#             location_name = station_row.get(
#                 "location_name", station_row.get("city", "Unknown")
#             )
#             pm25_true = station_row.get("pm25", station_row.get("pm25_raw"))
#
#             if pm25_true is None:
#                 continue
#
#             no2_t = extract_near(df_tempo, lat, lon, time=time_t, value_col="no2")
#             no2_lag1 = extract_near(df_tempo, lat, lon, time=time_t1, value_col="no2")
#             o3_t = extract_near(df_o3, lat, lon, time=o3_time_t, value_col="o3")
#             o3_lag1 = extract_near(df_o3, lat, lon, time=o3_time_t1, value_col="o3")
#
#             if None in (no2_t, no2_lag1, o3_t, o3_lag1):
#                 continue
#
#             hour = time_t.hour
#             dow = time_t.dayofweek
#
#             pred_result = predictor.predict(no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)
#
#             results.append(
#                 {
#                     "location_name": location_name,
#                     "lat": lat,
#                     "lon": lon,
#                     "pm25_predicted": pred_result["pm25_pred"],
#                     "pm25_observed": float(pm25_true),
#                     "error": pred_result["pm25_pred"] - float(pm25_true),
#                     "abs_error": abs(pred_result["pm25_pred"] - float(pm25_true)),
#                 }
#             )
#
#         if not results:
#             raise HTTPException(
#                 status_code=404, detail="No valid predictions could be made"
#             )
#
#         errors = [r["error"] for r in results]
#         abs_errors = [r["abs_error"] for r in results]
#
#         metrics = {
#             "mae": float(np.mean(abs_errors)),
#             "rmse": float(np.sqrt(np.mean([e**2 for e in errors]))),
#             "mbe": float(np.mean(errors)),
#             "n_stations": len(results),
#         }
#
#         return {
#             "comparison": results,
#             "metrics": metrics,
#             "prediction_time": time_t.isoformat(),
#             "observation_time": df_ground_truth["time"].max().isoformat(),
#         }
#
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(
#             status_code=500, detail=f"Comparison failed: {exc}"
#         ) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "open_aq:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
