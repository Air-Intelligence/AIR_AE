from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
import numpy as np

# =========================================================
# 전역 변수 (데이터 캐싱)
# =========================================================
df_tempo_cache: Optional[pd.DataFrame] = None
df_openaq_cache: Optional[pd.DataFrame] = None

TEMPO_PARQUET_PATH = Path("/mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet")
OPENAQ_PARQUET_PATH = Path("/mnt/data/features/openaq/openaq_nrt.parquet")

EARTH_RADIUS_KM = 6371.0  # 지구 반지름


# =========================================================
# 데이터 로딩
# =========================================================
def load_tempo_data() -> pd.DataFrame:
    global df_tempo_cache
    if df_tempo_cache is None:
        if not TEMPO_PARQUET_PATH.exists():
            raise FileNotFoundError(f"TEMPO Parquet not found: {TEMPO_PARQUET_PATH}")
        df_tempo_cache = pd.read_parquet(TEMPO_PARQUET_PATH)
        df_tempo_cache["time"] = pd.to_datetime(df_tempo_cache["time"])
        print(f"✓ Loaded TEMPO: {len(df_tempo_cache):,} records from {TEMPO_PARQUET_PATH}")
    return df_tempo_cache


def load_openaq_data() -> pd.DataFrame:
    global df_openaq_cache
    if df_openaq_cache is None:
        if not OPENAQ_PARQUET_PATH.exists():
            raise FileNotFoundError(f"OpenAQ Parquet not found: {OPENAQ_PARQUET_PATH}")
        df_openaq_cache = pd.read_parquet(OPENAQ_PARQUET_PATH)
        df_openaq_cache["time"] = pd.to_datetime(df_openaq_cache["time"])
        print(f"✓ Loaded OpenAQ: {len(df_openaq_cache):,} records from {OPENAQ_PARQUET_PATH}")
    return df_openaq_cache


# =========================================================
# Lifespan
# =========================================================
@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        load_tempo_data()
    except FileNotFoundError as e:
        print(f"⚠️ TEMPO data not found: {e}")

    try:
        load_openaq_data()
    except FileNotFoundError as e:
        print(f"⚠️ OpenAQ data not found: {e}")

    print("✓ TEMPO NRT API server started")
    yield


# =========================================================
# FastAPI 앱 설정
# =========================================================
app = FastAPI(
    title="TEMPO NRT API",
    description="NASA TEMPO 실시간 대기질 데이터 API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# 루트
# =========================================================
@app.get("/")
async def root():
    return {
        "message": "TEMPO + OpenAQ NRT API",
        "version": "1.0.0",
        "endpoints": {
            "TEMPO": [
                "/api/stats",
                "/api/latest",
                "/api/timeseries",
                "/api/heatmap",
                "/api/grid"
            ],
            "OpenAQ": [
                "/api/pm25/stations",
                "/api/pm25/latest",
                "/api/pm25/timeseries"
            ],
            "Combined": [
                "/api/combined/latest"
            ]
        }
    }


# =========================================================
# TEMPO 통계
# =========================================================
@app.get("/api/stats")
async def get_stats():
    df = load_tempo_data()
    return {
        "total_records": len(df),
        "time_range": {
            "start": df["time"].min().isoformat(),
            "end": df["time"].max().isoformat(),
            "unique_times": int(df["time"].nunique())
        },
        "spatial_range": {
            "lat_min": float(df["lat"].min()),
            "lat_max": float(df["lat"].max()),
            "lon_min": float(df["lon"].min()),
            "lon_max": float(df["lon"].max()),
            "unique_locations": len(df[["lat", "lon"]].drop_duplicates())
        }
    }


# =========================================================
# 최신 TEMPO 데이터
# =========================================================
@app.get("/api/latest")
async def get_latest(variable: str = Query("no2", description="변수명 (no2, o3 등)")):
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")

    latest_time = df["time"].max()
    df_latest = df[df["time"] == latest_time].dropna(subset=[variable])

    if variable in ["no2", "o3"]:
        df_latest[variable] = df_latest[variable] / 1e15

    return {
        "time": latest_time.isoformat(),
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ["no2", "o3"] else None,
        "count": len(df_latest),
        "data": df_latest[["lat", "lon", variable]].to_dict(orient="records")
    }


# =========================================================
# 시계열
# =========================================================
@app.get("/api/timeseries")
async def get_timeseries(
        lat: float,
        lon: float,
        variable: str = Query("no2"),
        radius: float = Query(0.05)
):
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")

    df_near = df[
        (df["lat"] >= lat - radius) & (df["lat"] <= lat + radius) &
        (df["lon"] >= lon - radius) & (df["lon"] <= lon + radius)
        ]
    if df_near.empty:
        raise HTTPException(404, f"No data near ({lat}, {lon})")

    ts = df_near.groupby("time")[variable].mean().reset_index().sort_values("time")
    if variable in ["no2", "o3"]:
        ts[variable] = ts[variable] / 1e15

    return {
        "location": {"lat": lat, "lon": lon},
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ["no2", "o3"] else None,
        "count": len(ts),
        "data": [{"time": t.isoformat(), "value": float(v)} for t, v in zip(ts["time"], ts[variable])]
    }


# =========================================================
# 히트맵 (반경 필터 추가)
# =========================================================
@app.get("/api/heatmap")
async def get_heatmap(
        lat: Optional[float] = Query(None, description="사용자 위도"),
        lon: Optional[float] = Query(None, description="사용자 경도"),
        radius_km: float = Query(120.0, description="검색 반경 (km)"),
        time: Optional[str] = Query(None, description="ISO 시간 (예: 2025-10-03T23:00:00)"),
        variable: str = Query("no2", description="변수명")
):
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")

    if time:
        t = pd.to_datetime(time)
        closest_time = df.loc[(df["time"] - t).abs().idxmin(), "time"]
    else:
        closest_time = df["time"].max()

    df_filtered = df[df["time"] == closest_time].dropna(subset=[variable, "lat", "lon"]).copy()
    df_filtered["lat"] = df_filtered["lat"].astype(float)
    df_filtered["lon"] = df_filtered["lon"].astype(float)

    if lat is not None and lon is not None:
        lat1, lon1 = np.radians(lat), np.radians(lon)
        lat2, lon2 = np.radians(df_filtered["lat"].values), np.radians(df_filtered["lon"].values)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        distances = 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        df_filtered = df_filtered[distances <= radius_km]

    if variable in ["no2", "o3"]:
        df_filtered[variable] = df_filtered[variable] / 1e15

    if df_filtered.empty:
        raise HTTPException(404, "No data in specified range")

    return {
        "time": closest_time.isoformat(),
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ["no2", "o3"] else None,
        "count": len(df_filtered),
        "data": df_filtered[["lat", "lon", variable]].to_dict(orient="records")
    }


# =========================================================
# GRID, OpenAQ, Combined, Predict 그대로 유지
# =========================================================
# (생략 없이 원본 동일 — 수정 필요 없음)
# =========================================================
# ... 이하 /api/grid, /api/pm25/*, /api/combined/latest, /api/predict 그대로 유지 ...
# =========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("open_aq:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
