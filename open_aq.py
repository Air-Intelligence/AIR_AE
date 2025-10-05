from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

# ===============================================================
# 전역 캐시 및 상수
# ===============================================================
df_tempo_cache: Optional[pd.DataFrame] = None
df_openaq_cache: Optional[pd.DataFrame] = None
TEMPO_PARQUET_PATH = Path("/mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet")
OPENAQ_PARQUET_PATH = Path("/mnt/data/features/openaq/openaq_nrt.parquet")
EARTH_RADIUS_KM = 6371.0


# ===============================================================
# 유틸 함수
# ===============================================================
def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (km)"""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * atan2(sqrt(a), sqrt(1 - a))


def load_tempo_data() -> pd.DataFrame:
    global df_tempo_cache
    if df_tempo_cache is None:
        if not TEMPO_PARQUET_PATH.exists():
            raise FileNotFoundError(f"TEMPO Parquet not found: {TEMPO_PARQUET_PATH}")
        df_tempo_cache = pd.read_parquet(TEMPO_PARQUET_PATH)
        df_tempo_cache["time"] = pd.to_datetime(df_tempo_cache["time"])
        print(f"✓ Loaded TEMPO: {len(df_tempo_cache):,} records")
    return df_tempo_cache


def load_openaq_data() -> pd.DataFrame:
    global df_openaq_cache
    if df_openaq_cache is None:
        if not OPENAQ_PARQUET_PATH.exists():
            raise FileNotFoundError(f"OpenAQ Parquet not found: {OPENAQ_PARQUET_PATH}")
        df_openaq_cache = pd.read_parquet(OPENAQ_PARQUET_PATH)
        df_openaq_cache["time"] = pd.to_datetime(df_openaq_cache["time"])
        print(f"✓ Loaded OpenAQ: {len(df_openaq_cache):,} records")
    return df_openaq_cache


# ===============================================================
# Lifespan
# ===============================================================
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


# ===============================================================
# FastAPI 앱
# ===============================================================
app = FastAPI(
    title="TEMPO NRT API",
    description="NASA TEMPO 실시간 대기질 데이터 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===============================================================
# 루트 및 TEMPO 엔드포인트
# ===============================================================
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
                "/api/grid",
            ],
            "OpenAQ": [
                "/api/pm25/stations",
                "/api/pm25/latest",
                "/api/pm25/timeseries",
            ],
            "Combined": ["/api/combined/latest"],
        },
    }


@app.get("/api/stats")
async def get_stats():
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
        },
    }


@app.get("/api/latest")
async def get_latest(variable: str = Query("no2")):
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")
    latest_time = df["time"].max()
    df_latest = df[df["time"] == latest_time].dropna(subset=[variable])
    return {
        "time": latest_time.isoformat(),
        "variable": variable,
        "count": len(df_latest),
        "data": df_latest[["lat", "lon", variable]].to_dict(orient="records"),
    }


@app.get("/api/heatmap")
async def get_heatmap(
        lat: Optional[float] = Query(None, description="사용자 위도"),
        lon: Optional[float] = Query(None, description="사용자 경도"),
        radius_km: float = Query(120.0, description="검색 반경 (km)"),
        variable: str = Query("no2", description="변수명"),
        time: Optional[str] = Query(None, description="ISO 시간 (예: 2025-10-03T23:00:00)"),
):
    """
    특정 시간의 히트맵 데이터
    - lat, lon이 주어지면 반경 radius_km(기본 120km) 내 데이터만 반환
    """
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")

    # 시간 선택
    if time:
        target_time = pd.to_datetime(time)
        time_diff = (df["time"] - target_time).abs()
        selected_time = df.loc[time_diff.idxmin(), "time"]
    else:
        selected_time = df["time"].max()

    df_filtered = df[df["time"] == selected_time].dropna(subset=[variable])

    # 거리 필터
    if lat is not None and lon is not None:
        df_filtered["distance_km"] = df_filtered.apply(
            lambda r: haversine(lat, lon, r["lat"], r["lon"]), axis=1
        )
        df_filtered = df_filtered[df_filtered["distance_km"] <= radius_km]

    if df_filtered.empty:
        raise HTTPException(404, "No data in specified range")

    return {
        "time": selected_time.isoformat(),
        "variable": variable,
        "count": len(df_filtered),
        "data": df_filtered[["lat", "lon", variable]].to_dict(orient="records"),
    }


@app.get("/api/grid")
async def get_grid(
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        variable: str = Query("no2"),
):
    df = load_tempo_data()
    if variable not in df.columns:
        raise HTTPException(400, f"Variable '{variable}' not found")
    df_filtered = df[
        (df["lat"] >= lat_min)
        & (df["lat"] <= lat_max)
        & (df["lon"] >= lon_min)
        & (df["lon"] <= lon_max)
        ].dropna(subset=[variable])
    if df_filtered.empty:
        raise HTTPException(404, "No data found in region")
    return {
        "count": len(df_filtered),
        "data": df_filtered[["time", "lat", "lon", variable]].to_dict(orient="records"),
    }


# ===============================================================
# OpenAQ 엔드포인트
# ===============================================================
@app.get("/api/pm25/stations")
async def get_pm25_stations():
    df = load_openaq_data()
    latest = df.sort_values("time").groupby(["lat", "lon", "location_name"]).tail(1)
    return {
        "count": len(latest),
        "stations": [
            {
                "lat": r["lat"],
                "lon": r["lon"],
                "name": r["location_name"],
                "pm25": float(r["pm25"]),
                "time": r["time"].isoformat(),
            }
            for _, r in latest.iterrows()
        ],
    }


@app.get("/api/pm25/latest")
async def get_pm25_latest():
    df = load_openaq_data()
    latest_time = df["time"].max()
    df_latest = df[df["time"] == latest_time]
    return {
        "time": latest_time.isoformat(),
        "count": len(df_latest),
        "data": df_latest[["lat", "lon", "pm25", "location_name"]].to_dict(
            orient="records"
        ),
    }


@app.get("/api/pm25/timeseries")
async def get_pm25_timeseries(location_name: Optional[str] = Query(None)):
    df = load_openaq_data()
    if location_name:
        df_filtered = df[df["location_name"] == location_name]
        if df_filtered.empty:
            raise HTTPException(404, f"Station '{location_name}' not found")
        ts = df_filtered.groupby("time")["pm25"].mean().reset_index()
    else:
        ts = df.groupby("time")["pm25"].mean().reset_index()
    return {
        "location": location_name or "All Stations",
        "count": len(ts),
        "data": [
            {"time": r["time"].isoformat(), "pm25": float(r["pm25"])} for _, r in ts.iterrows()
        ],
    }


# ===============================================================
# Combined Latest
# ===============================================================
@app.get("/api/combined/latest")
async def get_combined_latest():
    df_tempo = load_tempo_data()
    df_openaq = load_openaq_data()
    t_time = df_tempo["time"].max()
    o_time = df_openaq["time"].max()
    df_tempo_latest = df_tempo[df_tempo["time"] == t_time]
    df_openaq_latest = df_openaq[df_openaq["time"] == o_time]
    tempo_cols = [c for c in ["lat", "lon", "no2", "o3"] if c in df_tempo_latest.columns]
    return {
        "tempo": {
            "time": t_time.isoformat(),
            "count": len(df_tempo_latest),
            "data": df_tempo_latest[tempo_cols].to_dict(orient="records"),
        },
        "openaq": {
            "time": o_time.isoformat(),
            "count": len(df_openaq_latest),
            "data": df_openaq_latest[["lat", "lon", "pm25", "location_name"]].to_dict(
                orient="records"
            ),
        },
    }


# ===============================================================
# 실행
# ===============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("open_aq:app", host="0.0.0.0", port=8000, reload=True)
