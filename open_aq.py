from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
import math

# 글로벌 변수 캐싱
df_openaq_cache: Optional[pd.DataFrame] = None
OPENAQ_PARQUET_PATH = Path("/mnt/data/features/openaq/openaq_nrt.parquet")


def load_openaq_data() -> pd.DataFrame:
    """OpenAQ NRT Parquet 로드 및 캐싱"""
    global df_openaq_cache
    if df_openaq_cache is None:
        if not OPENAQ_PARQUET_PATH.exists():
            raise FileNotFoundError(f"OpenAQ Parquet not found: {OPENAQ_PARQUET_PATH}")
        df_openaq_cache = pd.read_parquet(OPENAQ_PARQUET_PATH)
        df_openaq_cache["time"] = pd.to_datetime(df_openaq_cache["time"])
        print(f"✓ Loaded OpenAQ: {len(df_openaq_cache):,} records")
    return df_openaq_cache


def haversine(lat1, lon1, lat2, lon2) -> float:
    """두 위경도 좌표 간 거리 (km)"""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """서버 시작/종료 이벤트"""
    try:
        load_openaq_data()
    except FileNotFoundError as e:
        print(f"⚠️ OpenAQ data not found: {e}")
    print("✓ OpenAQ PM2.5 API server started")
    yield


# FastAPI 앱 생성
app = FastAPI(
    title="OpenAQ PM2.5 API",
    description="OpenAQ 실측 데이터 기반 대기질 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "OpenAQ PM2.5 API",
        "version": "1.0.0",
        "endpoints": [
            "/api/pm25/nearest"
        ]
    }


@app.get("/api/pm25/nearest")
async def get_pm25_nearest(
        lat: float = Query(..., description="사용자 위도"),
        lon: float = Query(..., description="사용자 경도"),
        radius_km: float = Query(5.0, description="검색 반경 (km)")
):
    """
    사용자 위치(lat, lon)를 기준으로 반경 내 관측소들의 위도, 경도, PM2.5 수치 반환
    """
    df = load_openaq_data()
    latest_time = df["time"].max()
    df_latest = df[df["time"] == latest_time].copy()

    stations = []
    for _, row in df_latest.iterrows():
        d = haversine(lat, lon, row["lat"], row["lon"])
        if d <= radius_km:
            stations.append({
                "lat": row["lat"],
                "lon": row["lon"],
                "pm25": float(row["pm25"])
            })

    if not stations:
        raise HTTPException(status_code=404,
                            detail=f"No stations found within {radius_km} km of ({lat}, {lon})")

    return {
        "location": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "time": latest_time.isoformat(),
        "stations_count": len(stations),
        "stations": stations
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("open_aq:app", host="0.0.0.0", port=8000, reload=True)
