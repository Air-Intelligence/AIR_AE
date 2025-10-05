"""
TEMPO NRT 데이터 FastAPI 백엔드
실시간 대기질 데이터 제공 API
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# FastAPI 앱 생성
app = FastAPI(
    title="TEMPO NRT API",
    description="NASA TEMPO 실시간 대기질 데이터 API",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 변수로 데이터 캐싱 (서버 시작 시 한 번만 로드)
df_tempo_cache: Optional[pd.DataFrame] = None
df_openaq_cache: Optional[pd.DataFrame] = None

TEMPO_PARQUET_PATH = Path("/mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet")
OPENAQ_PARQUET_PATH = Path("/mnt/data/features/openaq/openaq_nrt.parquet")


def load_tempo_data() -> pd.DataFrame:
    """TEMPO NRT Parquet 로드 및 캐싱"""
    global df_tempo_cache

    if df_tempo_cache is None:
        if not TEMPO_PARQUET_PATH.exists():
            raise FileNotFoundError(f"TEMPO Parquet not found: {TEMPO_PARQUET_PATH}")

        df_tempo_cache = pd.read_parquet(TEMPO_PARQUET_PATH)
        df_tempo_cache['time'] = pd.to_datetime(df_tempo_cache['time'])
        print(f"✓ Loaded TEMPO: {len(df_tempo_cache):,} records from {TEMPO_PARQUET_PATH}")

    return df_tempo_cache


def load_openaq_data() -> pd.DataFrame:
    """OpenAQ NRT Parquet 로드 및 캐싱"""
    global df_openaq_cache

    if df_openaq_cache is None:
        if not OPENAQ_PARQUET_PATH.exists():
            raise FileNotFoundError(f"OpenAQ Parquet not found: {OPENAQ_PARQUET_PATH}")

        df_openaq_cache = pd.read_parquet(OPENAQ_PARQUET_PATH)
        df_openaq_cache['time'] = pd.to_datetime(df_openaq_cache['time'])
        print(f"✓ Loaded OpenAQ: {len(df_openaq_cache):,} records from {OPENAQ_PARQUET_PATH}")

    return df_openaq_cache


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 데이터 미리 로드"""
    try:
        load_tempo_data()
    except FileNotFoundError as e:
        print(f"⚠️  TEMPO data not found: {e}")

    try:
        load_openaq_data()
    except FileNotFoundError as e:
        print(f"⚠️  OpenAQ data not found: {e}")

    print("✓ TEMPO NRT API server started")


@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "TEMPO + OpenAQ NRT API",
        "version": "1.0.0",
        "endpoints": {
            "TEMPO (위성 데이터)": [
                "/api/stats",
                "/api/latest",
                "/api/timeseries",
                "/api/heatmap",
                "/api/grid"
            ],
            "OpenAQ (PM2.5 실측)": [
                "/api/pm25/stations",
                "/api/pm25/latest",
                "/api/pm25/timeseries"
            ],
            "결합 데이터": [
                "/api/combined/latest"
            ]
        }
    }


@app.get("/api/stats")
async def get_stats():
    """
    데이터 통계 정보
    - 시간 범위, 공간 범위, 데이터 개수 등
    """
    df = load_tempo_data()

    return {
        "total_records": len(df),
        "time_range": {
            "start": df['time'].min().isoformat(),
            "end": df['time'].max().isoformat(),
            "unique_times": int(df['time'].nunique())
        },
        "spatial_range": {
            "lat_min": float(df['lat'].min()),
            "lat_max": float(df['lat'].max()),
            "lon_min": float(df['lon'].min()),
            "lon_max": float(df['lon'].max()),
            "unique_locations": len(df[['lat','lon']].drop_duplicates())
        },
        "variables": {
            "no2": {
                "min": float(df['no2'].min()),
                "max": float(df['no2'].max()),
                "mean": float(df['no2'].mean())
            },
            "o3": {
                "min": float(df['o3'].min()),
                "max": float(df['o3'].max()),
                "mean": float(df['o3'].mean())
            }
        }
    }


@app.get("/api/latest")
async def get_latest(
    variable: str = Query("no2", description="변수명 (no2, o3, uv_aerosol_index 등)")
):
    """
    최신 시간대의 전체 그리드 데이터
    - 지도 히트맵 표시용
    """
    df = load_tempo_data()

    # 사용 가능한 변수 체크
    if variable not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found. Available: {list(df.columns)}"
        )

    # 최신 시간 데이터만 필터링
    latest_time = df['time'].max()
    df_latest = df[df['time'] == latest_time].copy()

    # 결측치 제거
    df_latest = df_latest.dropna(subset=[variable])

    return {
        "time": latest_time.isoformat(),
        "variable": variable,
        "count": len(df_latest),
        "data": df_latest[['lat', 'lon', variable]].to_dict(orient='records')
    }


@app.get("/api/timeseries")
async def get_timeseries(
    lat: float = Query(..., description="위도"),
    lon: float = Query(..., description="경도"),
    variable: str = Query("no2", description="변수명"),
    radius: float = Query(0.05, description="검색 반경 (도 단위)")
):
    """
    특정 위치의 시계열 데이터
    - 차트 표시용
    """
    df = load_tempo_data()

    if variable not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Variable '{variable}' not found"
        )

    # 지정 위치 근처 데이터 필터링 (반경 내)
    df_nearby = df[
        (df['lat'] >= lat - radius) & (df['lat'] <= lat + radius) &
        (df['lon'] >= lon - radius) & (df['lon'] <= lon + radius)
    ].copy()

    if len(df_nearby) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data found near ({lat}, {lon})"
        )

    # 시간별 평균 계산
    ts = df_nearby.groupby('time')[variable].mean().reset_index()
    ts = ts.sort_values('time')

    return {
        "location": {"lat": lat, "lon": lon},
        "variable": variable,
        "count": len(ts),
        "data": [
            {
                "time": row['time'].isoformat(),
                "value": float(row[variable])
            }
            for _, row in ts.iterrows()
        ]
    }


@app.get("/api/heatmap")
async def get_heatmap(
    time: Optional[str] = Query(None, description="ISO 시간 (예: 2025-10-03T23:00:00)"),
    variable: str = Query("no2", description="변수명")
):
    """
    특정 시간의 히트맵 데이터
    - time이 없으면 최신 데이터 반환
    """
    df = load_tempo_data()

    if variable not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found")

    # 시간 필터링
    if time:
        target_time = pd.to_datetime(time)
        # 가장 가까운 시간 찾기
        time_diff = (df['time'] - target_time).abs()
        closest_time = df.loc[time_diff.idxmin(), 'time']
        df_filtered = df[df['time'] == closest_time].copy()
    else:
        # 최신 시간
        latest_time = df['time'].max()
        df_filtered = df[df['time'] == latest_time].copy()

    # 결측치 제거
    df_filtered = df_filtered.dropna(subset=[variable])

    return {
        "time": df_filtered['time'].iloc[0].isoformat(),
        "variable": variable,
        "count": len(df_filtered),
        "bounds": {
            "lat_min": float(df_filtered['lat'].min()),
            "lat_max": float(df_filtered['lat'].max()),
            "lon_min": float(df_filtered['lon'].min()),
            "lon_max": float(df_filtered['lon'].max())
        },
        "data": df_filtered[['lat', 'lon', variable]].to_dict(orient='records')
    }


@app.get("/api/grid")
async def get_grid(
    lat_min: float = Query(..., description="최소 위도"),
    lat_max: float = Query(..., description="최대 위도"),
    lon_min: float = Query(..., description="최소 경도"),
    lon_max: float = Query(..., description="최대 경도"),
    variable: str = Query("no2", description="변수명"),
    time_start: Optional[str] = Query(None, description="시작 시간"),
    time_end: Optional[str] = Query(None, description="종료 시간")
):
    """
    특정 영역의 그리드 데이터
    - 시간 범위 지정 가능
    """
    df = load_tempo_data()

    if variable not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found")

    # 공간 필터링
    df_filtered = df[
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    ].copy()

    # 시간 필터링
    if time_start:
        df_filtered = df_filtered[df_filtered['time'] >= pd.to_datetime(time_start)]
    if time_end:
        df_filtered = df_filtered[df_filtered['time'] <= pd.to_datetime(time_end)]

    # 결측치 제거
    df_filtered = df_filtered.dropna(subset=[variable])

    if len(df_filtered) == 0:
        raise HTTPException(status_code=404, detail="No data found in specified region")

    return {
        "bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max
        },
        "time_range": {
            "start": df_filtered['time'].min().isoformat(),
            "end": df_filtered['time'].max().isoformat()
        },
        "variable": variable,
        "count": len(df_filtered),
        "data": df_filtered[['time', 'lat', 'lon', variable]].to_dict(orient='records')
    }


@app.get("/api/pm25/stations")
async def get_pm25_stations():
    """
    OpenAQ PM2.5 관측소 목록
    - 지도에 마커 표시용
    """
    df = load_openaq_data()

    # 최신 데이터만 (각 관측소별 최근 측정값)
    latest = df.sort_values('time').groupby(['lat', 'lon', 'location_name']).tail(1)

    return {
        "count": len(latest),
        "stations": [
            {
                "lat": row['lat'],
                "lon": row['lon'],
                "name": row['location_name'],
                "pm25": float(row['pm25']),
                "time": row['time'].isoformat()
            }
            for _, row in latest.iterrows()
        ]
    }


@app.get("/api/pm25/latest")
async def get_pm25_latest():
    """
    OpenAQ PM2.5 최신 실측값
    - 전체 관측소의 최신 데이터
    """
    df = load_openaq_data()

    # 최신 시간
    latest_time = df['time'].max()
    df_latest = df[df['time'] == latest_time].copy()

    return {
        "time": latest_time.isoformat(),
        "count": len(df_latest),
        "data": df_latest[['lat', 'lon', 'pm25', 'location_name']].to_dict(orient='records')
    }


@app.get("/api/pm25/timeseries")
async def get_pm25_timeseries(
    location_name: Optional[str] = Query(None, description="관측소 이름")
):
    """
    특정 관측소의 PM2.5 시계열
    - location_name이 없으면 전체 평균
    """
    df = load_openaq_data()

    if location_name:
        df_filtered = df[df['location_name'] == location_name].copy()

        if len(df_filtered) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Station '{location_name}' not found"
            )

        # 시간별 평균 (같은 관측소의 중복 데이터 처리)
        ts = df_filtered.groupby('time')['pm25'].mean().reset_index()
    else:
        # 전체 관측소 평균
        ts = df.groupby('time')['pm25'].mean().reset_index()

    ts = ts.sort_values('time')

    return {
        "location": location_name or "All Stations (Average)",
        "count": len(ts),
        "data": [
            {
                "time": row['time'].isoformat(),
                "pm25": float(row['pm25'])
            }
            for _, row in ts.iterrows()
        ]
    }


@app.get("/api/combined/latest")
async def get_combined_latest():
    """
    TEMPO + OpenAQ 결합 데이터 (최신)
    - TEMPO: NO2, O3 위성 데이터 (전체 그리드)
    - OpenAQ: PM2.5 실측 데이터 (관측소만)
    """
    # TEMPO 최신 데이터
    df_tempo = load_tempo_data()
    latest_tempo_time = df_tempo['time'].max()
    df_tempo_latest = df_tempo[df_tempo['time'] == latest_tempo_time].copy()

    # OpenAQ 최신 데이터
    df_openaq = load_openaq_data()
    latest_openaq_time = df_openaq['time'].max()
    df_openaq_latest = df_openaq[df_openaq['time'] == latest_openaq_time].copy()

    return {
        "tempo": {
            "time": latest_tempo_time.isoformat(),
            "count": len(df_tempo_latest),
            "data": df_tempo_latest[['lat', 'lon', 'no2', 'o3']].to_dict(orient='records')
        },
        "openaq": {
            "time": latest_openaq_time.isoformat(),
            "count": len(df_openaq_latest),
            "data": df_openaq_latest[['lat', 'lon', 'pm25', 'location_name']].to_dict(orient='records')
        }
    }


if __name__ == "__main__":
    import uvicorn

    # 개발 서버 실행
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 코드 변경 시 자동 재시작
        log_level="info"
    )
