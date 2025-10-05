from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
from contextlib import asynccontextmanager
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 글로벌 변수로 데이터 캐싱 (서버 시작 시 한 번만 로드)
df_tempo_cache: Optional[pd.DataFrame] = None
df_openaq_cache: Optional[pd.DataFrame] = None
df_o3_static_cache: Optional[pd.DataFrame] = None
df_airnow_cache: Optional[pd.DataFrame] = None
df_openaq_latest_cache: Optional[pd.DataFrame] = None

# download_realtime_data.py로 다운받은 TEMPO NO₂ 실시간 데이터 사용
TEMPO_PARQUET_PATH = Path("/mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet")
OPENAQ_PARQUET_PATH = Path("/mnt/data/features/openaq/openaq_nrt.parquet")
O3_STATIC_PARQUET_PATH = Path("/mnt/data/features/tempo/o3_static.parquet")
AIRNOW_CSV_PATH = Path("/mnt/data/raw/AirNow/current_observations.csv")
OPENAQ_LATEST_CSV_PATH = Path("/mnt/data/raw/OpenAQ/latest_observations.csv")

from pydantic import BaseModel, Field
import joblib
from pathlib import Path

MODEL_PATH = Path("/mnt/data/models/residual_lgbm.pkl")
_lgbm = None
def _get_model():
    global _lgbm
    if _lgbm is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Model not found: {MODEL_PATH}")
        _lgbm = joblib.load(MODEL_PATH)
        print(f"✓ Loaded LGBM: {MODEL_PATH}")
    return _lgbm

class PredictReq(BaseModel):
    lat: float
    lon: float
    when: Optional[str] = Field(None, description="ISO8601 UTC; omit = now")

def _snapshot(df: pd.DataFrame, when: pd.Timestamp):
    d = df[df["time"] <= when]
    if d.empty:
        t = df["time"].max()
        return df[df["time"] == t]
    t = d["time"].max()
    return df[df["time"] == t]

def _nearest_or_mean(df: pd.DataFrame, lat: float, lon: float, value_col: str, radius: float = 0.05):
    # 반경 평균 → 데이터 없으면 최근접
    nearby = df[(df["lat"].between(lat - radius, lat + radius)) &
                (df["lon"].between(lon - radius, lon + radius))]
    if len(nearby):
        return float(nearby[value_col].mean())
    idx = ((df["lat"]-lat)**2 + (df["lon"]-lon)**2).idxmin()
    return float(df.loc[idx, value_col])

@app.post("/api/predict/pm25")
def predict_pm25_lgbm(req: PredictReq):
    import pandas as pd
    when = pd.to_datetime(req.when) if req.when else pd.Timestamp.utcnow()

    # 1) 데이터 로드
    df_no2 = load_tempo_data()                 # NRT NO2
    df_o3  = load_tempo_o3_data()             # 정적 O3 (또는 최신으로 교체 가능)
    df_obs = load_openaq_latest_data()        # OpenAQ 최신 PM2.5

    # 2) 시간 스냅샷
    s_no2 = _snapshot(df_no2, when)
    s_o3  = _snapshot(df_o3,  when)
    t_obs = df_obs["time"].max()
    s_obs = df_obs[df_obs["time"] == t_obs]

    # 3) 피처 추출 (반경 평균)
    f_no2 = _nearest_or_mean(s_no2, req.lat, req.lon, "no2")
    f_o3  = _nearest_or_mean(s_o3,  req.lat, req.lon, "o3")
    f_obs = _nearest_or_mean(s_obs, req.lat, req.lon, "pm25")

    X = pd.DataFrame([{
        "lat": req.lat, "lon": req.lon,
        "hour": when.hour, "dow": when.dayofweek,
        "tempo_no2": f_no2, "tempo_o3": f_o3, "pm25_obs": f_obs
    }]).fillna(0.0)

    model = _get_model()
    try:
        y = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "when": when.isoformat(),
        "pred_pm25": y,
        "features": {**X.iloc[0].to_dict(), "obs_time": str(t_obs)},
        "model": MODEL_PATH.name
    }


TEMPO_O3_PATH = Path("/mnt/data/features/tempo/o3_static.parquet")
df_tempo_o3_cache = None

def _find_existing_file(candidates):
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(f"File not found in candidates: {candidates}")

def _latest_at_or_before(df: pd.DataFrame, when: pd.Timestamp, time_col: str = "time") -> pd.DataFrame:
    d = df[df[time_col] <= when]
    if d.empty:
        # 없으면 전체에서 최대 시각으로 대체
        t = df[time_col].max()
        return df[df[time_col] == t]
    t = d[time_col].max()
    return df[df[time_col] == t]

def _nearest_row(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    d = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    return df.loc[d.idxmin()]


def load_tempo_o3_data(force: bool = False) -> pd.DataFrame:
    """TEMPO O3 데이터 로드 및 캐싱"""
    global df_tempo_o3_cache
    if df_tempo_o3_cache is None or force:
        if not TEMPO_O3_PATH.exists():
            raise FileNotFoundError(f"TEMPO O3 파일이 없습니다: {TEMPO_O3_PATH}")
        df_tempo_o3_cache = pd.read_parquet(TEMPO_O3_PATH)
        df_tempo_o3_cache["time"] = pd.to_datetime(df_tempo_o3_cache["time"], errors="coerce")
        print(f"✓ Loaded TEMPO O3: {len(df_tempo_o3_cache):,} rows from {TEMPO_O3_PATH}")
    return df_tempo_o3_cache

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


def load_o3_static_data() -> pd.DataFrame:
    """TEMPO O3 Static Parquet 로드 및 캐싱 (Standard V04 과거 데이터)"""
    global df_o3_static_cache

    if df_o3_static_cache is None:
        if not O3_STATIC_PARQUET_PATH.exists():
            raise FileNotFoundError(f"O3 Static Parquet not found: {O3_STATIC_PARQUET_PATH}")

        df_o3_static_cache = pd.read_parquet(O3_STATIC_PARQUET_PATH)
        df_o3_static_cache['time'] = pd.to_datetime(df_o3_static_cache['time'])
        print(f"✓ Loaded O3 Static: {len(df_o3_static_cache):,} records from {O3_STATIC_PARQUET_PATH}")
        print(f"  ⚠️  O3는 정적 데이터 (실시간 아님): {df_o3_static_cache['time'].min()} ~ {df_o3_static_cache['time'].max()}")

    return df_o3_static_cache


def load_airnow_data() -> pd.DataFrame:
    """AirNow 현재 관측값 CSV 로드 및 캐싱"""
    global df_airnow_cache

    if df_airnow_cache is None:
        if not AIRNOW_CSV_PATH.exists():
            raise FileNotFoundError(f"AirNow CSV not found: {AIRNOW_CSV_PATH}")

        df_airnow_cache = pd.read_csv(AIRNOW_CSV_PATH)
        df_airnow_cache['time'] = pd.to_datetime(df_airnow_cache['time'])
        print(f"✓ Loaded AirNow: {len(df_airnow_cache):,} records from {AIRNOW_CSV_PATH}")
        print(f"  Latest observation: {df_airnow_cache['time'].max()}")

    return df_airnow_cache


def load_openaq_latest_data() -> pd.DataFrame:
    """OpenAQ 최신 관측값 CSV 로드 및 캐싱"""
    global df_openaq_latest_cache

    if df_openaq_latest_cache is None:
        if not OPENAQ_LATEST_CSV_PATH.exists():
            raise FileNotFoundError(f"OpenAQ Latest CSV not found: {OPENAQ_LATEST_CSV_PATH}")

        df_openaq_latest_cache = pd.read_csv(OPENAQ_LATEST_CSV_PATH)
        df_openaq_latest_cache['time'] = pd.to_datetime(df_openaq_latest_cache['time'])
        print(f"✓ Loaded OpenAQ Latest: {len(df_openaq_latest_cache):,} records from {OPENAQ_LATEST_CSV_PATH}")
        print(f"  Latest observation: {df_openaq_latest_cache['time'].max()}")
        print(f"  Stations: {df_openaq_latest_cache['location_name'].nunique()}")

    return df_openaq_latest_cache


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """서버 시작/종료 시 실행되는 lifespan 이벤트"""
    # Startup
    try:
        load_tempo_data()
    except FileNotFoundError as e:
        print(f"⚠️  TEMPO data not found: {e}")

    try:
        load_openaq_data()
    except FileNotFoundError as e:
        print(f"⚠️  OpenAQ data not found: {e}")

    try:
        load_o3_static_data()
    except FileNotFoundError as e:
        print(f"⚠️  O3 Static data not found: {e}")
        print(f"   Run: python scripts/download/download_o3_static.py")

    try:
        load_airnow_data()
    except FileNotFoundError as e:
        print(f"⚠️  AirNow data not found: {e}")
        print(f"   Run: python scripts/download/download_airnow.py")

    try:
        load_openaq_latest_data()
    except FileNotFoundError as e:
        print(f"⚠️  OpenAQ Latest data not found: {e}")
        print(f"   Run: python scripts/download/download_openaq_latest.py")

    print("✓ TEMPO NRT API server started")

    yield

    # Shutdown (필요시 정리 작업 추가)


# FastAPI 앱 생성
app = FastAPI(
    title="TEMPO NRT API",
    description="NASA TEMPO 실시간 대기질 데이터 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "TEMPO + OpenAQ + AirNow NRT API",
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
            "AirNow (PM2.5 실측)": [
                "/api/airnow/latest",
                "/api/airnow/stations"
            ],
            "결합 데이터": [
                "/api/combined/latest"
            ],
            "예측 및 비교": [
                "/api/predict",
                "/api/compare"
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
            "unique_locations": len(df[['lat', 'lon']].drop_duplicates())
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
    - NO2, O3는 1e15로 스케일링 (단위: ×10¹⁵ molecules/cm²)
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

    # NO2, O3 스케일링 (1e15로 나눔)
    if variable in ['no2', 'o3']:
        df_latest[variable] = df_latest[variable] / 1e15

    return {
        "time": latest_time.isoformat(),
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ['no2', 'o3'] else None,
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

    # NO2, O3 스케일링 (1e15로 나눔)
    if variable in ['no2', 'o3']:
        ts[variable] = ts[variable] / 1e15

    return {
        "location": {"lat": lat, "lon": lon},
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ['no2', 'o3'] else None,
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

    # NO2, O3 스케일링 (1e15로 나눔)
    if variable in ['no2', 'o3']:
        df_filtered[variable] = df_filtered[variable] / 1e15

    return {
        "time": df_filtered['time'].iloc[0].isoformat(),
        "variable": variable,
        "unit": "×10¹⁵ molecules/cm²" if variable in ['no2', 'o3'] else None,
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

    # NO2, O3 스케일링 (1e15로 나눔)
    if variable in ['no2', 'o3']:
        df_filtered[variable] = df_filtered[variable] / 1e15

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
        "unit": "×10¹⁵ molecules/cm²" if variable in ['no2', 'o3'] else None,
        "count": len(df_filtered),
        "data": df_filtered[['time', 'lat', 'lon', variable]].to_dict(orient='records')
    }

@app.get("/api/pm25/latest_csv")
async def get_pm25_latest_csv(force: bool = Query(False, description="캐시 무시하고 파일 재로드")):
    """
    OpenAQ 최신 관측값 CSV 기반 엔드포인트
    - /mnt/data/raw/OpenAQ/latest_observations.csv 사용
    - force=true 이면 캐시 무시하고 디스크에서 재로딩
    """
    global df_openaq_latest_cache

    if force:
        df_openaq_latest_cache = None  # 캐시 무효화

    df = load_openaq_latest_data()

    latest_time = df['time'].max()
    df_latest = df[df['time'] == latest_time].copy()

    return {
        "time": latest_time.isoformat(),
        "count": len(df_latest),
        "data": df_latest[['lat', 'lon', 'pm25', 'location_name', 'location_id']].to_dict(orient='records')
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

    # NO2, O3 스케일링 (1e15로 나눔)
    df_tempo_latest['no2'] = df_tempo_latest['no2'] / 1e15
    df_tempo_latest['o3'] = df_tempo_latest['o3'] / 1e15

    # OpenAQ 최신 데이터
    df_openaq = load_openaq_data()
    latest_openaq_time = df_openaq['time'].max()
    df_openaq_latest = df_openaq[df_openaq['time'] == latest_openaq_time].copy()

    return {
        "tempo": {
            "time": latest_tempo_time.isoformat(),
            "count": len(df_tempo_latest),
            "unit": "×10¹⁵ molecules/cm²",
            "data": df_tempo_latest[['lat', 'lon', 'no2', 'o3']].to_dict(orient='records')
        },
        "openaq": {
            "time": latest_openaq_time.isoformat(),
            "count": len(df_openaq_latest),
            "unit": "µg/m³",
            "data": df_openaq_latest[['lat', 'lon', 'pm25', 'location_name']].to_dict(orient='records')
        }
    }


@app.post("/api/predict")
async def predict_pm25(
    lat: float = Query(..., description="위도"),
    lon: float = Query(..., description="경도"),
    city: str = Query("San Francisco", description="도시명")
):
    """
    PM2.5 예측 (1시간 후)

    입력: lat, lon, city
    출력: PM2.5 예측값 + 신뢰구간
    """
    from src.features import extract_near
    from src.model import get_predictor
    from datetime import datetime

    try:
        # 1. TEMPO NO₂ 로드 (캐시 또는 실시간 데이터)
        df_tempo = load_tempo_data()

        # 최신 2개 시간 추출 (t, t-1)
        tempo_times = sorted(df_tempo['time'].unique())
        if len(tempo_times) < 2:
            raise HTTPException(
                status_code=503,
                detail="Insufficient TEMPO data (need at least 2 time points)"
            )

        time_t = tempo_times[-1]
        time_t1 = tempo_times[-2]

        # 2. (lat, lon) 근처 NO₂ 추출
        no2_t = extract_near(df_tempo, lat, lon, time=time_t, value_col='no2')
        no2_lag1 = extract_near(df_tempo, lat, lon, time=time_t1, value_col='no2')

        if no2_t is None or no2_lag1 is None:
            raise HTTPException(
                status_code=404,
                detail=f"No TEMPO NO₂ data near ({lat}, {lon})"
            )

        # 3. O₃ 정적 데이터 로드 (TEMPO O3 Standard V04)
        df_o3 = load_o3_static_data()

        o3_times = sorted(df_o3['time'].unique())
        if len(o3_times) < 2:
            raise HTTPException(
                status_code=503,
                detail="Insufficient O₃ static data"
            )

        o3_time_t = o3_times[-1]
        o3_time_t1 = o3_times[-2]

        # 4. (lat, lon) 근처 O₃ 추출
        o3_t = extract_near(df_o3, lat, lon, time=o3_time_t, value_col='o3')
        o3_lag1 = extract_near(df_o3, lat, lon, time=o3_time_t1, value_col='o3')

        if o3_t is None or o3_lag1 is None:
            raise HTTPException(
                status_code=404,
                detail=f"No O₃ data near ({lat}, {lon})"
            )

        # 5. 시간 피처
        hour = time_t.hour
        dow = time_t.dayofweek

        # 6. 예측
        predictor = get_predictor(model_dir="/mnt/data/models")
        result = predictor.predict(no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)

        # 7. 응답 생성
        return {
            "predicted_pm25": result['pm25_pred'],
            "confidence_lower": result['confidence_lower'],
            "confidence_upper": result['confidence_upper'],
            "prediction_time": time_t.isoformat(),
            "location": {
                "lat": lat,
                "lon": lon,
                "city": city
            },
            "inputs": {
                "no2_current": no2_t,
                "no2_lag1": no2_lag1,
                "o3_current": o3_t,
                "o3_lag1": o3_lag1,
                "hour": hour,
                "dow": dow
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# @app.get("/api/airnow/latest")
# async def get_airnow_latest():
#     """
#     AirNow 최신 PM2.5 관측값
#     - Bay Area 5개 도시의 실시간 PM2.5 데이터
#     """
#     df = load_airnow_data()

#     return {
#         "time": df['time'].max().isoformat(),
#         "count": len(df),
#         "data": [
#             {
#                 "city": row['city'],
#                 "lat": row['lat'],
#                 "lon": row['lon'],
#                 "pm25": float(row['pm25_raw']),
#                 "aqi": int(row['pm25']),
#                 "category": row['category'],
#                 "site_name": row['site_name'],
#                 "time": row['time'].isoformat()
#             }
#             for _, row in df.iterrows()
#         ]
#     }


# @app.get("/api/airnow/stations")
# async def get_airnow_stations():
#     """
#     AirNow 관측소 정보
#     - Bay Area 5개 도시의 관측소 위치 및 최신 데이터
#     """
#     df = load_airnow_data()

#     return {
#         "count": len(df),
#         "stations": [
#             {
#                 "city": row['city'],
#                 "lat": row['lat'],
#                 "lon": row['lon'],
#                 "site_name": row['site_name'],
#                 "pm25": float(row['pm25_raw']),
#                 "aqi": int(row['pm25']),
#                 "category": row['category'],
#                 "time": row['time'].isoformat()
#             }
#             for _, row in df.iterrows()
#         ]
#     }


@app.get("/api/compare")
async def compare_predictions():
    """
    모델 예측값 vs OpenAQ 실측값 비교
    - Bay Area 모든 관측소에 대해 예측 수행 후 OpenAQ 데이터와 비교
    - MAE, RMSE 등 성능 메트릭 제공
    """
    from src.features import extract_near
    from src.model import get_predictor
    import numpy as np

    try:
        # 1. OpenAQ 최신 실측값 로드 (AirNow 대신)
        try:
            df_ground_truth = load_openaq_latest_data()
        except FileNotFoundError:
            # Fallback to AirNow if available
            df_ground_truth = load_airnow_data()

        # 2. TEMPO NO₂, O₃ 데이터 로드
        df_tempo = load_tempo_data()
        df_o3 = load_o3_static_data()

        # 최신 시간 추출
        tempo_times = sorted(df_tempo['time'].unique())
        o3_times = sorted(df_o3['time'].unique())

        if len(tempo_times) < 2 or len(o3_times) < 2:
            raise HTTPException(
                status_code=503,
                detail="Insufficient satellite data for prediction"
            )

        time_t = tempo_times[-1]
        time_t1 = tempo_times[-2]
        o3_time_t = o3_times[-1]
        o3_time_t1 = o3_times[-2]

        # 3. 각 관측소별 예측 수행
        predictor = get_predictor(model_dir="/mnt/data/models")
        results = []

        for _, station_row in df_ground_truth.iterrows():
            lat = station_row['lat']
            lon = station_row['lon']
            location_name = station_row.get('location_name', station_row.get('city', 'Unknown'))
            pm25_true = station_row.get('pm25', station_row.get('pm25_raw', None))

            if pm25_true is None:
                continue

            # NO₂ 추출
            no2_t = extract_near(df_tempo, lat, lon, time=time_t, value_col='no2')
            no2_lag1 = extract_near(df_tempo, lat, lon, time=time_t1, value_col='no2')

            # O₃ 추출
            o3_t = extract_near(df_o3, lat, lon, time=o3_time_t, value_col='o3')
            o3_lag1 = extract_near(df_o3, lat, lon, time=o3_time_t1, value_col='o3')

            if no2_t is None or no2_lag1 is None or o3_t is None or o3_lag1 is None:
                continue

            # 시간 피처
            hour = time_t.hour
            dow = time_t.dayofweek

            # 예측
            pred_result = predictor.predict(no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)

            results.append({
                "location_name": location_name,
                "lat": lat,
                "lon": lon,
                "pm25_predicted": pred_result['pm25_pred'],
                "pm25_observed": float(pm25_true),
                "error": pred_result['pm25_pred'] - float(pm25_true),
                "abs_error": abs(pred_result['pm25_pred'] - float(pm25_true))
            })

        if len(results) == 0:
            raise HTTPException(
                status_code=404,
                detail="No valid predictions could be made"
            )

        # 4. 성능 메트릭 계산
        errors = [r['error'] for r in results]
        abs_errors = [r['abs_error'] for r in results]

        metrics = {
            "mae": float(np.mean(abs_errors)),
            "rmse": float(np.sqrt(np.mean([e**2 for e in errors]))),
            "mbe": float(np.mean(errors)),
            "n_stations": len(results)
        }

        return {
            "comparison": results,
            "metrics": metrics,
            "prediction_time": time_t.isoformat(),
            "observation_time": df_ground_truth['time'].max().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    # 개발 서버 실행
    uvicorn.run(
        "open_aq:app",
        host="0.0.0.0",
        port=8000,  # 포트: 8000
        reload=True,  # 코드 변경 시 자동 재시작
        log_level="info"
    )
