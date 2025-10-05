"""
피처 엔지니어링 모듈

기능:
1. KD-Tree 공간 매칭 (TEMPO 위성 데이터 ↔ OpenAQ 지상 관측소)
2. 시간 정렬 (UTC 1시간 간격)
3. 6개 피처 생성 (no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)
4. 실시간 예측용 extract_near() 함수
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def join_tempo_openaq(
    df_tempo: pd.DataFrame,
    df_openaq: pd.DataFrame,
    max_distance_km: float = 10.0
) -> pd.DataFrame:
    """
    KD-Tree 기반 공간 매칭 + 시간 정렬

    Args:
        df_tempo: TEMPO 데이터 [time, lat, lon, no2/o3]
        df_openaq: OpenAQ 데이터 [time, lat, lon, o3/pm25, location_name]
        max_distance_km: 최대 매칭 거리 (기본 10km)

    Returns:
        결합된 DataFrame [time_utc, lat, lon, tempo_var, openaq_var]
    """

    # 1. 시간을 UTC 1시간 단위로 정렬
    df_tempo = df_tempo.copy()
    df_openaq = df_openaq.copy()

    df_tempo['time'] = pd.to_datetime(df_tempo['time']).dt.floor('1H')
    df_openaq['time'] = pd.to_datetime(df_openaq['time']).dt.floor('1H')

    # 2. 공통 시간대 찾기
    common_times = sorted(set(df_tempo['time'].unique()) & set(df_openaq['time'].unique()))

    if not common_times:
        logger.error("No common time periods found")
        return pd.DataFrame()

    logger.info(f"Found {len(common_times)} common time periods")

    # 3. 시간대별로 KD-Tree 매칭
    results = []

    for t in common_times:
        df_t = df_tempo[df_tempo['time'] == t].copy()
        df_o = df_openaq[df_openaq['time'] == t].copy()

        if len(df_t) == 0 or len(df_o) == 0:
            continue

        # TEMPO 그리드 포인트 (위도/경도 → 라디안)
        tempo_coords = np.radians(df_t[['lat', 'lon']].values)

        # OpenAQ 관측소 위치
        openaq_coords = np.radians(df_o[['lat', 'lon']].values)

        # KD-Tree 구축 (TEMPO 그리드)
        tree = cKDTree(tempo_coords)

        # 각 OpenAQ 관측소에 대해 가장 가까운 TEMPO 그리드 찾기
        distances, indices = tree.query(openaq_coords, k=1)

        # 거리를 km로 변환 (지구 반지름 6371km)
        distances_km = distances * 6371

        # max_distance_km 이내만 매칭
        mask = distances_km <= max_distance_km

        # 매칭된 데이터 결합
        for i, (dist_km, idx) in enumerate(zip(distances_km, indices)):
            if mask[i]:
                # TEMPO 데이터에서 변수명 자동 감지
                tempo_var_name = None
                for col in df_t.columns:
                    if col in ['no2', 'o3']:
                        tempo_var_name = col
                        break

                # OpenAQ 데이터에서 변수명 자동 감지
                openaq_var_name = None
                for col in df_o.columns:
                    if col in ['o3', 'pm25']:
                        openaq_var_name = col
                        break

                if tempo_var_name and openaq_var_name:
                    results.append({
                        'time': t,
                        'lat': df_o.iloc[i]['lat'],
                        'lon': df_o.iloc[i]['lon'],
                        f'tempo_{tempo_var_name}': df_t.iloc[idx][tempo_var_name],
                        f'openaq_{openaq_var_name}': df_o.iloc[i][openaq_var_name],
                        'distance_km': dist_km
                    })

    if not results:
        logger.error("No matches found within max_distance_km")
        return pd.DataFrame()

    df_joined = pd.DataFrame(results)
    logger.info(f"✓ Joined {len(df_joined)} records (avg distance: {df_joined['distance_km'].mean():.2f} km)")

    return df_joined


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    6개 피처 생성

    Input columns:
        - time: datetime
        - no2 (또는 tempo_no2)
        - o3 (또는 tempo_o3, openaq_o3)

    Output columns:
        - no2_t: 현재 NO₂
        - no2_lag1: 1시간 전 NO₂
        - o3_t: 현재 O₃
        - o3_lag1: 1시간 전 O₃
        - hour: 시각 (0-23)
        - dow: 요일 (0-6)
    """

    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)

    # 변수명 자동 감지
    no2_col = 'no2' if 'no2' in df.columns else 'tempo_no2'
    o3_col = 'o3' if 'o3' in df.columns else (
        'tempo_o3' if 'tempo_o3' in df.columns else 'openaq_o3'
    )

    # 1. 현재값
    df['no2_t'] = df[no2_col]
    df['o3_t'] = df[o3_col]

    # 2. Lag-1 (1시간 전)
    df['no2_lag1'] = df.groupby(['lat', 'lon'])[no2_col].shift(1)
    df['o3_lag1'] = df.groupby(['lat', 'lon'])[o3_col].shift(1)

    # 3. 시간 피처
    df['hour'] = df['time'].dt.hour
    df['dow'] = df['time'].dt.dayofweek

    # 4. NaN 제거 (lag-1이 없는 첫 시간 제거)
    df = df.dropna(subset=['no2_lag1', 'o3_lag1'])

    # 5. 최종 피처만 선택
    feature_cols = ['time', 'lat', 'lon', 'no2_t', 'no2_lag1', 'o3_t', 'o3_lag1', 'hour', 'dow']

    # 존재하는 컬럼만 선택
    final_cols = [c for c in feature_cols if c in df.columns]
    df_features = df[final_cols].copy()

    logger.info(f"✓ Created {len(df_features)} feature records (6 features per record)")
    logger.info(f"  NaN ratio: {df_features.isnull().sum().sum() / df_features.size * 100:.2f}%")

    return df_features


def extract_near(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    time: Optional[pd.Timestamp] = None,
    radius_km: float = 10.0,
    value_col: str = None
) -> Optional[float]:
    """
    실시간 예측용: (lat, lon) 근처의 값 추출

    Args:
        df: 데이터 (TEMPO 또는 OpenAQ)
        lat, lon: 목표 위치
        time: 시간 (None이면 최신 시간)
        radius_km: 검색 반경 (km)
        value_col: 추출할 값 컬럼명 (None이면 자동 감지)

    Returns:
        추출된 값 (거리 역제곱 가중 평균) 또는 None
    """

    df = df.copy()

    # 시간 필터링
    if time is not None:
        df = df[df['time'] == time]
    else:
        # 최신 시간
        latest_time = df['time'].max()
        df = df[df['time'] == latest_time]

    if len(df) == 0:
        logger.warning(f"No data found for time={time}")
        return None

    # 값 컬럼 자동 감지
    if value_col is None:
        for col in ['no2', 'o3', 'pm25']:
            if col in df.columns:
                value_col = col
                break

    if value_col not in df.columns:
        logger.error(f"Value column '{value_col}' not found")
        return None

    # 거리 계산 (Haversine)
    df['distance_km'] = haversine_distance(
        lat, lon,
        df['lat'].values, df['lon'].values
    )

    # radius_km 이내만
    df_near = df[df['distance_km'] <= radius_km].copy()

    if len(df_near) == 0:
        logger.warning(f"No data within {radius_km}km of ({lat}, {lon})")
        return None

    # 거리 역제곱 가중 평균
    df_near['weight'] = 1 / (df_near['distance_km'] ** 2 + 1e-6)  # 0 방지
    weighted_avg = (df_near[value_col] * df_near['weight']).sum() / df_near['weight'].sum()

    return float(weighted_avg)


def haversine_distance(
    lat1: float, lon1: float,
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Haversine 거리 계산 (km)

    Args:
        lat1, lon1: 기준점
        lat2, lon2: 대상점들 (numpy array)

    Returns:
        거리 배열 (km)
    """
    R = 6371  # 지구 반지름 (km)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c
