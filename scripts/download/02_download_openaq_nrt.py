
"""
OpenAQ 실시간 PM2.5 데이터 다운로드 (최근 3일)

목적:
- FastAPI 백엔드용 PM2.5 실측 데이터 제공
- Bay Area 지상 관측소에서 최근 3일 PM2.5 측정값 수집
- TEMPO 위성 데이터와 결합하여 완전한 대기질 정보 제공

데이터 흐름:
1. OpenAQ API v3에서 Bay Area 센서 목록 조회
2. 각 센서별 최근 3일 PM2.5 측정값 다운로드
3. 센서 메타데이터(위치, 이름) 병합
4. QC 적용 후 Parquet 저장
5. FastAPI가 이 Parquet 읽어서 실시간 제공

출력: /mnt/data/features/openaq/openaq_nrt.parquet
"""
import requests  # OpenAQ API 호출용
import pandas as pd  # 데이터 처리
from datetime import datetime, timedelta, timezone  # 시간 범위 계산
from pathlib import Path  # 파일 경로 처리
import time  # API rate limiting용
from typing import List, Dict  # 타입 힌트
import config  # 전역 설정 (BBOX, API Key 등)
import utils  # 공통 유틸리티 (QC, 로깅)

logger = utils.setup_logging(__name__)

# ============================================================================
# OpenAQ API v3 설정
# ============================================================================
BASE_URL = "https://api.openaq.org/v3"  # API 기본 URL (v3부터 HTTPS 필수)
HEADERS = {"X-API-Key": config.OPENAQ_API_KEY}  # 인증 헤더 (config.py에서 읽음)


def get_sensors_in_bbox(bbox: Dict, parameter_id: int = 2) -> List[Dict]:
    """
    OpenAQ API v3: BBOX 내의 PM2.5 센서 목록 조회

    Bay Area 영역 내에 있는 모든 PM2.5 센서(지상 관측소)를 찾습니다.
    센서 정보에는 ID, 이름, 위치(위도/경도)가 포함됩니다.

    Args:
        bbox: Bounding box dict with keys: west, south, east, north
        parameter_id: Parameter ID (2 = PM2.5, OpenAQ 표준)

    Returns:
        센서 정보 리스트 [{'sensor_id': 123, 'name': '...', 'lat': 37.5, 'lon': -122.0}]
    """
    logger.info(f"Fetching sensors in BBOX {bbox['name']}...")

    # API 엔드포인트: /locations (관측소 위치 검색)
    url = f"{BASE_URL}/locations"

    # BBOX 문자열 생성 (west,south,east,north 순서)
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    # API 쿼리 파라미터
    params = {
        'bbox': bbox_str,  # 검색 영역
        'parameters_id': parameter_id,  # PM2.5만 필터링
        'limit': 1000  # 한 페이지당 최대 1000개
    }

    all_sensors = []
    page = 1

    # 페이지네이션 루프 (결과가 1000개 넘으면 여러 페이지)
    while True:
        try:
            # API 호출 (인증 헤더 포함)
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()  # 4xx, 5xx 에러 시 예외 발생
            data = response.json()

            # 결과가 없으면 종료
            if 'results' not in data or len(data['results']) == 0:
                break

            # 각 location(관측소)에서 센서 추출
            for loc in data['results']:
                if 'sensors' not in loc:
                    continue

                # 한 관측소에 여러 센서가 있을 수 있음 (PM2.5, O3, NO2 등)
                for sensor in loc['sensors']:
                    # PM2.5 센서만 선택
                    if sensor.get('parameter', {}).get('id') == parameter_id:
                        all_sensors.append({
                            'sensor_id': sensor['id'],  # 센서 고유 ID
                            'name': sensor.get('name', 'Unknown'),
                            'location_name': loc.get('name', 'Unknown'),  # 관측소 이름
                            'lat': loc.get('coordinates', {}).get('latitude'),
                            'lon': loc.get('coordinates', {}).get('longitude')
                        })

            logger.info(f"  Page {page}: {len(data['results'])} locations")

            # 페이지네이션: 다음 페이지가 있는지 확인
            meta = data.get('meta', {})
            if meta.get('found', 0) <= page * params['limit']:
                break  # 모든 결과를 가져왔으면 종료

            # 다음 페이지 요청
            params['page'] = page + 1
            page += 1
            time.sleep(0.5)  # API rate limiting (초당 2회 제한 회피)

        except Exception as e:
            logger.error(f"Error fetching sensors (page {page}): {e}")
            break

    logger.info(f"✓ Found {len(all_sensors)} PM2.5 sensors")
    return all_sensors


def fetch_latest_from_parameters(bbox: Dict, parameter_id: int = 2) -> pd.DataFrame:
    """
    /parameters/{id}/latest 엔드포인트로 최신 측정값 조회 (우선순위 1)

    Args:
        bbox: Bounding box dict
        parameter_id: Parameter ID (2 = PM2.5)

    Returns:
        최신 측정값 DataFrame (time, sensor_id, pm25, lat, lon)
    """
    logger.info(f"Fetching latest values from /parameters/{parameter_id}/latest...")

    url = f"{BASE_URL}/parameters/{parameter_id}/latest"
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    params = {
        'bbox': bbox_str,
        'limit': 1000
    }

    all_measurements = []
    page = 1

    while True:
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data or len(data['results']) == 0:
                break

            for m in data['results']:
                try:
                    # Extract time, value, sensor_id, coordinates
                    time_val = m.get('period', {}).get('datetimeFrom', {}).get('utc')
                    if not time_val:
                        time_val = m.get('period', {}).get('datetimeTo', {}).get('utc')

                    pm25_val = m.get('value')
                    sensor_id = m.get('sensor', {}).get('id') or m.get('sensorId')

                    coords = m.get('coordinates', {})
                    lat = coords.get('latitude')
                    lon = coords.get('longitude')

                    if time_val and pm25_val is not None and sensor_id:
                        all_measurements.append({
                            'time': time_val,
                            'sensor_id': sensor_id,
                            'pm25': pm25_val,
                            'lat': lat,
                            'lon': lon
                        })
                except Exception as e:
                    logger.debug(f"Failed to parse result: {e}")
                    continue

            # Pagination - 결과가 limit보다 적으면 마지막 페이지
            if len(data['results']) < params['limit']:
                break

            page += 1
            params['page'] = page
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Error fetching latest (page {page}): {e}")
            break

    if len(all_measurements) == 0:
        logger.warning("No latest measurements found")
        return pd.DataFrame()

    df = pd.DataFrame(all_measurements)
    df['time'] = pd.to_datetime(df['time'])

    # Sort by time descending and keep only recent data (last 72 hours)
    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=72)
    df = df[df['time'] >= cutoff_time].copy()
    df = df.sort_values('time', ascending=False).reset_index(drop=True)

    logger.info(f"✓ Retrieved {len(df):,} latest measurements from {df['sensor_id'].nunique()} sensors")
    return df


def fetch_hourly_fallback(sensor_ids: List[int], sensors: List[Dict], limit: int = 1) -> pd.DataFrame:
    """
    /sensors/{id}/measurements/hourly 폴백 (우선순위 2)

    latest 엔드포인트가 비어있거나 오래된 데이터일 때 사용
    각 센서의 최신 1개 hourly 측정값만 가져옴

    Args:
        sensor_ids: 센서 ID 리스트
        sensors: 센서 메타데이터 리스트 (lat, lon 포함)
        limit: 센서당 가져올 최신 측정값 개수

    Returns:
        hourly 측정값 DataFrame (lat, lon 포함)
    """
    logger.info(f"Fallback: Fetching hourly data for {len(sensor_ids)} sensors...")

    # 센서 ID -> 위치 정보 매핑
    sensor_map = {s['sensor_id']: s for s in sensors}

    all_measurements = []
    failed_sensors = []

    for idx, sensor_id in enumerate(sensor_ids, 1):
        url = f"{BASE_URL}/sensors/{sensor_id}/measurements/hourly"
        params = {'limit': limit}

        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'results' in data:
                for m in data['results']:
                    try:
                        time_val = m.get('period', {}).get('datetimeFrom', {}).get('utc')
                        pm25_val = m.get('value')

                        if time_val and pm25_val is not None:
                            # 센서 메타데이터에서 lat/lon 가져오기
                            sensor_info = sensor_map.get(sensor_id, {})
                            all_measurements.append({
                                'time': time_val,
                                'sensor_id': sensor_id,
                                'pm25': pm25_val,
                                'lat': sensor_info.get('lat'),
                                'lon': sensor_info.get('lon')
                            })
                    except:
                        continue

            if idx % 10 == 0 or idx == len(sensor_ids):
                logger.info(f"  Progress: {idx}/{len(sensor_ids)} sensors")

            time.sleep(0.6)

        except Exception as e:
            logger.debug(f"Sensor {sensor_id} failed: {e}")
            failed_sensors.append(sensor_id)
            continue

    if len(all_measurements) == 0:
        logger.warning("No hourly measurements retrieved")
        return pd.DataFrame()

    df = pd.DataFrame(all_measurements)
    df['time'] = pd.to_datetime(df['time'])

    # 디버그: lat/lon이 없는 행 확인
    missing_coords = df[df['lat'].isna() | df['lon'].isna()]
    if len(missing_coords) > 0:
        logger.warning(f"⚠ {len(missing_coords)} measurements missing lat/lon (sensor_ids not in metadata)")
        logger.debug(f"Missing sensor_ids: {missing_coords['sensor_id'].unique().tolist()[:10]}")

    logger.info(f"✓ Fallback retrieved {len(df):,} measurements from {len(df['sensor_id'].unique())} sensors")
    logger.info(f"  - With coordinates: {df['lat'].notna().sum():,} measurements")
    return df


def fetch_measurements_realtime(
    bbox: Dict,
    sensor_ids: List[int],
    sensors: List[Dict],
    hours_back: int = 72
) -> pd.DataFrame:
    """
    실시간 PM2.5 측정값 조회 (latest 우선, hourly 폴백)

    전략:
    1. /parameters/{id}/latest?bbox=... 로 최신값 조회 (빠름)
    2. 결과가 비거나 72시간보다 오래됐으면 hourly로 폴백

    Args:
        bbox: Bounding box
        sensor_ids: 센서 ID 리스트
        sensors: 센서 메타데이터 리스트 (lat, lon 포함)
        hours_back: 최근 몇 시간 데이터 (72시간 = 3일)

    Returns:
        PM2.5 측정값 DataFrame (lat, lon 포함)
    """
    logger.info(f"Fetching NRT measurements (last {hours_back}h)...")

    # 1. Latest 엔드포인트 시도
    df_latest = fetch_latest_from_parameters(bbox, parameter_id=2)

    # Latest 성공 시 반환
    if len(df_latest) > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        recent_count = (df_latest['time'] >= cutoff).sum()

        if recent_count > 0:
            logger.info(f"✓ Using latest endpoint ({recent_count} recent measurements)")
            return df_latest

    # 2. Hourly 폴백
    logger.warning("Latest endpoint insufficient, using hourly fallback...")
    # hours_back에 맞춰서 센서당 측정값 개수 설정
    limit_per_sensor = min(hours_back, 720)  # 최대 30일 (720시간)
    df_hourly = fetch_hourly_fallback(sensor_ids, sensors, limit=limit_per_sensor)

    if len(df_hourly) == 0:
        logger.error("Both latest and hourly endpoints failed")
        return pd.DataFrame()

    # 최근 hours_back 시간 내 데이터만 필터
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    df_filtered = df_hourly[df_hourly['time'] >= cutoff].copy()

    # 디버그: 필터링 전후 비교
    logger.info(f"  Time filter: {len(df_hourly):,} → {len(df_filtered):,} measurements (cutoff: {cutoff})")

    return df_filtered


def merge_sensor_metadata(
    measurements_df: pd.DataFrame,
    sensors: List[Dict]
) -> pd.DataFrame:
    """
    측정값에 센서 메타데이터 (위치, 이름) 병합

    측정값 DataFrame은 sensor_id만 있고 위치 정보가 없습니다.
    센서 목록에서 각 ID의 위도/경도/이름을 찾아서 병합합니다.

    Args:
        measurements_df: 측정값 DataFrame (time, sensor_id, pm25, lat?, lon?)
        sensors: 센서 메타데이터 리스트 (sensor_id, lat, lon, location_name)

    Returns:
        병합된 DataFrame (time, lat, lon, pm25, location_name)
        FastAPI가 바로 사용할 수 있는 형태
    """
    # 센서 메타데이터를 DataFrame으로 변환
    sensors_df = pd.DataFrame(sensors)

    # sensor_id 기준으로 LEFT JOIN (측정값 유지, 센서 정보 추가)
    df = measurements_df.merge(
        sensors_df[['sensor_id', 'lat', 'lon', 'location_name']],
        on='sensor_id',
        how='left',
        suffixes=('', '_meta')
    )

    # 디버그: merge 전후 개수 확인
    logger.info(f"  Merge: {len(measurements_df)} measurements → {len(df)} rows")
    missing_location = df['location_name'].isna().sum()
    if missing_location > 0:
        logger.warning(f"  ⚠ {missing_location} measurements missing location_name (sensor_id not in metadata)")

    # lat/lon이 측정값에 이미 있으면 우선 사용, 없으면 메타데이터 사용
    if 'lat_meta' in df.columns:
        df['lat'] = df['lat'].fillna(df['lat_meta'])
        df['lon'] = df['lon'].fillna(df['lon_meta'])
        df = df.drop(columns=['lat_meta', 'lon_meta'])

    # 필요한 컬럼만 선택 (sensor_id는 내부 처리용이므로 제거)
    df = df[['time', 'lat', 'lon', 'pm25', 'location_name']]

    # 위치 정보, PM2.5 값, location_name이 없는 행 제거
    before_drop = len(df)
    df = df.dropna(subset=['lat', 'lon', 'pm25', 'location_name'])
    after_drop = len(df)
    if before_drop > after_drop:
        logger.warning(f"  ⚠ Dropped {before_drop - after_drop} rows due to missing data")

    return df


def main():
    """메인 실행 함수"""
    logger.info("="*60)
    logger.info("OpenAQ 실시간 PM2.5 다운로드 (NRT)")
    logger.info("="*60)

    # ========================================================================
    # 1. Bay Area 센서 목록 조회
    # ========================================================================
    sensors = get_sensors_in_bbox(config.BBOX, parameter_id=2)

    if len(sensors) == 0:
        logger.error("No PM2.5 sensors found in BBOX")
        return

    # ========================================================================
    # 2. 최근 3일 측정값 조회
    # ========================================================================
    sensor_ids = [s['sensor_id'] for s in sensors]  # 센서 ID만 추출
    hours_back = getattr(config, 'NRT_RECENT_DAYS', 3) * 24  # 3일 = 72시간

    df_measurements = fetch_measurements_realtime(config.BBOX, sensor_ids, sensors, hours_back=hours_back)

    if len(df_measurements) == 0:
        logger.error("No measurements retrieved")
        return

    # ========================================================================
    # 3. 센서 메타데이터 병합
    # ========================================================================
    # 측정값(time, sensor_id, pm25) + 센서 정보(lat, lon, name)
    # → (time, lat, lon, pm25, location_name)
    df = merge_sensor_metadata(df_measurements, sensors)

    # ========================================================================
    # 4. QC (품질 관리)
    # ========================================================================
    logger.info("\nQC 적용 중...")

    # PM2.5 유효 범위 체크 (config.py에서 읽음)
    # 예: 0 ~ 500 µg/m³ 범위 밖의 값은 이상치로 제거
    qc_thresholds = {
        'pm25': (config.QC_THRESHOLDS['pm25_min'], config.QC_THRESHOLDS['pm25_max'])
    }
    df = utils.apply_qc(df, thresholds=qc_thresholds)

    # ========================================================================
    # 5. Parquet 저장
    # ========================================================================
    output_path = config.FEATURES_OPENAQ / "openaq_nrt.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 디렉터리 생성 (없으면)

    logger.info(f"\nParquet 저장 중: {output_path}")
    utils.save_parquet(df, output_path, downcast=True)  # float64 → float32 변환하여 용량 절감

    # ========================================================================
    # 6. 결과 요약
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("OpenAQ NRT 다운로드 완료")
    logger.info("="*60)
    logger.info(f"  시간 범위:      {df['time'].min()} ~ {df['time'].max()}")
    logger.info(f"  총 측정값:      {len(df):,} 개")
    logger.info(f"  센서 수:        {df['location_name'].nunique()} 개")
    logger.info(f"  PM2.5 범위:     {df['pm25'].min():.1f} ~ {df['pm25'].max():.1f} µg/m³")
    logger.info(f"  평균 PM2.5:     {df['pm25'].mean():.1f} µg/m³")
    logger.info(f"  출력 파일:      {output_path}")
    logger.info(f"  파일 크기:      {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("="*60)

    logger.info("\n✓ OpenAQ NRT 데이터 준비 완료! FastAPI에서 사용 가능합니다.")


if __name__ == "__main__":
    main()
