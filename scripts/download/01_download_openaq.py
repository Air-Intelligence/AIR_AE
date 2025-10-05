"""
Download PM2.5 data from OpenAQ API for Bay Area stations
OpenAQ는 전 세계 대기질 모니터링 데이터를 제공하는 오픈 API입니다.
이 스크립트는 Bay Area 지역의 PM2.5 관측값(라벨 데이터)을 다운로드합니다.

Updated for OpenAQ API v3 (v2 retired on 2025-01-31)
"""
import requests  # OpenAQ API 호출용
import pandas as pd  # 데이터 처리
from datetime import datetime, timedelta  # 날짜 범위 설정
import time  # API rate limiting용
from typing import List, Dict, Optional  # 타입 힌트
import config  # 전역 설정 (날짜, BBOX, 도시 목록 등)
import utils  # 공통 유틸리티 함수 (QC, 로깅 등)

logger = utils.setup_logging(__name__)

# OpenAQ API v3 설정
BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": config.OPENAQ_API_KEY}


def get_sensors_in_bbox(bbox: Dict, parameter_id: int = 2) -> List[Dict]:
    """
    OpenAQ API v3: BBOX 내의 PM2.5 센서 목록 조회

    Args:
        bbox: Bounding box dict with keys: west, south, east, north
        parameter_id: Parameter ID (2 = PM2.5)

    Returns:
        List of sensor dicts with id, name, lat, lon
    """
    logger.info(f"Fetching sensors in BBOX {bbox['name']}...")

    # v3 API: /locations 엔드포인트에서 bbox로 검색
    url = f"{BASE_URL}/locations"

    # bbox 파라미터: minX,minY,maxX,maxY (경도,위도,경도,위도)
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    params = {
        'bbox': bbox_str,
        'parameters_id': parameter_id,  # PM2.5만 필터링
        'limit': 1000  # 한 번에 최대 1000개
    }

    all_sensors = []
    page = 1

    while True:
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # v3 응답 구조: {"results": [...], "meta": {...}}
            if 'results' not in data or len(data['results']) == 0:
                break

            # 각 location에서 PM2.5 센서 추출
            for loc in data['results']:
                # location 내에 여러 센서가 있을 수 있음
                if 'sensors' not in loc:
                    continue

                for sensor in loc['sensors']:
                    # PM2.5 센서만 선택
                    if sensor.get('parameter', {}).get('id') == parameter_id:
                        all_sensors.append({
                            'sensor_id': sensor['id'],
                            'name': sensor.get('name', 'Unknown'),
                            'location_name': loc.get('name', 'Unknown'),
                            'lat': loc.get('coordinates', {}).get('latitude'),
                            'lon': loc.get('coordinates', {}).get('longitude')
                        })

            logger.info(f"  Page {page}: {len(data['results'])} locations")

            # 페이지네이션: meta.found가 limit보다 크면 다음 페이지 존재
            meta = data.get('meta', {})
            found = meta.get('found', 0)
            # found가 문자열인 경우 정수로 변환
            if isinstance(found, str):
                try:
                    found = int(found)
                except ValueError:
                    found = 0

            if found <= len(all_sensors):
                break

            # 다음 페이지
            page += 1
            params['page'] = page
            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            logger.error(f"  Error fetching sensors: {e}")
            break

    logger.info(f"✓ Found {len(all_sensors)} PM2.5 sensors in {bbox['name']}")
    return all_sensors


def get_sensor_hourly_data(
    sensor_id: int,
    sensor_name: str,
    date_from: datetime,
    date_to: datetime,
    lat: float,
    lon: float
) -> List[Dict]:
    """
    OpenAQ API v3: 특정 센서의 시간별 PM2.5 데이터 조회

    Args:
        sensor_id: Sensor ID
        sensor_name: Sensor name (for logging)
        date_from: Start date
        date_to: End date
        lat, lon: Sensor coordinates

    Returns:
        List of hourly measurement dicts
    """
    # v3 API: /sensors/{id}/hours 엔드포인트 (시간별 평균 제공)
    url = f"{BASE_URL}/sensors/{sensor_id}/hours"

    params = {
        'date_from': date_from.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        'date_to': date_to.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
        'limit': 1000
    }

    all_data = []
    page = 1

    while True:
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data or len(data['results']) == 0:
                break

            # 첫 번째 응답의 구조 로깅 (디버깅용)
            if page == 1 and len(data['results']) > 0:
                logger.debug(f"First result keys: {list(data['results'][0].keys())}")
                logger.debug(f"Meta structure: {data.get('meta', {})}")

            # 각 시간별 데이터 파싱
            for result in data['results']:
                try:
                    # v3 응답 구조: period.datetimeTo['utc']에서 시간 추출
                    period = result['period']
                    datetime_obj = period['datetimeTo']
                    if isinstance(datetime_obj, dict) and 'utc' in datetime_obj:
                        time_val = pd.to_datetime(datetime_obj['utc'])
                    else:
                        time_val = pd.to_datetime(datetime_obj)

                    value_val = result['value']

                    all_data.append({
                        'time': time_val,
                        'location': sensor_name,
                        'lat': lat,
                        'lon': lon,
                        'value': value_val,
                        'unit': 'µg/m³'
                    })

                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse record from {sensor_name}: {e}")
                    continue

            # 페이지네이션
            meta = data.get('meta', {})
            found = meta.get('found', 0)
            # found가 문자열인 경우 정수로 변환
            if isinstance(found, str):
                try:
                    found = int(found)
                except ValueError:
                    found = 0

            if found <= len(all_data):
                break

            page += 1
            params['page'] = page
            time.sleep(0.3)

        except requests.exceptions.RequestException as e:
            logger.error(f"  Error fetching data for {sensor_name}: {e}")
            break

    return all_data


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process hourly data from v3 API (already hourly averaged)

    Args:
        df: Raw hourly DataFrame from v3 API

    Returns:
        Processed DataFrame
    """
    logger.info("Processing hourly data...")

    # v3 API는 이미 시간별 평균을 제공하므로 추가 집계 불필요
    # 컬럼명만 변경: value → pm25 (이후 파이프라인과의 호환성)
    df = df.rename(columns={'value': 'pm25'})

    # 데이터 타입 검증 및 변환
    try:
        df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

        # NaN 제거
        before_clean = len(df)
        df = df.dropna(subset=['pm25', 'lat', 'lon', 'time'])
        after_clean = len(df)

        if before_clean > after_clean:
            logger.warning(f"Removed {before_clean - after_clean} records with invalid data")

    except Exception as e:
        logger.error(f"Data type conversion failed: {e}")
        return pd.DataFrame()

    # 시간 순으로 정렬
    df = df.sort_values('time').reset_index(drop=True)

    logger.info(f"✓ Processed: {len(df):,} hourly records from {df['location'].nunique()} sensors")

    return df




def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("OpenAQ PM2.5 Download (API v3)")
    logger.info("="*60)

    # ========================================================================
    # 1. Bay Area BBOX 내의 PM2.5 센서 검색 (v3 API)
    # ========================================================================
    # v3에서는 도시명 대신 BBOX로 센서를 검색
    sensors = get_sensors_in_bbox(
        bbox=config.BBOX,
        parameter_id=2  # PM2.5
    )

    if len(sensors) == 0:
        logger.error("No PM2.5 sensors found in BBOX. Exiting.")
        return

    # ========================================================================
    # 2. 각 센서에서 시간별 PM2.5 데이터 다운로드
    # ========================================================================
    all_data = []

    for i, sensor in enumerate(sensors, 1):
        logger.info(f"[{i}/{len(sensors)}] Fetching data from {sensor['location_name']}...")

        sensor_data = get_sensor_hourly_data(
            sensor_id=sensor['sensor_id'],
            sensor_name=sensor['location_name'],
            date_from=config.DATE_START,
            date_to=config.DATE_END,
            lat=sensor['lat'],
            lon=sensor['lon']
        )

        if len(sensor_data) > 0:
            all_data.extend(sensor_data)
            logger.info(f"  ✓ {len(sensor_data):,} hourly records")
        else:
            logger.warning(f"  No data for {sensor['location_name']}")

        # Rate limiting
        time.sleep(0.5)

    # DataFrame으로 변환
    df = pd.DataFrame(all_data)

    if len(df) == 0:
        logger.error("No data downloaded. Exiting.")
        return

    logger.info(f"\n✓ Total downloaded: {len(df):,} records from {len(sensors)} sensors")

    # ========================================================================
    # 3. 데이터 처리 (타입 변환, 정렬 등)
    # ========================================================================
    # v3 API는 이미 시간별 평균을 제공하므로 추가 집계 불필요
    df = process_data(df)

    if len(df) == 0:
        logger.error("No valid data after processing. Exiting.")
        return

    # ========================================================================
    # 4. 품질 관리 (QC) 적용
    # ========================================================================
    # pm25_min (0), pm25_max (500) 범위 밖의 비현실적 값 제거
    df = utils.apply_qc(df, {
        'pm25': (config.QC_THRESHOLDS['pm25_min'],
                 config.QC_THRESHOLDS['pm25_max'])
    })

    # 결측치 비율 보고
    utils.report_missing(df)

    # ========================================================================
    # 5. CSV로 저장
    # ========================================================================
    # OpenAQ 데이터는 용량이 작아서(~10 MB) CSV로도 충분
    # (위성 데이터는 용량이 커서 Parquet 사용)
    try:
        df.to_csv(config.OPENAQ_CSV, index=False)
        size_mb = config.OPENAQ_CSV.stat().st_size / 1024**2
        logger.info(f"\n✓ Saved to {config.OPENAQ_CSV} ({size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to save CSV: {e}")
        return

    # ========================================================================
    # 요약 통계 출력
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info("="*60)

    try:
        # 안전하게 통계 출력 (컬럼이 없을 수도 있음)
        if 'time' in df.columns and len(df) > 0:
            logger.info(f"  Date range:  {df['time'].min()} to {df['time'].max()}")
        if 'location' in df.columns:
            logger.info(f"  Locations:   {df['location'].nunique()}")
        logger.info(f"  Total hours: {len(df):,}")
        if 'pm25' in df.columns and len(df) > 0:
            logger.info(f"  PM2.5 mean:  {df['pm25'].mean():.1f} µg/m³")
            logger.info(f"  PM2.5 std:   {df['pm25'].std():.1f} µg/m³")
    except Exception as e:
        logger.warning(f"Could not compute all statistics: {e}")

    logger.info("="*60)


if __name__ == "__main__":
    main()
