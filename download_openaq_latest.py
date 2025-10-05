"""
OpenAQ API v3로 Bay Area 최신 PM2.5 관측값 다운로드

목적:
- AirNow 대체용 실시간 PM2.5 ground truth 데이터
- Bay Area 전체 관측소의 최신 PM2.5 측정값 다운로드
- FastAPI에서 모델 예측값과 비교

출력:
- /mnt/data/raw/OpenAQ/latest_observations.csv

사용법:
    python scripts/download/download_openaq_latest.py
"""

import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict
import time

# 프로젝트 루트를 sys.path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils

logger = utils.setup_logging(__name__)

# OpenAQ API v3 설정
BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": config.OPENAQ_API_KEY}


def get_latest_measurements_bbox(bbox: Dict, parameter_id: int = 2) -> List[Dict]:
    """
    OpenAQ API v3: BBOX 내 모든 센서의 최신 PM2.5 측정값 조회

    2단계 접근:
    1. /v3/locations로 PM2.5 센서가 있는 location 목록 조회
    2. 각 location의 /latest 엔드포인트로 최신 측정값 조회

    Args:
        bbox: Bounding box dict with keys: west, south, east, north
        parameter_id: Parameter ID (2 = PM2.5)

    Returns:
        List of measurement dicts
    """
    logger.info(f"Fetching latest PM2.5 measurements in {bbox['name']}...")

    # Step 1: Get locations with PM2.5 sensors
    locations_url = f"{BASE_URL}/locations"
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    params = {
        'bbox': bbox_str,
        'parameter': 'pm25',
        'limit': 100  # 처음 100개 location만
    }

    try:
        response = requests.get(locations_url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or len(data['results']) == 0:
            logger.error("No locations found in bbox")
            return []

        locations = data['results']
        logger.info(f"  Found {len(locations)} locations with PM2.5 sensors")

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch locations: {e}")
        return []

    # Step 2: Get latest measurement for each location
    all_measurements = []

    for i, loc in enumerate(locations, 1):
        location_id = loc.get('id')
        location_name = loc.get('name', 'Unknown')
        coordinates = loc.get('coordinates', {})
        lat = coordinates.get('latitude')
        lon = coordinates.get('longitude')

        if not location_id or lat is None or lon is None:
            continue

        # Call /v3/locations/{id}/latest
        latest_url = f"{BASE_URL}/locations/{location_id}/latest"

        try:
            response = requests.get(latest_url, headers=HEADERS, params={'parameter': 'pm25'}, timeout=10)
            response.raise_for_status()
            latest_data = response.json()

            if 'results' not in latest_data or len(latest_data['results']) == 0:
                continue

            # Get first PM2.5 measurement
            measurement = latest_data['results'][0]
            value = measurement.get('value')

            # datetime은 dict 형태일 수 있음
            datetime_obj = measurement.get('datetime')
            if isinstance(datetime_obj, dict):
                datetime_str = datetime_obj.get('utc') or datetime_obj.get('local')
            else:
                datetime_str = datetime_obj

            # 디버깅: 첫 번째 측정값 구조 확인
            if i == 1:
                import json
                logger.info(f"  Sample measurement: {json.dumps(measurement, indent=2, default=str)[:500]}")

            if value is not None:  # datetime_str 체크 제거 (없어도 저장)
                all_measurements.append({
                    'location_name': location_name,
                    'lat': lat,
                    'lon': lon,
                    'pm25': value,
                    'time': datetime_str or datetime.now(timezone.utc).isoformat(),
                    'location_id': location_id,
                    'unit': measurement.get('unit', 'µg/m³')
                })

            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(locations)} locations processed")

            time.sleep(1.0)  # Rate limiting (increased to avoid 429)

        except Exception as e:
            logger.warning(f"  Failed to get latest for {location_name}: {e}")
            continue

    logger.info(f"✓ Total measurements collected: {len(all_measurements)}")
    return all_measurements


def main():
    """
    메인 실행 함수: OpenAQ v3로 최신 PM2.5 관측값 다운로드

    실행 순서:
        1. Bay Area BBOX 내 모든 PM2.5 센서 검색
        2. 각 센서의 최신 측정값 수집
        3. CSV 파일로 저장

    출력:
        - /mnt/data/raw/OpenAQ/latest_observations.csv
    """
    logger.info("=" * 60)
    logger.info("OpenAQ v3 최신 PM2.5 관측값 다운로드")
    logger.info("=" * 60)

    # ========================================================================
    # 1. 최신 측정값 다운로드
    # ========================================================================
    measurements = get_latest_measurements_bbox(config.BBOX, parameter_id=2)

    if len(measurements) == 0:
        logger.error("다운로드된 측정값이 없습니다")
        logger.info("💡 BBOX를 확인하거나 더 넓은 영역으로 시도하세요")
        sys.exit(1)

    # ========================================================================
    # 2. DataFrame 변환
    # ========================================================================
    df = pd.DataFrame(measurements)

    # 시간 컬럼 확인 및 변환
    # time이 dict인 경우 (API 응답 구조 변경 대응)
    if len(df) > 0 and isinstance(df['time'].iloc[0], dict):
        logger.warning("Time column contains dict, extracting datetime string...")
        df['time'] = df['time'].apply(lambda x: x.get('datetime') if isinstance(x, dict) else x)

    df['time'] = pd.to_datetime(df['time'])
    df['time'] = utils.to_utc_naive(df['time'])

    # 중복 제거 (같은 관측소의 여러 센서)
    df = df.sort_values('time', ascending=False).groupby(['lat', 'lon', 'location_name']).first().reset_index()

    logger.info(f"✓ Unique stations after deduplication: {len(df)}")

    # ========================================================================
    # 3. CSV 저장
    # ========================================================================
    output_dir = Path("/mnt/data/raw/OpenAQ")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "latest_observations.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"\n✓ 저장 완료: {output_path}")

    # ========================================================================
    # 4. 요약
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("다운로드 요약:")
    logger.info("=" * 60)
    logger.info(f"  다운로드 시간:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"  관측소 수:      {len(df)}")
    logger.info(f"  PM2.5 평균:     {df['pm25'].mean():.1f} µg/m³")
    logger.info(f"  PM2.5 범위:     [{df['pm25'].min():.1f}, {df['pm25'].max():.1f}] µg/m³")
    logger.info(f"  최신 시간:      {df['time'].max()}")
    logger.info(f"  출력 파일:      {output_path}")
    logger.info("=" * 60)

    # 관측소 목록 출력
    logger.info("\n관측소 목록:")
    for _, row in df.iterrows():
        logger.info(f"  - {row['location_name']}: {row['pm25']:.1f} µg/m³ ({row['time']})")

    logger.info("\n다음 단계:")
    logger.info("  - FastAPI에서 이 파일을 로드하여 실시간 비교 제공")
    logger.info("  - 주기적으로 이 스크립트를 실행하여 최신 데이터 유지")


if __name__ == "__main__":
    main()
