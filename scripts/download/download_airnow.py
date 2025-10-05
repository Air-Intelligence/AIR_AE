"""
AirNow API를 통한 실시간 PM2.5 관측값 다운로드

목적:
- Bay Area 5개 도시의 현재 PM2.5 실측값을 AirNow API에서 다운로드
- FastAPI에서 모델 예측값과 비교할 ground truth 데이터 제공

API 문서:
- https://docs.airnowapi.org/webservices

출력:
- /mnt/data/raw/AirNow/current_observations.csv

사용법:
    python scripts/download/download_airnow.py
"""

import sys
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional

# 프로젝트 루트를 sys.path에 추가
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils

logger = utils.setup_logging(__name__)

# AirNow API 설정
BASE_URL = "https://www.airnowapi.org/aq/observation"


def get_current_observation_by_latlon(
    lat: float,
    lon: float,
    distance: int = 25,
    api_key: str = None,
    max_retries: int = 3
) -> Optional[Dict]:
    """
    AirNow API: 위도/경도로 현재 대기질 관측값 조회

    Args:
        lat: 위도
        lon: 경도
        distance: 검색 반경 (마일, 기본값 25)
        api_key: AirNow API 키
        max_retries: 최대 재시도 횟수

    Returns:
        PM2.5 관측 데이터 딕셔너리 또는 None
    """
    if api_key is None:
        api_key = config.AIRNOW_API_KEY

    # API 키 확인
    if api_key == "YOUR_AIRNOW_API_KEY_HERE":
        logger.error("AirNow API 키가 설정되지 않았습니다. config.py에서 AIRNOW_API_KEY를 설정하세요.")
        return None

    # 엔드포인트: /latLong/current
    url = f"{BASE_URL}/latLong/current/"

    params = {
        "format": "application/json",
        "latitude": lat,
        "longitude": lon,
        "distance": distance,
        "API_KEY": api_key,
    }

    for attempt in range(max_retries):
        try:
            # 타임아웃 60초로 증가
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # 빈 응답 체크
            if not data or len(data) == 0:
                logger.warning(f"API returned empty data (lat={lat}, lon={lon})")
                return None

            # PM2.5 데이터 필터링
            pm25_obs = None
            for obs in data:
                if obs.get("ParameterName") == "PM2.5":
                    pm25_obs = {
                        "lat": obs.get("Latitude"),
                        "lon": obs.get("Longitude"),
                        "pm25": obs.get("AQI"),  # AQI 값
                        "pm25_raw": obs.get("Value"),  # 원시 농도 (µg/m³)
                        "category": obs.get("Category", {}).get("Name"),
                        "site_name": obs.get("ReportingArea"),
                        "datetime_observed": obs.get("DateObserved") + " " + obs.get("HourObserved") + ":00",
                    }
                    break

            return pm25_obs

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}, retrying...")
                continue
            else:
                logger.error(f"API 요청 실패 (lat={lat}, lon={lon}): Timeout after {max_retries} attempts")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패 (lat={lat}, lon={lon}): {e}")
            return None
        except Exception as e:
            logger.error(f"데이터 파싱 실패 (lat={lat}, lon={lon}): {e}")
            return None

    return None


def download_current_observations(cities: Dict[str, Dict]) -> pd.DataFrame:
    """
    여러 도시의 현재 PM2.5 관측값 다운로드

    Args:
        cities: 도시명과 좌표 딕셔너리
                예: {"San Francisco": {"lat": 37.77, "lon": -122.42}}

    Returns:
        PM2.5 관측값 DataFrame
    """
    logger.info(f"AirNow API를 통해 {len(cities)}개 도시의 현재 관측값 다운로드 중...")

    all_obs = []

    for city_name, coords in cities.items():
        logger.info(f"  {city_name} (lat={coords['lat']}, lon={coords['lon']})...")

        obs = get_current_observation_by_latlon(
            lat=coords["lat"],
            lon=coords["lon"],
            distance=config.AIRNOW_DISTANCE,
        )

        if obs:
            obs["city"] = city_name
            all_obs.append(obs)
            logger.info(f"    ✓ PM2.5 = {obs['pm25_raw']:.1f} µg/m³ (AQI {obs['pm25']}) at {obs['site_name']}")
        else:
            logger.warning(f"    ✗ 데이터 없음")

    if len(all_obs) == 0:
        logger.error("다운로드된 관측값이 없습니다")
        return pd.DataFrame()

    # DataFrame 생성
    df = pd.DataFrame(all_obs)

    # 시간 컬럼을 datetime으로 변환
    df["time"] = pd.to_datetime(df["datetime_observed"], format="%Y-%m-%d %H:%M")
    df = df.drop(columns=["datetime_observed"])

    logger.info(f"\n✓ 총 {len(df)}개 도시의 관측값 다운로드 완료")

    return df


def main():
    """
    메인 실행 함수: AirNow API로 현재 PM2.5 관측값 다운로드

    실행 순서:
        1. Bay Area 5개 도시의 현재 PM2.5 데이터 다운로드
        2. CSV 파일로 저장

    출력:
        - /mnt/data/raw/AirNow/current_observations.csv
    """
    logger.info("=" * 60)
    logger.info("AirNow 현재 관측값 다운로드")
    logger.info("=" * 60)

    # API 키 확인
    if config.AIRNOW_API_KEY == "YOUR_AIRNOW_API_KEY_HERE":
        logger.error("AirNow API 키가 설정되지 않았습니다.")
        logger.info("💡 https://docs.airnowapi.org 에서 API 키를 발급받으세요.")
        logger.info("💡 config.py에서 AIRNOW_API_KEY를 설정하세요.")
        sys.exit(1)

    # ========================================================================
    # 1. 현재 관측값 다운로드
    # ========================================================================
    df = download_current_observations(config.AIRNOW_CITIES)

    if df.empty:
        logger.error("다운로드 실패")
        sys.exit(1)

    # ========================================================================
    # 2. CSV 저장
    # ========================================================================
    output_dir = Path("/mnt/data/raw/AirNow")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "current_observations.csv"
    df.to_csv(output_path, index=False)

    logger.info(f"\n✓ 저장 완료: {output_path}")

    # ========================================================================
    # 3. 요약
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("다운로드 요약:")
    logger.info("=" * 60)
    logger.info(f"  다운로드 시간:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"  도시 수:        {len(df)}")
    logger.info(f"  PM2.5 평균:     {df['pm25_raw'].mean():.1f} µg/m³")
    logger.info(f"  PM2.5 범위:     [{df['pm25_raw'].min():.1f}, {df['pm25_raw'].max():.1f}] µg/m³")
    logger.info(f"  출력 파일:      {output_path}")
    logger.info("=" * 60)

    logger.info("\n다음 단계:")
    logger.info("  - FastAPI에서 이 파일을 로드하여 실시간 비교 제공")
    logger.info("  - 주기적으로 이 스크립트를 실행하여 최신 데이터 유지")


if __name__ == "__main__":
    main()
