
"""
TEMPO NO₂ + OpenAQ O₃ 실시간 데이터 수집

목적:
- PM2.5 예측을 위한 실시간 input 데이터 수집
- TEMPO NO₂ (V04 NRT, 위성 데이터)
- OpenAQ O₃ (지상 관측소 데이터)

출력:
- /mnt/data/features/realtime/tempo_no2.parquet
- /mnt/data/features/realtime/openaq_o3.parquet

주기:
- 수동 실행 (스케줄러 없음)
- 최근 72시간 데이터 수집
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List
import requests
import pandas as pd

import earthaccess
import config
import utils

logger = utils.setup_logging(__name__)

# ============================================================================
# TEMPO NO₂ 실시간 수집 (02_download_tempo_nrt.py 로직 재사용)
# ============================================================================

def auth_via_netrc() -> None:
    """
    Earthdata 인증 (~/.netrc 사용)

    ~/.netrc 파일에 NASA Earthdata 계정 정보 필요:
        machine urs.earthdata.nasa.gov
            login YOUR_USERNAME
            password YOUR_PASSWORD
    """
    earthaccess.login(strategy="netrc")
    logger.info("✓ Earthdata authenticated via ~/.netrc")


def bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """
    BBOX 딕셔너리 → 튜플 변환

    earthaccess.search_data()는 (west, south, east, north) 튜플 형식 요구
    """
    return (bbox["west"], bbox["south"], bbox["east"], bbox["north"])


def resolve_collection(key: str, default_short_name: str) -> str:
    """
    config.TEMPO_COLLECTIONS에서 collection short_name 가져오기
    키가 없으면 기본값 사용 (fallback)

    Args:
        key: config에서 찾을 키 (예: "NO2_NRT")
        default_short_name: 키가 없을 때 사용할 기본값 (예: "TEMPO_NO2_L3")

    Returns:
        Collection short name 문자열
    """
    try:
        # config에서 문자열 직접 반환 (V04 NRT는 문자열로 저장됨)
        return config.TEMPO_COLLECTIONS.get(key, default_short_name)
    except Exception:
        return default_short_name


def search_granules(short_name: str, start: datetime, end: datetime, bbox: dict) -> List:
    """
    TEMPO NRT granule 검색

    Args:
        short_name: TEMPO collection 이름 (예: "TEMPO_NO2_L3")
        start: 검색 시작 시간 (UTC)
        end: 검색 종료 시간 (UTC)
        bbox: 검색 영역 (Bay Area)

    Returns:
        검색된 granule 리스트 (위성 데이터 파일 메타데이터)
    """
    bt = bbox_tuple(bbox)

    # NASA CMR (Common Metadata Repository)에서 granule 검색
    results = earthaccess.search_data(
        short_name=short_name,
        version="V04",  # V04 = NRT (Near Real-Time, 2025-09-17 이후)
        temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        bounding_box=bt,
        cloud_hosted=True,  # AWS S3에서 직접 다운로드
    )

    logger.info(
        f"{short_name}: found {len(results)} granules "
        f"({start.date()} ~ {end.date()})"
    )
    return results


def download_granules(granules: List, outdir: Path, threads: int = 12) -> list[str]:
    """
    Granule 병렬 다운로드

    Args:
        granules: 검색된 granule 리스트
        outdir: 다운로드 출력 디렉터리
        threads: 병렬 다운로드 스레드 수 (기본 12개)

    Returns:
        성공적으로 다운로드된 파일 경로 리스트

    참고:
        - earthaccess가 자동으로 중복 다운로드 방지
        - 파일이 이미 존재하면 건너뜀
    """
    if not granules:
        logger.warning("No granules to download")
        return []

    # 출력 디렉터리 생성
    outdir.mkdir(parents=True, exist_ok=True)

    # earthaccess로 병렬 다운로드 실행
    files = earthaccess.download(granules, str(outdir), threads=threads) or []

    # 성공한 파일만 필터링 (크기가 0보다 큰 파일)
    ok = []
    for f in files:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:
            ok.append(str(p))

    if ok:
        sample = ", ".join(Path(f).name for f in ok[:3])
        logger.info(f"✓ Downloaded {len(ok)} files → {outdir} (sample: {sample}...)")
    else:
        logger.warning("No files downloaded")

    return ok


# ============================================================================
# OpenAQ O₃ 실시간 수집 (02_download_openaq_nrt.py 로직 재사용)
# ============================================================================

BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": config.OPENAQ_API_KEY}


def fetch_openaq_locations(bbox: dict, parameter_id: int = 5) -> List[dict]:
    """
    OpenAQ 관측소 목록 조회

    Args:
        bbox: 검색 영역 (Bay Area)
        parameter_id: 측정 변수 (5 = O₃, 2 = PM2.5)

    Returns:
        관측소(location) 리스트 (각 관측소는 여러 센서 포함)
    """
    bbox_str = f"{bbox['west']},{bbox['south']},{bbox['east']},{bbox['north']}"

    url = f"{BASE_URL}/locations"
    params = {
        'bbox': bbox_str,
        'parameters_id': parameter_id,  # 5 = O₃
        'limit': 1000
    }

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    locations = data.get('results', [])
    logger.info(f"✓ Found {len(locations)} O₃ stations in {bbox.get('name', 'region')}")

    return locations


def extract_sensors(locations: List[dict], parameter_id: int = 5) -> List[dict]:
    """
    관측소에서 O₃ 센서만 추출

    Args:
        locations: 관측소 리스트
        parameter_id: 5 = O₃

    Returns:
        센서 정보 리스트 (sensor_id, location_name, lat, lon)

    참고:
        하나의 관측소(location)에 여러 센서가 있을 수 있음
        우리는 O₃ 센서만 필요
    """
    sensors = []

    for loc in locations:
        for sensor in loc.get('sensors', []):
            if sensor.get('parameter', {}).get('id') == parameter_id:
                sensors.append({
                    'sensor_id': sensor['id'],
                    'location_name': loc.get('name', 'Unknown'),
                    'lat': loc.get('coordinates', {}).get('latitude'),
                    'lon': loc.get('coordinates', {}).get('longitude')
                })

    logger.info(f"✓ Extracted {len(sensors)} O₃ sensors")
    return sensors


def fetch_measurements(sensor_id: int, start: datetime, end: datetime) -> List[dict]:
    """
    단일 센서의 O₃ 측정값 조회

    Args:
        sensor_id: 센서 ID
        start: 시작 시간
        end: 종료 시간

    Returns:
        측정값 리스트 [{time, o3, sensor_id}, ...]

    참고:
        - OpenAQ API v3의 응답 구조는 복잡함
        - 시간 필드가 'datetime' 또는 'period' 안에 있을 수 있음
    """
    url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
    params = {
        'date_from': start.isoformat(),
        'date_to': end.isoformat(),
        'limit': 1000
    }

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        measurements = []
        for m in data.get('results', []):
            # 시간 필드 추출 (API 응답 구조가 일정하지 않음)
            time_val = None
            if 'datetime' in m:
                if isinstance(m['datetime'], dict):
                    time_val = m['datetime'].get('utc')
                else:
                    time_val = m['datetime']
            elif 'period' in m:
                time_val = m.get('period', {}).get('datetimeFrom', {}).get('utc')

            if time_val:
                measurements.append({
                    'time': time_val,
                    'o3': m.get('value'),
                    'sensor_id': sensor_id
                })

        return measurements

    except Exception as e:
        logger.warning(f"Failed to fetch measurements for sensor {sensor_id}: {e}")
        return []


def collect_openaq_o3(bbox: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """
    OpenAQ O₃ 데이터 수집 → DataFrame 반환

    전체 프로세스:
        1. bbox 내 관측소 검색
        2. 각 관측소에서 O₃ 센서 추출
        3. 각 센서의 측정값 수집
        4. DataFrame으로 변환 및 시간 정렬

    Args:
        bbox: 검색 영역
        start: 시작 시간
        end: 종료 시간

    Returns:
        DataFrame [time, lat, lon, o3, location_name]
    """
    # 1. 관측소 조회
    locations = fetch_openaq_locations(bbox, parameter_id=5)

    # 2. 센서 추출
    sensors = extract_sensors(locations, parameter_id=5)

    if not sensors:
        logger.error("No O₃ sensors found")
        return pd.DataFrame()

    # 3. 측정값 수집 (최대 50개 센서로 제한)
    all_measurements = []

    for sensor in sensors[:50]:  # API rate limit 고려
        measurements = fetch_measurements(sensor['sensor_id'], start, end)

        # 센서 위치 정보와 측정값 결합
        for m in measurements:
            all_measurements.append({
                'time': m['time'],
                'lat': sensor['lat'],
                'lon': sensor['lon'],
                'o3': m['o3'],
                'location_name': sensor['location_name']
            })

    # 4. DataFrame 변환
    if not all_measurements:
        logger.error("No measurements collected")
        return pd.DataFrame()

    df = pd.DataFrame(all_measurements)
    df['time'] = pd.to_datetime(df['time'])

    # UTC 1시간 간격으로 정렬 (TEMPO와 시간 맞추기 위함)
    df['time'] = df['time'].dt.floor('1H')
    # 같은 시간/위치의 중복 데이터는 평균 처리
    df = df.groupby(['time', 'lat', 'lon', 'location_name'])['o3'].mean().reset_index()

    logger.info(f"✓ Collected {len(df)} O₃ records")

    return df


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    메인 함수: TEMPO NO₂ + OpenAQ O₃ 실시간 수집

    실행 순서:
        1. UTC 현재 시각 기준 최근 72시간 범위 계산
        2. TEMPO NO₂ NRT 데이터 다운로드 (NASA Earthdata)
        3. OpenAQ O₃ 데이터 수집 (OpenAQ API v3)
        4. 결과를 Parquet 파일로 저장

    출력 파일:
        - /mnt/data/features/realtime/tempo_no2_raw/*.nc (NetCDF)
        - /mnt/data/features/realtime/openaq_o3.parquet

    사용법:
        python download_realtime_data.py

    참고:
        - 이 스크립트는 수동 실행 (스케줄러 없음)
        - 실행 전 ~/.netrc에 NASA Earthdata 계정 설정 필요
        - config.OPENAQ_API_KEY 필요
    """
    logger.info("=" * 60)
    logger.info("실시간 데이터 수집: TEMPO NO₂ + OpenAQ O₃")
    logger.info("=" * 60)

    # ========================================================================
    # 시간 범위 설정: UTC 현재 기준 최근 72시간
    # ========================================================================
    utc_now = datetime.now(timezone.utc)
    start = (utc_now - timedelta(hours=72)).replace(tzinfo=None)  # naive datetime
    end = utc_now.replace(tzinfo=None)

    logger.info(f"Period (UTC): {start} ~ {end} (72 hours)")
    logger.info(f"Region: {config.BBOX.get('name', 'N/A')}")

    # ========================================================================
    # 1. TEMPO NO₂ 수집 (위성 데이터)
    # ========================================================================
    logger.info("\n[1/2] TEMPO NO₂ NRT 수집...")

    try:
        # NASA Earthdata 인증
        auth_via_netrc()

        # config에서 collection 이름 가져오기 (resolve_collection 사용)
        no2_short = resolve_collection("NO2_NRT", "TEMPO_NO2_L3")

        # granule 검색 (CMR API)
        no2_granules = search_granules(no2_short, start, end, config.BBOX)

        # 병렬 다운로드
        out_tempo = Path("/mnt/data/features/realtime/tempo_no2_raw")
        no2_files = download_granules(no2_granules, out_tempo, threads=12)

        logger.info(f"✓ TEMPO NO₂: {len(no2_files)} files → {out_tempo}")

    except Exception as e:
        logger.error(f"TEMPO NO₂ 수집 실패: {e}")
        sys.exit(1)

    # ========================================================================
    # 2. TEMPO O₃ NRT 수집 (위성 데이터)
    # ========================================================================
    logger.info("\n[2/2] TEMPO O₃ NRT 수집...")

    try:
        # config에서 collection 이름 가져오기
        o3_short = resolve_collection("O3_NRT", "TEMPO_O3TOT_L3")

        # granule 검색 (CMR API)
        o3_granules = search_granules(o3_short, start, end, config.BBOX)

        # 병렬 다운로드
        out_o3 = config.RAW_TEMPO_O3_NRT  # /mnt/data/raw/tempo_v04/o3
        o3_files = download_granules(o3_granules, out_o3, threads=12)

        logger.info(f"✓ TEMPO O₃: {len(o3_files)} files → {out_o3}")

    except Exception as e:
        logger.error(f"TEMPO O₃ 수집 실패: {e}")
        # O3 실패해도 NO2가 있으면 계속 진행
        logger.warning("O₃ 없이 계속 진행...")
        o3_files = []

    # ========================================================================
    # 완료 메시지
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("✓ 실시간 데이터 수집 완료")
    logger.info("=" * 60)
    logger.info(f"  TEMPO NO₂: {len(no2_files)} files")
    logger.info(f"  TEMPO O₃: {len(o3_files)} files")
    logger.info("=" * 60)
    logger.info("\n다음 단계:")
    logger.info("  1. TEMPO NetCDF 파일을 parquet으로 변환 (04_preprocess_nrt.py)")
    logger.info("  2. 모델 학습 (train_pm25_model.py)")
    logger.info("  3. API 서버 실행 (python open_aq.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
