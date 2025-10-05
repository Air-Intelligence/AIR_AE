"""
TEMPO V04 (NRT - Near Real Time) 실시간 데이터 다운로드
웹 실시간 표시용 최근 며칠치 NO2 & O3 데이터만 다운로드

주요 특징:
- earthaccess + .netrc 인증 사용
- 최근 N일치만 다운로드 (config.NRT_RECENT_DAYS, 기본 3일)
- V03 학습 데이터와 분리된 경로에 저장 (tempo_v04/...)
- Cron으로 주기적 실행 시 자동으로 새 데이터만 추가 다운로드
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import earthaccess
import config
import utils

logger = utils.setup_logging(__name__)


def auth_via_netrc() -> None:
    """
    ~/.netrc 파일을 통한 Earthdata 인증

    .netrc 파일 형식:
        machine urs.earthdata.nasa.gov
            login YOUR_USERNAME
            password YOUR_PASSWORD
    """
    earthaccess.login(strategy="netrc")
    logger.info("✓ Earthdata authenticated via ~/.netrc")


def ensure_dir(p: Path) -> None:
    """디렉터리가 없으면 생성 (부모 디렉터리까지 포함)"""
    p.mkdir(parents=True, exist_ok=True)


def bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """
    BBOX 딕셔너리를 튜플로 변환 (west, south, east, north)
    earthaccess.search_data()는 튜플 형식을 요구함
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


def search_granules(
    short_name: str, start: datetime, end: datetime, bbox: dict
) -> List:
    """
    TEMPO NRT (V04) granule 검색

    Args:
        short_name: Collection 이름 (예: "TEMPO_NO2_L3_V04")
        start: 검색 시작 날짜 (UTC)
        end: 검색 종료 날짜 (UTC)
        bbox: 검색 영역 (west, south, east, north)

    Returns:
        검색된 granule 리스트 (earthaccess granule 객체)
    """
    bt = bbox_tuple(bbox)

    # NASA Earthdata CMR에서 granule 검색
    # granule = 위성 데이터의 최소 배포 단위 (보통 시간별 파일)
    results = earthaccess.search_data(
        short_name=short_name,
        version="V04",  # V04는 2025-09-17 이후 데이터
        temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        bounding_box=bt,
        cloud_hosted=True,  # 수정됨: V04 클라우드 호스팅 컬렉션
    )

    logger.info(
        f"{short_name}: found {len(results)} granules "
        f"({start.date()} ~ {end.date()}, region={bbox.get('name','N/A')})"
    )
    return results


def download_granules(granules: List, outdir: Path, threads: int) -> list[str]:
    """
    Granule 병렬 다운로드

    earthaccess가 자동으로 처리하는 기능:
    - 이미 다운로드된 파일은 건너뛰기 (중복 방지)
    - 다운로드 중단 시 재개(resume)
    - 네트워크 오류 시 자동 재시도(retry)

    Args:
        granules: 다운로드할 granule 리스트
        outdir: 출력 디렉터리
        threads: 병렬 다운로드 스레드 수

    Returns:
        성공적으로 다운로드된 파일 경로 리스트
    """
    if not granules:
        logger.warning("No granules to download")
        return []

    # 출력 디렉터리 생성
    ensure_dir(outdir)

    # 병렬 다운로드 실행
    files = earthaccess.download(granules, str(outdir), threads=threads) or []

    # 정상적으로 다운로드된 파일만 필터링
    ok = []
    for f in files:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:  # 파일 존재 & 크기가 0보다 큼
            ok.append(str(p))

    # 다운로드 결과 로깅
    if ok:
        sample = ", ".join(Path(f).name for f in ok[:3])
        logger.info(f"✓ Downloaded {len(ok)} files → {outdir} (sample: {sample} ...)")
        total_gb = sum(Path(f).stat().st_size for f in ok) / 1024**3
        logger.info(f"  Total size: {total_gb:.2f} GB")
    else:
        logger.warning("No files downloaded (check auth/collection/time range)")

    return ok


def main() -> None:
    logger.info("=" * 60)
    logger.info("TEMPO V04 (NRT) recent download for LIVE UI")
    logger.info("=" * 60)

    # ========================================================================
    # 1. Earthdata 인증
    # ========================================================================
    auth_via_netrc()

    # ========================================================================
    # 2. 다운로드 기간 설정 (UTC 기준 최근 N일)
    # ========================================================================
    # config에서 NRT_RECENT_DAYS 읽기 (없으면 기본값 3일)
    n_days = getattr(config, "NRT_RECENT_DAYS", 3)

    # UTC 현재 시각 기준으로 시작/종료 날짜 계산
    utc_now = datetime.now(timezone.utc)
    start = (utc_now - timedelta(days=n_days)).replace(tzinfo=None)  # naive UTC
    end = utc_now.replace(tzinfo=None)  # naive UTC

    logger.info(f"Period (UTC): {start.date()} ~ {end.date()} (last {n_days} days)")
    logger.info(f"Region: {config.BBOX.get('name', 'N/A')}")

    # ========================================================================
    # 3. Collection 및 출력 경로 설정
    # ========================================================================
    # Collection short_name (config에 없으면 기본값 사용)
    no2_short = resolve_collection("NO2_NRT", "TEMPO_NO2_L3")
    o3_short = resolve_collection("O3_NRT", "TEMPO_O3TOT_L3")

    # 출력 디렉터리 (config에 없으면 기본값 사용)
    out_no2 = getattr(config, "RAW_TEMPO_NO2_NRT", Path("tables/tempo_v04/no2"))
    out_o3 = getattr(config, "RAW_TEMPO_O3_NRT", Path("tables/tempo_v04/o3"))

    # 병렬 다운로드 스레드 수 (클라우드 다운로드 최적화)
    threads = 12

    # ========================================================================
    # 4. Granule 검색 및 다운로드
    # ========================================================================
    try:
        # NO2, O3 granule 검색
        no2_rs = search_granules(no2_short, start, end, config.BBOX)
        o3_rs = search_granules(o3_short, start, end, config.BBOX)

        # 병렬 다운로드 실행
        no2_files = download_granules(no2_rs, out_no2, threads=threads)
        o3_files = download_granules(o3_rs, out_o3, threads=threads)

        # ========================================================================
        # 5. 다운로드 완료 요약
        # ========================================================================
        logger.info("\n" + "=" * 60)
        logger.info("✓ NRT (V04) download complete")
        logger.info("=" * 60)
        logger.info(f"  NO2_NRT files: {len(no2_files)} → {out_no2}")
        logger.info(f"  O3_NRT  files: {len(o3_files)}  → {out_o3}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"NRT download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
