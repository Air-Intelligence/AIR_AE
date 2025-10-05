"""
Download TEMPO L3 NO₂ and O₃ data using earthaccess
TEMPO(Tropospheric Emissions: Monitoring of Pollution)는 NASA의 대기질 관측 위성입니다.
이 스크립트는 NO₂(이산화질소)와 O₃(오존) 기둥 농도 데이터를 다운로드합니다.
"""

import earthaccess  # NASA Earthdata 다운로드 라이브러리
from datetime import datetime  # 날짜 처리
from pathlib import Path  # 파일 경로 처리
from typing import List  # 타입 힌트
import config  # 전역 설정 (날짜, BBOX, collection ID 등)
import utils  # 공통 유틸리티 (로깅)

logger = utils.setup_logging(__name__)


def setup_earthdata_auth():
    """
    Set up Earthdata authentication
    Requires ~/.netrc with Earthdata credentials:
        machine urs.earthdata.nasa.gov
            login YOUR_USERNAME
            password YOUR_PASSWORD
    """
    logger.info("Setting up Earthdata authentication...")

    try:
        # .netrc 파일에서 인증 정보 읽기
        # Windows: C:\Users\사용자명\.netrc
        # Linux/Mac: ~/.netrc (권한 600 필수)
        auth = earthaccess.login(strategy="netrc")
        logger.info("✓ Authenticated with Earthdata")
        return auth
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("\nPlease create ~/.netrc with:")
        logger.error("  machine urs.earthdata.nasa.gov")
        logger.error("      login YOUR_USERNAME")
        logger.error("      password YOUR_PASSWORD")
        raise


def search_granules(
    collection: str, date_start: datetime, date_end: datetime, bbox: dict = None
) -> List:
    """
    Search for granules in ASDC collection with 3-step fallback

    Args:
        collection: Collection dict with short_name and version
        date_start: Start date
        date_end: End date
        bbox: Bounding box (west, south, east, north)

    Returns:
        List of granule objects
    """
    logger.info(f"Searching for {collection['short_name']} granules...")

    # BBOX를 튜플 형식으로 변환 (west, south, east, north)
    if bbox is None:
        bbox_tuple = (
            config.BBOX["west"],
            config.BBOX["south"],
            config.BBOX["east"],
            config.BBOX["north"],
        )
    else:
        bbox_tuple = (bbox["west"], bbox["south"], bbox["east"], bbox["north"])

    temporal_tuple = (date_start.strftime("%Y-%m-%d"), date_end.strftime("%Y-%m-%d"))

    # 1차 시도: 기간 + 영역
    try:
        logger.info("1차 검색: 기간 + 영역 조건")
        results = earthaccess.search_data(
            short_name=collection["short_name"],
            version=collection["version"],
            temporal=temporal_tuple,
            bounding_box=bbox_tuple,
            cloud_hosted=True,  # 수정됨: daac 삭제, cloud_hosted 추가
        )
        logger.info(f"1차 검색 결과: {len(results)} granules")
        if len(results) > 0:
            return results
    except Exception as e:
        logger.warning(f"1차 검색 실패: {e}")

    # 2차 시도: 기간만
    try:
        logger.warning("Granule 없음 → 2차 검색(기간만) 시도 중...")
        results = earthaccess.search_data(
            short_name=collection["short_name"],
            version=collection["version"],
            temporal=temporal_tuple,
            cloud_hosted=True,  # 수정됨: daac 삭제, cloud_hosted 추가
        )
        logger.info(f"2차 검색 결과: {len(results)} granules")
        if len(results) > 0:
            return results
    except Exception as e:
        logger.warning(f"2차 검색 실패: {e}")

    # 3차 시도: 영역만
    try:
        logger.warning("Granule 없음 → 3차 검색(영역만) 시도 중...")
        results = earthaccess.search_data(
            short_name=collection["short_name"],
            version=collection["version"],
            bounding_box=bbox_tuple,
            cloud_hosted=True,  # 수정됨: daac 삭제, cloud_hosted 추가
        )
        logger.info(f"3차 검색 결과: {len(results)} granules")
        return results
    except Exception as e:
        logger.error(f"3차 검색 실패: {e}")
        return []


def download_granules(
    granules: List, output_dir: Path, max_workers: int = 8
) -> List[str]:
    """
    Download granules with parallel connections

    Args:
        granules: List of granule objects from search
        output_dir: Output directory
        max_workers: Number of parallel downloads

    Returns:
        List of downloaded file paths
    """
    if len(granules) == 0:
        logger.warning("No granules to download")
        return []

    logger.info(f"Downloading {len(granules)} granules to {output_dir}...")
    logger.info(f"Using {max_workers} parallel workers")

    try:
        # earthaccess가 자동으로 재시도(retry)와 이어받기(resume) 처리
        # threads: 병렬 다운로드 수 (기본 8개)
        # 다운로드 속도 향상을 위해 여러 파일을 동시에 받음
        files = earthaccess.download(granules, str(output_dir), threads=max_workers)

        logger.info(f"✓ Downloaded {len(files)} files")
        return files

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return []


def download_tempo_product(product: str, collection: str, output_dir: Path):
    """
    Download a TEMPO product (NO₂ or O₃)

    Args:
        product: Product name (e.g., 'NO2', 'O3')
        collection: Collection short name
        output_dir: Output directory
    """
    logger.info("=" * 60)
    logger.info(f"Downloading TEMPO {product}")
    logger.info("=" * 60)

    # ========================================================================
    # 1. Granule 검색
    # ========================================================================
    # config.py에 정의된 날짜 범위와 BBOX로 검색
    granules = search_granules(
        collection=collection,
        date_start=config.DATE_START,
        date_end=config.DATE_END,
        bbox=config.BBOX,
    )

    if len(granules) == 0:
        logger.error(f"No granules found for {product}")
        return

    # ========================================================================
    # 2. Granule 다운로드
    # ========================================================================
    # 병렬 다운로드로 속도 향상 (기본 8개 동시 다운로드)
    files = download_granules(
        granules,
        output_dir,
        max_workers=config.ARIA2_PARAMS["max_concurrent_downloads"],
    )

    if len(files) == 0:
        logger.error(f"Failed to download {product} data")
        return

    # ========================================================================
    # 3. 다운로드 결과 요약
    # ========================================================================
    # 전체 용량 계산 (GB 단위)
    total_size = sum(Path(f).stat().st_size for f in files if Path(f).exists())
    size_gb = total_size / 1024**3

    logger.info(f"\n✓ {product} download complete:")
    logger.info(f"  Files:  {len(files)}")
    logger.info(f"  Size:   {size_gb:.2f} GB")
    logger.info(f"  Location: {output_dir}")


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("TEMPO L3 Data Download")
    logger.info("=" * 60)
    logger.info(f"Period: {config.DATE_START.date()} to {config.DATE_END.date()}")
    logger.info(f"Region: {config.BBOX['name']}")
    logger.info("=" * 60)

    # ========================================================================
    # 1. Earthdata 인증
    # ========================================================================
    # .netrc 파일에서 NASA Earthdata 계정 정보 읽기
    setup_earthdata_auth()

    # ========================================================================
    # 2. TEMPO NO₂ 다운로드
    # ========================================================================
    # 이산화질소(NO₂) 기둥 농도 데이터
    # PM2.5 예측의 주요 특성 중 하나
    download_tempo_product(
        product="NO2",
        collection=config.TEMPO_COLLECTIONS["NO2"],  # 'TEMPO_NO2_L3_V03'
        output_dir=config.RAW_TEMPO_NO2,
    )

    # ========================================================================
    # 3. TEMPO O₃ 다운로드
    # ========================================================================
    # 오존(O₃) 기둥 농도 데이터
    # 대기 화학 반응의 지표
    download_tempo_product(
        product="O3",
        collection=config.TEMPO_COLLECTIONS["O3"],  # 'TEMPO_O3_L3_V03'
        output_dir=config.RAW_TEMPO_O3,
    )

    # ========================================================================
    # 4. (선택) TEMPO CLDO4 다운로드
    # ========================================================================
    # 클라우드 분율 데이터 (Phase 2에서 성능 향상을 위해 추가 가능)
    # config.py에서 ENABLE_CLDO4 = True로 설정 시 다운로드
    if config.ENABLE_CLDO4:
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Downloading TEMPO CLDO4")
        logger.info("=" * 60)

        # CLDO4는 용량이 크므로 기간을 줄일 수 있음 (기본 4주)
        if config.CLDO4_WEEKS < config.WEEKS:
            from datetime import timedelta

            date_end_cldo4 = config.DATE_START + timedelta(weeks=config.CLDO4_WEEKS)
            logger.info(f"Using reduced period for CLDO4: {config.CLDO4_WEEKS} weeks")
        else:
            date_end_cldo4 = config.DATE_END

        granules_cldo4 = search_granules(
            collection=config.TEMPO_COLLECTIONS["CLDO4"],
            date_start=config.DATE_START,
            date_end=date_end_cldo4,
            bbox=config.BBOX,
        )

        if len(granules_cldo4) > 0:
            download_granules(
                granules_cldo4,
                config.RAW_TEMPO_CLDO4,
                max_workers=config.ARIA2_PARAMS["max_concurrent_downloads"],
            )

    logger.info("\n" + "=" * 60)
    logger.info("✓ TEMPO download complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
