"""
TEMPO O3 Standard V04 정적 데이터 다운로드

목적:
- TEMPO O3 NRT가 없으므로 Standard V04 과거 데이터를 대안으로 사용
- 학습 데이터와 다른 기간의 데이터 다운로드 (최근 3일)
- FastAPI에서 NO2 NRT와 조합하여 제공

출력:
- /mnt/data/features/tempo/o3_static/*.nc (NetCDF)
- /mnt/data/features/tempo/o3_static.parquet (전처리 후)

사용법:
    python download_o3_static.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import earthaccess
import xarray as xr
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
import config
import utils

logger = utils.setup_logging(__name__)


def auth_via_netrc() -> None:
    """
    Earthdata 인증 (~/.netrc 사용)
    """
    earthaccess.login(strategy="netrc")
    logger.info("✓ Earthdata authenticated via ~/.netrc")


def bbox_tuple(bbox: dict) -> tuple[float, float, float, float]:
    """BBOX 딕셔너리 → 튜플 변환"""
    return (bbox["west"], bbox["south"], bbox["east"], bbox["north"])


def search_o3_granules(start: datetime, end: datetime, bbox: dict) -> List:
    """
    TEMPO O3 Standard V04 granule 검색

    Args:
        start: 검색 시작 시간 (UTC)
        end: 검색 종료 시간 (UTC)
        bbox: 검색 영역 (Bay Area)

    Returns:
        검색된 granule 리스트
    """
    bt = bbox_tuple(bbox)

    # NASA CMR에서 TEMPO O3 Standard V04 검색
    results = earthaccess.search_data(
        short_name="TEMPO_O3TOT_L3",
        version="V04",  # Standard V04
        temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
        bounding_box=bt,
        cloud_hosted=True,
    )

    logger.info(
        f"TEMPO O3 V04: found {len(results)} granules "
        f"({start.date()} ~ {end.date()})"
    )
    return results


def download_granules(granules: List, outdir: Path, threads: int = 12) -> list[str]:
    """
    Granule 병렬 다운로드

    Args:
        granules: 검색된 granule 리스트
        outdir: 다운로드 출력 디렉터리
        threads: 병렬 다운로드 스레드 수

    Returns:
        성공적으로 다운로드된 파일 경로 리스트
    """
    if not granules:
        logger.warning("No granules to download")
        return []

    outdir.mkdir(parents=True, exist_ok=True)

    files = earthaccess.download(granules, str(outdir), threads=threads) or []

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


def process_o3_netcdf(nc_files: List[str]) -> pd.DataFrame:
    """
    TEMPO O3 NetCDF 파일을 Parquet으로 전처리

    Args:
        nc_files: NetCDF 파일 경로 리스트

    Returns:
        Tidy DataFrame (time, lat, lon, o3)
    """
    logger.info(f"\nO3 NetCDF 전처리 중... ({len(nc_files)}개 파일)")

    all_dfs = []

    for i, file in enumerate(nc_files, 1):
        try:
            # V04는 /product/ 그룹에 데이터 있음
            # 1. 루트에서 좌표 읽기
            with xr.open_dataset(file, chunks="auto") as ds_root:
                coords_dict = {
                    'longitude': ds_root['longitude'],
                    'latitude': ds_root['latitude'],
                    'time': ds_root['time']
                }

            # 2. /product/ 그룹에서 데이터 읽기
            with xr.open_dataset(file, group='/product', chunks="auto") as ds:
                ds = ds.assign_coords(coords_dict)

                # Bay Area로 서브셋팅
                ds = utils.subset_bbox(ds)

                # 변수명 표준화: column_amount_o3 → o3
                if 'column_amount_o3' in ds:
                    ds = ds.rename({'column_amount_o3': 'o3'})
                elif 'vertical_column_total' in ds:
                    ds = ds.rename({'vertical_column_total': 'o3'})

                # Tidy format 변환
                df = utils.netcdf_to_tidy(ds, var_mapping=None)

            all_dfs.append(df)

            if i % 10 == 0 or i == len(nc_files):
                logger.info(f"  {i}/{len(nc_files)} 파일 처리 완료")

        except Exception as e:
            logger.error(f"  파일 처리 실패 ({Path(file).name}): {e}")
            continue

    if len(all_dfs) == 0:
        logger.error("추출된 데이터 없음")
        return pd.DataFrame()

    # 모든 DataFrame 합치기
    df = pd.concat(all_dfs, ignore_index=True)

    # UTC 시간 정규화
    df["time"] = utils.to_utc_naive(df["time"])

    # 중복 제거
    df = utils.dedup_grid(df)

    logger.info(f"✓ O3 전처리 완료: {len(df):,} 행")

    return df


def main() -> None:
    """
    메인 함수: TEMPO O3 Standard V04 정적 데이터 다운로드 및 전처리

    실행 순서:
        1. 최근 2주 기간의 TEMPO O3 Standard V04 데이터 검색
        2. NetCDF 파일 다운로드
        3. 전처리 후 Parquet 저장

    출력:
        - /mnt/data/features/tempo/o3_static/*.nc
        - /mnt/data/features/tempo/o3_static.parquet
    """
    logger.info("=" * 60)
    logger.info("TEMPO O3 Standard V04 정적 데이터 다운로드")
    logger.info("=" * 60)

    # ========================================================================
    # 시간 범위 설정: 최근 3일 (Standard V04는 며칠 지연됨)
    # ========================================================================
    utc_now = datetime.now(timezone.utc)
    # Standard V04는 보통 1주일 정도 지연되므로, 1~2주 전 데이터 다운로드
    start = (utc_now - timedelta(days=10)).replace(tzinfo=None)
    end = (utc_now - timedelta(days=7)).replace(tzinfo=None)

    logger.info(f"Period (UTC): {start.date()} ~ {end.date()} (3일)")
    logger.info(f"Region: {config.BBOX.get('name', 'Bay Area')}")
    logger.info("⚠️  Standard V04는 실시간이 아닌 과거 데이터입니다")

    # ========================================================================
    # 1. TEMPO O3 Standard V04 다운로드
    # ========================================================================
    logger.info("\n[1/2] TEMPO O3 V04 검색 및 다운로드...")

    try:
        # NASA Earthdata 인증
        auth_via_netrc()

        # O3 granule 검색
        o3_granules = search_o3_granules(start, end, config.BBOX)

        if not o3_granules:
            logger.error("검색된 O3 granule이 없습니다")
            logger.info("💡 더 이전 기간으로 시도해보세요 (Standard V04는 지연됨)")
            sys.exit(1)

        # 다운로드
        out_dir = Path("/mnt/data/features/tempo/o3_static")
        o3_files = download_granules(o3_granules, out_dir, threads=12)

        if not o3_files:
            logger.error("다운로드된 파일이 없습니다")
            sys.exit(1)

        logger.info(f"✓ O3 다운로드: {len(o3_files)} files → {out_dir}")

    except Exception as e:
        logger.error(f"O3 다운로드 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # ========================================================================
    # 2. NetCDF → Parquet 전처리
    # ========================================================================
    logger.info("\n[2/2] O3 전처리 (NetCDF → Parquet)...")

    try:
        df_o3 = process_o3_netcdf(o3_files)

        if df_o3.empty:
            logger.error("전처리 후 데이터 없음")
            sys.exit(1)

        # QC 적용
        logger.info("QC 적용 중...")
        df_o3 = utils.apply_qc(df_o3)

        # O3 결측치 제거
        df_o3 = df_o3.dropna(subset=["o3"])

        logger.info(f"✓ QC 후: {len(df_o3):,} 행")

        # Parquet 저장
        output_path = Path("/mnt/data/features/tempo/o3_static.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        utils.save_parquet(df_o3, output_path, downcast=True)

        logger.info(f"✓ Parquet 저장: {output_path}")

    except Exception as e:
        logger.error(f"전처리 실패: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)

    # ========================================================================
    # 완료 메시지
    # ========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("✓ O3 정적 데이터 준비 완료")
    logger.info("=" * 60)
    logger.info(f"  시간 범위:      {df_o3['time'].min()} ~ {df_o3['time'].max()}")
    logger.info(f"  총 데이터 행:   {len(df_o3):,}")
    logger.info(f"  고유 시간대:    {df_o3['time'].nunique():,}")
    logger.info(f"  고유 위치:      {len(df_o3[['lat','lon']].drop_duplicates()):,}")
    logger.info(f"  출력 파일:      {output_path}")
    logger.info(f"  파일 크기:      {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("=" * 60)
    logger.info("\n다음 단계:")
    logger.info("  1. FastAPI 수정 (O3 정적 데이터 로딩)")
    logger.info("  2. API 서버 실행 (python open_aq.py)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
