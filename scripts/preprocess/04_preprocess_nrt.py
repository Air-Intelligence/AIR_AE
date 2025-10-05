"""
TEMPO V04 NRT (Near Real-Time) 데이터 전처리 - 롤링 업데이트

목적:
- V04 NO2 & O3 실시간 데이터 전처리 (최근 3일치 유지)
- 새로운 파일만 처리하여 기존 Parquet에 추가
- 3일 이전 오래된 데이터 자동 삭제
- Cron으로 매시간 실행 가능

처리 흐름:
1. 기존 Parquet 읽기 (있으면)
2. 이미 처리된 시간 확인
3. 새 NetCDF 파일만 전처리 (중복 방지)
4. 기존 데이터 + 새 데이터 병합
5. 3일 이전 데이터 삭제 (롤링 윈도우)
6. Parquet 저장 (~1GB)

주요 차이점 (vs 04_preprocess_merge.py):
- V04 collection 사용 (2025-09-17 이후)
- O3 변수명: vertical_column_total → column_amount_o3
- MERRA-2 제외 (실시간 업데이트 느림)
- 롤링 업데이트 (덮어쓰기가 아닌 추가)
- 출력: /mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from typing import Dict, List, Set
from datetime import datetime, timedelta
import config
import utils

logger = utils.setup_logging(__name__)


def get_processed_times(parquet_path: Path) -> Set[pd.Timestamp]:
    """
    기존 Parquet에서 이미 처리된 시간 목록 추출

    Args:
        parquet_path: Parquet 파일 경로

    Returns:
        처리된 시간 집합 (Set)
    """
    if not parquet_path.exists():
        logger.info("기존 Parquet 없음 → 전체 파일 처리")
        return set()

    try:
        df = pd.read_parquet(parquet_path)
        df["time"] = pd.to_datetime(df["time"])
        processed_times = set(df["time"].unique())
        logger.info(f"기존 Parquet에서 {len(processed_times)}개 시간대 확인")
        return processed_times
    except Exception as e:
        logger.warning(f"기존 Parquet 읽기 실패: {e} → 전체 재처리")
        return set()


def filter_new_files(files: List[str], processed_times: Set[pd.Timestamp]) -> List[str]:
    """
    이미 처리된 시간의 파일 제외 (중복 방지)

    Args:
        files: NetCDF 파일 경로 리스트
        processed_times: 이미 처리된 시간 집합

    Returns:
        새로운 파일만 필터링된 리스트
    """
    if not processed_times:
        # 기존 데이터 없으면 전체 파일 처리
        return files

    new_files = []
    for file in files:
        try:
            # 파일에서 시간 읽기 (메타데이터만, 빠름)
            with xr.open_dataset(file) as ds:
                file_time = pd.to_datetime(ds.time.values[0])

            # 이미 처리된 시간이 아니면 추가
            if file_time not in processed_times:
                new_files.append(file)
        except Exception as e:
            logger.warning(f"파일 시간 확인 실패 ({Path(file).name}): {e}")
            # 확인 실패 시 안전하게 포함
            new_files.append(file)

    logger.info(f"전체 {len(files)}개 중 새 파일 {len(new_files)}개 발견")
    return new_files


def process_tempo_v04_files(
    file_pattern: str,
    var_name_mapping: Dict[str, str],
    product_name: str,
    group_path: str = None,
    processed_times: Set[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    TEMPO V04 NetCDF 파일을 Tidy DataFrame으로 전처리
    이미 처리된 파일은 스킵 (롤링 업데이트)

    Args:
        file_pattern: NC 파일 glob 패턴
        var_name_mapping: NetCDF 변수명 → 표준 이름 매핑
        product_name: 제품명 (로깅용, 예: NO2, O3)
        group_path: HDF5 그룹 경로 (예: '/product')
        processed_times: 이미 처리된 시간 집합 (중복 방지)

    Returns:
        Tidy DataFrame (time, lat, lon, 변수)
    """
    logger.info(f"TEMPO V04 {product_name} 처리 중...")

    # 파일 목록 가져오기
    files = sorted(glob(file_pattern))
    if len(files) == 0:
        logger.error(f"파일 없음: {file_pattern}")
        return pd.DataFrame()

    logger.info(f"발견된 파일: {len(files)}개")

    # 새 파일만 필터링 (중복 방지)
    if processed_times:
        files = filter_new_files(files, processed_times)
        if len(files) == 0:
            logger.info(f"{product_name}: 새 파일 없음, 스킵")
            return pd.DataFrame()

    all_dfs = []

    for i, file in enumerate(files, 1):
        try:
            # HDF5 그룹 경로 지정하여 NetCDF 열기
            if group_path:
                # V04: 좌표는 루트, 데이터는 /product/ 그룹에 분리되어 있음
                # 1. 루트에서 좌표 읽기
                with xr.open_dataset(file, chunks="auto") as ds_root:
                    coords_dict = {
                        'longitude': ds_root['longitude'],
                        'latitude': ds_root['latitude'],
                        'time': ds_root['time']
                    }

                # 2. /product/ 그룹에서 데이터 읽고 좌표 할당
                with xr.open_dataset(file, group=group_path, chunks="auto") as ds:
                    # 좌표 할당
                    ds = ds.assign_coords(coords_dict)

                    # Bay Area로 BBOX 서브셋팅 (용량 30배 절감)
                    ds = utils.subset_bbox(ds)

                    # 변수명 표준화 (예: vertical_column_troposphere → no2)
                    for raw_name, std_name in var_name_mapping.items():
                        if raw_name in ds:
                            ds = ds.rename({raw_name: std_name})

                    # 2D 그리드 → 1D 테이블 변환 (Tidy format)
                    df = utils.netcdf_to_tidy(ds, var_mapping=None)
            else:
                # 그룹 경로 없을 때 (루트에서 읽기)
                with xr.open_dataset(file, chunks="auto") as ds:
                    ds = utils.subset_bbox(ds)

                    for raw_name, std_name in var_name_mapping.items():
                        if raw_name in ds:
                            ds = ds.rename({raw_name: std_name})

                    df = utils.netcdf_to_tidy(ds, var_mapping=None)

            all_dfs.append(df)

            # 진행 상황 로깅 (10개마다)
            if i % 10 == 0 or i == len(files):
                logger.info(f"  {i}/{len(files)} 파일 처리 완료")

        except Exception as e:
            import traceback
            logger.error(f"  파일 처리 실패 ({Path(file).name}): {e}")
            logger.debug(f"  상세 에러:\n{traceback.format_exc()}")
            continue

    if len(all_dfs) == 0:
        logger.warning(f"{product_name}: 추출된 데이터 없음")
        return pd.DataFrame()

    # 모든 DataFrame 합치기
    df = pd.concat(all_dfs, ignore_index=True)

    # UTC 시간 정규화 (timezone 제거)
    df["time"] = utils.to_utc_naive(df["time"])

    # 중복 그리드 포인트 제거 (같은 시간/위치는 평균)
    df = utils.dedup_grid(df)

    logger.info(f"✓ {product_name}: {len(df):,} 행 추출")

    return df


def rolling_update(
    new_df: pd.DataFrame,
    parquet_path: Path,
    keep_days: int = 3
) -> pd.DataFrame:
    """
    롤링 업데이트: 기존 데이터 + 새 데이터 병합 후 오래된 데이터 삭제

    Args:
        new_df: 새로 처리된 DataFrame
        parquet_path: 기존 Parquet 경로
        keep_days: 유지할 날짜 수 (기본 3일)

    Returns:
        업데이트된 DataFrame (최근 N일치만)
    """
    logger.info(f"\n롤링 업데이트 (최근 {keep_days}일 유지)...")

    # 기존 Parquet 읽기
    if parquet_path.exists():
        try:
            old_df = pd.read_parquet(parquet_path)
            old_df["time"] = pd.to_datetime(old_df["time"])

            # 좌표명 표준화 (latitude→lat, longitude→lon)
            if 'latitude' in old_df.columns:
                old_df = old_df.rename(columns={'latitude': 'lat'})
            if 'longitude' in old_df.columns:
                old_df = old_df.rename(columns={'longitude': 'lon'})

            logger.info(f"  기존 데이터: {len(old_df):,} 행")
        except Exception as e:
            logger.warning(f"  기존 Parquet 읽기 실패: {e} → 새 데이터만 사용")
            old_df = pd.DataFrame()
    else:
        logger.info("  기존 Parquet 없음 → 새 데이터만 사용")
        old_df = pd.DataFrame()

    # 새 데이터와 기존 데이터 병합
    if len(old_df) > 0:
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
        logger.info(f"  병합 후: {len(combined_df):,} 행")
    else:
        combined_df = new_df
        logger.info(f"  새 데이터만: {len(combined_df):,} 행")

    # 중복 제거 (같은 시간/위치)
    combined_df = utils.dedup_grid(combined_df)
    logger.info(f"  중복 제거 후: {len(combined_df):,} 행")

    # 최근 N일 데이터만 유지
    cutoff_time = combined_df["time"].max() - timedelta(days=keep_days)
    filtered_df = combined_df[combined_df["time"] >= cutoff_time].copy()

    removed = len(combined_df) - len(filtered_df)
    logger.info(f"  {keep_days}일 이전 데이터 {removed:,}행 삭제")
    logger.info(f"  최종 데이터: {len(filtered_df):,} 행 (시간 범위: {filtered_df['time'].min()} ~ {filtered_df['time'].max()})")

    return filtered_df


def main():
    """메인 실행 함수 (Cron으로 매시간 실행)"""
    logger.info("="*60)
    logger.info("TEMPO V04 NRT 데이터 전처리 (롤링 업데이트)")
    logger.info("="*60)

    # 출력 경로: /mnt/data/features/tempo/nrt_roll3d/nrt_merged.parquet
    output_path = config.FEATURES_TEMPO_NRT / "nrt_merged.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. 기존 Parquet에서 이미 처리된 시간 확인
    # ========================================================================
    processed_times = get_processed_times(output_path)

    # ========================================================================
    # 2. TEMPO V04 NO₂ 처리 (새 파일만)
    # ========================================================================
    # download_realtime_data.py로 다운받은 경로 사용
    tempo_no2_path = Path("/mnt/data/features/realtime/tempo_no2_raw")

    df_no2 = process_tempo_v04_files(
        file_pattern=str(tempo_no2_path / "*.nc"),
        var_name_mapping={'vertical_column_troposphere': 'no2'},
        product_name='NO2_NRT',
        group_path='/product',  # V04 NO2는 /product/ 그룹에 있음
        processed_times=processed_times
    )

    # ========================================================================
    # 3. TEMPO V04 O₃ 처리 (새 파일만)
    # ========================================================================
    df_o3 = process_tempo_v04_files(
        file_pattern=str(config.RAW_TEMPO_O3_NRT / "*.nc"),
        var_name_mapping={'column_amount_o3': 'o3'},  # V04: column_amount_o3
        product_name='O3_NRT',
        group_path='/product',  # V04 O3도 /product/ 그룹에 있음
        processed_times=processed_times
    )

    # ========================================================================
    # 4. 새 데이터 병합 (NO2 + O3)
    # ========================================================================
    # 새 파일이 없으면 종료
    if len(df_no2) == 0 and len(df_o3) == 0:
        logger.info("\n새로운 데이터 없음. 종료.")
        return

    # 새 데이터가 있으면 병합
    if len(df_no2) > 0 and len(df_o3) > 0:
        logger.info("\nNO2 + O3 병합 (시간 매칭)...")
        df_new = utils.merge_time_nearest(
            df_no2, df_o3,
            tol=config.LABEL_JOIN["time_tolerance"]  # 30분 tolerance
        )
    elif len(df_no2) > 0:
        logger.info("\nNO2만 처리 (O3 새 데이터 없음)")
        df_new = df_no2
    else:
        logger.info("\nO3만 처리 (NO2 새 데이터 없음)")
        df_new = df_o3

    if len(df_new) == 0:
        logger.error("병합 후 데이터 없음. 시간 정렬 확인 필요.")
        return

    logger.info(f"✓ 새 데이터 병합: {len(df_new):,} 행, {len(df_new.columns)} 열")

    # ========================================================================
    # 5. QC (품질 관리)
    # ========================================================================
    logger.info("\nQC 적용 중...")
    df_new = utils.apply_qc(df_new)

    # 필수 변수 결측치 제거 (존재하는 컬럼만)
    required_cols = [col for col in ["no2", "o3"] if col in df_new.columns]
    if required_cols:
        df_new = df_new.dropna(subset=required_cols)
        logger.info(f"결측치 제거 대상: {required_cols}")
    else:
        logger.warning("no2 또는 o3 컬럼이 없습니다")

    logger.info(f"✓ QC 후: {len(df_new):,} 행")

    # ========================================================================
    # 6. 롤링 업데이트 (기존 + 새 데이터, 3일치 유지)
    # ========================================================================
    df_final = rolling_update(
        new_df=df_new,
        parquet_path=output_path,
        keep_days=config.NRT_RECENT_DAYS  # 3일
    )

    # ========================================================================
    # 7. Parquet 저장
    # ========================================================================
    logger.info("\nParquet 저장 중...")
    utils.save_parquet(df_final, output_path, downcast=True)

    # ========================================================================
    # 8. 결과 요약
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("NRT 전처리 완료")
    logger.info("="*60)
    logger.info(f"  시간 범위:      {df_final['time'].min()} ~ {df_final['time'].max()}")
    logger.info(f"  총 데이터 행:   {len(df_final):,}")
    logger.info(f"  고유 시간대:    {df_final['time'].nunique():,}")
    logger.info(f"  고유 위치:      {len(df_final[['lat','lon']].drop_duplicates()):,}")
    logger.info(f"  변수:           {', '.join([c for c in df_final.columns if c not in ['time', 'lat', 'lon']])}")
    logger.info(f"  출력 파일:      {output_path}")
    logger.info(f"  파일 크기:      {output_path.stat().st_size / 1024**2:.1f} MB")
    logger.info("="*60)

    logger.info("\n✓ NRT 전처리 성공! FastAPI가 이 Parquet를 읽을 수 있습니다.")


if __name__ == "__main__":
    main()
