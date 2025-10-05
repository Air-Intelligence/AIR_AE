#!/usr/bin/env python3
"""
TEMPO L3 California Data Pipeline
==================================
NASA TEMPO L3 데이터(NO₂/HCHO/O₃/Cloud)를 캘리포니아 영역에 대해
최소 시간·용량으로 수집/전처리/병합하는 통합 파이프라인

예상 소요시간:
- 테스트(4주, 3변수): ~30분
- 전체(3개월, 4변수): ~2-3시간 (THREADS=8 기준)

예상 용량:
- 원본 netCDF: ~5-10GB
- 최종 Parquet: ~200-500MB
"""

import earthaccess
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - 여기만 수정하세요
# ============================================================================

CONFIG = {
    # 공간 범위: 캘리포니아 (West, South, East, North)
    'BBOX': (-122.5, 37.2, -121.5, 38.0),  # Bay Area (SF, Oakland, San Jose)

    # 시간 범위: 2~3개월 (처음엔 4주로 테스트 권장)
    'DATE_RANGE': ('2023-08-01', '2023-10-31'),  # 3개월 (2023년 여름~가을)
    #'DATE_RANGE': ('2024-06-01', '2024-08-31'),  # 3개월
    #'DATE_RANGE': ('2024-06-01', '2024-06-28'),  # 4주 (테스트용)

    # 다운로드 병렬 스레드 (8~12 권장, VM 코어 수에 맞게 조정)
    'THREADS': 12,

    # 출력 디렉토리
    'OUT_DIR': './tempo_l3_bayarea_202308-202310',

    # 시간 리샘플링 ('3H' 또는 None)
    'RESAMPLE': '3H',

    # 다운로드할 변수 (처음엔 2~3개로 테스트)
    'VARS': ['NO2', 'HCHO', 'O3', 'CLOUD'],  # 전체 4변수
    # 'VARS': ['NO2', 'O3', 'CLOUD'],  # 3변수

    # TEMPO 제품 버전 (V03 또는 V04, V04가 최신)
    # 'VERSION': 'V04',
    'VERSION': 'V03',
}

# TEMPO L3 컬렉션 정의 (CSV 기반)
TEMPO_L3_COLLECTIONS = {
    'NO2': {
        'V03': 'TEMPO_NO2_L3',
        'V04': 'TEMPO_NO2_L3',
    },
    'HCHO': {
        'V03': 'TEMPO_HCHO_L3',
        'V04': 'TEMPO_HCHO_L3',
    },
    'O3': {
        'V03': 'TEMPO_O3TOT_L3',
        'V04': 'TEMPO_O3TOT_L3',
    },
    'CLOUD': {
        'V03': 'TEMPO_CLDO4_L3',
        'V04': 'TEMPO_CLDO4_L3',
    }
}

# 변수명 매핑 (파일 내 실제 키 → 표준명)
# 우선순위 순으로 시도
VARIABLE_MAPPINGS = {
    'NO2': [
        'vertical_column_troposphere',
        'tropospheric_vertical_column',
        'nitrogen_dioxide_tropospheric_column',
        'NO2_column',
    ],
    'HCHO': [
        'vertical_column',
        'formaldehyde_vertical_column',
        'HCHO_column',
    ],
    'O3': [
        'vertical_column',
        'ozone_total_vertical_column',
        'O3_column',
        'total_ozone_column',
    ],
    'CLOUD': [
        'cloud_fraction',
        'effective_cloud_fraction',
        'cloud_fraction_o2o2',
    ]
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def authenticate():
    """NASA Earthdata 인증"""
    print("🔐 NASA Earthdata 로그인 중...")
    print("   (처음 실행 시 계정 정보 입력 필요: https://urs.earthdata.nasa.gov/)")

    auth = earthaccess.login()
    if not auth:
        raise RuntimeError("❌ 인증 실패! EDL 계정 확인 필요")

    print("✅ 인증 성공!\n")
    return auth


def search_granules(var_name, config):
    """CMR에서 granule 검색"""
    short_name = TEMPO_L3_COLLECTIONS[var_name][config['VERSION']]
    version = config['VERSION']

    print(f"🔍 검색 중: {var_name} ({short_name} {version})")
    print(f"   기간: {config['DATE_RANGE'][0]} ~ {config['DATE_RANGE'][1]}")
    print(f"   영역: {config['BBOX']}")

    try:
        granules = earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=config['DATE_RANGE'],
            bounding_box=config['BBOX'],
        )

        print(f"   → 발견: {len(granules)}개 파일\n")
        return granules

    except Exception as e:
        print(f"   ⚠️  검색 실패: {e}\n")
        return []


def download_granules(granules, var_name, config):
    """병렬 다운로드 with retry"""
    if not granules:
        return []

    out_dir = Path(config['OUT_DIR']) / 'raw' / var_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  다운로드 시작: {len(granules)}개 파일 (THREADS={config['THREADS']})")
    print(f"   저장 경로: {out_dir}")

    try:
        files = earthaccess.download(
            granules,
            str(out_dir),
            threads=config['THREADS']
        )

        print(f"✅ 다운로드 완료: {len(files)}개 파일\n")
        return files

    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print("   → Retry 시도 중 (threads=4)...\n")

        try:
            files = earthaccess.download(
                granules,
                str(out_dir),
                threads=4  # 스레드 줄여서 재시도
            )
            print(f"✅ Retry 성공: {len(files)}개 파일\n")
            return files
        except:
            print(f"❌ Retry 실패. 수동 확인 필요.\n")
            return []


def find_variable_name(ds, var_type):
    """xarray Dataset에서 실제 변수명 자동 탐색"""
    candidates = VARIABLE_MAPPINGS[var_type]

    for name in candidates:
        if name in ds.data_vars:
            return name

    # 후보에 없으면 첫 번째 data_var 반환
    if len(ds.data_vars) > 0:
        fallback = list(ds.data_vars.keys())[0]
        print(f"   ⚠️  표준 변수명 없음. 대체: {fallback}")
        return fallback

    raise ValueError(f"변수를 찾을 수 없음: {list(ds.data_vars.keys())}")


def subset_and_convert(files, var_type, config):
    """BBOX 서브셋 + Tidy 변환"""
    if not files:
        return pd.DataFrame()

    print(f"📦 전처리 중: {var_type} ({len(files)}개 파일)")

    dfs = []
    w, s, e, n = config['BBOX']

    for i, fpath in enumerate(files, 1):
        try:
            # netCDF 열기
            ds = xr.open_dataset(fpath, decode_times=True)

            # 첫 파일에서 변수명 확인
            if i == 1:
                var_name = find_variable_name(ds, var_type)
                print(f"   변수명: {var_name}")

            # BBOX 서브셋 (lat/lon 좌표명 자동 탐색)
            lat_key = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_key = 'longitude' if 'longitude' in ds.coords else 'lon'

            ds_subset = ds.where(
                (ds[lat_key] >= s) & (ds[lat_key] <= n) &
                (ds[lon_key] >= w) & (ds[lon_key] <= e),
                drop=True
            )

            # Tidy 변환
            df = ds_subset[[var_name]].to_dataframe().reset_index()

            # 컬럼명 표준화
            df = df.rename(columns={
                lat_key: 'lat',
                lon_key: 'lon',
                var_name: var_type.lower()
            })

            # 필수 컬럼만 선택
            df = df[['time', 'lat', 'lon', var_type.lower()]]

            # 결측치/이상치 제거
            df = df.dropna()

            dfs.append(df)
            ds.close()

            if i % 10 == 0:
                print(f"   진행: {i}/{len(files)}...")

        except Exception as e:
            print(f"   ⚠️  파일 {i} 처리 실패: {e}")
            continue

    if not dfs:
        print(f"   ❌ 처리 가능한 파일 없음\n")
        return pd.DataFrame()

    # 전체 병합
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"   → 총 {len(df_all):,}행 생성")
    print(f"   → 시간 범위: {df_all['time'].min()} ~ {df_all['time'].max()}\n")

    return df_all


def resample_temporal(df, freq='3H'):
    """시간 리샘플링 (선택사항)"""
    if freq is None or df.empty:
        return df

    print(f"⏰ 시간 리샘플링: {freq}")

    df = df.set_index('time')
    df_resampled = df.groupby(['lat', 'lon']).resample(freq).mean().reset_index()

    print(f"   → {len(df_resampled):,}행\n")
    return df_resampled


def merge_all_variables(dfs_dict, config):
    """모든 변수를 시공간 기준으로 병합"""
    print("🔗 변수 병합 중...")

    # 첫 번째 변수를 base로 시작
    var_list = list(dfs_dict.keys())
    if not var_list:
        raise ValueError("병합할 데이터 없음")

    merged = dfs_dict[var_list[0]].copy()
    print(f"   Base: {var_list[0]} ({len(merged):,}행)")

    # 나머지 변수 순차 조인
    for var in var_list[1:]:
        df = dfs_dict[var]

        before = len(merged)
        merged = merged.merge(
            df,
            on=['time', 'lat', 'lon'],
            how='inner'  # 모든 변수가 있는 시공간만 유지
        )
        after = len(merged)

        print(f"   + {var}: {before:,} → {after:,}행")

    print(f"\n✅ 최종: {len(merged):,}행 × {len(merged.columns)}열")
    print(f"   컬럼: {list(merged.columns)}\n")

    return merged


def save_outputs(df, config):
    """Parquet + CSV 샘플 저장"""
    if df.empty:
        print("⚠️  저장할 데이터 없음\n")
        return

    out_dir = Path(config['OUT_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parquet (전체)
    parquet_path = out_dir / 'tempo_l3_ca_merged.parquet'
    print(f"💾 Parquet 저장 중: {parquet_path}")
    df.to_parquet(parquet_path, compression='snappy', index=False)

    size_mb = parquet_path.stat().st_size / 1024 / 1024
    print(f"   → {size_mb:.1f} MB\n")

    # CSV 샘플 (1만행)
    csv_path = out_dir / 'tempo_l3_ca_sample.csv'
    sample_size = min(10000, len(df))
    print(f"💾 CSV 샘플 저장 중: {csv_path} ({sample_size:,}행)")
    df.head(sample_size).to_csv(csv_path, index=False)
    print()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*70)
    print("  TEMPO L3 California Data Pipeline")
    print("="*70)
    print(f"설정:")
    print(f"  - BBOX: {CONFIG['BBOX']}")
    print(f"  - 기간: {CONFIG['DATE_RANGE']}")
    print(f"  - 변수: {CONFIG['VARS']}")
    print(f"  - 버전: {CONFIG['VERSION']}")
    print(f"  - 스레드: {CONFIG['THREADS']}")
    print(f"  - 리샘플: {CONFIG['RESAMPLE']}")
    print(f"  - 출력: {CONFIG['OUT_DIR']}")
    print("="*70)
    print()

    start_time = datetime.now()

    # 1. 인증
    authenticate()

    # 2. 각 변수별 파이프라인
    dfs = {}

    for var in CONFIG['VARS']:
        print(f"\n{'#'*70}")
        print(f"#  {var} 처리 시작")
        print(f"{'#'*70}\n")

        # 2.1 검색
        granules = search_granules(var, CONFIG)
        if not granules:
            print(f"⚠️  {var}: granule 없음. 스킵.\n")
            continue

        # 2.2 다운로드
        files = download_granules(granules, var, CONFIG)
        if not files:
            print(f"⚠️  {var}: 다운로드 실패. 스킵.\n")
            continue

        # 2.3 서브셋 + Tidy 변환
        df = subset_and_convert(files, var, CONFIG)
        if df.empty:
            print(f"⚠️  {var}: 전처리 실패. 스킵.\n")
            continue

        # 2.4 시간 리샘플
        df = resample_temporal(df, CONFIG['RESAMPLE'])

        dfs[var] = df

    # 3. 변수 병합
    if not dfs:
        print("\n❌ 처리된 변수 없음. 종료.\n")
        return

    print(f"\n{'='*70}")
    print("  변수 병합")
    print(f"{'='*70}\n")

    merged = merge_all_variables(dfs, CONFIG)

    # 4. 저장
    print(f"{'='*70}")
    print("  최종 저장")
    print(f"{'='*70}\n")

    save_outputs(merged, CONFIG)

    # 5. 통계 요약
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"{'='*70}")
    print("  완료!")
    print(f"{'='*70}")
    print(f"소요시간: {elapsed/60:.1f}분")
    print(f"최종 데이터: {len(merged):,}행 × {len(merged.columns)}열")
    print(f"출력: {CONFIG['OUT_DIR']}/tempo_l3_ca_merged.parquet")
    print(f"{'='*70}\n")

    # 6. 데이터 미리보기
    print("📊 데이터 미리보기:")
    print(merged.head(10))
    print()
    print("📈 기초 통계:")
    print(merged.describe())


if __name__ == '__main__':
    main()
