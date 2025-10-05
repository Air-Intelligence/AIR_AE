"""
NRT Parquet 파일 분석 스크립트
"""
import pandas as pd
import numpy as np

# Parquet 읽기
df = pd.read_parquet('nrt_merged.parquet')

print('='*70)
print('NRT MERGED PARQUET 분석')
print('='*70)

print('\n=== 기본 정보 ===')
print(f'총 행 수: {len(df):,}')
print(f'총 컬럼 수: {len(df.columns)}')
print(f'메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')

print('\n=== 컬럼 목록 및 타입 ===')
for col in df.columns:
    dtype = df[col].dtype
    non_null = df[col].notna().sum()
    print(f'{col:40s} {str(dtype):15s} (non-null: {non_null:,})')

print('\n=== 시간 범위 ===')
df['time'] = pd.to_datetime(df['time'])
print(f'시작: {df["time"].min()}')
print(f'종료: {df["time"].max()}')
print(f'기간: {(df["time"].max() - df["time"].min()).days}일')
print(f'고유 시간대: {df["time"].nunique()}개')

# 시간별 데이터 개수
time_counts = df.groupby('time').size()
print(f'시간당 평균 포인트: {time_counts.mean():.0f}개')
print(f'시간당 최소 포인트: {time_counts.min()}개')
print(f'시간당 최대 포인트: {time_counts.max()}개')

print('\n=== 공간 범위 ===')
print(f'위도 범위: {df["lat"].min():.4f} ~ {df["lat"].max():.4f}')
print(f'경도 범위: {df["lon"].min():.4f} ~ {df["lon"].max():.4f}')
unique_locations = df[['lat','lon']].drop_duplicates()
print(f'고유 위치: {len(unique_locations):,}개')

print('\n=== 주요 변수 통계 ===')
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col not in ['lat', 'lon']:
        valid_count = df[col].notna().sum()
        missing_pct = df[col].isna().sum() / len(df) * 100

        if valid_count > 0:
            print(f'\n{col}:')
            print(f'  범위: {df[col].min():.3e} ~ {df[col].max():.3e}')
            print(f'  평균: {df[col].mean():.3e}')
            print(f'  중앙값: {df[col].median():.3e}')
            print(f'  표준편차: {df[col].std():.3e}')
            print(f'  결측치: {df[col].isna().sum():,} ({missing_pct:.1f}%)')

print('\n=== 샘플 데이터 (처음 5행) ===')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(df.head(5))

print('\n=== 최신 데이터 (마지막 5행) ===')
print(df.tail(5))

print('\n' + '='*70)
print('분석 완료!')
print('='*70)
