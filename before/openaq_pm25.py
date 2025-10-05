#!/usr/bin/env python3
"""
OpenAQ PM2.5 Label Collection & Matching
========================================
TEMPO 데이터와 조인하기 위한 OpenAQ 지상 PM2.5 측정값 수집

출력: merged_with_pm25.parquet
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # 시간 범위 (TEMPO 메인 스크립트와 동일하게)
    'DATE_RANGE': ('2023-08-01', '2023-10-31'),

    # 캘리포니아 주요 도시 또는 전체 주
    'LOCATIONS': {
        'state': 'CA',  # 캘리포니아 전체
        # 또는 주요 도시만:
        # 'cities': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento', 'Fresno']
    },

    # OpenAQ API 파라미터
    'PARAMETER': 'pm25',  # PM2.5

    # TEMPO 격자 파일 경로 (메인 파이프라인 출력)
    'TEMPO_FILE': './tempo_l3_bayarea_202308-202310/tempo_l3_ca_merged.parquet',

    # 출력 경로
    'OUTPUT_FILE': './tempo_l3_bayarea_202308-202310/merged_with_pm25.parquet',

    # 최근접 매칭 거리 임계값 (km)
    'MAX_DISTANCE_KM': 50,
}


# ============================================================================
# OpenAQ API Functions
# ============================================================================

def fetch_openaq_pm25(date_start, date_end, state='CA'):
    """
    OpenAQ API v2로 PM2.5 데이터 수집

    API 문서: https://docs.openaq.org/
    """
    print(f"🌍 OpenAQ PM2.5 수집 중...")
    print(f"   지역: {state}")
    print(f"   기간: {date_start} ~ {date_end}\n")

    base_url = "https://api.openaq.org/v2/measurements"

    all_data = []
    page = 1
    limit = 10000  # API 최대

    while True:
        params = {
            'parameter': 'pm25',
            'country': 'US',
            'location': state,  # 캘리포니아 주
            'date_from': date_start,
            'date_to': date_end,
            'limit': limit,
            'page': page,
        }

        try:
            print(f"   페이지 {page} 요청 중...", end=' ')
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            results = data.get('results', [])

            if not results:
                print("완료")
                break

            all_data.extend(results)
            print(f"{len(results)}개 수집")

            # 다음 페이지 확인
            meta = data.get('meta', {})
            if meta.get('found', 0) <= page * limit:
                break

            page += 1

        except Exception as e:
            print(f"\n⚠️  API 오류: {e}")
            break

    print(f"\n✅ 총 {len(all_data):,}개 측정값 수집\n")
    return all_data


def parse_openaq_data(raw_data):
    """OpenAQ JSON → pandas DataFrame"""
    if not raw_data:
        return pd.DataFrame()

    records = []

    for item in raw_data:
        try:
            records.append({
                'time': pd.to_datetime(item['date']['utc']),
                'pm25': item['value'],
                'lat': item['coordinates']['latitude'],
                'lon': item['coordinates']['longitude'],
                'location': item['location'],
                'city': item.get('city', ''),
                'unit': item['unit'],
            })
        except (KeyError, TypeError):
            continue

    df = pd.DataFrame(records)

    # 기본 QC
    df = df[df['pm25'] >= 0]  # 음수 제거
    df = df[df['pm25'] < 1000]  # 이상치 제거

    # 시간 정렬
    df = df.sort_values('time').reset_index(drop=True)

    print(f"📊 파싱 완료: {len(df):,}행")
    print(f"   시간 범위: {df['time'].min()} ~ {df['time'].max()}")
    print(f"   측정소 수: {df['location'].nunique()}개")
    print(f"   PM2.5 범위: {df['pm25'].min():.1f} ~ {df['pm25'].max():.1f} µg/m³\n")

    return df


# ============================================================================
# Spatial Matching (TEMPO ↔ OpenAQ)
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """위경도 간 거리 계산 (km)"""
    R = 6371  # 지구 반지름 (km)

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def match_nearest_grid(tempo_df, openaq_df, max_distance_km=50):
    """
    OpenAQ 측정소를 TEMPO 격자에 최근접 매칭

    방법:
    1. 각 OpenAQ 측정소에 대해 가장 가까운 TEMPO 격자 찾기
    2. 시간별로 조인
    """
    print("🔗 TEMPO ↔ OpenAQ 공간 매칭 중...")

    # TEMPO 고유 격자점
    tempo_grids = tempo_df[['lat', 'lon']].drop_duplicates()
    print(f"   TEMPO 격자: {len(tempo_grids):,}개")

    # OpenAQ 고유 측정소
    openaq_stations = openaq_df[['lat', 'lon', 'location']].drop_duplicates()
    print(f"   OpenAQ 측정소: {len(openaq_stations):,}개\n")

    # 각 측정소에 대해 최근접 격자 찾기
    matches = []

    for idx, station in openaq_stations.iterrows():
        s_lat, s_lon = station['lat'], station['lon']

        # 모든 TEMPO 격자와의 거리 계산
        distances = haversine_distance(
            s_lat, s_lon,
            tempo_grids['lat'].values,
            tempo_grids['lon'].values
        )

        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist <= max_distance_km:
            tempo_grid = tempo_grids.iloc[min_dist_idx]

            matches.append({
                'openaq_location': station['location'],
                'openaq_lat': s_lat,
                'openaq_lon': s_lon,
                'tempo_lat': tempo_grid['lat'],
                'tempo_lon': tempo_grid['lon'],
                'distance_km': min_dist,
            })

        if (idx + 1) % 50 == 0:
            print(f"   진행: {idx+1}/{len(openaq_stations)}...")

    match_df = pd.DataFrame(matches)

    print(f"\n✅ 매칭 완료: {len(match_df)}개 측정소")
    print(f"   평균 거리: {match_df['distance_km'].mean():.1f} km")
    print(f"   최대 거리: {match_df['distance_km'].max():.1f} km\n")

    return match_df


def merge_tempo_openaq(tempo_df, openaq_df, match_df, time_tolerance='1H'):
    """
    시공간 조인

    전략:
    1. 공간: match_df 사용
    2. 시간: merge_asof (최근접 시간 매칭)
    """
    print("🔀 TEMPO + OpenAQ 시공간 조인 중...\n")

    # OpenAQ에 TEMPO 격자 좌표 추가
    openaq_mapped = openaq_df.merge(
        match_df[['openaq_location', 'tempo_lat', 'tempo_lon']],
        left_on='location',
        right_on='openaq_location',
        how='inner'
    )

    print(f"   OpenAQ (격자 매핑 후): {len(openaq_mapped):,}행")

    # 시간 정렬 (merge_asof 요구사항)
    tempo_df = tempo_df.sort_values('time')
    openaq_mapped = openaq_mapped.sort_values('time')

    # 격자별로 그룹화하여 조인
    merged_list = []

    for (lat, lon), group in openaq_mapped.groupby(['tempo_lat', 'tempo_lon']):
        # 해당 격자의 TEMPO 데이터
        tempo_grid = tempo_df[
            (tempo_df['lat'] == lat) &
            (tempo_df['lon'] == lon)
        ].copy()

        if tempo_grid.empty:
            continue

        # 최근접 시간 조인
        merged_grid = pd.merge_asof(
            tempo_grid,
            group[['time', 'pm25', 'location']],
            on='time',
            tolerance=pd.Timedelta(time_tolerance),
            direction='nearest'
        )

        merged_list.append(merged_grid)

    if not merged_list:
        print("⚠️  조인 결과 없음\n")
        return pd.DataFrame()

    merged_all = pd.concat(merged_list, ignore_index=True)

    # PM2.5 측정값 있는 행만 유지
    merged_all = merged_all.dropna(subset=['pm25'])

    print(f"✅ 최종 병합: {len(merged_all):,}행")
    print(f"   PM2.5 측정소: {merged_all['location'].nunique()}개")
    print(f"   시간 범위: {merged_all['time'].min()} ~ {merged_all['time'].max()}\n")

    return merged_all


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  OpenAQ PM2.5 + TEMPO Merge Pipeline")
    print("="*70)
    print()

    # 1. OpenAQ 데이터 수집
    raw_data = fetch_openaq_pm25(
        CONFIG['DATE_RANGE'][0],
        CONFIG['DATE_RANGE'][1],
        CONFIG['LOCATIONS']['state']
    )

    openaq_df = parse_openaq_data(raw_data)

    if openaq_df.empty:
        print("❌ OpenAQ 데이터 없음. 종료.\n")
        return

    # 2. TEMPO 데이터 로드
    print(f"📂 TEMPO 데이터 로딩: {CONFIG['TEMPO_FILE']}\n")

    if not Path(CONFIG['TEMPO_FILE']).exists():
        print(f"❌ TEMPO 파일 없음: {CONFIG['TEMPO_FILE']}")
        print("   → 먼저 tempo_ca_pipeline.py를 실행하세요.\n")
        return

    tempo_df = pd.read_parquet(CONFIG['TEMPO_FILE'])
    print(f"✅ TEMPO: {len(tempo_df):,}행 × {len(tempo_df.columns)}열\n")

    # 3. 공간 매칭
    match_df = match_nearest_grid(
        tempo_df,
        openaq_df,
        CONFIG['MAX_DISTANCE_KM']
    )

    if match_df.empty:
        print("❌ 매칭 결과 없음. 종료.\n")
        return

    # 4. 시공간 병합
    merged_df = merge_tempo_openaq(
        tempo_df,
        openaq_df,
        match_df
    )

    if merged_df.empty:
        print("❌ 병합 실패. 종료.\n")
        return

    # 5. 저장
    out_path = Path(CONFIG['OUTPUT_FILE'])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 저장 중: {out_path}\n")
    merged_df.to_parquet(out_path, compression='snappy', index=False)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"✅ 완료! ({size_mb:.1f} MB)\n")

    # 6. 요약
    print("="*70)
    print("  데이터 요약")
    print("="*70)
    print(merged_df.info())
    print()
    print("📊 PM2.5 통계:")
    print(merged_df['pm25'].describe())
    print()
    print("📋 샘플:")
    print(merged_df.head(10))


if __name__ == '__main__':
    main()
