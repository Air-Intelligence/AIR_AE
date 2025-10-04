import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from math import radians, cos, sin, sqrt, atan2
from datetime import datetime

# --------------------------------------------------------------------------
# FastAPI 앱을 시작합니다.
# --------------------------------------------------------------------------
app = FastAPI(
    title="미세먼지 API",
    description="특정 위치 주변의 PM2.5 데이터를 제공하는 API입니다.",
    version="1.0.0"
)

# --------------------------------------------------------------------------
# 두 지점 간의 거리를 계산하는 Haversine 공식 함수입니다.
# --------------------------------------------------------------------------
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위도/경도 좌표 간의 거리를 킬로미터(km) 단위로 계산합니다."""
    R = 6371  # 지구의 반경 (km)

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# --------------------------------------------------------------------------
# 데이터를 불러오는 함수입니다. (예시 데이터 사용)
# 실제 환경에서는 데이터베이스나 파일에서 데이터를 불러오도록 수정해야 합니다.
# --------------------------------------------------------------------------
def load_openaq_data() -> pd.DataFrame:
    """
    OpenAQ 데이터를 불러와 Pandas DataFrame으로 반환하는 함수입니다.
    이 함수는 실제 데이터 로딩 로직으로 대체되어야 합니다.
    현재는 시연을 위한 예시 데이터를 생성하여 반환합니다.
    """
    data = {
        'location_name': ['서울시청', '종로구', '강남역', '광화문', '여의도공원'],
        'lat': [37.5665, 37.5796, 37.4979, 37.5759, 37.5253],
        'lon': [126.9780, 126.9770, 127.0276, 126.9768, 126.9254],
        'pm25': [15.5, 12.3, 18.1, 14.0, 11.5],
        'time': pd.to_datetime([datetime.now()] * 5) # 현재 시각으로 통일
    }
    return pd.DataFrame(data)

# --------------------------------------------------------------------------
# API 엔드포인트: 특정 반경 내의 모든 관측소 데이터 반환
# --------------------------------------------------------------------------
@app.get("/api/pm25/in-range")
async def get_pm25_in_range(
        lat: float = Query(..., description="중심 위도 (예: 37.56)", example=37.56),
        lon: float = Query(..., description="중심 경도 (예: 126.97)", example=126.97),
        radius_km: float = Query(5.0, description="검색할 반경 (km 단위)", gt=0, example=5.0)
):
    """
    주어진 위도/경도를 중심으로 특정 반경(radius_km) 내에 있는
    모든 PM2.5 관측소 데이터를 반환합니다.
    """
    df = load_openaq_data()

    # 최신 시각 데이터만 필터링합니다.
    latest_time = df['time'].max()
    df_latest = df[df['time'] == latest_time].copy()

    if df_latest.empty:
        raise HTTPException(status_code=404, detail="사용 가능한 PM2.5 데이터가 없습니다.")

    # 각 관측소까지의 거리를 계산하여 'distance_km' 컬럼에 추가합니다.
    df_latest['distance_km'] = df_latest.apply(
        lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1
    )

    # 지정된 반경 내에 있는 관측소들만 필터링합니다.
    stations_in_range_df = df_latest[df_latest['distance_km'] <= radius_km].copy()

    # 결과를 사용자에게 편리하도록 가까운 순으로 정렬합니다.
    stations_in_range_df.sort_values('distance_km', inplace=True)

    # 응답 형식에 맞게 결과를 리스트로 변환합니다.
    results = [
        {
            "name": row['location_name'],
            "lat": row['lat'],
            "lon": row['lon'],
            "pm25": float(row['pm25']),
            "distance_km": round(float(row['distance_km']), 4), # 소수점 4자리까지 반올림
            "time": row['time'].isoformat(),
        }
        for _, row in stations_in_range_df.iterrows()
    ]

    return {
        "query_details": {
            "center_location": {"lat": lat, "lon": lon},
            "radius_km": radius_km
        },
        "station_count": len(results),
        "stations_in_range": results
    }
