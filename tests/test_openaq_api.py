"""
OpenAQ API v3 응답 구조 확인 테스트

목적:
- /v3/sensors/{id}/measurements 엔드포인트의 실제 응답 JSON 확인
- 필드명 및 중첩 구조 파악
- 02_download_openaq_nrt.py 수정을 위한 사전 조사
"""
import requests
import json
from datetime import datetime, timedelta, timezone
import config

# ============================================================================
# OpenAQ API v3 설정
# ============================================================================
BASE_URL = "https://api.openaq.org/v3"
HEADERS = {"X-API-Key": config.OPENAQ_API_KEY}

# ============================================================================
# 1. 테스트용 센서 찾기 (Bay Area에서 하나만)
# ============================================================================
print("="*70)
print("OpenAQ API v3 응답 구조 테스트")
print("="*70)

# Bay Area BBOX에서 센서 1개 조회
bbox_str = f"{config.BBOX['west']},{config.BBOX['south']},{config.BBOX['east']},{config.BBOX['north']}"
locations_url = f"{BASE_URL}/locations"
params = {
    'bbox': bbox_str,
    'parameters_id': 2,  # PM2.5
    'limit': 1  # 1개만
}

print("\n[1] 테스트용 센서 조회 중...")
response = requests.get(locations_url, headers=HEADERS, params=params, timeout=30)
response.raise_for_status()
data = response.json()

# 첫 번째 센서 추출
if 'results' not in data or len(data['results']) == 0:
    print("❌ No sensors found in Bay Area")
    exit(1)

location = data['results'][0]
sensor_id = None

if 'sensors' in location:
    for sensor in location['sensors']:
        if sensor.get('parameter', {}).get('id') == 2:
            sensor_id = sensor['id']
            break

if not sensor_id:
    print("❌ No PM2.5 sensor found")
    exit(1)

print(f"✓ 테스트 센서 ID: {sensor_id}")
print(f"  관측소 이름: {location.get('name', 'Unknown')}")
print(f"  위치: ({location.get('coordinates', {}).get('latitude')}, {location.get('coordinates', {}).get('longitude')})")

# ============================================================================
# 2. 측정값 조회 (최근 24시간)
# ============================================================================
print(f"\n[2] 센서 {sensor_id}의 측정값 조회 중 (최근 24시간)...")

# 시간 범위 설정
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

# API 호출
measurements_url = f"{BASE_URL}/sensors/{sensor_id}/measurements"
params = {
    'date_from': start_time.isoformat(),
    'date_to': end_time.isoformat(),
    'limit': 100  # 100개만 (테스트용)
}

print(f"  URL: {measurements_url}")
print(f"  시간 범위: {start_time.isoformat()} ~ {end_time.isoformat()}")

response = requests.get(measurements_url, headers=HEADERS, params=params, timeout=60)
print(f"  HTTP 상태 코드: {response.status_code}")

if response.status_code != 200:
    print(f"❌ API 호출 실패: {response.status_code}")
    print(f"  응답: {response.text}")
    exit(1)

data = response.json()

# ============================================================================
# 3. 응답 구조 분석
# ============================================================================
print("\n" + "="*70)
print("응답 JSON 구조")
print("="*70)

# Pretty print 전체 JSON (첫 2개 결과만)
if 'results' in data and len(data['results']) > 0:
    sample_data = {
        'meta': data.get('meta', {}),
        'results': data['results'][:2]  # 처음 2개만
    }
    print(json.dumps(sample_data, indent=2, ensure_ascii=False))
else:
    print(json.dumps(data, indent=2, ensure_ascii=False))

# ============================================================================
# 4. 필드 분석
# ============================================================================
print("\n" + "="*70)
print("필드 분석")
print("="*70)

print("\n[Meta 정보]")
if 'meta' in data:
    for key, value in data['meta'].items():
        print(f"  {key}: {value}")

print("\n[Results 구조]")
if 'results' in data and len(data['results']) > 0:
    first_result = data['results'][0]
    print(f"  총 결과 수: {len(data['results'])}개")
    print(f"  첫 번째 결과 키: {list(first_result.keys())}")

    print("\n[첫 번째 측정값 상세]")
    for key, value in first_result.items():
        if isinstance(value, dict):
            print(f"  {key}: (dict)")
            for sub_key, sub_value in value.items():
                print(f"    └─ {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")

# ============================================================================
# 5. 파싱 테스트
# ============================================================================
print("\n" + "="*70)
print("파싱 테스트 (02_download_openaq_nrt.py용 코드)")
print("="*70)

if 'results' in data and len(data['results']) > 0:
    print("\n# 추천 파싱 코드:")
    print("for m in data['results']:")

    # 시간 필드 찾기
    first = data['results'][0]
    if 'datetime' in first:
        if isinstance(first['datetime'], dict):
            if 'utc' in first['datetime']:
                print("    time = m['datetime']['utc']")
            else:
                print(f"    # datetime 키: {list(first['datetime'].keys())}")
        else:
            print("    time = m['datetime']")
    elif 'period' in first:
        print("    time = m['period']['datetimeFrom']['utc']  # 또는 적절한 경로")

    # 값 필드 찾기
    if 'value' in first:
        print("    pm25 = m['value']")

    print("    sensor_id = {sensor_id}  # URL에서 가져옴")

    print("\n예시 데이터:")
    for i, result in enumerate(data['results'][:3]):
        time_val = "?"
        if 'datetime' in result:
            time_val = result['datetime']['utc'] if isinstance(result['datetime'], dict) else result['datetime']
        elif 'period' in result:
            time_val = result.get('period', {}).get('datetimeFrom', {}).get('utc', '?')

        pm25_val = result.get('value', '?')
        print(f"  [{i+1}] time={time_val}, pm25={pm25_val}")

else:
    print("❌ No results in response")

print("\n" + "="*70)
print("테스트 완료!")
print("="*70)
