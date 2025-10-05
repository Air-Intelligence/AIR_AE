# TEMPO L3 Pipeline - 성능 최적화 & 트러블슈팅 가이드

## 🚀 빠른 시작 (해커톤 2일 완성)

### Phase 1: 테스트 실행 (30분)
```bash
# 환경 설정
pip install earthaccess xarray netCDF4 h5netcdf pandas pyarrow requests

# 메인 스크립트에서 다음 설정 사용:
# CONFIG = {
#     'DATE_RANGE': ('2024-06-01', '2024-06-28'),  # 4주만
#     'VARS': ['NO2', 'O3', 'CLOUD'],              # 3변수만
#     'VERSION': 'V04',                             # 최신 버전만
#     'THREADS': 8,
# }

python tempo_ca_pipeline.py
```

**예상 결과:**
- 다운로드: ~2-5GB
- 최종 Parquet: ~50-100MB
- 시간: ~20-40분 (네트워크 속도 의존)

---

### Phase 2: 전체 실행 (2-3시간)
```python
# CONFIG 수정:
'DATE_RANGE': ('2024-06-01', '2024-08-31'),  # 3개월
'VARS': ['NO2', 'HCHO', 'O3', 'CLOUD'],      # 4변수
'THREADS': 12,  # VM 코어 수에 맞게
```

**예상 결과:**
- 다운로드: ~8-15GB
- 최종 Parquet: ~300-600MB
- 시간: ~2-3시간

---

## ⚡ 성능 최적화 체크리스트

### 1. 데이터 범위 축소 (우선순위 ★★★)

#### 변수 개수 줄이기
```python
# ❌ 느림 (4변수)
'VARS': ['NO2', 'HCHO', 'O3', 'CLOUD']

# ✅ 빠름 (2변수)
'VARS': ['NO2', 'O3']
```
**효과:** 시간 50% 단축, 용량 50% 감소

---

#### 기간 단축
```python
# ❌ 느림 (6개월)
'DATE_RANGE': ('2024-01-01', '2024-06-30')

# ✅ 빠름 (1개월)
'DATE_RANGE': ('2024-06-01', '2024-06-30')
```
**효과:** granule 수에 비례 (1개월 → ~3,000개 파일)

---

#### BBOX 축소 (도시권만)
```python
# ❌ 전체 캘리포니아
'BBOX': (-125, 32, -113, 42)

# ✅ Bay Area만
'BBOX': (-122.5, 37.2, -121.5, 38.0)

# ✅ LA Basin만
'BBOX': (-118.7, 33.7, -117.6, 34.3)
```
**효과:** 용량 80% 감소, 전처리 3배 빠름

---

### 2. 병렬 처리 최적화 (우선순위 ★★)

#### 스레드 수 조정
```python
# VM 코어 수 확인
import os
cores = os.cpu_count()
print(f"사용 가능 코어: {cores}")

# 권장 설정
'THREADS': min(cores, 12)  # 최대 12
```

**참고:**
- AWS/GCP VM: 8-16 코어 권장
- 로컬 PC: 4-8 코어
- 너무 높으면 네트워크 병목 발생

---

#### 메모리 부족 시
```python
# xarray에서 chunk 사용
ds = xr.open_dataset(fpath, chunks={'time': 10, 'y': 100, 'x': 100})

# 또는 파일 개수 제한
granules = granules[:100]  # 최대 100개만
```

---

### 3. 저장 최적화 (우선순위 ★★)

#### Parquet만 사용 (CSV 제외)
```python
# ❌ CSV 전체 저장 (느림)
df.to_csv('output.csv')  # 5GB, 5분 소요

# ✅ Parquet 압축 (빠름)
df.to_parquet('output.parquet', compression='snappy')  # 200MB, 10초
```

---

#### 샘플 데이터만 CSV로
```python
# 전체는 Parquet, 확인용만 CSV
df.to_parquet('full.parquet')
df.head(10000).to_csv('sample.csv')
```

---

### 4. 버전 선택 (우선순위 ★)

#### V04만 사용 (최신)
```python
# ✅ V04만 (2023-08-01 ~ 현재)
'VERSION': 'V04'

# ❌ V03 + V04 병행 (충돌 가능)
```

**참고:**
- V03: 2023-08-01 ~ 2024-05-31 (구버전)
- V04: 2023-08-01 ~ 현재 (PROVISIONAL)
- NRT: Near Real-Time (최근 2주, 실시간)

---

## 🐛 트러블슈팅

### 1. EDL 로그인 실패

#### 증상
```
❌ 인증 실패! EDL 계정 확인 필요
```

#### 해결 방법

**방법 1: 대화형 로그인**
```python
auth = earthaccess.login(strategy='interactive')
# → 사용자명/비밀번호 입력 프롬프트
```

**방법 2: 환경변수**
```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password
```

**방법 3: .netrc 파일 생성**
```bash
# ~/.netrc 파일 생성 (Linux/Mac)
echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" > ~/.netrc
chmod 600 ~/.netrc

# Windows: C:\Users\YOUR_NAME\_netrc
```

**방법 4: 계정 신규 생성**
1. https://urs.earthdata.nasa.gov/users/new 방문
2. 가입 후 이메일 인증
3. 5분 후 재시도

---

### 2. Granule 검색 결과 0개

#### 증상
```
🔍 검색 중: NO2 (TEMPO_NO2_L3 V04)
   → 발견: 0개 파일
```

#### 원인 & 해결

**원인 1: 날짜 범위 초과**
```python
# ❌ V04는 2023-08-01부터 시작
'DATE_RANGE': ('2023-01-01', '2023-12-31')

# ✅ 수정
'DATE_RANGE': ('2023-08-01', '2024-12-31')
```

**원인 2: 버전 불일치**
```python
# ❌ V05는 아직 없음
'VERSION': 'V05'

# ✅ V03 또는 V04만 사용
'VERSION': 'V04'
```

**원인 3: BBOX 오류**
```python
# ❌ 순서 틀림
'BBOX': (32, -125, 42, -113)  # (S, W, N, E)

# ✅ 올바른 순서: (W, S, E, N)
'BBOX': (-125, 32, -113, 42)
```

**디버깅 팁:**
```python
# CMR 직접 검색해보기
results = earthaccess.search_data(
    short_name='TEMPO_NO2_L3',
    version='V04',
)
print(f"전체 granule 수: {len(results)}")
```

---

### 3. netCDF 변수명 불일치

#### 증상
```
KeyError: 'vertical_column_troposphere'
```

#### 원인
버전/제품마다 변수명이 다를 수 있음

#### 해결: 변수 탐색
```python
import xarray as xr

# 파일 열기
ds = xr.open_dataset('TEMPO_NO2_20240601_example.nc')

# 모든 변수명 확인
print("사용 가능 변수:")
for var in ds.data_vars:
    print(f"  - {var}: {ds[var].dims}")

# 실제 변수명으로 수정
```

**일반적인 변수명:**
| 제품 | V03 변수명 | V04 변수명 |
|------|------------|------------|
| NO2  | `vertical_column_troposphere` | `tropospheric_vertical_column` |
| HCHO | `vertical_column` | `formaldehyde_vertical_column` |
| O3   | `vertical_column` | `ozone_total_vertical_column` |
| Cloud| `cloud_fraction` | `effective_cloud_fraction` |

**스크립트는 자동 탐색 포함:**
```python
# VARIABLE_MAPPINGS에 후보 목록 정의
# 순차 시도하여 자동 매칭
```

---

### 4. 메모리 부족 (MemoryError)

#### 증상
```
MemoryError: Unable to allocate array
```

#### 해결 방법

**방법 1: Chunk 사용**
```python
# subset_and_convert() 함수 수정
ds = xr.open_dataset(fpath, chunks={'time': 10, 'y': 100, 'x': 100})
```

**방법 2: 배치 처리**
```python
# 월별로 나눠서 처리
for month in ['06', '07', '08']:
    CONFIG['DATE_RANGE'] = (f'2024-{month}-01', f'2024-{month}-30')
    # ... 처리
```

**방법 3: VM 메모리 증설**
- AWS/GCP: 16GB → 32GB 인스턴스로 변경

---

### 5. 다운로드 속도 느림

#### 증상
```
PROCESSING TASKS | : 10%|█ | 100/1000 [30:00<?, ?it/s]
```

#### 해결

**방법 1: 스레드 증가**
```python
'THREADS': 12  # 4 → 8 → 12로 점진적 증가
```

**방법 2: VM 네트워크 속도 확인**
```bash
# 다운로드 속도 테스트
curl -o /dev/null https://data.asdc.earthdata.nasa.gov/test_file
```

**방법 3: Retry 간격 조정**
```python
# download_granules() 함수에 추가
earthaccess.download(
    granules,
    str(out_dir),
    threads=CONFIG['THREADS'],
    retry=3,  # 재시도 횟수
)
```

**방법 4: 지역 선택**
- AWS us-west-2 (오레곤): NASA ASDC와 가까움
- 한국 VM: 속도 느림 → VPN/프록시 고려

---

### 6. OpenAQ API 오류

#### 증상
```
⚠️  API 오류: Connection timeout
```

#### 해결

**방법 1: Timeout 증가**
```python
response = requests.get(base_url, params=params, timeout=60)  # 30 → 60초
```

**방법 2: 페이지 제한**
```python
# 전체 기간 대신 월별로
'DATE_RANGE': ('2024-06-01', '2024-06-30')  # 1개월씩
```

**방법 3: OpenAQ 스킵**
```bash
# TEMPO 데이터만 먼저 수집
python tempo_ca_pipeline.py

# PM2.5는 나중에
# python openaq_pm25.py
```

---

## 📊 예상 용량 & 시간 가이드

### 용량 예측표

| 기간 | 변수 수 | 원본 netCDF | 최종 Parquet | CSV (전체) |
|------|---------|-------------|--------------|-----------|
| 1주  | 2개     | ~1GB        | ~30MB        | ~500MB    |
| 4주  | 3개     | ~5GB        | ~150MB       | ~2GB      |
| 3개월| 4개     | ~15GB       | ~500MB       | ~8GB      |
| 6개월| 4개     | ~30GB       | ~1GB         | ~15GB     |

---

### 시간 예측표 (THREADS=8 기준)

| 작업 단계        | 1주  | 4주   | 3개월 |
|------------------|------|-------|-------|
| Granule 검색     | 10초 | 20초  | 30초  |
| 다운로드         | 5분  | 20분  | 1시간 |
| BBOX 서브셋      | 2분  | 10분  | 30분  |
| Tidy 변환        | 1분  | 5분   | 20분  |
| 변수 병합        | 10초 | 30초  | 2분   |
| Parquet 저장     | 5초  | 20초  | 1분   |
| **총계**         | ~10분| ~40분 | ~2시간|

---

## 🎯 해커톤 최적 전략

### Day 1 (데이터 수집)
```python
# 아침: 테스트 실행 (4주, 2변수)
'DATE_RANGE': ('2024-06-01', '2024-06-28')
'VARS': ['NO2', 'O3']
# → 30분 소요, 결과 확인

# 오후: 전체 실행 시작 (3개월, 4변수)
'DATE_RANGE': ('2024-06-01', '2024-08-31')
'VARS': ['NO2', 'HCHO', 'O3', 'CLOUD']
# → 2-3시간 소요, 백그라운드 실행

# 저녁: OpenAQ PM2.5 수집
python openaq_pm25.py
# → 30분 소요
```

### Day 2 (모델 학습 & 시연)
- 오전: ML 모델 학습 (Parquet 로드 → 학습)
- 오후: 시각화 & 대시보드
- 저녁: 시연 준비

---

## 💡 추가 팁

### 1. 백그라운드 실행 (Linux/Mac)
```bash
nohup python tempo_ca_pipeline.py > log.txt 2>&1 &
tail -f log.txt  # 로그 실시간 확인
```

### 2. 진행 상황 저장
```python
# 각 변수 처리 후 중간 저장
df.to_parquet(f'{var}_intermediate.parquet')
# → 중단 시 재개 가능
```

### 3. Docker 컨테이너 (재현성)
```dockerfile
FROM python:3.10
RUN pip install earthaccess xarray netCDF4 pandas pyarrow
COPY tempo_ca_pipeline.py /app/
CMD ["python", "/app/tempo_ca_pipeline.py"]
```

### 4. GPU 불필요
- TEMPO 다운로드/전처리: CPU만 사용
- ML 학습 단계에서만 GPU 고려

---

## 🆘 긴급 문제 해결

### 해커톤 중 막혔을 때

1. **BBOX 축소** → Bay Area만 (5분 내 다운로드)
2. **기간 축소** → 1주만 (10분 내 완료)
3. **변수 축소** → NO2만 (데모용 충분)
4. **OpenAQ 스킵** → TEMPO만으로 EDA 진행

### 최소 실행 설정 (긴급용)
```python
CONFIG = {
    'BBOX': (-122.5, 37.2, -121.5, 38.0),  # SF만
    'DATE_RANGE': ('2024-06-01', '2024-06-07'),  # 1주
    'VARS': ['NO2'],  # 1변수
    'VERSION': 'V04',
    'THREADS': 4,
    'RESAMPLE': None,  # 리샘플 스킵
}
```
**→ 5분 내 완료 보장**

---

## 📚 참고 링크

- **TEMPO 공식 문서**: https://tempo.si.edu/
- **earthaccess 문서**: https://earthaccess.readthedocs.io/
- **ASDC Data Portal**: https://asdc.larc.nasa.gov/project/TEMPO
- **OpenAQ API**: https://docs.openaq.org/
- **NASA Earthdata 가입**: https://urs.earthdata.nasa.gov/

---

**마지막 업데이트**: 2024-06-01
**작성자**: TEMPO Hackathon Pipeline
