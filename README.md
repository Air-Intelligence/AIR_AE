# Bay Area Air Quality Prediction & Real-Time API

**NASA TEMPO 위성 데이터 기반 PM2.5 예측 시스템 + 실시간 대기질 API**

---

## 📋 개요

캘리포니아 Bay Area의 대기질(PM2.5)을 NASA TEMPO 위성 데이터와 MERRA-2 기상 데이터를 활용하여 예측하는 end-to-end 머신러닝 파이프라인 및 실시간 API 서버입니다.

### 주요 특징
- ✅ **학습 파이프라인**: TEMPO V03 데이터 기반 6주 학습
- ✅ **실시간 API**: FastAPI 기반 대기질 데이터 제공 및 PM2.5 예측
- ✅ **다중 데이터 소스**: TEMPO NO₂/O₃ (NRT) + OpenAQ PM2.5 + AirNow
- ✅ **머신러닝 모델**: LightGBM 잔차 보정 모델
- ✅ **웹 시각화 준비**: RESTful API 엔드포인트 제공

---

## 📂 프로젝트 구조

```
프로젝트/
├── config.py                    # 전역 설정 (BBOX, 날짜, API 키, 경로)
├── utils.py                     # 공통 함수 (QC, 저장, 로드, 특성화)
│
├── scripts/                     # 데이터 파이프라인 스크립트
│   ├── download/               # 데이터 다운로드
│   │   ├── 01_download_openaq.py       # OpenAQ PM2.5 (학습용)
│   │   ├── 02_download_tempo.py        # TEMPO V03 (학습용)
│   │   ├── 02_download_tempo_nrt.py    # TEMPO V04 NRT (실시간)
│   │   ├── 02_download_openaq_nrt.py   # OpenAQ NRT (실시간)
│   │   ├── 03_download_merra2.py       # MERRA-2 기상 데이터
│   │   ├── download_o3_static.py       # O3 정적 데이터
│   │   ├── download_openaq_latest.py   # OpenAQ 최신 관측값
│   │   └── download_airnow.py          # AirNow API
│   │
│   ├── preprocess/             # 전처리 및 특성 엔지니어링
│   │   ├── 04_preprocess_merge.py      # 전처리 & 병합
│   │   ├── 04_preprocess_nrt.py        # NRT 데이터 전처리
│   │   ├── 05_join_labels.py           # PM2.5 라벨 조인
│   │   └── 06_feature_engineering.py   # 특성 생성 (래그, 시간)
│   │
│   └── train/                  # 모델 학습 및 평가
│       ├── 07_train_baseline.py        # Baseline (Persistence)
│       ├── 08_train_residual.py        # LightGBM 잔차 모델
│       └── 09_evaluate.py              # 평가 & 시각화
│
├── api/                         # FastAPI 서버
│   └── open_aq.py              # 실시간 대기질 API
│
├── src/                         # 핵심 모듈
│   ├── features.py             # 특성 추출 함수
│   └── model.py                # 예측 모델 래퍼
│
├── analysis/                    # 분석 스크립트
│   ├── analyze_nrt.py          # NRT 데이터 분석
│   └── validate_pipeline.py    # 파이프라인 검증
│
├── tests/                       # 테스트 코드
│
├── /mnt/data/                  # 데이터 저장소 (1TB HDD)
│   ├── raw/                    # 원시 데이터
│   │   ├── OpenAQ/
│   │   ├── TEMPO_NO2/
│   │   ├── TEMPO_O3/
│   │   ├── tempo_v04/          # NRT V04 데이터
│   │   └── MERRA2/
│   ├── features/               # 전처리된 특성
│   │   ├── tempo/train_6w/     # 학습용 TEMPO
│   │   ├── tempo/nrt_roll3d/   # NRT TEMPO (72시간)
│   │   ├── merra2/
│   │   └── openaq/
│   ├── tables/                 # Parquet 테이블
│   ├── models/                 # 학습된 모델 (pkl)
│   └── plots/                  # 시각화 결과 (png)
└── README.md                   # 본 문서
```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

#### Python 패키지 설치
```bash
# 핵심 패키지
pip install pandas numpy xarray earthaccess lightgbm scikit-learn matplotlib seaborn pyarrow requests

# API 서버용
pip install fastapi uvicorn joblib scipy

# 선택적 (성능 향상)
pip install xgboost  # XGBoost 사용 시
```

#### Earthdata 인증 설정
NASA Earthdata 계정 필요 → [가입](https://urs.earthdata.nasa.gov/users/new)

**Windows**: `C:\Users\<사용자>\.netrc` 파일 생성 (확장자 없음)
```
machine urs.earthdata.nasa.gov
    login YOUR_USERNAME
    password YOUR_PASSWORD
```

**Linux/Mac**: `~/.netrc` 파일 생성 후 권한 설정
```bash
chmod 600 ~/.netrc
```

---

### 2️⃣ 설정 조정 (선택)

`config.py` 파일에서 다음을 조정 가능:

```python
# 기간 (기본: 6주)
DATE_START = datetime(2024, 8, 1)
DATE_END = datetime(2024, 9, 15)

# 공간 (기본: Bay Area)
BBOX = {
    'west': -122.75,
    'south': 36.9,
    'east': -121.6,
    'north': 38.3
}

# 모델 선택 (기본: LightGBM만)
USE_LIGHTGBM = True
USE_XGBOOST = False  # True로 설정 시 XGBoost 추가
USE_RANDOMFOREST = False
USE_ENSEMBLE = False  # 다중 모델 평균
```

---

### 3️⃣ 학습 파이프라인 실행

#### 전체 학습 (2023년 데이터, 6주)
```bash
# 1. 학습 데이터 다운로드
python scripts/download/01_download_openaq.py
python scripts/download/02_download_tempo.py  # TEMPO V03
python scripts/download/03_download_merra2.py

# 2. 전처리 및 특성 생성
python scripts/preprocess/04_preprocess_merge.py
python scripts/preprocess/05_join_labels.py
python scripts/preprocess/06_feature_engineering.py

# 3. 모델 학습
python scripts/train/07_train_baseline.py
python scripts/train/08_train_residual.py  # → residual_lgbm.pkl 생성

# 4. 평가
python scripts/train/09_evaluate.py
```

### 4️⃣ 실시간 API 서버 실행

#### NRT 데이터 다운로드
```bash
# O3 정적 데이터 (최근 3일)
python scripts/download/download_o3_static.py

# TEMPO NRT 데이터 (V04, 최근 3일)
python scripts/download/02_download_tempo_nrt.py

# OpenAQ NRT 데이터
python scripts/download/02_download_openaq_nrt.py

# OpenAQ 최신 관측값
python scripts/download/download_openaq_latest.py

# NRT 데이터 전처리
python scripts/preprocess/04_preprocess_nrt.py
```

#### FastAPI 서버 시작
```bash
# 개발 모드 (자동 리로드)
python open_aq.py

# 또는
uvicorn open_aq:app --host 0.0.0.0 --port 8000 --reload

# 프로덕션 모드
uvicorn open_aq:app --host 0.0.0.0 --port 8000
```

서버 접속: http://localhost:8000

---

### 5️⃣ API 엔드포인트

#### TEMPO 위성 데이터
- `GET /api/stats` - 데이터셋 통계
- `GET /api/latest?variable=no2` - 최신 NO₂/O₃ 데이터
- `GET /api/timeseries?lat=37.77&lon=-122.41&variable=no2` - 시계열 조회
- `GET /api/heatmap?variable=no2&time=2025-10-03T23:00:00` - 히트맵 데이터
- `GET /api/grid?lat_min=37.5&lat_max=38.0&lon_min=-122.5&lon_max=-122.0&variable=no2` - 그리드 데이터

#### OpenAQ PM2.5 관측
- `GET /api/pm25/stations` - 모니터링 스테이션 목록
- `GET /api/pm25/latest` - 최신 PM2.5 관측값
- `GET /api/pm25/timeseries?location_name=San Francisco` - 스테이션별 시계열
- `GET /api/pm25/latest_csv` - 최신 관측값 (CSV 기반)

#### PM2.5 예측
- `POST /api/predict/pm25` - LGBM 모델 예측
  ```json
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "when": "2025-10-03T23:00:00"  // optional, 생략 시 현재
  }
  ```

- `POST /api/predict` - 상세 예측 (신뢰구간 포함)
  ```json
  {
    "lat": 37.7749,
    "lon": -122.4194,
    "city": "San Francisco"
  }
  ```

- `GET /api/compare` - 예측 vs 관측 비교

#### 통합 데이터
- `GET /api/combined/latest` - TEMPO + OpenAQ 최신 데이터

### 6️⃣ 학습 결과 확인

#### 평가 지표
```bash
cat /mnt/data/models/evaluation_metrics.json
```

#### 시각화
- `/mnt/data/plots/timeseries_pred_vs_obs.png` - 시계열 비교
- `/mnt/data/plots/scatter_pred_vs_obs.png` - 산점도 (관측 vs 예측)
- `/mnt/data/plots/residuals_histogram.png` - 잔차 분포
- `/mnt/data/plots/feature_importance.png` - 특성 중요도

---

## 📊 데이터 소스

| 데이터 | 변수 | 해상도 | 용량 | 용도 |
|--------|------|--------|------|------|
| **TEMPO L3 V03** | NO₂, O₃ | 시간별, ~5 km | ~4 GB (6주) | 학습 |
| **TEMPO L3 V04 NRT** | NO₂, O₃ | 시간별, ~5 km | ~200 MB (3일) | 실시간 예측 |
| **MERRA-2** | PBLH, U10M, V10M | 1시간, 0.5° × 0.625° | ~1 GB (6주) | 학습 (기상) |
| **OpenAQ** | PM2.5 | 1시간, 지점별 | ~10 MB | 학습 + 검증 |
| **AirNow** | PM2.5 | 1시간, 지점별 | API | 실시간 검증 |

---

## 🔧 문제 해결

### 다운로드 실패
**증상**: `Authentication failed` 또는 `No granules found`

**해결**:
1. `.netrc` 파일 경로/권한 확인
2. Earthdata 계정에서 ASDC/GES DISC 승인 확인
3. `config.py`에서 `DATE_START/END` 범위 확인

### 메모리 부족
**증상**: `MemoryError` 또는 느린 처리

**해결**:
1. `config.py`에서 기간을 6주 → 4주로 축소
2. `04_preprocess_merge.py`에서 파일별 처리로 수정
3. BBOX 범위 축소 (Bay Area → San Francisco만)

### 성능 부족
**증상**: MAE > 10 µg/m³ 또는 R² < 0.5

**해결**:
1. `config.py`에서 `ENABLE_CLDO4 = True` (클라우드 데이터 추가)
2. `USE_XGBOOST = True` 또는 `USE_ENSEMBLE = True`
3. 기간을 8주로 확장

---

## 📈 시스템 아키텍처

### 학습 파이프라인 (Offline)
```
1. 데이터 수집
   ├─ TEMPO V03 (2023-09-01 ~ 2023-10-15, 6주)
   │  ├─ NO₂ tropospheric column
   │  └─ O₃ total column
   ├─ MERRA-2 기상 (PBLH, U10M, V10M)
   └─ OpenAQ PM2.5 (Bay Area 5개 도시)

2. 전처리
   ├─ BBOX 서브셋팅 (Bay Area)
   ├─ Tidy 변환 (time, lat, lon, variable)
   ├─ 시간 정렬 (1시간 간격)
   ├─ 병합 (spatial join)
   ├─ QC (음수/비현실값 제거, Winsorization)
   └─ Parquet 저장

3. 특성 엔지니어링
   ├─ 래그 특성 (t-1, t-3 시간)
   ├─ 시간 인코딩 (hour, day of week)
   └─ PM2.5 라벨 조인 (KD-tree, 10km)

4. 모델 학습
   ├─ Baseline: Persistence (t-1)
   ├─ Residual: PM2.5 - PM2.5(t-1)
   ├─ LightGBM 학습 (4주 학습, 2주 검증)
   └─ 모델 저장 (residual_lgbm.pkl)

5. 평가
   ├─ R², MAE, RMSE
   └─ 시각화 (시계열, 산점도, 특성 중요도)
```

### 실시간 API (Online)
```
1. NRT 데이터 수집 (정기 실행)
   ├─ TEMPO V04 NRT (최근 3일, cron)
   ├─ OpenAQ NRT (최근 3일)
   └─ OpenAQ Latest (API, 1시간마다)

2. 전처리
   ├─ Rolling 3일 윈도우
   ├─ Parquet 업데이트
   └─ 캐시 무효화

3. FastAPI 서버
   ├─ 시작 시 데이터 로드
   ├─ 파일 변경 시 자동 리로드
   └─ RESTful API 제공

4. 예측 엔드포인트
   ├─ 위치(lat, lon) 입력
   ├─ 최신 TEMPO NO₂/O₃ 추출
   ├─ OpenAQ PM2.5(t-1) 추출
   ├─ LGBM 모델 추론
   └─ PM2.5 예측값 반환
```

---

## 🎯 성능 목표

| 지표 | 최소 목표 | 이상적 목표 |
|------|-----------|-------------|
| **MAE** | < 10 µg/m³ | < 8 µg/m³ |
| **R²** | > 0.5 | > 0.65 |
| **개선율** | Baseline 대비 10%+ | Baseline 대비 20%+ |

---

## 📝 인용

이 파이프라인에서 사용하는 데이터:

- **TEMPO**: NASA TEMPO Mission, ASDC
- **MERRA-2**: NASA GMAO, GES DISC
- **OpenAQ**: OpenAQ API v2

---

## 🤝 기여

해커톤 팀원:
- 데이터 수집: [이름]
- 전처리: [이름]
- 모델링: [이름]
- 시각화: [이름]

---

## 📄 라이선스

MIT License - 교육 및 연구 목적으로 자유롭게 사용 가능

---

## 🔑 주요 기술 스택

| 구분 | 기술 |
|------|------|
| **데이터 수집** | earthaccess, requests, OpenAQ API, AirNow API |
| **데이터 처리** | pandas, xarray, numpy, scipy |
| **머신러닝** | LightGBM, scikit-learn |
| **API 서버** | FastAPI, uvicorn, pydantic |
| **저장소** | Parquet (pyarrow), joblib |
| **시각화** | matplotlib, seaborn |

---

## 🚀 구현 완료 기능

### ✅ 학습 파이프라인
- [x] TEMPO V03 데이터 다운로드 (NO₂, O₃)
- [x] MERRA-2 기상 데이터 수집
- [x] OpenAQ PM2.5 수집 및 전처리
- [x] 공간-시간 정합 (KD-tree)
- [x] 특성 엔지니어링 (래그, 시간 인코딩)
- [x] LightGBM 잔차 모델 학습
- [x] 평가 및 시각화

### ✅ 실시간 API
- [x] TEMPO V04 NRT 데이터 수집
- [x] OpenAQ NRT 데이터 수집
- [x] FastAPI 서버 구축
- [x] 14개 API 엔드포인트 구현
  - TEMPO 위성 데이터 조회 (5개)
  - OpenAQ 관측 데이터 조회 (4개)
  - PM2.5 예측 (3개)
  - 통합 데이터 (1개)
- [x] 파일 캐싱 및 자동 리로드
- [x] CORS 설정

### ✅ 데이터 관리
- [x] 1TB HDD 활용 (/mnt/data)
- [x] Parquet 압축 저장
- [x] 디스크 사용량 모니터링
- [x] V03 학습 데이터와 V04 NRT 데이터 분리

---

## 🚦 향후 개선 방향

### 우선순위 높음
- [ ] 프론트엔드 대시보드 (React/Leaflet)
- [ ] Docker 컨테이너화
- [ ] 자동화된 NRT 데이터 업데이트 (cron/scheduler)
- [ ] 모델 성능 모니터링 및 재학습 파이프라인

### 우선순위 중간
- [ ] SHAP 값 기반 모델 해석
- [ ] 공간 홀드아웃 검증
- [ ] 앙상블 모델 (XGBoost, Random Forest)
- [ ] 예측 불확실성 정량화

### 우선순위 낮음
- [ ] MERRA-2 NRT 통합
- [ ] TEMPO CLDO4 (구름) 데이터 활용
- [ ] 추가 오염물질 예측 (NO₂, O₃)
- [ ] 모바일 앱 개발

---

## 📝 개발 이력

| 날짜 | 주요 변경사항 |
|------|--------------|
| 2025-10-05 | FastAPI 서버 구축, NRT 데이터 파이프라인 완성 |
| 2025-10-04 | TEMPO V04 NRT 지원, OpenAQ Latest API 추가 |
| 2025-10-03 | LightGBM 학습 완료, 평가 스크립트 작성 |
| 2025-10-02 | 특성 엔지니어링, 데이터 전처리 |
| 2025-10-01 | 프로젝트 구조 설계, TEMPO V03 다운로드 |

---

**개발 기간**: 2025-10-01 ~ 2025-10-05
**버전**: 1.0.0
**프로젝트**: NASA 해커톤 2025 - Air Quality Intelligence
