# Bay Area Air Quality Prediction Pipeline

**2일 해커톤용 TEMPO 위성 데이터 기반 PM2.5 예측 시스템**

---

## 📋 개요

캘리포니아 Bay Area의 대기질(PM2.5)을 NASA TEMPO 위성 데이터와 MERRA-2 기상 데이터를 활용하여 예측하는 end-to-end 머신러닝 파이프라인입니다.

### 주요 특징
- ✅ **최소 다운로드**: NO₂/O₃ 6주치만 우선 사용 (~ 5 GB)
- ✅ **빠른 실행**: 병렬 다운로드, Parquet 압축, 모듈화 설계
- ✅ **다중 모델**: LightGBM (기본) + XGBoost/RandomForest 옵션
- ✅ **잔차 보정**: Persistence 베이스라인 대비 개선
- ✅ **자동 평가**: 시각화 + 지표 산출 자동화

---

## 📂 프로젝트 구조

```
프로젝트/
├── config.py                    # 전역 설정 (BBOX, 날짜, 경로)
├── utils.py                     # 공통 함수 (QC, 저장, 로드, 특성화)
├── 01_download_openaq.py        # OpenAQ PM2.5 다운로드
├── 02_download_tempo.py         # TEMPO NO₂/O₃ 다운로드
├── 03_download_merra2.py        # MERRA-2 PBLH/U10M/V10M 다운로드
├── 04_preprocess_merge.py       # 전처리 & 병합
├── 05_feature_engineering.py    # 특성 생성 (래그, MA, 시간 인코딩)
├── 06_train_baseline.py         # Baseline (Persistence) 학습
├── 07_train_residual.py         # LightGBM/XGBoost/RF 학습
├── 08_evaluate.py               # 평가 & 시각화
├── run_all.bat                  # 전체 실행 스크립트 (Windows)
├── README.md                    # 본 문서
├── raw/                         # 원시 데이터 저장
│   ├── OpenAQ/
│   ├── TEMPO_NO2/
│   ├── TEMPO_O3/
│   └── MERRA2/
├── tables/                      # 처리된 데이터 (Parquet)
├── models/                      # 학습된 모델 (pkl)
└── plots/                       # 시각화 결과 (png)
```

---

## 🚀 빠른 시작

### 1️⃣ 환경 설정

#### Python 패키지 설치
```bash
pip install pandas numpy xarray earthaccess lightgbm scikit-learn matplotlib seaborn pyarrow requests
```

**선택적 (성능 향상)**:
```bash
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

### 3️⃣ 전체 파이프라인 실행

#### Windows
```bash
run_all.bat
```

#### Linux/Mac (개별 실행)
```bash
python 01_download_openaq.py
python 02_download_tempo.py
python 03_download_merra2.py
python 04_preprocess_merge.py
python 05_feature_engineering.py
python 06_train_baseline.py
python 07_train_residual.py
python 08_evaluate.py
```

---

### 4️⃣ 결과 확인

#### 평가 지표
```bash
cat models/evaluation_metrics.json
```

예상 출력:
```json
{
  "Baseline": {
    "r2": 0.45,
    "mae": 12.3,
    "rmse": 16.8
  },
  "Lightgbm": {
    "r2": 0.68,
    "mae": 8.7,
    "rmse": 11.2
  }
}
```

#### 시각화
- `plots/timeseries_pred_vs_obs.png` - 시계열 비교
- `plots/scatter_pred_vs_obs.png` - 산점도 (관측 vs 예측)
- `plots/residuals_histogram.png` - 잔차 분포
- `plots/feature_importance.png` - 특성 중요도

---

## 📊 데이터 소스

| 데이터 | 변수 | 해상도 | 용량 (6주) |
|--------|------|--------|------------|
| **TEMPO L3** | NO₂, O₃ | 시간별, ~5 km | ~4 GB |
| **MERRA-2** | PBLH, U10M, V10M | 1시간, 0.5° × 0.625° | ~1 GB |
| **OpenAQ** | PM2.5 | 1시간, 지점별 | ~10 MB |

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

## 📈 파이프라인 흐름

```
1. 다운로드
   ├─ OpenAQ PM2.5 (5개 도시, 6주) → CSV
   ├─ TEMPO NO₂/O₃ (Bay Area, 6주) → NetCDF
   └─ MERRA-2 PBLH/U10M/V10M (6주) → NetCDF

2. 전처리
   ├─ BBOX 서브셋팅
   ├─ Tidy 변환 (time, lat, lon, var)
   ├─ 시간 정렬 (3시간 리샘플 옵션)
   ├─ 병합 (inner join on time+lat+lon)
   ├─ QC (음수/비현실값 제거)
   └─ Parquet 저장

3. 특성 엔지니어링
   ├─ 풍속/풍향 계산 (U10M, V10M → ws, wd)
   ├─ 시간 인코딩 (hour_sin/cos, dow_sin/cos)
   ├─ 래그 특성 (t-1, t-3, t-6, t-12)
   ├─ 이동평균 (3h, 6h)
   └─ PM2.5 라벨 병합

4. 모델링
   ├─ Baseline: PM2.5(t) = PM2.5(t-1)
   ├─ Residual = PM2.5(t) - PM2.5(t-1)
   ├─ LightGBM/XGBoost/RF로 residual 예측
   └─ 최종: PM2.5(t) = PM2.5(t-1) + residual_pred

5. 평가
   ├─ 시간 단절 검증 (앞 4주 학습, 뒤 2주 검증)
   ├─ R², MAE, RMSE 계산
   ├─ 시각화 (시계열, 산점도, 잔차, 특성 중요도)
   └─ JSON 내보내기
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

## 🚦 다음 단계 (시간 남을 시)

- [ ] Streamlit 대시보드 추가
- [ ] 공간 홀드아웃 검증
- [ ] SHAP 값으로 모델 해석
- [ ] 실시간 예측 API
- [ ] Docker 컨테이너화

---

**생성일**: 2025-01-04
**버전**: 1.0
**해커톤**: NASA Space Apps Challenge 2025
