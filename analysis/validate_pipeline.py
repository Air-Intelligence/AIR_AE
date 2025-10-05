"""
PM2.5 예측 파이프라인 검증 스크립트

검증 항목:
1. 데이터 로드 (TEMPO, OpenAQ)
2. 피처 생성 (NaN < 10%)
3. 모델 예측 (0~500 µg/m³ 범위)
4. API 엔드포인트 (/api/predict)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import json

import config
import utils
from src.features import extract_near
from src.model import get_predictor

logger = utils.setup_logging(__name__)


def test_data_loading():
    """데이터 로드 검증"""
    logger.info("=" * 60)
    logger.info("[1/4] 데이터 로드 검증")
    logger.info("=" * 60)

    errors = []

    # TEMPO NO₂ NRT
    tempo_path = Path("/mnt/data/features/realtime/tempo_no2.parquet")
    if not tempo_path.exists():
        errors.append(f"TEMPO NO₂ not found: {tempo_path}")
    else:
        try:
            df = pd.read_parquet(tempo_path)
            logger.info(f"✓ TEMPO NO₂: {len(df):,} records")
        except Exception as e:
            errors.append(f"TEMPO NO₂ load error: {e}")

    # OpenAQ O₃
    o3_path = Path("/mnt/data/features/realtime/openaq_o3.parquet")
    if not o3_path.exists():
        errors.append(f"OpenAQ O₃ not found: {o3_path}")
    else:
        try:
            df = pd.read_parquet(o3_path)
            logger.info(f"✓ OpenAQ O₃: {len(df):,} records")
        except Exception as e:
            errors.append(f"OpenAQ O₃ load error: {e}")

    if errors:
        for err in errors:
            logger.error(f"  ❌ {err}")
        return False
    else:
        logger.info("✓ 데이터 로드 성공\n")
        return True


def test_feature_engineering():
    """피처 생성 검증"""
    logger.info("=" * 60)
    logger.info("[2/4] 피처 생성 검증")
    logger.info("=" * 60)

    try:
        # 실시간 데이터 로드
        tempo_path = Path("/mnt/data/features/realtime/tempo_no2.parquet")
        o3_path = Path("/mnt/data/features/realtime/openaq_o3.parquet")

        if not tempo_path.exists() or not o3_path.exists():
            logger.error("❌ 데이터 파일 없음")
            return False

        df_tempo = pd.read_parquet(tempo_path)
        df_o3 = pd.read_parquet(o3_path)

        df_tempo['time'] = pd.to_datetime(df_tempo['time'])
        df_o3['time'] = pd.to_datetime(df_o3['time'])

        # 최신 2개 시간 추출 (t, t-1)
        tempo_times = sorted(df_tempo['time'].unique())[-2:]
        o3_times = sorted(df_o3['time'].unique())[-2:]

        if len(tempo_times) < 2 or len(o3_times) < 2:
            logger.error("❌ 시간 데이터 부족 (최소 2시간 필요)")
            return False

        # 샘플 위치 (San Francisco)
        lat, lon = 37.7749, -122.4194

        # extract_near 테스트
        no2_t = extract_near(df_tempo, lat, lon, time=tempo_times[-1], value_col='no2')
        no2_lag1 = extract_near(df_tempo, lat, lon, time=tempo_times[-2], value_col='no2')

        o3_t = extract_near(df_o3, lat, lon, time=o3_times[-1], value_col='o3')
        o3_lag1 = extract_near(df_o3, lat, lon, time=o3_times[-2], value_col='o3')

        # 검증
        if no2_t is None or no2_lag1 is None:
            logger.error("❌ NO₂ 추출 실패")
            return False

        if o3_t is None or o3_lag1 is None:
            logger.error("❌ O₃ 추출 실패")
            return False

        logger.info(f"✓ no2_t: {no2_t:.2e}")
        logger.info(f"✓ no2_lag1: {no2_lag1:.2e}")
        logger.info(f"✓ o3_t: {o3_t:.2e}")
        logger.info(f"✓ o3_lag1: {o3_lag1:.2e}")

        # NaN 체크
        nan_count = sum([v is None for v in [no2_t, no2_lag1, o3_t, o3_lag1]])
        nan_ratio = nan_count / 4 * 100

        if nan_ratio > 10:
            logger.warning(f"⚠️  NaN ratio: {nan_ratio:.1f}% (> 10%)")
        else:
            logger.info(f"✓ NaN ratio: {nan_ratio:.1f}% (< 10%)\n")

        return True

    except Exception as e:
        logger.error(f"❌ 피처 생성 오류: {e}")
        return False


def test_model_prediction():
    """모델 예측 검증"""
    logger.info("=" * 60)
    logger.info("[3/4] 모델 예측 검증")
    logger.info("=" * 60)

    try:
        # 모델 로드
        predictor = get_predictor(model_dir="/mnt/data/models")

        # 샘플 입력
        no2_t = 3.5e15
        no2_lag1 = 3.2e15
        o3_t = 8.5e18
        o3_lag1 = 8.3e18
        hour = 14
        dow = 2

        # 예측
        result = predictor.predict(no2_t, no2_lag1, o3_t, o3_lag1, hour, dow)

        pm25_pred = result['pm25_pred']
        conf_lower = result['confidence_lower']
        conf_upper = result['confidence_upper']

        logger.info(f"✓ 예측 PM2.5: {pm25_pred:.2f} µg/m³")
        logger.info(f"  신뢰구간: [{conf_lower:.2f}, {conf_upper:.2f}]")

        # 범위 검증 (0~500)
        if not (0 <= pm25_pred <= 500):
            logger.error(f"❌ 예측값 범위 초과: {pm25_pred}")
            return False

        if not (0 <= conf_lower <= 500):
            logger.error(f"❌ 신뢰구간 하한 초과: {conf_lower}")
            return False

        if not (0 <= conf_upper <= 500):
            logger.error(f"❌ 신뢰구간 상한 초과: {conf_upper}")
            return False

        logger.info("✓ 예측값 범위 정상 (0~500 µg/m³)\n")
        return True

    except Exception as e:
        logger.error(f"❌ 모델 예측 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """API 엔드포인트 검증"""
    logger.info("=" * 60)
    logger.info("[4/4] API 엔드포인트 검증")
    logger.info("=" * 60)

    try:
        # API URL (로컬 서버 가정)
        api_url = "http://localhost:8000/api/predict"

        # 샘플 요청
        params = {
            'lat': 37.7749,
            'lon': -122.4194,
            'city': 'San Francisco'
        }

        logger.info(f"요청 URL: {api_url}")
        logger.info(f"파라미터: {params}")

        response = requests.post(api_url, params=params, timeout=10)

        if response.status_code != 200:
            logger.warning(f"⚠️  API 응답 코드: {response.status_code}")
            logger.warning("  (서버가 실행 중이 아닐 수 있습니다)")
            return False

        # JSON 파싱
        data = response.json()

        logger.info("✓ API 응답:")
        logger.info(f"  {json.dumps(data, indent=2)}")

        # 필수 필드 체크
        required_fields = ['predicted_pm25', 'confidence_lower', 'confidence_upper', 'prediction_time']

        for field in required_fields:
            if field not in data:
                logger.error(f"❌ 필수 필드 누락: {field}")
                return False

        logger.info("✓ API 응답 정상\n")
        return True

    except requests.exceptions.ConnectionError:
        logger.warning("⚠️  API 서버 연결 실패 (서버가 실행 중이 아닐 수 있습니다)")
        logger.warning("  python open_aq.py 로 서버를 먼저 실행하세요\n")
        return False

    except Exception as e:
        logger.error(f"❌ API 테스트 오류: {e}")
        return False


def main():
    logger.info("\n" + "=" * 60)
    logger.info("PM2.5 예측 파이프라인 검증")
    logger.info("=" * 60 + "\n")

    results = []

    # 1. 데이터 로드
    results.append(("데이터 로드", test_data_loading()))

    # 2. 피처 생성
    results.append(("피처 생성", test_feature_engineering()))

    # 3. 모델 예측
    results.append(("모델 예측", test_model_prediction()))

    # 4. API 테스트
    results.append(("API 테스트", test_api()))

    # 결과 요약
    logger.info("=" * 60)
    logger.info("검증 결과 요약")
    logger.info("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"  {name:20s} {status}")

    logger.info("=" * 60)

    # 전체 결과
    all_passed = all(r[1] for r in results)

    if all_passed:
        logger.info("✓ 모든 검증 통과!")
        sys.exit(0)
    else:
        logger.error("❌ 일부 검증 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
