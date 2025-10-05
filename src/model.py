"""
PM2.5 예측 모델 클래스

기능:
- LightGBM 모델 로드
- StandardScaler 로드
- 6개 피처 입력 → PM2.5 예측값 + 신뢰구간 반환
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PM25Predictor:
    """PM2.5 예측 클래스"""

    def __init__(self, model_dir: str = "/mnt/data/models"):
        """
        모델 초기화

        Args:
            model_dir: 모델 파일 디렉터리 경로
        """
        self.model_dir = Path(model_dir)

        # 모델 파일 경로
        self.model_path = self.model_dir / "pm25_lgbm.pkl"
        self.scaler_path = self.model_dir / "feature_scaler.pkl"
        self.info_path = self.model_dir / "feature_info.json"

        # 모델 로드
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.mae = None  # 학습 시 저장된 MAE (신뢰구간용)

        self._load_model()

    def _load_model(self):
        """모델 파일 로드"""
        try:
            # LightGBM 모델
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            self.model = joblib.load(self.model_path)
            logger.info(f"✓ Loaded LightGBM model from {self.model_path}")

            # StandardScaler
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✓ Loaded scaler from {self.scaler_path}")

            # Feature info (피처 순서, MAE 등)
            if not self.info_path.exists():
                raise FileNotFoundError(f"Feature info not found: {self.info_path}")

            with open(self.info_path, 'r') as f:
                self.feature_info = json.load(f)

            self.mae = self.feature_info.get('mae', 2.0)  # 기본값 2.0
            logger.info(f"✓ Loaded feature info (MAE: {self.mae:.2f})")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(
        self,
        no2_t: float,
        no2_lag1: float,
        o3_t: float,
        o3_lag1: float,
        hour: int,
        dow: int
    ) -> Dict[str, float]:
        """
        PM2.5 예측

        Args:
            no2_t: 현재 NO₂ (molecules/cm²)
            no2_lag1: 1시간 전 NO₂
            o3_t: 현재 O₃ (molecules/cm²)
            o3_lag1: 1시간 전 O₃
            hour: 시각 (0-23)
            dow: 요일 (0-6)

        Returns:
            {
                'pm25_pred': 예측 PM2.5 (µg/m³),
                'confidence_lower': 신뢰구간 하한,
                'confidence_upper': 신뢰구간 상한,
                'inputs': 입력값 딕셔너리
            }
        """

        # 입력값 검증
        if any(v is None for v in [no2_t, no2_lag1, o3_t, o3_lag1, hour, dow]):
            raise ValueError("All input features must be non-None")

        # 피처 순서 (학습 시와 동일하게)
        feature_order = self.feature_info.get('feature_order', [
            'no2_t', 'no2_lag1', 'o3_t', 'o3_lag1', 'hour', 'dow'
        ])

        # 입력 딕셔너리
        inputs = {
            'no2_t': float(no2_t),
            'no2_lag1': float(no2_lag1),
            'o3_t': float(o3_t),
            'o3_lag1': float(o3_lag1),
            'hour': int(hour),
            'dow': int(dow)
        }

        # 피처 순서대로 배열 생성
        X = np.array([[inputs[f] for f in feature_order]])

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측
        pred = self.model.predict(X_scaled)[0]

        # 신뢰구간 (±2 MAE)
        uncertainty = 2.0 * self.mae
        confidence_lower = max(0, pred - uncertainty)  # PM2.5는 0 이상
        confidence_upper = min(500, pred + uncertainty)  # 최대 500

        return {
            'pm25_pred': float(pred),
            'confidence_lower': float(confidence_lower),
            'confidence_upper': float(confidence_upper),
            'inputs': inputs
        }

    def predict_batch(
        self,
        no2_t: np.ndarray,
        no2_lag1: np.ndarray,
        o3_t: np.ndarray,
        o3_lag1: np.ndarray,
        hour: np.ndarray,
        dow: np.ndarray
    ) -> np.ndarray:
        """
        배치 예측 (여러 샘플 동시 예측)

        Args:
            각 인자는 numpy array (shape: [N,])

        Returns:
            예측값 배열 (shape: [N,])
        """

        # 피처 순서
        feature_order = self.feature_info.get('feature_order', [
            'no2_t', 'no2_lag1', 'o3_t', 'o3_lag1', 'hour', 'dow'
        ])

        # 배열 결합
        X = np.column_stack([no2_t, no2_lag1, o3_t, o3_lag1, hour, dow])

        # 스케일링
        X_scaled = self.scaler.transform(X)

        # 예측
        preds = self.model.predict(X_scaled)

        return preds


class PM25PredictorFallback:
    """
    모델 파일이 없을 때 사용하는 fallback 예측기
    단순 선형 근사
    """

    def __init__(self):
        logger.warning("Using fallback predictor (model files not found)")

    def predict(
        self,
        no2_t: float,
        no2_lag1: float,
        o3_t: float,
        o3_lag1: float,
        hour: int,
        dow: int
    ) -> Dict[str, float]:
        """간단한 선형 근사"""

        # NO₂와 O₃를 정규화 (대략적인 범위)
        no2_norm = no2_t / 1e16  # TEMPO NO₂ 일반적 범위
        o3_norm = o3_t / 1e18    # TEMPO O₃ 일반적 범위

        # 단순 선형 조합
        pm25_pred = 10 + no2_norm * 5 + o3_norm * 3

        # 0-100 범위로 클리핑
        pm25_pred = np.clip(pm25_pred, 0, 100)

        return {
            'pm25_pred': float(pm25_pred),
            'confidence_lower': float(pm25_pred * 0.8),
            'confidence_upper': float(pm25_pred * 1.2),
            'inputs': {
                'no2_t': float(no2_t),
                'no2_lag1': float(no2_lag1),
                'o3_t': float(o3_t),
                'o3_lag1': float(o3_lag1),
                'hour': int(hour),
                'dow': int(dow)
            }
        }


def get_predictor(model_dir: str = "/mnt/data/models") -> PM25Predictor:
    """
    PM25Predictor 인스턴스 반환 (싱글톤 패턴)

    Args:
        model_dir: 모델 디렉터리 경로

    Returns:
        PM25Predictor 또는 PM25PredictorFallback
    """
    try:
        return PM25Predictor(model_dir=model_dir)
    except FileNotFoundError:
        logger.warning("Model files not found, using fallback predictor")
        return PM25PredictorFallback()
