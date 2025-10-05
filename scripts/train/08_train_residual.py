"""
잔차 보정 모델 학습 (LightGBM, XGBoost, Random Forest)

전략:
1. 잔차 계산: residual = PM2.5(t) - PM2.5(t-1) [베이스라인 = 지속성 모델]
2. 위성 + 기상 특성으로 잔차를 예측하는 모델 학습
3. 최종 예측: PM2.5(t) = PM2.5(t-1) + 예측된_잔차
"""
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
from typing import Tuple, Dict
import config
import utils

logger = utils.setup_logging(__name__)


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시간 기준으로 학습/검증 데이터 분할"""
    logger.info("Splitting train/validation by time...")

    # 시간순 정렬
    df = df.sort_values('time')

    # 분할 기준 날짜 계산 (시작 시점 + TRAIN_WEEKS)
    split_date = df['time'].min() + timedelta(weeks=config.TRAIN_WEEKS)

    # 학습 데이터: 분할 기준 이전
    train_df = df[df['time'] < split_date].copy()
    # 검증 데이터: 분할 기준 이후
    val_df = df[df['time'] >= split_date].copy()

    logger.info(f"Train: {train_df['time'].min()} to {train_df['time'].max()} ({len(train_df):,} samples)")
    logger.info(f"Val:   {val_df['time'].min()} to {val_df['time'].max()} ({len(val_df):,} samples)")

    return train_df, val_df


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    특성(X), 타겟(잔차), PM2.5_lag1 준비

    Args:
        df: 입력 DataFrame

    Returns:
        (X, y_residual, pm25_lag1)
    """
    # 특성이 아닌 컬럼 제외
    exclude_cols = ['time', 'lat', 'lon', 'pm25', 'location', 'city', 'unit']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 특성 행렬 X 준비
    X = df[feature_cols].copy()

    # 잔차 계산: PM2.5(t) - PM2.5(t-1)
    pm25_lag1 = df['pm25_lag1'].copy()  # 1시간 전 PM2.5 값
    y_residual = df['pm25'] - pm25_lag1  # 현재 - 이전 = 변화량

    logger.info(f"Features: {len(feature_cols)} columns")
    logger.info(f"Target: residual (PM2.5 - PM2.5_lag1)")

    return X, y_residual, pm25_lag1


def train_lightgbm(X_train, y_train, X_val, y_val) -> object:
    """LightGBM 모델 학습"""
    import lightgbm as lgb

    logger.info("\n" + "="*60)
    logger.info("Training LightGBM")
    logger.info("="*60)

    # LightGBM Dataset 생성
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # 모델 학습 (조기 종료 포함)
    model = lgb.train(
        config.LGBM_PARAMS,  # 하이퍼파라미터 (config.py에서 설정)
        train_data,
        valid_sets=[train_data, val_data],  # 학습/검증 데이터
        valid_names=['train', 'val'],
        callbacks=[
            lgb.log_evaluation(period=50),  # 50라운드마다 로그 출력
            lgb.early_stopping(stopping_rounds=config.LGBM_PARAMS['early_stopping_rounds'])  # 조기 종료
        ]
    )

    logger.info("✓ LightGBM training complete")

    return model


def train_xgboost(X_train, y_train, X_val, y_val) -> object:
    """XGBoost 모델 학습"""
    import xgboost as xgb

    logger.info("\n" + "="*60)
    logger.info("Training XGBoost")
    logger.info("="*60)

    # XGBoost DMatrix 생성 (XGBoost 전용 데이터 형식)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 모델 학습 (조기 종료 포함)
    model = xgb.train(
        config.XGB_PARAMS,  # 하이퍼파라미터
        dtrain,
        num_boost_round=config.XGB_PARAMS['n_estimators'],  # 부스팅 라운드 수
        evals=[(dtrain, 'train'), (dval, 'val')],  # 평가 데이터셋
        early_stopping_rounds=config.XGB_PARAMS['early_stopping_rounds'],  # 조기 종료
        verbose_eval=50  # 50라운드마다 로그 출력
    )

    logger.info("✓ XGBoost training complete")

    return model


def train_randomforest(X_train, y_train, X_val, y_val) -> object:
    """Random Forest 모델 학습"""
    from sklearn.ensemble import RandomForestRegressor

    logger.info("\n" + "="*60)
    logger.info("Training Random Forest")
    logger.info("="*60)

    # Random Forest 회귀 모델 생성 및 학습
    model = RandomForestRegressor(**config.RF_PARAMS, random_state=42)

    model.fit(X_train, y_train)  # 학습 (RF는 조기 종료 없음)

    logger.info("✓ Random Forest training complete")

    return model


def evaluate_residual_model(
    model,
    X_val,
    y_residual_val,
    pm25_lag1_val,
    pm25_true_val,
    model_type: str
) -> Dict:
    """
    잔차 모델 평가

    Args:
        model: 학습된 모델
        X_val: 검증 데이터 특성
        y_residual_val: 실제 잔차값
        pm25_lag1_val: PM2.5(t-1) - 재구성에 사용
        pm25_true_val: 실제 PM2.5(t) 값
        model_type: 모델 이름

    Returns:
        메트릭 딕셔너리, PM2.5 예측값
    """
    logger.info(f"\nEvaluating {model_type}...")

    # 잔차 예측 (모델마다 예측 방법이 다름)
    if model_type == 'LightGBM':
        residual_pred = model.predict(X_val, num_iteration=model.best_iteration)
    elif model_type == 'XGBoost':
        import xgboost as xgb
        dval = xgb.DMatrix(X_val)
        residual_pred = model.predict(dval)
    else:  # Random Forest
        residual_pred = model.predict(X_val)

    # PM2.5 재구성: PM2.5_pred = PM2.5(t-1) + 예측된_잔차
    # 베이스라인(지속성) + 잔차 보정 = 최종 예측
    pm25_pred = pm25_lag1_val + residual_pred

    # 평가 메트릭 계산 (MAE, RMSE, R²)
    metrics = utils.calculate_metrics(pm25_true_val, pm25_pred)
    metrics['model'] = model_type

    utils.print_metrics(metrics, model_name=model_type)

    return metrics, pm25_pred


def main():
    """메인 실행 함수"""
    logger.info("="*60)
    logger.info("Residual Correction Model Training")
    logger.info("="*60)

    # ========================================================================
    # 1. 특성 엔지니어링된 데이터 로드
    # ========================================================================
    logger.info(f"Loading data from {config.FEATURES_PARQUET}...")
    df = utils.load_parquet(config.FEATURES_PARQUET)

    # 필수 컬럼 확인 (pm25, pm25_lag1)
    if 'pm25' not in df.columns or 'pm25_lag1' not in df.columns:
        logger.error("pm25 or pm25_lag1 not found. Cannot train residual model.")
        return

    # 잔차 계산에 필요한 pm25_lag1이 없는 행 제거
    df = df.dropna(subset=['pm25', 'pm25_lag1'])
    logger.info(f"Samples with valid PM2.5 and lag: {len(df):,}")

    # ========================================================================
    # 2. 학습/검증 데이터 분할 (시간 기준)
    # ========================================================================
    train_df, val_df = split_train_val(df)

    # ========================================================================
    # 3. 특성(X)과 타겟(y) 준비
    # ========================================================================
    logger.info("\nPreparing features and targets...")

    # 학습 데이터
    X_train, y_train, pm25_lag1_train = prepare_features_target(train_df)
    # 검증 데이터
    X_val, y_val, pm25_lag1_val = prepare_features_target(val_df)

    # 검증 데이터의 실제 PM2.5 값 (평가용)
    pm25_true_val = val_df['pm25'].values

    # NaN 제거 (학습 데이터)
    # X의 각 행에 NaN이 하나라도 있거나 y가 NaN인 경우 제외
    train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    pm25_lag1_train = pm25_lag1_train[train_mask]

    # NaN 제거 (검증 데이터)
    val_mask = ~(X_val.isnull().any(axis=1) | y_val.isnull())
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    pm25_lag1_val = pm25_lag1_val[val_mask]
    pm25_true_val = pm25_true_val[val_mask]

    logger.info(f"Train samples (after dropping NaNs): {len(X_train):,}")
    logger.info(f"Val samples (after dropping NaNs):   {len(X_val):,}")

    # ========================================================================
    # 4. 모델 학습
    # ========================================================================
    models = {}          # 학습된 모델 저장
    predictions = {}     # 각 모델의 예측값 저장
    all_metrics = {}     # 각 모델의 평가 메트릭 저장

    # LightGBM (기본 모델 - 항상 학습)
    if config.USE_LIGHTGBM:
        try:
            # 모델 학습
            model_lgbm = train_lightgbm(X_train, y_train, X_val, y_val)
            models['LightGBM'] = model_lgbm

            # 모델 평가 및 예측
            metrics_lgbm, pred_lgbm = evaluate_residual_model(
                model_lgbm, X_val, y_val, pm25_lag1_val, pm25_true_val, 'LightGBM'
            )
            all_metrics['LightGBM'] = metrics_lgbm
            predictions['LightGBM'] = pred_lgbm

            # 모델 저장 (pickle 파일로 저장)
            with open(config.LGBM_MODEL, 'wb') as f:
                pickle.dump(model_lgbm, f)
            logger.info(f"✓ LightGBM model saved to {config.LGBM_MODEL}")

        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")

    # XGBoost (선택 사항 - config.USE_XGBOOST로 제어)
    if config.USE_XGBOOST:
        try:
            # 모델 학습
            model_xgb = train_xgboost(X_train, y_train, X_val, y_val)
            models['XGBoost'] = model_xgb

            # 모델 평가 및 예측
            metrics_xgb, pred_xgb = evaluate_residual_model(
                model_xgb, X_val, y_val, pm25_lag1_val, pm25_true_val, 'XGBoost'
            )
            all_metrics['XGBoost'] = metrics_xgb
            predictions['XGBoost'] = pred_xgb

            # 모델 저장
            with open(config.XGB_MODEL, 'wb') as f:
                pickle.dump(model_xgb, f)
            logger.info(f"✓ XGBoost model saved to {config.XGB_MODEL}")

        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")

    # Random Forest (선택 사항 - config.USE_RANDOMFOREST로 제어)
    if config.USE_RANDOMFOREST:
        try:
            # 모델 학습
            model_rf = train_randomforest(X_train, y_train, X_val, y_val)
            models['RandomForest'] = model_rf

            # 모델 평가 및 예측
            metrics_rf, pred_rf = evaluate_residual_model(
                model_rf, X_val, y_val, pm25_lag1_val, pm25_true_val, 'RandomForest'
            )
            all_metrics['RandomForest'] = metrics_rf
            predictions['RandomForest'] = pred_rf

            # 모델 저장
            with open(config.RF_MODEL, 'wb') as f:
                pickle.dump(model_rf, f)
            logger.info(f"✓ Random Forest model saved to {config.RF_MODEL}")

        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")

    # ========================================================================
    # 5. 앙상블 (선택 사항 - 여러 모델의 예측 평균)
    # ========================================================================
    if config.USE_ENSEMBLE and len(predictions) > 1:
        logger.info("\n" + "="*60)
        logger.info("Creating Ensemble (Average)")
        logger.info("="*60)

        # 모든 모델의 예측값 평균 계산
        pred_ensemble = np.mean(list(predictions.values()), axis=0)

        # 앙상블 모델 평가
        metrics_ensemble = utils.calculate_metrics(pm25_true_val, pred_ensemble)
        metrics_ensemble['model'] = 'Ensemble'
        all_metrics['Ensemble'] = metrics_ensemble

        utils.print_metrics(metrics_ensemble, model_name="Ensemble")

        predictions['Ensemble'] = pred_ensemble

    # ========================================================================
    # 6. 검증 데이터 예측 결과 저장
    # ========================================================================
    # 검증 데이터 기본 정보 (시간, 위치, 실제 PM2.5)
    val_df_out = val_df[val_mask][['time', 'lat', 'lon', 'pm25']].copy()
    val_df_out['pm25_lag1'] = pm25_lag1_val.values

    # 각 모델의 예측값을 컬럼으로 추가
    for model_name, preds in predictions.items():
        val_df_out[f'pm25_pred_{model_name.lower()}'] = preds

    # Parquet 파일로 저장
    residual_pred_path = config.TABLES_DIR / 'residual_predictions.parquet'
    utils.save_parquet(val_df_out, residual_pred_path, downcast=True)
    logger.info(f"\n✓ Predictions saved to {residual_pred_path}")

    # ========================================================================
    # 7. 학습 요약
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Training Summary:")
    logger.info("="*60)

    # 모든 모델의 성능 메트릭 출력
    for model_name, metrics in all_metrics.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  MAE:  {metrics['mae']:.2f} µg/m³")  # 평균 절대 오차
        logger.info(f"  RMSE: {metrics['rmse']:.2f} µg/m³")  # 평균 제곱근 오차
        logger.info(f"  R²:   {metrics['r2']:.4f}")  # 결정 계수 (1에 가까울수록 좋음)

    logger.info("="*60)


if __name__ == "__main__":
    main()
