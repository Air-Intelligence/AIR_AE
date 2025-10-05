"""
PM2.5 예측 모델 학습 파이프라인

데이터:
- 학습: TEMPO NO₂ + TEMPO O₃ (V03, 2023-09-01 ~ 2023-10-13)
- 라벨: OpenAQ PM2.5

절차:
1. 학습 데이터 로드 (6주)
2. KD-Tree 공간 매칭
3. 피처 생성 (6개)
4. Train/Val 분할 (4주 / 2주)
5. LightGBM 학습
6. 평가 (MAE, RMSE, R²)
7. 모델 저장 (3개 파일)
8. 시각화
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

import config
import utils
from src.features import join_tempo_openaq, create_features

logger = utils.setup_logging(__name__)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    학습 데이터 로드

    Returns:
        df_tempo_no2: TEMPO NO₂ (6주)
        df_tempo_o3: TEMPO O₃ (6주)
        df_openaq_pm25: OpenAQ PM2.5 (라벨)
    """
    logger.info("=" * 60)
    logger.info("학습 데이터 로드 중...")
    logger.info("=" * 60)

    # TEMPO NO₂ (학습용 parquet)
    tempo_no2_path = config.FEATURES_TEMPO_TRAIN / "no2_merged.parquet"
    if not tempo_no2_path.exists():
        logger.error(f"TEMPO NO₂ not found: {tempo_no2_path}")
        sys.exit(1)

    df_tempo_no2 = pd.read_parquet(tempo_no2_path)
    df_tempo_no2['time'] = pd.to_datetime(df_tempo_no2['time'])
    logger.info(f"✓ Loaded TEMPO NO₂: {len(df_tempo_no2):,} records")

    # TEMPO O₃ (학습용 parquet)
    tempo_o3_path = config.FEATURES_TEMPO_TRAIN / "o3_merged.parquet"
    if not tempo_o3_path.exists():
        logger.error(f"TEMPO O₃ not found: {tempo_o3_path}")
        sys.exit(1)

    df_tempo_o3 = pd.read_parquet(tempo_o3_path)
    df_tempo_o3['time'] = pd.to_datetime(df_tempo_o3['time'])
    logger.info(f"✓ Loaded TEMPO O₃: {len(df_tempo_o3):,} records")

    # OpenAQ PM2.5 (라벨)
    openaq_path = config.FEATURES_OPENAQ / "openaq_pm25.parquet"
    if not openaq_path.exists():
        # CSV fallback
        openaq_path = config.OPENAQ_CSV
        if not openaq_path.exists():
            logger.error(f"OpenAQ PM2.5 not found: {openaq_path}")
            sys.exit(1)
        df_openaq_pm25 = pd.read_csv(openaq_path)
    else:
        df_openaq_pm25 = pd.read_parquet(openaq_path)

    df_openaq_pm25['time'] = pd.to_datetime(df_openaq_pm25['time'])
    logger.info(f"✓ Loaded OpenAQ PM2.5: {len(df_openaq_pm25):,} records")

    return df_tempo_no2, df_tempo_o3, df_openaq_pm25


def main():
    logger.info("=" * 60)
    logger.info("PM2.5 예측 모델 학습 파이프라인")
    logger.info("=" * 60)

    # ========================================================================
    # 1. 데이터 로드
    # ========================================================================
    df_tempo_no2, df_tempo_o3, df_openaq_pm25 = load_training_data()

    # ========================================================================
    # 2. TEMPO NO₂ + TEMPO O₃ KD-Tree 매칭
    # ========================================================================
    logger.info("\n[1/7] TEMPO NO₂ + TEMPO O₃ 결합 중...")

    df_tempo_joined = join_tempo_openaq(
        df_tempo_no2,
        df_tempo_o3,
        max_distance_km=10.0
    )

    if df_tempo_joined.empty:
        logger.error("TEMPO 데이터 결합 실패")
        sys.exit(1)

    # 컬럼명 정리 (tempo_no2 → no2, tempo_o3 → o3)
    df_tempo_joined = df_tempo_joined.rename(columns={
        'tempo_no2': 'no2',
        'tempo_o3': 'o3'
    })

    logger.info(f"✓ TEMPO joined: {len(df_tempo_joined):,} records")

    # ========================================================================
    # 3. TEMPO + OpenAQ PM2.5 매칭
    # ========================================================================
    logger.info("\n[2/7] TEMPO + OpenAQ PM2.5 매칭 중...")

    df_with_labels = join_tempo_openaq(
        df_tempo_joined,
        df_openaq_pm25,
        max_distance_km=10.0
    )

    if df_with_labels.empty:
        logger.error("라벨 매칭 실패")
        sys.exit(1)

    # 컬럼명 정리
    df_with_labels = df_with_labels.rename(columns={
        'tempo_no2': 'no2',
        'tempo_o3': 'o3',
        'openaq_pm25': 'pm25'
    })

    logger.info(f"✓ Labeled data: {len(df_with_labels):,} records")

    # ========================================================================
    # 4. 피처 생성
    # ========================================================================
    logger.info("\n[3/7] 피처 생성 중 (6개)...")

    df_features = create_features(df_with_labels)

    if df_features.empty:
        logger.error("피처 생성 실패")
        sys.exit(1)

    # pm25 라벨 추가
    df_features = df_features.merge(
        df_with_labels[['time', 'lat', 'lon', 'pm25']],
        on=['time', 'lat', 'lon'],
        how='left'
    )

    # NaN 제거
    df_features = df_features.dropna()

    logger.info(f"✓ Features: {len(df_features):,} records, {df_features.shape[1]} columns")

    # ========================================================================
    # 5. Train/Val 분할 (4주 / 2주)
    # ========================================================================
    logger.info("\n[4/7] Train/Val 분할 중...")

    # 시간 기준 정렬
    df_features = df_features.sort_values('time').reset_index(drop=True)

    # 전체 기간 계산
    total_days = (df_features['time'].max() - df_features['time'].min()).days
    train_days = config.TRAIN_WEEKS * 7
    val_days = config.VAL_WEEKS * 7

    # 분할 시점
    split_time = df_features['time'].min() + pd.Timedelta(days=train_days)

    df_train = df_features[df_features['time'] < split_time].copy()
    df_val = df_features[df_features['time'] >= split_time].copy()

    logger.info(f"✓ Train: {len(df_train):,} records ({df_train['time'].min().date()} ~ {df_train['time'].max().date()})")
    logger.info(f"✓ Val:   {len(df_val):,} records ({df_val['time'].min().date()} ~ {df_val['time'].max().date()})")

    # ========================================================================
    # 6. 피처/라벨 분리
    # ========================================================================
    feature_cols = ['no2_t', 'no2_lag1', 'o3_t', 'o3_lag1', 'hour', 'dow']

    X_train = df_train[feature_cols].values
    y_train = df_train['pm25'].values

    X_val = df_val[feature_cols].values
    y_val = df_val['pm25'].values

    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_val shape:   {X_val.shape}")

    # ========================================================================
    # 7. 정규화 (StandardScaler)
    # ========================================================================
    logger.info("\n[5/7] 피처 정규화 중...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logger.info("✓ StandardScaler fitted")

    # ========================================================================
    # 8. LightGBM 학습
    # ========================================================================
    logger.info("\n[6/7] LightGBM 학습 중...")

    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

    params = config.LGBM_PARAMS.copy()

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    logger.info("✓ LightGBM training complete")

    # ========================================================================
    # 9. 평가
    # ========================================================================
    logger.info("\n[7/7] 모델 평가 중...")

    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    # Train 평가
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    # Val 평가
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    logger.info("=" * 60)
    logger.info("평가 결과")
    logger.info("=" * 60)
    logger.info(f"Train  MAE:  {train_mae:.2f} µg/m³")
    logger.info(f"Train  RMSE: {train_rmse:.2f} µg/m³")
    logger.info(f"Train  R²:   {train_r2:.4f}")
    logger.info("-" * 60)
    logger.info(f"Val    MAE:  {val_mae:.2f} µg/m³")
    logger.info(f"Val    RMSE: {val_rmse:.2f} µg/m³")
    logger.info(f"Val    R²:   {val_r2:.4f}")
    logger.info("=" * 60)

    # ========================================================================
    # 10. 모델 저장
    # ========================================================================
    logger.info("\n모델 저장 중...")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) LightGBM 모델
    model_path = config.MODELS_DIR / "pm25_lgbm.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✓ Model saved: {model_path}")

    # 2) StandardScaler
    scaler_path = config.MODELS_DIR / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved: {scaler_path}")

    # 3) Feature info
    feature_info = {
        'feature_order': feature_cols,
        'mae': val_mae,
        'rmse': val_rmse,
        'r2': val_r2,
        'train_date_range': [
            df_train['time'].min().isoformat(),
            df_train['time'].max().isoformat()
        ],
        'val_date_range': [
            df_val['time'].min().isoformat(),
            df_val['time'].max().isoformat()
        ]
    }

    info_path = config.MODELS_DIR / "feature_info.json"
    with open(info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    logger.info(f"✓ Feature info saved: {info_path}")

    # ========================================================================
    # 11. 시각화
    # ========================================================================
    logger.info("\n시각화 중...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Train 시계열
    axes[0, 0].plot(df_train['time'], y_train, label='Observed', alpha=0.7)
    axes[0, 0].plot(df_train['time'], y_train_pred, label='Predicted', alpha=0.7)
    axes[0, 0].set_title(f'Train: Observed vs Predicted (MAE={train_mae:.2f})')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('PM2.5 (µg/m³)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # (2) Val 시계열
    axes[0, 1].plot(df_val['time'], y_val, label='Observed', alpha=0.7)
    axes[0, 1].plot(df_val['time'], y_val_pred, label='Predicted', alpha=0.7)
    axes[0, 1].set_title(f'Val: Observed vs Predicted (MAE={val_mae:.2f})')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('PM2.5 (µg/m³)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # (3) Scatter plot (Val)
    axes[1, 0].scatter(y_val, y_val_pred, alpha=0.3)
    axes[1, 0].plot([0, y_val.max()], [0, y_val.max()], 'r--', label='1:1 line')
    axes[1, 0].set_title(f'Val: Scatter (R²={val_r2:.3f})')
    axes[1, 0].set_xlabel('Observed PM2.5 (µg/m³)')
    axes[1, 0].set_ylabel('Predicted PM2.5 (µg/m³)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # (4) Feature importance
    importance = model.feature_importance(importance_type='gain')
    axes[1, 1].barh(feature_cols, importance)
    axes[1, 1].set_title('Feature Importance (Gain)')
    axes[1, 1].set_xlabel('Importance')

    plt.tight_layout()

    plot_path = config.PLOTS_DIR / "pm25_training_results.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"✓ Plot saved: {plot_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ 학습 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
