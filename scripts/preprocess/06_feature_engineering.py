"""
Feature Engineering for PM2.5 Prediction
- Add lag features (1, 3, 6, 12 hours)
- Add moving averages (3, 6 hours)
- Add time encoding (hour, day-of-week)
- Prepare final training dataset
"""
import pandas as pd
from pathlib import Path
import config
import utils

logger = utils.setup_logging(__name__)


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Feature Engineering")
    logger.info("="*60)

    # ========================================================================
    # 1. Load labeled dataset
    # ========================================================================
    input_path = config.DATA_DIR / "dataset_train.parquet"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run 05_join_labels.py first")
        return

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded: {len(df):,} samples")

    # Rename grid_* columns to standard names for easier processing
    rename_map = {
        'grid_no2': 'no2',
        'grid_o3': 'o3'
    }
    df = df.rename(columns=rename_map)

    # ========================================================================
    # 2. Sort by time and location (required for lag/MA features)
    # ========================================================================
    df = df.sort_values(['lat', 'lon', 'time']).reset_index(drop=True)

    # ========================================================================
    # 3. Add time encoding features
    # ========================================================================
    logger.info("\nAdding time encoding...")
    df = utils.create_time_encoding(df, time_col='time')

    # ========================================================================
    # 4. Add lag features for PM2.5
    # ========================================================================
    logger.info("\nAdding lag features...")

    # PM2.5 lags (most important for forecasting)
    # config.LAG_HOURS = [1, 3]로 설정됨
    df = utils.create_lags(df, 'pm25', lags=config.LAG_HOURS, group_cols=['lat', 'lon'])

    # ========================================================================
    # 5. Add moving average features (SKIPPED - MA_WINDOWS = [])
    # ========================================================================
    # MA 특성은 실시간 구현 간소화를 위해 제거됨
    if config.MA_WINDOWS:
        logger.info("\nAdding moving average features...")
        df = utils.create_moving_averages(df, 'pm25', windows=config.MA_WINDOWS, group_cols=['lat', 'lon'])
    else:
        logger.info("\nSkipping moving average features (MA_WINDOWS is empty)")

    # ========================================================================
    # 6. Drop rows with NaN (from lag/MA operations)
    # ========================================================================
    logger.info("\nCleaning dataset...")

    n_before = len(df)
    # Keep only rows with all features (lags will create NaN at the start)
    df = df.dropna()
    n_after = len(df)

    logger.info(f"  Dropped {n_before - n_after:,} rows with NaN (from lag/MA)")
    logger.info(f"  Remaining: {n_after:,} samples")

    if len(df) == 0:
        logger.error("No data remaining after feature engineering. Check lag/MA settings.")
        return

    # ========================================================================
    # 7. Report missing values
    # ========================================================================
    utils.report_missing(df)

    # ========================================================================
    # 8. Save final dataset
    # ========================================================================
    output_path = config.FEATURES_PARQUET
    output_path.parent.mkdir(parents=True, exist_ok=True)

    utils.save_parquet(df, output_path, downcast=True)

    # ========================================================================
    # 9. Summary
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Feature Engineering Summary:")
    logger.info("="*60)

    feature_cols = [c for c in df.columns if c not in ['time', 'lat', 'lon', 'pm25', 'dist_km']]

    logger.info(f"  Total samples:     {len(df):,}")
    logger.info(f"  Time range:        {df['time'].min()} to {df['time'].max()}")
    logger.info(f"  Unique locations:  {len(df[['lat','lon']].drop_duplicates()):,}")
    logger.info(f"  Total features:    {len(feature_cols)}")
    logger.info(f"  Feature list:")

    # Group features by type
    lag_features = [c for c in feature_cols if 'lag' in c]
    ma_features = [c for c in feature_cols if 'ma' in c]
    time_features = [c for c in feature_cols if 'sin' in c or 'cos' in c]
    base_features = [c for c in feature_cols if c not in lag_features + ma_features + time_features]

    logger.info(f"    Base:       {', '.join(base_features)}")
    logger.info(f"    Lag:        {', '.join(lag_features)}")
    logger.info(f"    MA:         {', '.join(ma_features)}")
    logger.info(f"    Time:       {', '.join(time_features)}")

    logger.info(f"\n  Label (PM2.5):")
    logger.info(f"    Mean:       {df['pm25'].mean():.2f} µg/m³")
    logger.info(f"    Std:        {df['pm25'].std():.2f} µg/m³")
    logger.info(f"    Range:      [{df['pm25'].min():.1f}, {df['pm25'].max():.1f}]")

    logger.info(f"\n  Output:       {output_path}")
    logger.info("="*60)

    logger.info("\n✓ Feature engineering complete!")


if __name__ == "__main__":
    main()
