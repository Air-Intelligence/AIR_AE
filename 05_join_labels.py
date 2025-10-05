"""
Join OpenAQ PM2.5 labels with TEMPO/MERRA-2 features
- Time matching with tolerance
- Spatial matching using KDTree
- Report matching statistics
"""
import pandas as pd
from pathlib import Path
import config
import utils

logger = utils.setup_logging(__name__)


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Joining Labels with Features")
    logger.info("="*60)

    # ========================================================================
    # 1. Load data
    # ========================================================================
    logger.info("Loading data...")

    # Load features (TEMPO + MERRA-2 merged)
    features_path = config.MERGED_PARQUET
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.error("Please run 04_preprocess_merge.py first")
        return

    feat = pd.read_parquet(features_path)
    logger.info(f"  Features: {len(feat):,} rows")

    # Load labels (OpenAQ PM2.5)
    labels_path = config.OPENAQ_CSV
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        logger.error("Please run 01_download_openaq.py first")
        return

    labl = pd.read_csv(labels_path)
    logger.info(f"  Labels:   {len(labl):,} rows")

    # ========================================================================
    # 2. Normalize column names and time
    # ========================================================================
    logger.info("\nNormalizing data...")

    # Ensure consistent column names (lowercase)
    if 'pm25' not in labl.columns and 'value' in labl.columns:
        labl = labl.rename(columns={'value': 'pm25'})

    # Convert time to UTC-naive
    feat["time"] = utils.to_utc_naive(feat["time"])
    labl["time"] = utils.to_utc_naive(labl["time"])

    logger.info(f"  Feature time range: {feat['time'].min()} to {feat['time'].max()}")
    logger.info(f"  Label time range:   {labl['time'].min()} to {labl['time'].max()}")

    # ========================================================================
    # 3. Time matching with tolerance
    # ========================================================================
    logger.info("\nMatching by time...")

    merged = utils.merge_time_nearest(
        labl, feat,
        tol=config.LABEL_JOIN["time_tolerance"]
    )

    # Check time matching success
    n_matched_time = merged.dropna(subset=['no2', 'o3']).shape[0]
    time_match_rate = n_matched_time / len(labl) * 100 if len(labl) > 0 else 0
    logger.info(f"  Time matching success: {n_matched_time:,}/{len(labl):,} ({time_match_rate:.1f}%)")

    if n_matched_time == 0:
        logger.error("No time matches found. Check time ranges and tolerance.")
        return

    # ========================================================================
    # 4. Spatial matching
    # ========================================================================
    logger.info("\nMatching by space...")

    # Prepare grid dataframe (unique lat/lon with features)
    grid_df = merged[['lat', 'lon', 'no2', 'o3']].dropna().drop_duplicates()

    if len(grid_df) == 0:
        logger.error("No valid grid points for spatial matching")
        return

    # Attach nearest grid features to labels
    result = utils.attach_nearest_grid(merged, grid_df)

    # ========================================================================
    # 5. Final data preparation
    # ========================================================================
    logger.info("\nPreparing final dataset...")

    # Select columns: label (pm25) + features + metadata
    final_cols = [
        'time', 'lat', 'lon',
        'pm25',  # Label
        'grid_no2', 'grid_o3',  # Features
        'dist_km'  # Distance to nearest grid
    ]

    # Keep only rows with all required columns
    available_cols = [c for c in final_cols if c in result.columns]
    dataset = result[available_cols].copy()

    # Drop rows with missing label or features
    dataset = dataset.dropna(subset=['pm25', 'grid_no2', 'grid_o3'])

    # ========================================================================
    # 6. Report statistics
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Matching Statistics:")
    logger.info("="*60)

    if len(dataset) > 0:
        logger.info(f"  Original labels:     {len(labl):,}")
        logger.info(f"  After time match:    {n_matched_time:,} ({time_match_rate:.1f}%)")
        logger.info(f"  Final dataset:       {len(dataset):,}")
        logger.info(f"  Spatial match rate:  {len(dataset)/len(labl)*100:.1f}%")
        logger.info(f"  Mean distance:       {dataset['dist_km'].mean():.2f} km")
        logger.info(f"  Median distance:     {dataset['dist_km'].median():.2f} km")
        logger.info(f"  Max distance:        {dataset['dist_km'].max():.2f} km")
        logger.info(f"  Unique locations:    {len(dataset[['lat','lon']].drop_duplicates()):,}")
        logger.info(f"  Time range:          {dataset['time'].min()} to {dataset['time'].max()}")
    else:
        logger.error("No final matches. Check spatial tolerance and data coverage.")
        return

    # ========================================================================
    # 7. Save output
    # ========================================================================
    output_path = config.DATA_DIR / "dataset_train.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(output_path, index=False)

    size_mb = output_path.stat().st_size / 1024**2
    logger.info(f"\n✓ Saved: {output_path} ({size_mb:.2f} MB)")

    # ========================================================================
    # 8. Data summary
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Dataset Summary:")
    logger.info("="*60)
    logger.info(f"  Samples:        {len(dataset):,}")
    logger.info(f"  Features:       {len([c for c in dataset.columns if c.startswith('grid_')])}")
    logger.info(f"  PM2.5 mean:     {dataset['pm25'].mean():.2f} µg/m³")
    logger.info(f"  PM2.5 std:      {dataset['pm25'].std():.2f} µg/m³")
    logger.info(f"  PM2.5 range:    [{dataset['pm25'].min():.1f}, {dataset['pm25'].max():.1f}]")
    logger.info("="*60)

    logger.info("\n✓ Label joining complete!")


if __name__ == "__main__":
    main()
