"""
Preprocess and merge TEMPO NO2 and O3 data
- BBOX subsetting
- Tidy transformation
- Time alignment
- QC
- Merge on time+lat+lon grid
"""
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 프로젝트 루트를 Python path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from glob import glob
from typing import Dict, List
import config
import utils

logger = utils.setup_logging(__name__)


def process_tempo_files(
    file_pattern: str,
    var_name_mapping: Dict[str, str],
    product_name: str
) -> pd.DataFrame:
    """
    Process TEMPO NetCDF files into tidy DataFrame

    Args:
        file_pattern: Glob pattern for NC files
        var_name_mapping: Mapping of NetCDF var names to standard names
        product_name: Product name for logging (NO2, O3, etc.)

    Returns:
        Tidy DataFrame
    """
    logger.info(f"Processing TEMPO {product_name}...")

    files = sorted(glob(file_pattern))
    if len(files) == 0:
        logger.error(f"No files found matching: {file_pattern}")
        return pd.DataFrame()

    logger.info(f"Found {len(files)} files")

    all_dfs = []

    for i, file in enumerate(files, 1):
        try:
            # Open dataset with lazy loading
            with xr.open_dataset(file, chunks="auto") as ds:
                # Subset to BBOX
                ds = utils.subset_bbox(ds)

                # Rename variables using mapping
                for raw_name, std_name in var_name_mapping.items():
                    if raw_name in ds:
                        ds = ds.rename({raw_name: std_name})

                # Convert to tidy
                df = utils.netcdf_to_tidy(ds, var_mapping=None)  # Already renamed

            all_dfs.append(df)

            if i % 50 == 0 or i == len(files):
                logger.info(f"  Processed {i}/{len(files)} files")

        except Exception as e:
            logger.error(f"  Error processing {Path(file).name}: {e}")
            continue

    if len(all_dfs) == 0:
        logger.error(f"No data extracted from {product_name} files")
        return pd.DataFrame()

    # Concatenate all
    df = pd.concat(all_dfs, ignore_index=True)

    # UTC normalization and deduplication
    df["time"] = utils.to_utc_naive(df["time"])
    df = utils.dedup_grid(df)

    logger.info(f"✓ {product_name}: {len(df):,} rows")

    return df




def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Preprocessing and Merging Data")
    logger.info("="*60)

    # ========================================================================
    # 1. Process TEMPO NO₂
    # ========================================================================
    df_no2 = process_tempo_files(
        file_pattern=str(config.RAW_TEMPO_NO2 / "*.nc"),
        var_name_mapping={'vertical_column_troposphere': 'no2'},
        product_name='NO2'
    )

    if len(df_no2) == 0:
        logger.error("TEMPO NO₂ processing failed. Aborting.")
        return

    # ========================================================================
    # 2. Process TEMPO O₃
    # ========================================================================
    df_o3 = process_tempo_files(
        file_pattern=str(config.RAW_TEMPO_O3 / "*.nc"),
        var_name_mapping={'vertical_column_total': 'o3'},  # Fixed: O3 uses vertical_column_total
        product_name='O3'
    )

    if len(df_o3) == 0:
        logger.error("TEMPO O₃ processing failed. Aborting.")
        return


    # ========================================================================
    # 3. Resample time grids (optional)
    # ========================================================================
    if config.RESAMPLE_FREQ is not None:
        logger.info(f"\nResampling to {config.RESAMPLE_FREQ}...")
        df_no2 = utils.resample_time(df_no2)
        df_o3 = utils.resample_time(df_o3)

    # ========================================================================
    # 4. Merge datasets using nearest time matching
    # ========================================================================
    logger.info("\nMerging NO2 and O3...")

    # Merge NO2 and O3
    df_merged = utils.merge_time_nearest(
        df_no2, df_o3,
        tol=config.LABEL_JOIN["time_tolerance"]
    )

    if len(df_merged) == 0:
        logger.error("Merge resulted in empty DataFrame. Check time/spatial alignment.")
        return

    logger.info(f"✓ Merged: {len(df_merged):,} rows, {len(df_merged.columns)} columns")

    # ========================================================================
    # 5. Apply QC
    # ========================================================================
    logger.info("\nApplying QC...")

    df_merged = utils.apply_qc(df_merged)

    logger.info(f"Merged columns: {df_merged.columns.tolist()}")
    logger.info(f"First few rows:\n{df_merged.head()}")

    # Drop rows with missing key features
    df_merged = df_merged.dropna(subset=["no2", "o3"])

    # Optional: Winsorize outliers
    # df_merged = utils.winsorize_dataframe(df_merged)

    # ========================================================================
    # 6. Report missing values
    # ========================================================================
    utils.report_missing(df_merged)

    # ========================================================================
    # 7. Save to Parquet
    # ========================================================================
    logger.info("\nSaving merged data...")

    # Ensure output directory exists
    config.MERGED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    utils.save_parquet(df_merged, config.MERGED_PARQUET, downcast=True)

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Preprocessing Summary:")
    logger.info("="*60)
    logger.info(f"  Time range:  {df_merged['time'].min()} to {df_merged['time'].max()}")
    logger.info(f"  Grid points: {len(df_merged):,}")
    logger.info(f"  Unique times: {df_merged['time'].nunique():,}")
    logger.info(f"  Unique locs:  {len(df_merged[['lat','lon']].drop_duplicates()):,}")
    logger.info(f"  Variables:   {', '.join([c for c in df_merged.columns if c not in ['time', 'lat', 'lon']])}")
    logger.info(f"  Output:      {config.MERGED_PARQUET}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
