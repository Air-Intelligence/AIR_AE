"""
Utility functions for Bay Area air quality prediction pipeline
"""
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import logging
from typing import Union, Dict, List, Tuple
from scipy.stats.mstats import winsorize

import config

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(name: str = __name__) -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


# ============================================================================
# DATA LOADING & SAVING
# ============================================================================

def save_parquet(df: pd.DataFrame, path: Union[str, Path],
                 downcast: bool = True) -> None:
    """
    Save DataFrame to Parquet with optional downcasting

    Args:
        df: DataFrame to save
        path: Output file path
        downcast: If True, downcast float64 to float32 to save space
    """
    logger = logging.getLogger(__name__)

    if downcast:
        # Downcast float64 to float32
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
        logger.info(f"Downcasted {len(float_cols)} columns to float32")

    df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)

    size_mb = Path(path).stat().st_size / 1024**2
    logger.info(f"✓ Saved to {path} ({size_mb:.1f} MB)")


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Load DataFrame from Parquet"""
    logger = logging.getLogger(__name__)
    df = pd.read_parquet(path)
    logger.info(f"✓ Loaded {path} ({len(df):,} rows, {len(df.columns)} columns)")
    return df


# ============================================================================
# QUALITY CONTROL (QC)
# ============================================================================

def apply_qc(df: pd.DataFrame, thresholds: Dict = None) -> pd.DataFrame:
    """
    Apply quality control: remove negative/unrealistic values

    Args:
        df: Input DataFrame
        thresholds: Dict of {variable: (min, max)} thresholds

    Returns:
        QC'd DataFrame
    """
    logger = logging.getLogger(__name__)

    if thresholds is None:
        thresholds = config.QC_THRESHOLDS

    n_before = len(df)

    # Extract variable names from thresholds (format: 'no2_min', 'no2_max')
    variables = set()
    for key in thresholds.keys():
        if key.endswith('_min') or key.endswith('_max'):
            var_name = key.rsplit('_', 1)[0]  # 'no2_min' → 'no2'
            variables.add(var_name)

    # Apply thresholds
    for var in variables:
        if var in df.columns:
            min_key = f'{var}_min'
            max_key = f'{var}_max'

            if min_key in thresholds and max_key in thresholds:
                min_val = thresholds[min_key]
                max_val = thresholds[max_key]

                # 음수를 0으로 대체 (측정 불확도로 인한 음수 처리)
                n_negative = (df[var] < min_val).sum()
                if n_negative > 0:
                    df.loc[df[var] < min_val, var] = min_val
                    logger.info(f"  {var}: {n_negative:,} negative values → {min_val}")

                # 최대값 초과는 NaN으로 처리 (이상치)
                n_too_high = (df[var] > max_val).sum()
                if n_too_high > 0:
                    df.loc[df[var] > max_val, var] = np.nan
                    logger.info(f"  {var}: {n_too_high:,} extreme values (>{max_val}) → NaN")

    # Drop rows with all NaNs
    df = df.dropna(how='all')

    n_after = len(df)
    logger.info(f"QC: {n_before:,} → {n_after:,} rows ({n_before-n_after:,} removed)")

    return df


def winsorize_dataframe(df: pd.DataFrame,
                        lower: float = None,
                        upper: float = None,
                        exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Apply winsorization to clip outliers

    Args:
        df: Input DataFrame
        lower: Lower percentile (e.g., 0.001 for 0.1%)
        upper: Upper percentile (e.g., 0.999 for 99.9%)
        exclude_cols: Columns to exclude (e.g., time, lat, lon)

    Returns:
        Winsorized DataFrame
    """
    logger = logging.getLogger(__name__)

    if lower is None:
        lower = config.WINSORIZE_LOWER
    if upper is None:
        upper = config.WINSORIZE_UPPER

    if exclude_cols is None:
        exclude_cols = ['time', 'lat', 'lon']

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_winsorize = [c for c in numeric_cols if c not in exclude_cols]

    for col in cols_to_winsorize:
        df[col] = winsorize(df[col], limits=(lower, upper))

    logger.info(f"Winsorized {len(cols_to_winsorize)} columns "
               f"(limits: {lower:.3f}, {upper:.3f})")

    return df


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report missing value statistics

    Returns:
        DataFrame with missing value stats per column
    """
    logger = logging.getLogger(__name__)

    missing_stats = pd.DataFrame({
        'column': df.columns,
        'n_missing': df.isnull().sum().values,
        'pct_missing': (df.isnull().sum() / len(df) * 100).values
    }).sort_values('n_missing', ascending=False)

    logger.info("\n" + "="*60)
    logger.info("Missing Value Report:")
    logger.info("="*60)
    for _, row in missing_stats.iterrows():
        if row['n_missing'] > 0:
            logger.info(f"  {row['column']:20s}: {row['n_missing']:8,.0f} "
                       f"({row['pct_missing']:5.1f}%)")
    logger.info("="*60 + "\n")

    return missing_stats


# ============================================================================
# XARRAY / NETCDF UTILITIES
# ============================================================================

def subset_bbox(ds: xr.Dataset, bbox: Dict = None) -> xr.Dataset:
    """
    Subset xarray Dataset to bounding box

    Args:
        ds: Input xarray Dataset
        bbox: Dict with 'west', 'south', 'east', 'north' keys

    Returns:
        Subsetted Dataset
    """
    logger = logging.getLogger(__name__)

    if bbox is None:
        bbox = config.BBOX

    # Handle longitude wrapping (0-360 vs -180 to 180)
    lon_dim = 'lon' if 'lon' in ds.dims else 'longitude'
    lat_dim = 'lat' if 'lat' in ds.dims else 'latitude'

    lon_vals = ds[lon_dim].values

    # Check if longitude is 0-360
    if lon_vals.min() >= 0 and lon_vals.max() > 180:
        # Convert bbox to 0-360
        west = bbox['west'] if bbox['west'] >= 0 else bbox['west'] + 360
        east = bbox['east'] if bbox['east'] >= 0 else bbox['east'] + 360
    else:
        west = bbox['west']
        east = bbox['east']

    south = bbox['south']
    north = bbox['north']

    # V04 불규칙 그리드 체크 (2D 좌표 배열인지 확인)
    if ds[lon_dim].ndim == 2:
        logger.info(f"불규칙 그리드 감지 (V04) - Boolean masking 방식 사용")
        # Boolean masking 방식 (V04용)
        mask = (
            (ds[lon_dim] >= west) & (ds[lon_dim] <= east) &
            (ds[lat_dim] >= south) & (ds[lat_dim] <= north)
        )
        ds_sub = ds.where(mask, drop=True)

        # 그리드 포인트 수 계산 (실제 남은 유효 셀 개수)
        n_points = int(mask.sum().values)
        logger.info(f"BBOX subset: {bbox['name']} → {n_points:,} grid points (masked)")
    else:
        logger.info(f"정규 그리드 감지 (V03) - Slice 방식 사용")
        # Slice 방식 (V03용, 기존 코드)
        ds_sub = ds.sel(
            {lon_dim: slice(west, east),
             lat_dim: slice(south, north)}
        )

        n_points = len(ds_sub[lat_dim]) * len(ds_sub[lon_dim])
        logger.info(f"BBOX subset: {bbox['name']} → {n_points:,} grid points")

    return ds_sub


def netcdf_to_tidy(ds: xr.Dataset,
                   var_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Convert xarray Dataset to tidy (long-form) DataFrame

    Args:
        ds: Input xarray Dataset
        var_mapping: Dict mapping original variable names to standard names

    Returns:
        Tidy DataFrame with columns: time, lat, lon, var1, var2, ...
    """
    logger = logging.getLogger(__name__)

    # Stack dimensions to long format
    df = ds.to_dataframe().reset_index()

    # Rename variables if mapping provided
    if var_mapping:
        df = df.rename(columns=var_mapping)
        logger.info(f"Renamed variables: {var_mapping}")

    # Drop MultiIndex if exists
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # Standardize coordinate names (latitude→lat, longitude→lon)
    coord_mapping = {}
    if 'latitude' in df.columns:
        coord_mapping['latitude'] = 'lat'
    if 'longitude' in df.columns:
        coord_mapping['longitude'] = 'lon'
    if coord_mapping:
        df = df.rename(columns=coord_mapping)
        logger.info(f"Standardized coordinates: {coord_mapping}")

    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    logger.info(f"Converted to tidy format: {len(df):,} rows")

    return df


# ============================================================================
# TIME AND SPATIAL UTILITIES
# ============================================================================

def to_utc_naive(s: pd.Series) -> pd.Series:
    """
    Convert time series to UTC timezone-naive datetime

    Args:
        s: Pandas Series with datetime values

    Returns:
        UTC timezone-naive Series
    """
    s = pd.to_datetime(s, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def dedup_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate grid points by averaging

    Args:
        df: DataFrame with time, lat, lon columns

    Returns:
        Deduplicated DataFrame
    """
    logger = logging.getLogger(__name__)

    n_before = len(df)
    df = df.dropna(subset=["time", "lat", "lon"])
    df = df.groupby(["time", "lat", "lon"], as_index=False).mean(numeric_only=True)
    n_after = len(df)

    if n_before > n_after:
        logger.info(f"Deduplication: {n_before:,} → {n_after:,} grid points")

    return df


def merge_time_nearest(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tol: str = "30min",
    direction: str = "nearest"
) -> pd.DataFrame:
    """
    Merge DataFrames on nearest time with tolerance

    Args:
        left: Left DataFrame with 'time' column
        right: Right DataFrame with 'time' column
        tol: Time tolerance (e.g., "30min", "1H")
        direction: "nearest", "forward", or "backward"

    Returns:
        Merged DataFrame
    """
    logger = logging.getLogger(__name__)

    left_sorted = left.sort_values("time").copy()
    right_sorted = right.sort_values("time").copy()

    # 공통 좌표 컬럼 (lat, lon)은 left만 사용하고 right에서 제거
    common_coords = ['lat', 'lon', 'latitude', 'longitude']
    right_data_only = right_sorted.drop(columns=[c for c in common_coords if c in right_sorted.columns], errors='ignore')

    merged = pd.merge_asof(
        left_sorted,
        right_data_only,
        on="time",
        tolerance=pd.Timedelta(tol),
        direction=direction
    )

    logger.info(f"Time merge: {len(left):,} ⋈ {len(right):,} → {len(merged):,} rows (tol={tol})")

    return merged


def attach_nearest_grid(
    obs_df: pd.DataFrame,
    grid_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach nearest grid values to observation points using KDTree

    Args:
        obs_df: Observation DataFrame with lat, lon columns
        grid_df: Grid DataFrame with lat, lon and feature columns

    Returns:
        obs_df with grid features attached and distance column
    """
    from sklearn.neighbors import BallTree
    import numpy as np

    logger = logging.getLogger(__name__)

    # Build BallTree on grid points (in radians for haversine)
    grid_coords = np.deg2rad(grid_df[["lat", "lon"]].to_numpy())
    kdt = BallTree(grid_coords, metric='haversine')

    # Query nearest grid point for each observation
    obs_coords = np.deg2rad(obs_df[["lat", "lon"]].to_numpy())
    dist, idx = kdt.query(obs_coords, k=1)

    # Convert haversine distance to km
    earth_radius_km = 6371.0088
    obs_df = obs_df.copy()
    obs_df["nearest_idx"] = idx.ravel()
    obs_df["dist_km"] = dist.ravel() * earth_radius_km

    # Attach grid features
    grid_sel = grid_df.reset_index(drop=True).iloc[obs_df["nearest_idx"]].reset_index(drop=True)
    grid_sel = grid_sel.add_prefix("grid_")

    result = pd.concat([obs_df.reset_index(drop=True), grid_sel], axis=1)

    logger.info(f"Spatial matching: {len(obs_df):,} obs → mean distance {obs_df['dist_km'].mean():.2f} km")

    return result


# ============================================================================
# TIME SERIES UTILITIES
# ============================================================================

def resample_time(df: pd.DataFrame,
                  freq: str = None,
                  time_col: str = 'time') -> pd.DataFrame:
    """
    Resample time series to specified frequency

    Args:
        df: Input DataFrame with time column
        freq: Resampling frequency (e.g., '3H', '1D')
        time_col: Name of time column

    Returns:
        Resampled DataFrame
    """
    logger = logging.getLogger(__name__)

    if freq is None:
        freq = config.RESAMPLE_FREQ

    if freq is None:
        logger.info("Skipping time resampling (RESAMPLE_FREQ = None)")
        return df

    # Set time as index
    df = df.set_index(time_col)

    # Identify grouping columns (non-numeric)
    group_cols = [c for c in df.columns
                  if c not in df.select_dtypes(include=[np.number]).columns]

    if group_cols:
        # Group by spatial coordinates and resample
        df = df.groupby(group_cols).resample(freq).mean()
        df = df.reset_index()
    else:
        # Simple resample
        df = df.resample(freq).mean().reset_index()

    logger.info(f"Resampled to {freq}: {len(df):,} rows")

    return df


def align_time_grids(dfs: List[pd.DataFrame],
                     method: str = 'inner') -> pd.DataFrame:
    """
    Align multiple DataFrames on common time grid

    Args:
        dfs: List of DataFrames with 'time', 'lat', 'lon' columns
        method: Join method ('inner', 'outer', 'left')

    Returns:
        Merged DataFrame
    """
    logger = logging.getLogger(__name__)

    if len(dfs) == 0:
        raise ValueError("No DataFrames provided")

    if len(dfs) == 1:
        return dfs[0]

    # Merge sequentially
    merged = dfs[0]
    for i, df in enumerate(dfs[1:], 1):
        n_before = len(merged)
        merged = merged.merge(
            df,
            on=['time', 'lat', 'lon'],
            how=method,
            suffixes=('', f'_dup{i}')
        )
        n_after = len(merged)
        logger.info(f"Merge step {i}: {n_before:,} ⋈ {len(df):,} → {n_after:,} rows")

    return merged


# ============================================================================
# FEATURE ENGINEERING HELPERS
# ============================================================================

def create_time_encoding(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """
    Add cyclical time encoding (sin/cos) for hour and day-of-week

    Args:
        df: Input DataFrame with time column
        time_col: Name of time column

    Returns:
        DataFrame with added time features
    """
    logger = logging.getLogger(__name__)

    df = df.copy()

    # Ensure datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Hour of day (0-23)
    hour = df[time_col].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0-6)
    dow = df[time_col].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    logger.info("Added time encoding: hour_sin/cos, dow_sin/cos")

    return df


def create_lags(df: pd.DataFrame,
                var: str,
                lags: List[int] = None,
                group_cols: List[str] = None) -> pd.DataFrame:
    """
    Create lagged features for a variable

    Args:
        df: Input DataFrame
        var: Variable name to lag
        lags: List of lag hours (e.g., [1, 3, 6, 12])
        group_cols: Columns to group by (e.g., ['lat', 'lon'])

    Returns:
        DataFrame with added lag features
    """
    logger = logging.getLogger(__name__)

    if lags is None:
        lags = config.LAG_HOURS

    df = df.copy()

    if group_cols:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df.groupby(group_cols)[var].shift(lag)
    else:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)

    logger.info(f"Created {len(lags)} lag features for {var}: {lags}")

    return df


def create_moving_averages(df: pd.DataFrame,
                          var: str,
                          windows: List[int] = None,
                          group_cols: List[str] = None) -> pd.DataFrame:
    """
    Create moving average features

    Args:
        df: Input DataFrame
        var: Variable name
        windows: List of window sizes in hours (e.g., [3, 6])
        group_cols: Columns to group by

    Returns:
        DataFrame with added MA features
    """
    logger = logging.getLogger(__name__)

    if windows is None:
        windows = config.MA_WINDOWS

    df = df.copy()

    for window in windows:
        if group_cols:
            df[f'{var}_ma{window}'] = df.groupby(group_cols)[var].rolling(
                window=window, min_periods=1
            ).mean().reset_index(level=group_cols, drop=True)
        else:
            df[f'{var}_ma{window}'] = df[var].rolling(
                window=window, min_periods=1
            ).mean()

    logger.info(f"Created {len(windows)} MA features for {var}: {windows}")

    return df


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics

    Returns:
        Dict with R², MAE, RMSE, MBE
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # Remove NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mbe': np.mean(y_pred - y_true),  # Mean Bias Error
        'n_samples': len(y_true)
    }

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """Pretty print metrics"""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*60)
    logger.info(f"{model_name} Performance:")
    logger.info("="*60)
    logger.info(f"  R²:   {metrics['r2']:7.4f}")
    logger.info(f"  MAE:  {metrics['mae']:7.2f} µg/m³")
    logger.info(f"  RMSE: {metrics['rmse']:7.2f} µg/m³")
    logger.info(f"  MBE:  {metrics['mbe']:7.2f} µg/m³")
    logger.info(f"  N:    {metrics['n_samples']:7,}")
    logger.info("="*60 + "\n")


# ============================================================================
# STORAGE & SAFETY GUARDS
# ============================================================================

import shutil
import os
import fcntl
import time

def check_disk_usage(path: Union[str, Path] = "/") -> Dict[str, float]:
    """
    디스크 사용량 체크 (GB 단위)

    Args:
        path: 확인할 경로 (기본: 루트)

    Returns:
        {'total_gb': float, 'used_gb': float, 'free_gb': float, 'usage_pct': float}
    """
    logger = logging.getLogger(__name__)

    stat = shutil.disk_usage(path)
    total_gb = stat.total / (1024 ** 3)
    used_gb = stat.used / (1024 ** 3)
    free_gb = stat.free / (1024 ** 3)
    usage_pct = (stat.used / stat.total) * 100

    result = {
        'total_gb': total_gb,
        'used_gb': used_gb,
        'free_gb': free_gb,
        'usage_pct': usage_pct
    }

    logger.info(f"디스크 사용량: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_pct:.1f}%)")

    # 경고: 설정된 한계 초과 시
    if config.DISK_CHECK_ENABLED and used_gb > config.DISK_USAGE_LIMIT_GB:
        logger.warning(f"⚠️ 디스크 사용량이 한계({config.DISK_USAGE_LIMIT_GB}GB)를 초과했습니다!")

    return result


def atomic_save_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    validate_func=None,
    downcast: bool = True
) -> bool:
    """
    원자적 Parquet 저장 (임시 파일 → 검증 → rename)

    Args:
        df: 저장할 DataFrame
        path: 최종 저장 경로
        validate_func: 검증 함수 (df, path를 받아 True/False 반환)
        downcast: float64 → float32 변환 여부

    Returns:
        성공 시 True, 실패 시 False
    """
    logger = logging.getLogger(__name__)
    path = Path(path)
    tmp_path = Path(str(path) + ".tmp")

    try:
        # 1. Downcast (optional)
        if downcast:
            float_cols = df.select_dtypes(include=['float64']).columns
            if len(float_cols) > 0:
                df[float_cols] = df[float_cols].astype('float32')

        # 2. 임시 파일로 저장
        df.to_parquet(tmp_path, index=False, compression='snappy')
        logger.info(f"임시 파일 생성: {tmp_path} ({tmp_path.stat().st_size / 1024**2:.1f} MB)")

        # 3. 검증
        if validate_func:
            if not validate_func(df, tmp_path):
                logger.error(f"검증 실패: {tmp_path}")
                tmp_path.unlink()  # 임시 파일 삭제
                return False

        # 4. 원자적 rename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.rename(path)
        logger.info(f"✓ 저장 완료: {path}")
        return True

    except Exception as e:
        logger.error(f"저장 실패: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def validate_features(df: pd.DataFrame, path: Union[str, Path]) -> bool:
    """
    피처 파일 검증 (파일 크기, row 수, 시간 범위)

    Args:
        df: 검증할 DataFrame
        path: 파일 경로

    Returns:
        검증 통과 시 True
    """
    logger = logging.getLogger(__name__)
    path = Path(path)

    try:
        # 1. 파일 크기 체크 (최소 1KB)
        if path.stat().st_size < 1024:
            logger.error(f"파일 크기가 너무 작음: {path.stat().st_size} bytes")
            return False

        # 2. Row 수 체크 (최소 1개)
        if len(df) == 0:
            logger.error("DataFrame이 비어있음")
            return False

        # 3. 시간 컬럼 존재 확인
        if 'time' in df.columns:
            if df['time'].isna().all():
                logger.error("시간 컬럼이 모두 NaN")
                return False

            time_range = df['time'].max() - df['time'].min()
            logger.info(f"시간 범위: {df['time'].min()} ~ {df['time'].max()} ({time_range})")

        # 4. 결측률 체크
        missing_pct = (df.isna().sum() / len(df) * 100).mean()
        if missing_pct > 50:
            logger.warning(f"평균 결측률이 높음: {missing_pct:.1f}%")

        logger.info(f"✓ 검증 통과: {len(df):,} rows, {len(df.columns)} columns")
        return True

    except Exception as e:
        logger.error(f"검증 중 에러: {e}")
        return False


def deduplicate_by_time(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """
    시간 기준 중복 제거 (마지막 데이터 유지)

    Args:
        df: DataFrame
        time_col: 시간 컬럼명

    Returns:
        중복 제거된 DataFrame
    """
    logger = logging.getLogger(__name__)

    n_before = len(df)
    df = df.drop_duplicates(subset=[time_col], keep='last')
    n_after = len(df)

    if n_before > n_after:
        logger.info(f"중복 제거: {n_before:,} → {n_after:,} ({n_before - n_after:,}개 제거)")

    return df


def cleanup_old_files(directory: Union[str, Path], days: int, pattern: str = "*") -> int:
    """
    오래된 파일 삭제 (mtime 기준)

    Args:
        directory: 대상 디렉터리
        days: 며칠 초과 파일 삭제
        pattern: 파일 패턴 (예: "*.parquet")

    Returns:
        삭제된 파일 개수
    """
    logger = logging.getLogger(__name__)
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"디렉터리가 존재하지 않음: {directory}")
        return 0

    cutoff_time = time.time() - (days * 86400)  # 86400초 = 1일
    deleted_count = 0
    deleted_size = 0

    for file in directory.rglob(pattern):
        if file.is_file() and file.stat().st_mtime < cutoff_time:
            file_size = file.stat().st_size
            try:
                file.unlink()
                deleted_count += 1
                deleted_size += file_size
                logger.debug(f"삭제: {file}")
            except Exception as e:
                logger.error(f"삭제 실패: {file} - {e}")

    if deleted_count > 0:
        logger.info(f"✓ {deleted_count}개 파일 삭제 ({deleted_size / 1024**2:.1f} MB)")

    return deleted_count


class FileLock:
    """
    PID 파일 기반 중복 실행 방지

    Usage:
        with FileLock('/tmp/script.lock'):
            # 여기서 작업 수행
            pass
    """

    def __init__(self, lock_file: Union[str, Path]):
        self.lock_file = Path(lock_file)
        self.fp = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        try:
            self.fp = open(self.lock_file, 'w')
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fp.write(str(os.getpid()))
            self.fp.flush()
            self.logger.info(f"락 획득: {self.lock_file}")
            return self
        except IOError:
            self.logger.error(f"이미 실행 중: {self.lock_file}")
            raise RuntimeError(f"이미 실행 중인 프로세스가 있습니다: {self.lock_file}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fp:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
            self.lock_file.unlink(missing_ok=True)
            self.logger.info(f"락 해제: {self.lock_file}")


if __name__ == "__main__":
    # Test logging
    logger = setup_logging()
    logger.info("Utils module loaded successfully")
