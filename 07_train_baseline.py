"""
Train baseline (persistence) model
Prediction: PM2.5(t) = PM2.5(t-1)

This serves as the benchmark for evaluating the satellite-based model.
"""
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import config
import utils

logger = utils.setup_logging(__name__)


class PersistenceModel:
    """
    Persistence model: predict current value = previous value
    """
    def __init__(self, lag_hours: int = 1):
        """
        Args:
            lag_hours: Number of hours to look back (default: 1)
        """
        self.lag_hours = lag_hours
        self.name = f"Persistence (t-{lag_hours}h)"

    def fit(self, X, y):
        """No training needed for persistence model"""
        logger.info(f"{self.name}: No training required")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict using persistence

        Args:
            df: DataFrame with 'time' and 'pm25' columns

        Returns:
            Array of predictions (PM2.5 at t-lag_hours)
        """
        df = df.copy().sort_values('time')

        # Shift PM2.5 by lag_hours
        predictions = df['pm25'].shift(self.lag_hours).values

        return predictions

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Evaluate model on DataFrame

        Args:
            df: DataFrame with 'time' and 'pm25' columns

        Returns:
            Dict of metrics
        """
        y_true = df['pm25'].values
        y_pred = self.predict(df)

        metrics = utils.calculate_metrics(y_true, y_pred)
        metrics['model'] = self.name

        return metrics


def split_train_val(df: pd.DataFrame) -> tuple:
    """
    Split data into train and validation sets by time

    Args:
        df: Input DataFrame with 'time' column

    Returns:
        (train_df, val_df)
    """
    logger.info("Splitting train/validation by time...")

    df = df.sort_values('time')

    # Calculate split point
    split_date = df['time'].min() + timedelta(weeks=config.TRAIN_WEEKS)

    train_df = df[df['time'] < split_date].copy()
    val_df = df[df['time'] >= split_date].copy()

    logger.info(f"Train: {train_df['time'].min()} to {train_df['time'].max()} ({len(train_df):,} samples)")
    logger.info(f"Val:   {val_df['time'].min()} to {val_df['time'].max()} ({len(val_df):,} samples)")

    return train_df, val_df


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Baseline (Persistence) Model Training")
    logger.info("="*60)

    # ========================================================================
    # 1. Load feature-engineered data
    # ========================================================================
    logger.info(f"Loading data from {config.FEATURES_PARQUET}...")
    df = utils.load_parquet(config.FEATURES_PARQUET)

    # Check if PM2.5 exists
    if 'pm25' not in df.columns:
        logger.error("pm25 column not found in data. Cannot train baseline model.")
        return

    # ========================================================================
    # 2. Split train/validation
    # ========================================================================
    train_df, val_df = split_train_val(df)

    # ========================================================================
    # 3. Create and fit persistence model
    # ========================================================================
    model = PersistenceModel(lag_hours=1)

    # "Fit" (no-op for persistence)
    model.fit(None, None)

    # ========================================================================
    # 4. Evaluate on validation set
    # ========================================================================
    logger.info("\nEvaluating on validation set...")

    val_metrics = model.evaluate(val_df)

    utils.print_metrics(val_metrics, model_name="Baseline (Persistence)")

    # ========================================================================
    # 5. Save model
    # ========================================================================
    logger.info(f"\nSaving model to {config.BASELINE_MODEL}...")

    with open(config.BASELINE_MODEL, 'wb') as f:
        pickle.dump(model, f)

    logger.info("✓ Model saved")

    # ========================================================================
    # 6. Save validation predictions for later comparison
    # ========================================================================
    val_df_out = val_df[['time', 'lat', 'lon', 'pm25']].copy()
    val_df_out['pm25_pred_baseline'] = model.predict(val_df)

    baseline_pred_path = config.TABLES_DIR / 'baseline_predictions.parquet'
    utils.save_parquet(val_df_out, baseline_pred_path, downcast=True)

    logger.info(f"✓ Validation predictions saved to {baseline_pred_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Baseline Training Summary:")
    logger.info("="*60)
    logger.info(f"  Model:       {model.name}")
    logger.info(f"  Val MAE:     {val_metrics['mae']:.2f} µg/m³")
    logger.info(f"  Val R²:      {val_metrics['r2']:.4f}")
    logger.info(f"  Val RMSE:    {val_metrics['rmse']:.2f} µg/m³")
    logger.info("="*60)


if __name__ == "__main__":
    main()
