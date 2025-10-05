"""
Comprehensive evaluation and visualization
- Compare all models
- Generate plots (timeseries, scatter, residuals, feature importance)
- Export metrics to JSON
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
import config
import utils

logger = utils.setup_logging(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_predictions() -> pd.DataFrame:
    """Load predictions from all models"""
    logger.info("Loading predictions...")

    # Load baseline predictions
    baseline_path = config.TABLES_DIR / 'baseline_predictions.parquet'
    df_baseline = utils.load_parquet(baseline_path)

    # Load residual model predictions
    residual_path = config.TABLES_DIR / 'residual_predictions.parquet'
    df_residual = utils.load_parquet(residual_path)

    # Merge
    df = df_baseline.merge(
        df_residual.drop(columns=['pm25'], errors='ignore'),
        on=['time', 'lat', 'lon'],
        how='inner'
    )

    logger.info(f"Loaded {len(df):,} prediction samples")

    return df


def calculate_all_metrics(df: pd.DataFrame) -> dict:
    """Calculate metrics for all models"""
    logger.info("Calculating metrics for all models...")

    all_metrics = {}

    y_true = df['pm25'].values

    # Find all prediction columns
    pred_cols = [c for c in df.columns if c.startswith('pm25_pred_')]

    for col in pred_cols:
        model_name = col.replace('pm25_pred_', '').title()
        y_pred = df[col].values

        metrics = utils.calculate_metrics(y_true, y_pred)
        metrics['model'] = model_name

        all_metrics[model_name] = metrics

        utils.print_metrics(metrics, model_name=model_name)

    return all_metrics


def plot_timeseries(df: pd.DataFrame, n_hours: int = 168):
    """
    Plot timeseries comparison (first n_hours)

    Args:
        df: Predictions DataFrame
        n_hours: Number of hours to plot (default: 168 = 1 week)
    """
    logger.info(f"Generating timeseries plot ({n_hours} hours)...")

    df_plot = df.sort_values('time').head(n_hours)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot observations
    ax.plot(df_plot['time'], df_plot['pm25'], 'k-', linewidth=2, label='Observed', alpha=0.7)

    # Plot predictions
    pred_cols = [c for c in df_plot.columns if c.startswith('pm25_pred_')]
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for i, col in enumerate(pred_cols):
        model_name = col.replace('pm25_pred_', '').title()
        ax.plot(df_plot['time'], df_plot[col], '--', color=colors[i % len(colors)],
                linewidth=1.5, label=model_name, alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel('PM2.5 (µg/m³)')
    ax.set_title(f'PM2.5 Predictions vs Observations (First {n_hours} hours)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_TIMESERIES, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {config.PLOT_TIMESERIES}")
    plt.close()


def plot_scatter(df: pd.DataFrame):
    """Plot scatter plots (observed vs predicted) for all models"""
    logger.info("Generating scatter plots...")

    pred_cols = [c for c in df.columns if c.startswith('pm25_pred_')]
    n_models = len(pred_cols)

    if n_models == 0:
        logger.warning("No prediction columns found")
        return

    # Create subplots
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    y_true = df['pm25'].values

    for i, col in enumerate(pred_cols):
        ax = axes[i]
        model_name = col.replace('pm25_pred_', '').title()
        y_pred = df[col].values

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)

        # 1:1 line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

        # Calculate R² and MAE
        from sklearn.metrics import r2_score, mean_absolute_error
        r2 = r2_score(y_true[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)])
        mae = mean_absolute_error(y_true[~np.isnan(y_pred)], y_pred[~np.isnan(y_pred)])

        ax.set_xlabel('Observed PM2.5 (µg/m³)')
        ax.set_ylabel('Predicted PM2.5 (µg/m³)')
        ax.set_title(f'{model_name}\nR² = {r2:.3f}, MAE = {mae:.2f}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(config.PLOT_SCATTER, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {config.PLOT_SCATTER}")
    plt.close()


def plot_residuals(df: pd.DataFrame):
    """Plot residual histograms for all models"""
    logger.info("Generating residual plots...")

    pred_cols = [c for c in df.columns if c.startswith('pm25_pred_')]
    n_models = len(pred_cols)

    if n_models == 0:
        logger.warning("No prediction columns found")
        return

    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))

    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    y_true = df['pm25'].values

    for i, col in enumerate(pred_cols):
        ax = axes[i]
        model_name = col.replace('pm25_pred_', '').title()
        y_pred = df[col].values

        residuals = y_true - y_pred
        residuals = residuals[~np.isnan(residuals)]

        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)

        ax.set_xlabel('Residual (Obs - Pred) µg/m³')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name}\nMean = {residuals.mean():.2f}, Std = {residuals.std():.2f}')
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(config.PLOT_RESIDUALS, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {config.PLOT_RESIDUALS}")
    plt.close()


def plot_feature_importance():
    """Plot feature importance for LightGBM model"""
    logger.info("Generating feature importance plot...")

    if not config.LGBM_MODEL.exists():
        logger.warning("LightGBM model not found, skipping feature importance")
        return

    # Load model
    with open(config.LGBM_MODEL, 'rb') as f:
        model = pickle.load(f)

    # Get feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(importance_df['feature'], importance_df['importance'])
    ax.set_xlabel('Importance (Gain)')
    ax.set_title('Top 20 Feature Importance (LightGBM)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(config.PLOT_FEATURE_IMP, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved to {config.PLOT_FEATURE_IMP}")
    plt.close()


def export_metrics(metrics: dict):
    """Export metrics to JSON"""
    logger.info(f"Exporting metrics to {config.METRICS_JSON}...")

    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for model, m in metrics.items():
        metrics_serializable[model] = {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in m.items()
        }

    with open(config.METRICS_JSON, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"✓ Saved to {config.METRICS_JSON}")


def print_comparison_table(metrics: dict):
    """Print comparison table of all models"""
    logger.info("\n" + "="*80)
    logger.info("Model Comparison Table:")
    logger.info("="*80)

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Sort by MAE
    metrics_df = metrics_df.sort_values('mae')

    # Print
    print(metrics_df[['r2', 'mae', 'rmse', 'mbe']].to_string())

    logger.info("="*80)

    # Calculate improvement over baseline
    if 'Baseline' in metrics:
        baseline_mae = metrics['Baseline']['mae']

        logger.info("\nImprovement over Baseline:")
        logger.info("-" * 80)

        for model, m in metrics.items():
            if model != 'Baseline':
                improvement = (baseline_mae - m['mae']) / baseline_mae * 100
                logger.info(f"  {model:20s}: {improvement:+6.1f}% (MAE: {m['mae']:.2f} µg/m³)")

        logger.info("-" * 80)


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("Model Evaluation and Visualization")
    logger.info("="*60)

    # ========================================================================
    # 1. Load predictions
    # ========================================================================
    df = load_predictions()

    # ========================================================================
    # 2. Calculate metrics
    # ========================================================================
    metrics = calculate_all_metrics(df)

    # ========================================================================
    # 3. Generate plots
    # ========================================================================
    plot_timeseries(df, n_hours=168)  # 1 week
    plot_scatter(df)
    plot_residuals(df)
    plot_feature_importance()

    # ========================================================================
    # 4. Export metrics
    # ========================================================================
    export_metrics(metrics)

    # ========================================================================
    # 5. Print comparison table
    # ========================================================================
    print_comparison_table(metrics)

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete!")
    logger.info("="*60)
    logger.info(f"  Metrics:  {config.METRICS_JSON}")
    logger.info(f"  Plots:    {config.PLOTS_DIR}/")
    logger.info("="*60)


if __name__ == "__main__":
    main()
