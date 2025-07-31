import torch
import gc
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


def setup_cuda_environment():
    """Setup CUDA environment and return device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return device
    else:
        device = torch.device('cpu')
        print("Using CPU")
        return device


def aggressive_memory_cleanup():
    """Aggressive memory cleanup for both GPU and CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def monitor_memory():
    """Monitor and print memory usage"""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1e9

    print(f"Memory Usage - CPU: {cpu_memory:.2f} GB", end="")

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_memory_cached = torch.cuda.memory_reserved() / 1e9
        print(f" | GPU: {gpu_memory:.2f} GB (Cached: {gpu_memory_cached:.2f} GB)")
    else:
        print()


def compute_comprehensive_regression_metrics(y_true, y_pred):
    """Compute comprehensive evaluation metrics for regression"""
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite_mask.sum() == 0:
        print("Error: No finite values for evaluation")
        return {
            'mae': float('inf'),
            'rmse': float('inf'),
            'r2': -1.0,
            'mape': float('inf')
        }

    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)

    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))) * 100

    naive_forecast_error = np.mean(np.abs(np.diff(y_true_clean)))
    if naive_forecast_error > 0:
        mase = mae / naive_forecast_error
    else:
        mase = float('inf')

    smape = 100 * np.mean(
        2 * np.abs(y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8))

    medae = np.median(np.abs(y_true_clean - y_pred_clean))

    explained_var = 1 - np.var(y_true_clean - y_pred_clean) / np.var(y_true_clean)

    max_error = np.max(np.abs(y_true_clean - y_pred_clean))

    short_mask = y_true_clean <= 3.0
    medium_mask = (y_true_clean > 3.0) & (y_true_clean <= 7.0)
    long_mask = y_true_clean > 7.0

    range_metrics = {}
    if short_mask.sum() > 0:
        range_metrics['mae_short'] = mean_absolute_error(y_true_clean[short_mask], y_pred_clean[short_mask])
        range_metrics['count_short'] = short_mask.sum()

    if medium_mask.sum() > 0:
        range_metrics['mae_medium'] = mean_absolute_error(y_true_clean[medium_mask], y_pred_clean[medium_mask])
        range_metrics['count_medium'] = medium_mask.sum()

    if long_mask.sum() > 0:
        range_metrics['mae_long'] = mean_absolute_error(y_true_clean[long_mask], y_pred_clean[long_mask])
        range_metrics['count_long'] = long_mask.sum()

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'mase': mase,
        'smape': smape,
        'medae': medae,
        'explained_variance': explained_var,
        'max_error': max_error,
        **range_metrics
    }

    return metrics


def print_comprehensive_regression_results(metrics, title="REGRESSION EVALUATION RESULTS"):
    """Print comprehensive regression results in a formatted way"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    print(f"Core Regression Metrics:")
    print(f"   MAE (Mean Absolute Error):     {metrics['mae']:.4f} days")
    print(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.4f} days")
    print(f"   R² (Coefficient of Determination): {metrics['r2']:.4f}")

    print(f"\nPercentage Error Metrics:")
    print(f"   MAPE (Mean Absolute % Error):   {metrics['mape']:.2f}%")
    print(f"   sMAPE (Symmetric MAPE):         {metrics['smape']:.2f}%")

    print(f"\nAdditional Metrics:")
    print(f"   Median Absolute Error:          {metrics['medae']:.4f} days")
    print(f"   Explained Variance:             {metrics['explained_variance']:.4f}")
    print(f"   Max Error:                      {metrics['max_error']:.4f} days")
    if metrics['mase'] != float('inf'):
        print(f"   MASE (Mean Absolute Scaled Error): {metrics['mase']:.4f}")

    print(f"\nPerformance by LOS Range:")
    if 'mae_short' in metrics:
        print(f"   Short stays (≤3d):  MAE {metrics['mae_short']:.3f} ({metrics['count_short']} samples)")
    if 'mae_medium' in metrics:
        print(f"   Medium stays (3-7d): MAE {metrics['mae_medium']:.3f} ({metrics['count_medium']} samples)")
    if 'mae_long' in metrics:
        print(f"   Long stays (>7d):    MAE {metrics['mae_long']:.3f} ({metrics['count_long']} samples)")

    print(f"{'=' * 60}\n")


def create_regression_scatter_plot(y_true, y_pred, save_path=None):
    """Create scatter plot of predictions vs actual values"""
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]

    plt.figure(figsize=(10, 8))

    plt.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=20, color='steelblue')

    min_val = min(y_true_clean.min(), y_pred_clean.min())
    max_val = max(y_true_clean.max(), y_pred_clean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)

    plt.title(f'Predictions vs Actual LOS\nMAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Actual LOS (days)', fontsize=12)
    plt.ylabel('Predicted LOS (days)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")

    plt.show()


def create_residual_plot(y_true, y_pred, save_path=None):
    """Create residual plot to analyze prediction errors"""
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]

    residuals = y_true_clean - y_pred_clean

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(y_pred_clean, residuals, alpha=0.6, s=20, color='steelblue')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted LOS (days)')
    ax1.set_ylabel('Residuals (Actual - Predicted)')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals (days)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plot saved to {save_path}")

    plt.show()


def create_regression_metrics_plot(metrics, save_path=None):
    """Create a bar plot comparing different regression metrics"""
    key_metrics = {
        'R² Score': max(0, metrics['r2']),
        'Explained Variance': max(0, metrics['explained_variance']),
        'MAE (normalized)': 1 / (1 + metrics['mae']),
        'RMSE (normalized)': 1 / (1 + metrics['rmse']),
    }

    plt.figure(figsize=(10, 6))

    metric_names = list(key_metrics.keys())
    metric_values = list(key_metrics.values())

    bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.8)

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Regression Performance Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Score (Higher is Better)', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")

    plt.show()


def create_los_distribution_plot(y_true, y_pred, save_path=None):
    """Create distribution comparison plot for actual vs predicted LOS"""
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]

    plt.figure(figsize=(12, 8))

    bins = np.linspace(0, max(y_true_clean.max(), y_pred_clean.max()), 50)

    plt.hist(y_true_clean, bins=bins, alpha=0.7, label='Actual LOS', color='steelblue', density=True)
    plt.hist(y_pred_clean, bins=bins, alpha=0.7, label='Predicted LOS', color='orange', density=True)

    plt.xlabel('Length of Stay (days)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution Comparison: Actual vs Predicted LOS', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")

    plt.show()


def create_error_by_range_plot(y_true, y_pred, save_path=None):
    """Create plot showing error distribution by LOS ranges"""
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[finite_mask]
    y_pred_clean = y_pred[finite_mask]

    abs_errors = np.abs(y_true_clean - y_pred_clean)

    short_mask = y_true_clean <= 3.0
    medium_mask = (y_true_clean > 3.0) & (y_true_clean <= 7.0)
    long_mask = y_true_clean > 7.0

    error_data = []
    range_labels = []

    if short_mask.sum() > 0:
        error_data.append(abs_errors[short_mask])
        range_labels.append(f'Short (≤3d)\nn={short_mask.sum()}')

    if medium_mask.sum() > 0:
        error_data.append(abs_errors[medium_mask])
        range_labels.append(f'Medium (3-7d)\nn={medium_mask.sum()}')

    if long_mask.sum() > 0:
        error_data.append(abs_errors[long_mask])
        range_labels.append(f'Long (>7d)\nn={long_mask.sum()}')

    plt.figure(figsize=(10, 6))

    box_plot = plt.boxplot(error_data, labels=range_labels, patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors[:len(error_data)]):
        patch.set_facecolor(color)

    plt.ylabel('Absolute Error (days)', fontsize=12)
    plt.xlabel('LOS Range', fontsize=12)
    plt.title('Prediction Error Distribution by LOS Range', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error by range plot saved to {save_path}")

    plt.show()


def analyze_embeddings_statistics_regression(model, data, epoch, device):
    """Analyze embedding statistics during training for regression"""
    model.eval()
    with torch.no_grad():
        print(f"\nModel Statistics (Epoch {epoch + 1}):")

        for node_type, projection in model.input_projections.items():
            weight = projection.weight.data
            print(f"   {node_type} projection - Mean: {weight.mean():.6f}, Std: {weight.std():.6f}")

        regressor_weight = model.regressor.weight.data
        print(f"   Regressor - Mean: {regressor_weight.mean():.6f}, Std: {regressor_weight.std():.6f}")

        dead_neurons = (regressor_weight.abs() < 1e-6).sum().item()
        total_neurons = regressor_weight.numel()
        print(f"   Dead neurons: {dead_neurons}/{total_neurons} ({100 * dead_neurons / total_neurons:.2f}%)")


def save_regression_results_to_file(metrics, filepath):
    """Save regression evaluation results to a text file"""
    with open(filepath, 'w') as f:
        f.write("COMPREHENSIVE REGRESSION EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Core Regression Metrics:\n")
        f.write(f"  MAE (Mean Absolute Error):      {metrics['mae']:.6f} days\n")
        f.write(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.6f} days\n")
        f.write(f"  R² (Coefficient of Determination): {metrics['r2']:.6f}\n\n")

        f.write("Percentage Error Metrics:\n")
        f.write(f"  MAPE (Mean Absolute % Error):   {metrics['mape']:.6f}%\n")
        f.write(f"  sMAPE (Symmetric MAPE):         {metrics['smape']:.6f}%\n\n")

        f.write("Additional Metrics:\n")
        f.write(f"  Median Absolute Error:          {metrics['medae']:.6f} days\n")
        f.write(f"  Explained Variance:             {metrics['explained_variance']:.6f}\n")
        f.write(f"  Max Error:                      {metrics['max_error']:.6f} days\n")
        if metrics['mase'] != float('inf'):
            f.write(f"  MASE (Mean Absolute Scaled Error): {metrics['mase']:.6f}\n")
        f.write("\n")

        f.write("Performance by LOS Range:\n")
        if 'mae_short' in metrics:
            f.write(f"  Short stays (≤3d):  MAE {metrics['mae_short']:.6f} ({metrics['count_short']} samples)\n")
        if 'mae_medium' in metrics:
            f.write(f"  Medium stays (3-7d): MAE {metrics['mae_medium']:.6f} ({metrics['count_medium']} samples)\n")
        if 'mae_long' in metrics:
            f.write(f"  Long stays (>7d):    MAE {metrics['mae_long']:.6f} ({metrics['count_long']} samples)\n")

    print(f"Detailed regression results saved to {filepath}")


def create_tsne_visualization_regression(embeddings, targets, save_path=None):
    """Create t-SNE visualization of embeddings colored by regression targets"""
    if embeddings is None:
        print("No embeddings provided for t-SNE visualization")
        return

    print("Computing t-SNE visualization for regression...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=targets, cmap='viridis', alpha=0.7, s=20)

    plt.colorbar(scatter, label='Length of Stay (days)')
    plt.title('t-SNE Visualization of Node Embeddings\n(Colored by LOS)', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")

    plt.show()