#!/usr/bin/env python3
"""
GNN with Selective Graph Coarsening - Main Regression Training Script

This script provides a comprehensive framework for training Graph Neural Networks
with selective coarsening on medical heterogeneous graph data for length-of-stay regression.

Features:
- Product Quantization based selective coarsening
- Multiple GNN architectures (Memory Efficient, HAN, HGT) with edge weight support for regression
- Gradient accumulation training with comprehensive evaluation
- Advanced visualization and result analysis
- Ablation study for different coarsening parameters
- Command line argument support for easy experimentation

Usage:
    python main_coarsening_regression.py --model memory_efficient --hidden_dim 64 --num_layers 2
    python main_coarsening_regression.py --model han --hidden_dim 128 --num_layers 3 --heads 4
    python main_coarsening_regression.py --model hgt --hidden_dim 64 --num_heads 8
"""

import torch
import os
import argparse
from data_coarsening_regression import (
    load_data_safely, analyze_node_features, apply_selective_coarsening,
    compare_with_baseline
)
from models_coarsening_regression import (
    MemoryEfficientGNNRegression,
    HANRegressionModel,
    HGTRegressionModel,
    create_regression_model
)
from trainer_coarsening_regression import GradientAccumulationRegressionTrainer
from utils_coarsening_regression import (
    setup_cuda_environment, monitor_memory, aggressive_memory_cleanup
)


def parse_arguments():
    """Parse command line arguments for flexible experimentation"""
    parser = argparse.ArgumentParser(
        description='Medical GNN with Graph Coarsening for Length-of-Stay Regression',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--model', type=str, default='hgt',
                        choices=['memory_efficient', 'han', 'hgt'],
                        help='GNN architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size for embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate for regularization')

    parser.add_argument('--heads', type=int, default=2,
                        help='Number of attention heads for HAN model')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='Number of attention heads for HGT model')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta parameter for Huber loss')
    parser.add_argument('--visualize_every', type=int, default=10,
                        help='Visualization frequency during training')

    parser.add_argument('--data_path', type=str,
                        default=r"/nfs/hpc/share/bayitaas/stra/GNN/MIMIC_GraphCoarsening/GraphCreation_Code",
                        help='Path to data directory')
    parser.add_argument('--regression_file', type=str, default="regression_los_clinical_safe.pt",
                        help='Regression data file name')

    parser.add_argument('--coarsen_types', nargs='+', default=['careunit', 'inputevent'],
                        help='Node types to coarsen')
    parser.add_argument('--pq_m', type=int, default=8,
                        help='Number of subvectors for Product Quantization')
    parser.add_argument('--pq_k', type=int, default=16,
                        help='Number of centroids per subspace')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')

    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation data ratio')

    parser.add_argument('--run_ablation', action='store_true',
                        help='Run ablation study after main experiment')
    parser.add_argument('--run_baseline', action='store_true',
                        help='Run baseline comparison after main experiment')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Custom save directory (auto-generated if not provided)')

    return parser.parse_args()


def handle_nan_targets(los_values):
    """Handle NaN values in LOS targets"""
    print("\n" + "=" * 50)
    print("LOS TARGET CLEANING")
    print("=" * 50)

    original_count = len(los_values)
    nan_count = torch.isnan(los_values).sum()

    print(f"Original samples: {original_count:,}")
    print(f"NaN values found: {nan_count:,} ({100 * nan_count / original_count:.1f}%)")

    if nan_count > 0:
        if nan_count == original_count:
            print("ERROR: All values are NaN!")
            return None

        valid_mask = ~torch.isnan(los_values)
        valid_values = los_values[valid_mask]
        median_los = valid_values.median().item()

        print(f"Valid LOS statistics:")
        print(f"  Count: {len(valid_values):,}")
        print(f"  Mean: {valid_values.mean():.3f} days")
        print(f"  Median: {median_los:.3f} days")
        print(f"  Std: {valid_values.std():.3f} days")
        print(f"  Range: {valid_values.min():.3f} - {valid_values.max():.3f} days")

        cleaned_values = torch.where(torch.isnan(los_values), median_los, los_values)
        print(f"Filled {nan_count:,} NaN values with median: {median_los:.3f} days")

        return cleaned_values
    else:
        print("No NaN values found")
        return los_values


def analyze_los_targets(y_continuous):
    """Analyze the continuous LOS targets"""
    print("\n" + "=" * 50)
    print("FINAL LOS TARGET ANALYSIS")
    print("=" * 50)
    print(f"LOS Shape: {y_continuous.shape}")
    print(f"LOS Mean: {y_continuous.mean():.4f} days")
    print(f"LOS Std: {y_continuous.std():.4f} days")
    print(f"LOS Min: {y_continuous.min():.4f} days")
    print(f"LOS Max: {y_continuous.max():.4f} days")
    print(f"LOS Median: {y_continuous.median():.4f} days")

    percentiles = [25, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(y_continuous, p / 100.0)
        print(f"LOS {p}th percentile: {val:.4f} days")

    print(f"NaN count: {torch.isnan(y_continuous).sum()}")

    print(f"\nLOS Distribution:")
    short_mask = y_continuous <= 3.0
    medium_mask = (y_continuous > 3.0) & (y_continuous <= 7.0)
    long_mask = y_continuous > 7.0

    print(f"  Short (≤3 days): {short_mask.sum():,} ({100 * short_mask.sum() / len(y_continuous):.1f}%)")
    print(f"  Medium (3-7 days): {medium_mask.sum():,} ({100 * medium_mask.sum() / len(y_continuous):.1f}%)")
    print(f"  Long (>7 days): {long_mask.sum():,} ({100 * long_mask.sum() / len(y_continuous):.1f}%)")


def print_experiment_summary(args):
    """Print experiment configuration summary"""
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Model Architecture: {args.model.upper()}")
    print(f"Hidden Dimension: {args.hidden_dim}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"Dropout Rate: {args.dropout}")

    if args.model == 'han':
        print(f"Attention Heads (HAN): {args.heads}")
    elif args.model == 'hgt':
        print(f"Attention Heads (HGT): {args.num_heads}")

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Gradient Accumulation Steps: {args.accumulation_steps}")
    print(f"  Huber Loss Delta: {args.huber_delta}")
    print(f"  Early Stopping Patience: {args.patience}")

    print(f"\nCoarsening Configuration:")
    print(f"  Node Types to Coarsen: {args.coarsen_types}")
    print(f"  PQ Parameters: m={args.pq_m}, k={args.pq_k}")
    print(f"  Random State: {args.random_state}")

    print(f"\nData Configuration:")
    print(
        f"  Train/Val/Test Split: {args.train_ratio:.1f}/{args.val_ratio:.1f}/{1 - args.train_ratio - args.val_ratio:.1f}")
    print(f"  Data Path: {args.data_path}")
    print(f"  Regression File: {args.regression_file}")

    print(f"\nExperiment Control:")
    print(f"  Run Ablation Study: {'Yes' if args.run_ablation else 'No'}")
    print(f"  Run Baseline Comparison: {'Yes' if args.run_baseline else 'No'}")
    print(f"  Save Directory: {args.save_dir if args.save_dir else 'Auto-generated'}")
    print("=" * 80)


def main(args):
    """Main function with selective coarsening for regression"""
    print("=== Medical GNN with Selective Graph Coarsening - REGRESSION ===")

    print_experiment_summary(args)

    device = setup_cuda_environment()
    monitor_memory()

    regression_file = os.path.join(args.data_path, args.regression_file)

    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    print(f"Original data loaded:")
    print(f"Stay nodes: {original_data['stay'].x.size(0):,}")
    print(f"Node types: {list(original_data.x_dict.keys())}")
    print(f"Edge types: {list(original_data.edge_index_dict.keys())}")

    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    analyze_los_targets(y_continuous)

    print(f"\nOriginal Data Analysis:")
    analyze_node_features(original_data)

    print(f"\nApplying selective coarsening...")
    coarsened_data, coarsening_info = apply_selective_coarsening(
        original_data,
        coarsen_node_types=args.coarsen_types,
        m=args.pq_m,
        k=args.pq_k,
        random_state=args.random_state
    )

    coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

    compare_with_baseline(original_data, coarsened_data, coarsening_info)

    print(f"\nCoarsened Data Analysis:")
    analyze_node_features(coarsened_data)

    metadata = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))

    if args.save_dir is None:
        args.save_dir = f"./results_regression_coarsening_{args.model}_{args.hidden_dim}_{args.num_layers}"

    print(f"\nResults will be saved to: {args.save_dir}")
    print(f"{'=' * 50}")

    aggressive_memory_cleanup()

    model = create_regression_model(
        model_name=args.model,
        metadata=metadata,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        num_heads=args.num_heads
    )

    print("Moving coarsened data to GPU...")
    coarsened_data_gpu = coarsened_data.to(device)

    model.to(device)
    with torch.no_grad():
        model.eval()
        _ = model(coarsened_data_gpu.x_dict, coarsened_data_gpu.edge_index_dict)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    monitor_memory()

    trainer = GradientAccumulationRegressionTrainer(
        model, coarsened_data_gpu, coarsening_info, device,
        accumulation_steps=args.accumulation_steps,
        huber_delta=args.huber_delta,
        save_dir=args.save_dir
    )
    trainer.create_data_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    print(f"\nTraining {args.model.upper()} model on coarsened graph...")
    results = trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        visualize_every=args.visualize_every
    )
    test_metrics = results['test_metrics']

    print(f"\nFinal Results ({args.model.upper()} Model with Selective Coarsening):")
    print(f"MAE: {test_metrics['mae']:.3f} days")
    print(f"RMSE: {test_metrics['rmse']:.3f} days")
    print(f"R²: {test_metrics['r2']:.3f}")
    print(f"MAPE: {test_metrics['mape']:.1f}%")

    print(f"\nCoarsening Impact Summary:")
    total_original_nodes = sum(info['original_count'] for info in coarsening_info.values())
    total_coarsened_nodes = sum(info['coarsened_count'] for info in coarsening_info.values())
    total_reduction = (1 - total_coarsened_nodes / total_original_nodes) * 100

    print(f"Total node reduction: {total_reduction:.1f}%")
    print(f"Memory efficiency gained while preserving Stay node granularity")
    for node_type in args.coarsen_types:
        if node_type in coarsening_info:
            info = coarsening_info[node_type]
            print(f"{node_type.title()} nodes coarsened: {info['original_count']} -> {info['coarsened_count']}")

    del model, trainer, coarsened_data_gpu, original_data, coarsened_data
    aggressive_memory_cleanup()

    return test_metrics, coarsening_info


def run_ablation_study(args):
    """Run ablation study comparing different coarsening parameters for regression"""
    print(f"\n{'=' * 60}")
    print(f"ABLATION STUDY: Different Coarsening Parameters - {args.model.upper()} MODEL")
    print(f"{'=' * 60}")

    device = setup_cuda_environment()

    regression_file = os.path.join(args.data_path, args.regression_file)

    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    param_combinations = [
        {'m': 4, 'k': 8, 'name': 'Conservative'},
        {'m': 8, 'k': 16, 'name': 'Moderate'},
        {'m': 16, 'k': 32, 'name': 'Aggressive'},
    ]

    results = []

    for params in param_combinations:
        print(f"\n{'=' * 40}")
        print(f"Testing {params['name']} coarsening (m={params['m']}, k={params['k']}) with {args.model.upper()} model")
        print(f"{'=' * 40}")

        coarsened_data, coarsening_info = apply_selective_coarsening(
            original_data,
            coarsen_node_types=args.coarsen_types,
            m=params['m'],
            k=params['k'],
            random_state=args.random_state
        )

        coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

        metadata = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))
        model = create_regression_model(
            model_name=args.model,
            metadata=metadata,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            heads=args.heads,
            num_heads=args.num_heads
        )

        coarsened_data_gpu = coarsened_data.to(device)
        save_dir = f"./ablation_regression_{args.model}_{params['name'].lower()}_{params['m']}_{params['k']}"

        trainer = GradientAccumulationRegressionTrainer(
            model, coarsened_data_gpu, coarsening_info, device,
            accumulation_steps=args.accumulation_steps,
            huber_delta=args.huber_delta,
            save_dir=save_dir
        )
        trainer.create_data_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)

        training_results = trainer.train(epochs=25, lr=args.lr, weight_decay=args.weight_decay)
        test_metrics = training_results['test_metrics']

        total_nodes = sum(info['coarsened_count'] for info in coarsening_info.values())
        total_reduction = (1 - total_nodes / sum(info['original_count'] for info in coarsening_info.values())) * 100

        results.append({
            'name': params['name'],
            'm': params['m'],
            'k': params['k'],
            'mae': test_metrics['mae'],
            'rmse': test_metrics['rmse'],
            'r2': test_metrics['r2'],
            'mape': test_metrics['mape'],
            'node_reduction': total_reduction
        })

        del model, trainer, coarsened_data_gpu, coarsened_data
        aggressive_memory_cleanup()

    print(f"\n{'=' * 80}")
    print(f"ABLATION STUDY RESULTS - {args.model.upper()} MODEL")
    print(f"{'=' * 80}")
    print(f"{'Setting':<15} {'m':<3} {'k':<3} {'MAE':<6} {'RMSE':<6} {'R²':<6} {'MAPE':<6} {'Reduction':<9}")
    print(f"{'-' * 80}")

    for result in results:
        print(f"{result['name']:<15} {result['m']:<3} {result['k']:<3} "
              f"{result['mae']:<6.3f} {result['rmse']:<6.3f} "
              f"{result['r2']:<6.3f} {result['mape']:<6.1f} {result['node_reduction']:<9.1f}%")

    best_mae = min(results, key=lambda x: x['mae'])
    best_r2 = max(results, key=lambda x: x['r2'])
    print(f"\nBest MAE Performance: {best_mae['name']} (MAE: {best_mae['mae']:.3f} days)")
    print(f"Best R² Performance: {best_r2['name']} (R²: {best_r2['r2']:.3f})")

    del original_data
    return results


def run_baseline_comparison(args):
    """Run comparison between coarsened and non-coarsened models"""
    print(f"\n{'=' * 60}")
    print(f"BASELINE COMPARISON: Coarsened vs Non-Coarsened - {args.model.upper()} MODEL")
    print(f"{'=' * 60}")

    device = setup_cuda_environment()

    regression_file = os.path.join(args.data_path, args.regression_file)

    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    print(f"\n{'=' * 40}")
    print(f"Training BASELINE (Non-Coarsened) {args.model.upper()} Model")
    print(f"{'=' * 40}")

    metadata_baseline = (list(original_data.x_dict.keys()), list(original_data.edge_index_dict.keys()))
    model_baseline = create_regression_model(
        model_name=args.model,
        metadata=metadata_baseline,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        num_heads=args.num_heads
    )

    original_data_gpu = original_data.to(device)
    trainer_baseline = GradientAccumulationRegressionTrainer(
        model_baseline, original_data_gpu, {}, device,
        accumulation_steps=args.accumulation_steps,
        huber_delta=args.huber_delta,
        save_dir=f"./baseline_regression_{args.model}_no_coarsening"
    )
    trainer_baseline.create_data_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    baseline_results = trainer_baseline.train(epochs=30, lr=args.lr, weight_decay=args.weight_decay)
    baseline_metrics = baseline_results['test_metrics']

    print(f"Baseline {args.model.upper()} Results:")
    print(f"  MAE: {baseline_metrics['mae']:.3f} days")
    print(f"  RMSE: {baseline_metrics['rmse']:.3f} days")
    print(f"  R²: {baseline_metrics['r2']:.3f}")

    del model_baseline, trainer_baseline, original_data_gpu
    aggressive_memory_cleanup()

    print(f"\n{'=' * 40}")
    print(f"Training COARSENED {args.model.upper()} Model")
    print(f"{'=' * 40}")

    coarsened_data, coarsening_info = apply_selective_coarsening(
        original_data,
        coarsen_node_types=args.coarsen_types,
        m=args.pq_m, k=args.pq_k, random_state=args.random_state
    )
    coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

    metadata_coarsened = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))
    model_coarsened = create_regression_model(
        model_name=args.model,
        metadata=metadata_coarsened,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        num_heads=args.num_heads
    )

    coarsened_data_gpu = coarsened_data.to(device)
    trainer_coarsened = GradientAccumulationRegressionTrainer(
        model_coarsened, coarsened_data_gpu, coarsening_info, device,
        accumulation_steps=args.accumulation_steps,
        huber_delta=args.huber_delta,
        save_dir=f"./coarsened_regression_{args.model}_comparison"
    )
    trainer_coarsened.create_data_splits(train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    coarsened_results = trainer_coarsened.train(epochs=30, lr=args.lr, weight_decay=args.weight_decay)
    coarsened_metrics = coarsened_results['test_metrics']

    print(f"Coarsened {args.model.upper()} Results:")
    print(f"  MAE: {coarsened_metrics['mae']:.3f} days")
    print(f"  RMSE: {coarsened_metrics['rmse']:.3f} days")
    print(f"  R²: {coarsened_metrics['r2']:.3f}")

    print(f"\n{'=' * 60}")
    print(f"COMPARISON SUMMARY - {args.model.upper()} MODEL")
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Baseline':<12} {'Coarsened':<12} {'Improvement':<12}")
    print(f"{'-' * 60}")

    mae_improvement = ((baseline_metrics['mae'] - coarsened_metrics['mae']) / baseline_metrics['mae']) * 100
    rmse_improvement = ((baseline_metrics['rmse'] - coarsened_metrics['rmse']) / baseline_metrics['rmse']) * 100
    r2_improvement = ((coarsened_metrics['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2'])) * 100

    print(
        f"{'MAE (days)':<15} {baseline_metrics['mae']:<12.3f} {coarsened_metrics['mae']:<12.3f} {mae_improvement:<12.2f}%")
    print(
        f"{'RMSE (days)':<15} {baseline_metrics['rmse']:<12.3f} {coarsened_metrics['rmse']:<12.3f} {rmse_improvement:<12.2f}%")
    print(f"{'R²':<15} {baseline_metrics['r2']:<12.3f} {coarsened_metrics['r2']:<12.3f} {r2_improvement:<12.2f}%")

    total_original_nodes = sum(info['original_count'] for info in coarsening_info.values())
    total_coarsened_nodes = sum(info['coarsened_count'] for info in coarsening_info.values())
    total_reduction = (1 - total_coarsened_nodes / total_original_nodes) * 100

    print(f"\nNode Reduction: {total_reduction:.1f}%")
    print(f"Memory Efficiency: Achieved {total_reduction:.1f}% node reduction")
    if mae_improvement > 0:
        print(f"Performance Gain: {mae_improvement:.2f}% MAE improvement with coarsening")
    else:
        print(
            f"Performance Trade-off: {abs(mae_improvement):.2f}% MAE degradation for {total_reduction:.1f}% memory savings")

    del model_coarsened, trainer_coarsened, coarsened_data_gpu, original_data, coarsened_data
    aggressive_memory_cleanup()

    return {
        'baseline': baseline_metrics,
        'coarsened': coarsened_metrics,
        'node_reduction': total_reduction,
        'mae_improvement': mae_improvement,
        'rmse_improvement': rmse_improvement,
        'r2_improvement': r2_improvement
    }


if __name__ == "__main__":
    args = parse_arguments()

    print("Starting main regression experiment with graph coarsening...")
    test_metrics, coarsening_info = main(args)

    ablation_results = None
    comparison_results = None

    if args.run_ablation:
        print(f"\nRunning Ablation Study with {args.model.upper()} model...")
        ablation_results = run_ablation_study(args)

    if args.run_baseline:
        print(f"\nRunning Baseline Comparison with {args.model.upper()} model...")
        comparison_results = run_baseline_comparison(args)

    print(f"\nAll experiments completed successfully!")
    print(f"Model Used: {args.model.upper()}")
    print(f"Main Results saved in: {args.save_dir}")

    if args.run_ablation:
        print(f"Ablation results saved in: ./ablation_regression_{args.model}_*/")
    if args.run_baseline:
        print(
            f"Baseline comparison saved in: ./baseline_regression_{args.model}_*/ and ./coarsened_regression_{args.model}_*/")

    print(f"\nFinal {args.model.upper()} Model Performance:")
    print(f"  MAE: {test_metrics['mae']:.3f} days")
    print(f"  RMSE: {test_metrics['rmse']:.3f} days")
    print(f"  R²: {test_metrics['r2']:.3f}")
    print(f"  MAPE: {test_metrics['mape']:.1f}%")

    print(f"\nUsage Examples:")
    print(f"# Run with Memory Efficient GNN:")
    print(f"python main_coarsening_regression.py --model memory_efficient --hidden_dim 64 --num_layers 2")
    print(f"")
    print(f"# Run with HAN model and more attention heads:")
    print(f"python main_coarsening_regression.py --model han --hidden_dim 128 --heads 4 --run_ablation")
    print(f"")
    print(f"# Run with HGT model and baseline comparison:")
    print(f"python main_coarsening_regression.py --model hgt --hidden_dim 64 --num_heads 8 --run_baseline")
    print(f"")
    print(f"# Run comprehensive experiment with all studies:")
    print(f"python main_coarsening_regression.py --model han --hidden_dim 64 --run_ablation --run_baseline --epochs 75")
    print(f"")
    print(f"# Quick test with fewer epochs:")
    print(f"python main_coarsening_regression.py --model memory_efficient --epochs 20 --patience 5")