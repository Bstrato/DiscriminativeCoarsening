#!/usr/bin/env python3
"""
GNN with Selective Graph Coarsening - Main Regression Training Script

This script provides a comprehensive framework for training Graph Neural Networks
with selective coarsening on medical heterogeneous graph data for length-of-stay regression.

Features:
- Product Quantization based selective coarsening
- Memory-efficient GNN architecture with edge weight support for regression
- Gradient accumulation training with comprehensive evaluation
- Advanced visualization and result analysis
- Ablation study for different coarsening parameters

Usage:
    python mainv1_coarsening_regression.py
"""

import torch
import os
from data_coarsening_regression import (
    load_data_safely, analyze_node_features, apply_selective_coarsening,
    compare_with_baseline
)
from models_coarsening_regression import MemoryEfficientGNNRegression
from trainer_coarsening_regression import GradientAccumulationRegressionTrainer
from utils_coarsening_regression import (
    setup_cuda_environment, monitor_memory, aggressive_memory_cleanup
)


def handle_nan_targets(los_values):
    """Handle NaN values in LOS targets robustly"""
    print("\n" + "=" * 50)
    print("LOS TARGET CLEANING")
    print("=" * 50)

    original_count = len(los_values)
    nan_count = torch.isnan(los_values).sum()

    print(f"Original samples: {original_count:,}")
    print(f"NaN values found: {nan_count:,} ({100 * nan_count / original_count:.1f}%)")

    if nan_count > 0:
        if nan_count == original_count:
            print("❌ ERROR: All values are NaN!")
            return None

        # Calculate median of valid values
        valid_mask = ~torch.isnan(los_values)
        valid_values = los_values[valid_mask]
        median_los = valid_values.median().item()

        print(f"Valid LOS statistics:")
        print(f"  Count: {len(valid_values):,}")
        print(f"  Mean: {valid_values.mean():.3f} days")
        print(f"  Median: {median_los:.3f} days")
        print(f"  Std: {valid_values.std():.3f} days")
        print(f"  Range: {valid_values.min():.3f} - {valid_values.max():.3f} days")

        # Fill NaN with median
        cleaned_values = torch.where(torch.isnan(los_values), median_los, los_values)
        print(f"✅ Filled {nan_count:,} NaN values with median: {median_los:.3f} days")

        return cleaned_values
    else:
        print("✅ No NaN values found")
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

    # Percentiles
    percentiles = [25, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(y_continuous, p / 100.0)
        print(f"LOS {p}th percentile: {val:.4f} days")

    print(f"NaN count: {torch.isnan(y_continuous).sum()}")

    # Distribution analysis
    print(f"\nLOS Distribution:")
    short_mask = y_continuous <= 3.0
    medium_mask = (y_continuous > 3.0) & (y_continuous <= 7.0)
    long_mask = y_continuous > 7.0

    print(f"  Short (≤3 days): {short_mask.sum():,} ({100 * short_mask.sum() / len(y_continuous):.1f}%)")
    print(f"  Medium (3-7 days): {medium_mask.sum():,} ({100 * medium_mask.sum() / len(y_continuous):.1f}%)")
    print(f"  Long (>7 days): {long_mask.sum():,} ({100 * long_mask.sum() / len(y_continuous):.1f}%)")


def main():
    """Main function with selective coarsening for regression"""
    print("=== Medical GNN with Selective Graph Coarsening - REGRESSION ===")

    # Setup environment
    device = setup_cuda_environment()
    monitor_memory()

    # Data path - NOW USING REGRESSION FILE
    data_path = r"/nfs/hpc/share/bayitaas/stra/GNN/MIMIC_GraphCoarsening/GraphCreation_Code"
    regression_file = os.path.join(data_path, "regression_los_clinical_safe.pt")

    # Load original data
    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    print(f"Original data loaded:")
    print(f"Stay nodes: {original_data['stay'].x.size(0):,}")
    print(f"Node types: {list(original_data.x_dict.keys())}")
    print(f"Edge types: {list(original_data.edge_index_dict.keys())}")

    # Handle NaN values in continuous LOS targets
    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("❌ Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    # Analyze cleaned LOS targets
    analyze_los_targets(y_continuous)

    # Analyze original node features
    print(f"\nOriginal Data Analysis:")
    analyze_node_features(original_data)

    # Apply selective coarsening
    print(f"\nApplying selective coarsening...")
    coarsened_data, coarsening_info = apply_selective_coarsening(
        original_data,
        coarsen_node_types=['careunit', 'inputevent'],  # Only coarsen these types
        m=8,  # Number of subvectors for PQ
        k=16,  # Number of centroids per subspace
        random_state=42
    )

    # Transfer the y_continuous to coarsened data
    coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

    # Compare original vs coarsened
    compare_with_baseline(original_data, coarsened_data, coarsening_info)

    # Analyze coarsened node features
    print(f"\nCoarsened Data Analysis:")
    analyze_node_features(coarsened_data)

    # Metadata for coarsened graph
    metadata = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))

    # Model configuration
    config = {'hidden_dim': 64, 'num_layers': 2, 'accumulation_steps': 8, 'huber_delta': 1.0}

    print(f"\nModel Configuration: {config}")
    print(f"{'=' * 50}")

    aggressive_memory_cleanup()

    # Create model for coarsened graph
    model = MemoryEfficientGNNRegression(
        metadata,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    )

    # Move coarsened data to GPU
    print("Moving coarsened data to GPU...")
    coarsened_data_gpu = coarsened_data.to(device)

    # Initialize model
    model.to(device)
    with torch.no_grad():
        model.eval()
        _ = model(coarsened_data_gpu.x_dict, coarsened_data_gpu.edge_index_dict)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    monitor_memory()

    # Train and evaluate on coarsened graph
    trainer = GradientAccumulationRegressionTrainer(
        model, coarsened_data_gpu, coarsening_info, device,
        accumulation_steps=config['accumulation_steps'],
        huber_delta=config['huber_delta'],
        save_dir=f"./results_regression_coarsening_{config['hidden_dim']}_{config['num_layers']}"
    )
    trainer.create_data_splits()

    print(f"\nTraining on coarsened graph...")
    results = trainer.train(epochs=50, lr=0.005, patience=15, visualize_every=10)
    test_metrics = results['test_metrics']

    print(f"\n🏆 Final Results (Selective Coarsening - Regression):")
    print(f"MAE: {test_metrics['mae']:.3f} days")
    print(f"RMSE: {test_metrics['rmse']:.3f} days")
    print(f"R²: {test_metrics['r2']:.3f}")
    print(f"MAPE: {test_metrics['mape']:.1f}%")

    # Print coarsening impact summary
    print(f"\n📊 Coarsening Impact Summary:")
    total_original_nodes = sum(info['original_count'] for info in coarsening_info.values())
    total_coarsened_nodes = sum(info['coarsened_count'] for info in coarsening_info.values())
    total_reduction = (1 - total_coarsened_nodes / total_original_nodes) * 100

    print(f"Total node reduction: {total_reduction:.1f}%")
    print(f"Memory efficiency gained while preserving Stay node granularity")
    print(
        f"Care Unit nodes coarsened: {coarsening_info.get('careunit', {}).get('original_count', 0)} -> {coarsening_info.get('careunit', {}).get('coarsened_count', 0)}")
    print(
        f"Input Event nodes coarsened: {coarsening_info.get('inputevent', {}).get('original_count', 0)} -> {coarsening_info.get('inputevent', {}).get('coarsened_count', 0)}")

    # Cleanup
    del model, trainer, coarsened_data_gpu, original_data, coarsened_data
    aggressive_memory_cleanup()

    return test_metrics, coarsening_info


def run_ablation_study():
    """Run ablation study comparing different coarsening parameters for regression"""
    print(f"\n{'=' * 60}")
    print(f"ABLATION STUDY: Different Coarsening Parameters - REGRESSION")
    print(f"{'=' * 60}")

    # Setup environment
    device = setup_cuda_environment()

    # Data path
    data_path = r"/nfs/hpc/share/bayitaas/stra/GNN/MIMIC_GraphCoarsening/GraphCreation_Code"
    regression_file = os.path.join(data_path, "regression_los_clinical_safe.pt")

    # Load original data once
    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    # Handle NaN values in continuous LOS targets
    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("❌ Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    # Different parameter combinations to test
    param_combinations = [
        {'m': 4, 'k': 8, 'name': 'Conservative'},
        {'m': 8, 'k': 16, 'name': 'Moderate'},
        {'m': 16, 'k': 32, 'name': 'Aggressive'},
    ]

    results = []

    for params in param_combinations:
        print(f"\n{'=' * 40}")
        print(f"Testing {params['name']} coarsening (m={params['m']}, k={params['k']})")
        print(f"{'=' * 40}")

        # Apply coarsening with current parameters
        coarsened_data, coarsening_info = apply_selective_coarsening(
            original_data,
            coarsen_node_types=['careunit', 'inputevent'],
            m=params['m'],
            k=params['k'],
            random_state=42
        )

        # Transfer the y_continuous to coarsened data
        coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

        # Quick training (fewer epochs for ablation)
        metadata = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))
        model = MemoryEfficientGNNRegression(metadata, hidden_dim=64, num_layers=2)

        coarsened_data_gpu = coarsened_data.to(device)
        trainer = GradientAccumulationRegressionTrainer(
            model, coarsened_data_gpu, coarsening_info, device,
            save_dir=f"./ablation_regression_{params['name'].lower()}_{params['m']}_{params['k']}"
        )
        trainer.create_data_splits()

        training_results = trainer.train(epochs=25, lr=0.005)  # Fewer epochs for ablation
        test_metrics = training_results['test_metrics']

        # Store results
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

        # Cleanup
        del model, trainer, coarsened_data_gpu, coarsened_data
        aggressive_memory_cleanup()

    # Print ablation results
    print(f"\n{'=' * 80}")
    print(f"ABLATION STUDY RESULTS - REGRESSION")
    print(f"{'=' * 80}")
    print(f"{'Setting':<15} {'m':<3} {'k':<3} {'MAE':<6} {'RMSE':<6} {'R²':<6} {'MAPE':<6} {'Reduction':<9}")
    print(f"{'-' * 80}")

    for result in results:
        print(f"{result['name']:<15} {result['m']:<3} {result['k']:<3} "
              f"{result['mae']:<6.3f} {result['rmse']:<6.3f} "
              f"{result['r2']:<6.3f} {result['mape']:<6.1f} {result['node_reduction']:<9.1f}%")

    # Find best performing configuration
    best_mae = min(results, key=lambda x: x['mae'])
    best_r2 = max(results, key=lambda x: x['r2'])
    print(f"\nBest MAE Performance: {best_mae['name']} (MAE: {best_mae['mae']:.3f} days)")
    print(f"Best R² Performance: {best_r2['name']} (R²: {best_r2['r2']:.3f})")

    del original_data
    return results


def run_baseline_comparison():
    """Run comparison between coarsened and non-coarsened models"""
    print(f"\n{'=' * 60}")
    print(f"BASELINE COMPARISON: Coarsened vs Non-Coarsened")
    print(f"{'=' * 60}")

    # Setup environment
    device = setup_cuda_environment()

    # Data path
    data_path = r"/nfs/hpc/share/bayitaas/stra/GNN/MIMIC_GraphCoarsening/GraphCreation_Code"
    regression_file = os.path.join(data_path, "regression_los_clinical_safe.pt")

    # Load original data
    original_data = load_data_safely(regression_file)
    if original_data is None:
        return

    # Handle NaN values
    y_continuous = handle_nan_targets(original_data['stay'].y)
    if y_continuous is None:
        print("❌ Cannot proceed with all-NaN targets")
        return

    original_data['stay'].y_continuous = y_continuous

    # Train baseline (non-coarsened) model
    print(f"\n{'=' * 40}")
    print(f"Training BASELINE (Non-Coarsened) Model")
    print(f"{'=' * 40}")

    metadata_baseline = (list(original_data.x_dict.keys()), list(original_data.edge_index_dict.keys()))
    model_baseline = MemoryEfficientGNNRegression(metadata_baseline, hidden_dim=64, num_layers=2)

    original_data_gpu = original_data.to(device)
    trainer_baseline = GradientAccumulationRegressionTrainer(
        model_baseline, original_data_gpu, {}, device,
        save_dir="./baseline_regression_no_coarsening"
    )
    trainer_baseline.create_data_splits()

    baseline_results = trainer_baseline.train(epochs=30, lr=0.005)
    baseline_metrics = baseline_results['test_metrics']

    print(f"Baseline Results:")
    print(f"  MAE: {baseline_metrics['mae']:.3f} days")
    print(f"  RMSE: {baseline_metrics['rmse']:.3f} days")
    print(f"  R²: {baseline_metrics['r2']:.3f}")

    # Cleanup baseline
    del model_baseline, trainer_baseline, original_data_gpu
    aggressive_memory_cleanup()

    # Train coarsened model
    print(f"\n{'=' * 40}")
    print(f"Training COARSENED Model")
    print(f"{'=' * 40}")

    coarsened_data, coarsening_info = apply_selective_coarsening(
        original_data,
        coarsen_node_types=['careunit', 'inputevent'],
        m=8, k=16, random_state=42
    )
    coarsened_data['stay'].y_continuous = original_data['stay'].y_continuous

    metadata_coarsened = (list(coarsened_data.x_dict.keys()), list(coarsened_data.edge_index_dict.keys()))
    model_coarsened = MemoryEfficientGNNRegression(metadata_coarsened, hidden_dim=64, num_layers=2)

    coarsened_data_gpu = coarsened_data.to(device)
    trainer_coarsened = GradientAccumulationRegressionTrainer(
        model_coarsened, coarsened_data_gpu, coarsening_info, device,
        save_dir="./coarsened_regression_comparison"
    )
    trainer_coarsened.create_data_splits()

    coarsened_results = trainer_coarsened.train(epochs=30, lr=0.005)
    coarsened_metrics = coarsened_results['test_metrics']

    print(f"Coarsened Results:")
    print(f"  MAE: {coarsened_metrics['mae']:.3f} days")
    print(f"  RMSE: {coarsened_metrics['rmse']:.3f} days")
    print(f"  R²: {coarsened_metrics['r2']:.3f}")

    # Comparison summary
    print(f"\n{'=' * 60}")
    print(f"COMPARISON SUMMARY")
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

    # Node reduction summary
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

    # Cleanup
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
    # Run main experiment
    print("Starting main regression experiment with graph coarsening...")
    test_metrics, coarsening_info = main()

    print(f"\n" + "=" * 60)
    print("EXPERIMENT OPTIONS")
    print("=" * 60)
    print("1. Run ablation study to compare different coarsening parameters")
    print("2. Run baseline comparison (coarsened vs non-coarsened)")
    print("3. Both ablation study and baseline comparison")
    print("=" * 60)

    # For demonstration, run both studies
    # Uncomment the lines below to run additional experiments

    # Run ablation study
    print("\n🔬 Running Ablation Study...")
    ablation_results = run_ablation_study()

    # Run baseline comparison
    print("\n📊 Running Baseline Comparison...")
    comparison_results = run_baseline_comparison()

    print(f"\n✅ All experiments completed successfully!")
    print(f"Results saved in respective directories:")
    print(f"  - Main results: ./results_regression_coarsening_64_2/")
    print(f"  - Ablation results: ./ablation_regression_*/")
    print(f"  - Baseline comparison: ./baseline_regression_no_coarsening/ and ./coarsened_regression_comparison/")