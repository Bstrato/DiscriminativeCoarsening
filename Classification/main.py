#!/usr/bin/env python3
"""
Medical GNN Classification - Main Training Script
Training and evaluating Graph Neural Networks on medical heterogeneous graph data.
"""

import argparse
import os
import yaml
import torch
import warnings
from datetime import datetime

from data import load_data_safely, analyze_node_features, create_data_splits, get_class_weights
from models import get_model_class, create_model, MODEL_REGISTRY
from trainer import GradientAccumulationTrainer, ModelEvaluator
from utils import (
    setup_cuda_environment, monitor_memory, aggressive_memory_cleanup,
    print_comprehensive_results
)

warnings.filterwarnings('ignore')


class Config:
    """Configuration management for the training pipeline"""

    def __init__(self, config_dict=None):
        self.data_path = "/nfs/hpc/share/bayitaas/stra/GNN/MIMIC_GraphCoarsening/GraphCreation_Code"
        self.classification_file = "classification_los_clinical_safe.pt"

        self.model_type = "memory_efficient"
        self.hidden_dim = 64
        self.num_layers = 2
        self.num_classes = 3
        self.dropout = 0.2

        self.heads = 2
        self.num_heads = 2

        self.epochs = 50
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.accumulation_steps = 8
        self.patience = 10
        self.use_scheduler = True

        self.train_ratio = 0.7
        self.val_ratio = 0.15
        self.normalize_features = True
        self.augment_data = False

        self.visualize_every = 10
        self.class_names = ['Short Stay', 'Medium Stay', 'Long Stay']

        self.save_dir = f"./results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_model = True
        self.create_visualizations = True

        if config_dict:
            self.__dict__.update(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def save_config(self, save_path):
        """Save configuration to YAML file"""
        with open(save_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
        print(f"Configuration saved to {save_path}")


def setup_experiment(config):
    """Setup experiment directory and logging"""
    os.makedirs(config.save_dir, exist_ok=True)

    config_path = os.path.join(config.save_dir, 'config.yaml')
    config.save_config(config_path)

    print(f"Experiment setup complete")
    print(f"Results directory: {config.save_dir}")
    print(f"Configuration saved: {config_path}")

    return config.save_dir


def load_and_preprocess_data(config, device):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")

    classification_file = os.path.join(config.data_path, config.classification_file)

    data = load_data_safely(classification_file)
    if data is None:
        raise RuntimeError("Failed to load data")

    analyze_node_features(data)
    create_data_splits(data, device, config.train_ratio, config.val_ratio)

    print(f"Moving data to {device}...")
    data = data.to(device)

    metadata = (list(data.x_dict.keys()), list(data.edge_index_dict.keys()))

    print("Data loading and preprocessing complete")
    print(f"Stay nodes: {data['stay'].x.size(0):,}")
    print(f"Node types: {metadata[0]}")
    print(f"Edge types: {metadata[1]}")
    monitor_memory()

    return data, metadata


def create_and_initialize_model(config, metadata, data, device):
    """Create and initialize the model"""
    print(f"Creating {config.model_type.upper()} model...")

    model = create_model(
        model_name=config.model_type,
        metadata=metadata,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        heads=config.heads,
        num_heads=config.num_heads
    )

    model.to(device)

    print("Initializing model parameters...")
    model.eval()
    with torch.no_grad():
        try:
            _ = model(data.x_dict, data.edge_index_dict)
            print("Model parameters initialized successfully")
        except Exception as e:
            print(f"Warning during initialization: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model architecture: {config.model_type.upper()}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Number of layers: {config.num_layers}")

    monitor_memory()
    return model


def train_model(config, model, data, device):
    """Train the model with comprehensive evaluation"""
    print("Starting model training...")

    trainer = GradientAccumulationTrainer(
        model=model,
        data=data,
        device=device,
        num_classes=config.num_classes,
        accumulation_steps=config.accumulation_steps,
        class_names=config.class_names,
        save_dir=config.save_dir
    )

    results = trainer.train(
        epochs=config.epochs,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        use_scheduler=config.use_scheduler,
        visualize_every=config.visualize_every
    )

    return results


def run_comprehensive_model_comparison(config, data, metadata, device, models_to_compare=None):
    """Run comprehensive comparison between different model architectures"""
    print("Running comprehensive model architecture comparison...")

    if models_to_compare is None:
        models_to_compare = ['memory_efficient', 'rgcn', 'han', 'hgt']

    valid_models = []
    for model_name in models_to_compare:
        if model_name in MODEL_REGISTRY:
            valid_models.append(model_name)
        else:
            print(f"Warning: Unknown model '{model_name}', skipping...")

    if not valid_models:
        raise ValueError("No valid models specified for comparison")

    comparison_results = {}

    for model_name in valid_models:
        print(f"\n{'=' * 70}")
        print(f"Training {model_name.upper()} model")
        print(f"{'=' * 70}")

        temp_config = Config(config.__dict__.copy())
        temp_config.model_type = model_name
        temp_config.save_dir = os.path.join(config.save_dir, f"{model_name}_results")

        try:
            model = create_and_initialize_model(temp_config, metadata, data, device)
            results = train_model(temp_config, model, data, device)

            comparison_results[model_name] = {
                'metrics': results['test_metrics'],
                'training_history': results['training_history']
            }

            print(f"{model_name.upper()} training completed successfully")
            print(f"Test F1: {results['test_metrics']['f1_macro']:.4f}")
            print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

        except Exception as e:
            print(f"{model_name.upper()} training failed: {str(e)}")
            comparison_results[model_name] = {'error': str(e)}

        del model
        aggressive_memory_cleanup()

    create_model_comparison_report(comparison_results, config.save_dir)

    return comparison_results


def create_model_comparison_report(comparison_results, save_dir):
    """Create detailed comparison report and visualizations"""
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE MODEL ARCHITECTURE COMPARISON")
    print(f"{'=' * 80}")

    successful_results = {name: results for name, results in comparison_results.items()
                          if 'error' not in results}

    if not successful_results:
        print("No models completed successfully")
        return

    metrics_to_compare = [
        'accuracy', 'f1_macro', 'f1_weighted', 'jaccard_macro',
        'cohen_kappa', 'auroc_macro', 'auprc_macro'
    ]

    print(f"\n{'Model':<15}", end="")
    for metric in metrics_to_compare:
        print(f"{metric:<12}", end="")
    print()
    print("-" * (15 + 12 * len(metrics_to_compare)))

    best_scores = {metric: 0 for metric in metrics_to_compare}
    best_models = {metric: '' for metric in metrics_to_compare}

    for model_name, results in successful_results.items():
        print(f"{model_name:<15}", end="")
        metrics = results['metrics']

        for metric in metrics_to_compare:
            score = metrics.get(metric, 0)
            print(f"{score:<12.4f}", end="")

            if score > best_scores[metric]:
                best_scores[metric] = score
                best_models[metric] = model_name
        print()

    print(f"\nBest Performing Models:")
    for metric, model_name in best_models.items():
        if model_name:
            score = best_scores[metric]
            print(f"{metric:<15}: {model_name:<15} ({score:.4f})")

    comparison_file = os.path.join(save_dir, 'model_comparison_report.txt')
    with open(comparison_file, 'w') as f:
        f.write("COMPREHENSIVE MODEL ARCHITECTURE COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        for model_name, results in successful_results.items():
            f.write(f"{model_name.upper()} RESULTS:\n")
            f.write("-" * 40 + "\n")

            if 'metrics' in results:
                metrics = results['metrics']
                for metric_name, score in metrics.items():
                    f.write(f"  {metric_name:<20}: {score:.6f}\n")
                f.write("\n")
            else:
                f.write(f"  Error: {results.get('error', 'Unknown error')}\n\n")

    print(f"\nDetailed comparison report saved to: {comparison_file}")

    create_model_comparison_visualization(successful_results, save_dir)


def create_model_comparison_visualization(comparison_results, save_dir):
    """Create visualization comparing model performance"""
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_to_plot = ['accuracy', 'f1_macro', 'jaccard_macro', 'cohen_kappa', 'auroc_macro']
    model_names = list(comparison_results.keys())

    if len(model_names) < 2:
        print("Need at least 2 models for comparison visualization")
        return

    scores = np.zeros((len(model_names), len(metrics_to_plot)))

    for i, model_name in enumerate(model_names):
        metrics = comparison_results[model_name]['metrics']
        for j, metric in enumerate(metrics_to_plot):
            scores[i, j] = metrics.get(metric, 0)

    fig, ax = plt.subplots(figsize=(15, 8))

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(model_names)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(model_names)]

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores[i], width, label=model_name.upper(),
                      color=color, alpha=0.8)

        for bar, score in zip(bars, scores[i]):
            if score > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Architecture Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    viz_path = os.path.join(save_dir, 'model_comparison_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Model comparison visualization saved to: {viz_path}")


def evaluate_existing_model(config, model_path, data, metadata, device):
    """Evaluate an existing trained model"""
    print(f"Evaluating existing model: {model_path}")

    model_type = config.model_type

    model_kwargs = {
        'metadata': metadata,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'num_classes': config.num_classes,
        'dropout': config.dropout,
        'heads': config.heads,
        'num_heads': config.num_heads
    }

    evaluator = ModelEvaluator(
        model_path=model_path,
        data=data,
        device=device,
        class_names=config.class_names
    )

    model_class = get_model_class(model_type)
    results = evaluator.load_and_evaluate(model_class, model_kwargs)
    return results


def run_hyperparameter_search(config, data, metadata, device, model_name='memory_efficient'):
    """Run hyperparameter search for a specific model"""
    print(f"Running hyperparameter search for {model_name.upper()}...")

    hp_grid = {
        'hidden_dim': [32, 64, 128],
        'num_layers': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.005, 0.01]
    }

    if model_name in ['han', 'hgt']:
        hp_grid['heads'] = [1, 2, 4]
        hp_grid['num_heads'] = [1, 2, 4]

    best_score = 0
    best_config = None
    results_summary = []

    import itertools

    key_combinations = [
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_dim': 64, 'num_layers': 3, 'dropout': 0.2, 'learning_rate': 0.001},
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.1, 'learning_rate': 0.001},
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.005},
    ]

    for i, hp_combo in enumerate(key_combinations):
        print(f"\nHyperparameter combination {i + 1}/{len(key_combinations)}")
        print(f"{hp_combo}")

        temp_config = Config(config.__dict__.copy())
        temp_config.model_type = model_name
        temp_config.hidden_dim = hp_combo['hidden_dim']
        temp_config.num_layers = hp_combo['num_layers']
        temp_config.dropout = hp_combo['dropout']
        temp_config.learning_rate = hp_combo['learning_rate']
        temp_config.epochs = 20
        temp_config.save_dir = os.path.join(config.save_dir, f"hp_search_{model_name}_{i + 1}")

        if model_name in ['han', 'hgt']:
            temp_config.heads = hp_combo.get('heads', 2)
            temp_config.num_heads = hp_combo.get('num_heads', 2)

        try:
            model = create_and_initialize_model(temp_config, metadata, data, device)
            results = train_model(temp_config, model, data, device)

            f1_score = results['test_metrics']['f1_macro']
            results_summary.append({
                'config': hp_combo,
                'f1_score': f1_score,
                'full_results': results['test_metrics']
            })

            if f1_score > best_score:
                best_score = f1_score
                best_config = hp_combo.copy()

            print(f"F1 Score: {f1_score:.4f}")

        except Exception as e:
            print(f"Failed: {str(e)}")
            results_summary.append({
                'config': hp_combo,
                'error': str(e)
            })

        if 'model' in locals():
            del model
        aggressive_memory_cleanup()

    hp_results_file = os.path.join(config.save_dir, f'hyperparameter_search_{model_name}.txt')
    with open(hp_results_file, 'w') as f:
        f.write(f"HYPERPARAMETER SEARCH RESULTS - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Configuration (F1: {best_score:.4f}):\n")
        f.write(f"{best_config}\n\n")
        f.write("All Results:\n")
        f.write("-" * 40 + "\n")

        for result in results_summary:
            f.write(f"Config: {result['config']}\n")
            if 'f1_score' in result:
                f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            else:
                f.write(f"Error: {result['error']}\n")
            f.write("\n")

    print(f"\nBest hyperparameters for {model_name.upper()}:")
    print(f"Configuration: {best_config}")
    print(f"F1 Score: {best_score:.4f}")
    print(f"Results saved to: {hp_results_file}")

    return best_config, best_score, results_summary


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Medical GNN Training Pipeline')

    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--model', type=str, default='hgt',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--heads', type=int, default=2, help='Number of attention heads (HAN)')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads (HGT)')

    parser.add_argument('--compare_models', action='store_true',
                        help='Compare different model architectures')
    parser.add_argument('--models', nargs='+', default=['memory_efficient', 'han', 'hgt'],
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Models to compare when using --compare_models')
    parser.add_argument('--hyperparameter_search', action='store_true',
                        help='Run hyperparameter search for specified model')
    parser.add_argument('--evaluate_model', type=str,
                        help='Path to trained model to evaluate')

    parser.add_argument('--save_dir', type=str, help='Directory to save results')
    parser.add_argument('--no_visualizations', action='store_true',
                        help='Skip creating visualizations')

    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    if args.model:
        config.model_type = args.model
    if args.epochs:
        config.epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.accumulation_steps:
        config.accumulation_steps = args.accumulation_steps
    if args.heads:
        config.heads = args.heads
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.save_dir:
        config.save_dir = args.save_dir
    if args.no_visualizations:
        config.create_visualizations = False

    print("Setting up environment...")
    device = setup_cuda_environment()

    setup_experiment(config)

    try:
        data, metadata = load_and_preprocess_data(config, device)

        if args.evaluate_model:
            results = evaluate_existing_model(config, args.evaluate_model, data, metadata, device)

        elif args.compare_models:
            comparison_results = run_comprehensive_model_comparison(
                config, data, metadata, device, args.models
            )

        elif args.hyperparameter_search:
            best_config, best_score, search_results = run_hyperparameter_search(
                config, data, metadata, device, config.model_type
            )

        else:
            print(f"Training single model: {config.model_type.upper()}")
            model = create_and_initialize_model(config, metadata, data, device)
            results = train_model(config, model, data, device)

            print(f"\nTraining completed successfully!")
            print(f"Results saved to: {config.save_dir}")
            print(f"Best F1 Score: {results['test_metrics']['f1_macro']:.4f}")
            print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")

    except Exception as e:
        print(f"Execution failed with error: {str(e)}")
        raise

    finally:
        aggressive_memory_cleanup()
        print("Memory cleanup completed")


if __name__ == "__main__":
    main()