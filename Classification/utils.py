import torch
import gc
import psutil
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, f1_score, jaccard_score, cohen_kappa_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.preprocessing import label_binarize
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


def compute_comprehensive_metrics(y_true, y_pred, y_proba, class_names):
    """Compute comprehensive evaluation metrics"""
    num_classes = len(class_names)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
    jaccard_weighted = jaccard_score(y_true, y_pred, average='weighted')

    kappa = cohen_kappa_score(y_true, y_pred)

    if num_classes == 2:
        auroc_macro = roc_auc_score(y_true, y_proba[:, 1])
        auprc_macro = average_precision_score(y_true, y_proba[:, 1])
    else:
        try:
            auroc_macro = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            auroc_weighted = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
        except ValueError:
            auroc_macro = 0.0
            auroc_weighted = 0.0

        y_true_binarized = label_binarize(y_true, classes=range(num_classes))
        auprc_scores = []
        for i in range(num_classes):
            try:
                auprc = average_precision_score(y_true_binarized[:, i], y_proba[:, i])
                auprc_scores.append(auprc)
            except:
                auprc_scores.append(0.0)

        auprc_macro = np.mean(auprc_scores)
        auprc_weighted = np.average(auprc_scores, weights=np.bincount(y_true))

    f1_per_class = f1_score(y_true, y_pred, average=None)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'jaccard_macro': jaccard_macro,
        'jaccard_weighted': jaccard_weighted,
        'cohen_kappa': kappa,
        'auroc_macro': auroc_macro,
        'auprc_macro': auprc_macro
    }

    for i, class_name in enumerate(class_names):
        metrics[f'f1_{class_name.lower().replace(" ", "_")}'] = f1_per_class[i]

    if num_classes > 2:
        metrics['auroc_weighted'] = auroc_weighted
        metrics['auprc_weighted'] = auprc_weighted

    return metrics


def print_comprehensive_results(metrics, title="EVALUATION RESULTS"):
    """Print comprehensive results in a formatted way"""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")

    print(f"Core Metrics:")
    print(f"   Accuracy:      {metrics['accuracy']:.4f}")
    print(f"   F1 (Macro):    {metrics['f1_macro']:.4f}")
    print(f"   F1 (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"   F1 (Micro):    {metrics['f1_micro']:.4f}")

    print(f"\nAdvanced Metrics:")
    print(f"   Jaccard (Macro):    {metrics['jaccard_macro']:.4f}")
    print(f"   Jaccard (Weighted): {metrics['jaccard_weighted']:.4f}")
    print(f"   Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")

    print(f"\nAUC Metrics:")
    print(f"   AUROC (Macro):      {metrics['auroc_macro']:.4f}")
    print(f"   AUPRC (Macro):      {metrics['auprc_macro']:.4f}")

    if 'auroc_weighted' in metrics:
        print(f"   AUROC (Weighted):   {metrics['auroc_weighted']:.4f}")
        print(f"   AUPRC (Weighted):   {metrics['auprc_weighted']:.4f}")

    print(f"\nPer-Class F1 Scores:")
    class_metrics = {k: v for k, v in metrics.items() if
                     k.startswith('f1_') and 'macro' not in k and 'weighted' not in k and 'micro' not in k}
    for metric_name, score in class_metrics.items():
        class_name = metric_name.replace('f1_', '').replace('_', ' ').title()
        print(f"   {class_name:15}: {score:.4f}")

    print(f"{'=' * 60}\n")


def create_confusion_matrix_plot(y_true, y_pred, class_names, save_path=None):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})

    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")

    plt.show()

    print("\nRaw Confusion Matrix:")
    print(cm)


def create_tsne_visualization(embeddings, labels, class_names, save_path=None,
                              title="t-SNE Visualization of Node Embeddings", figsize=(12, 10)):
    """Create t-SNE visualization of embeddings"""
    if embeddings is None:
        print("No embeddings provided for t-SNE visualization")
        return

    print(f"Computing t-SNE visualization: {title}")

    if embeddings.shape[0] < 10:
        print(f"Too few samples ({embeddings.shape[0]}) for meaningful t-SNE visualization")
        return

    perplexity = min(30, embeddings.shape[0] // 4)
    if perplexity < 5:
        perplexity = 5

    print(f"   Samples: {embeddings.shape[0]}, Features: {embeddings.shape[1]}, Perplexity: {perplexity}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    colors = ['#FF073A', '#39FF14', '#0080FF', '#FF8C00', '#8A2BE2'][:len(class_names)]

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        if np.any(mask):
            scatter = plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                  c=[color], label=f'{class_name} (n={np.sum(mask)})',
                                  alpha=0.8, s=40, edgecolors='white', linewidth=1)

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE visualization saved to {save_path}")

    plt.show()

    print(f"Embedding Statistics:")
    print(f"   t-SNE range X: [{embeddings_2d[:, 0].min():.2f}, {embeddings_2d[:, 0].max():.2f}]")
    print(f"   t-SNE range Y: [{embeddings_2d[:, 1].min():.2f}, {embeddings_2d[:, 1].max():.2f}]")

    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            class_embeddings = embeddings_2d[mask]
            centroid = np.mean(class_embeddings, axis=0)
            distances = np.linalg.norm(class_embeddings - centroid, axis=1)
            print(f"   {class_name} cluster spread (std): {np.std(distances):.2f}")


def create_metrics_comparison_plot(metrics, save_path=None):
    """Create a bar plot comparing different metrics"""
    key_metrics = {
        'Accuracy': metrics['accuracy'],
        'F1 (Macro)': metrics['f1_macro'],
        'F1 (Weighted)': metrics['f1_weighted'],
        'Jaccard (Macro)': metrics['jaccard_macro'],
        'Cohen\'s Kappa': metrics['cohen_kappa'],
        'AUROC (Macro)': metrics['auroc_macro'],
        'AUPRC (Macro)': metrics['auprc_macro']
    }

    plt.figure(figsize=(12, 8))

    metric_names = list(key_metrics.keys())
    metric_values = list(key_metrics.values())

    bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.8)

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Metrics', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {save_path}")

    plt.show()


def analyze_embeddings_statistics(model, data, epoch, device):
    """Analyze embedding statistics during training"""
    model.eval()
    with torch.no_grad():
        print(f"\nModel Statistics (Epoch {epoch + 1}):")

        for node_type, projection in model.input_projections.items():
            weight = projection.weight.data
            print(f"   {node_type} projection - Mean: {weight.mean():.6f}, Std: {weight.std():.6f}")

        classifier_weight = model.classifier.weight.data
        print(f"   Classifier - Mean: {classifier_weight.mean():.6f}, Std: {classifier_weight.std():.6f}")

        dead_neurons = (classifier_weight.abs() < 1e-6).sum().item()
        total_neurons = classifier_weight.numel()
        print(f"   Dead neurons: {dead_neurons}/{total_neurons} ({100 * dead_neurons / total_neurons:.2f}%)")


def save_results_to_file(metrics, filepath):
    """Save evaluation results to a text file"""
    with open(filepath, 'w') as f:
        f.write("COMPREHENSIVE EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n\n")

        f.write("Core Metrics:\n")
        f.write(f"  Accuracy:      {metrics['accuracy']:.6f}\n")
        f.write(f"  F1 (Macro):    {metrics['f1_macro']:.6f}\n")
        f.write(f"  F1 (Weighted): {metrics['f1_weighted']:.6f}\n")
        f.write(f"  F1 (Micro):    {metrics['f1_micro']:.6f}\n\n")

        f.write("Advanced Metrics:\n")
        f.write(f"  Jaccard (Macro):    {metrics['jaccard_macro']:.6f}\n")
        f.write(f"  Jaccard (Weighted): {metrics['jaccard_weighted']:.6f}\n")
        f.write(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.6f}\n\n")

        f.write("AUC Metrics:\n")
        f.write(f"  AUROC (Macro):      {metrics['auroc_macro']:.6f}\n")
        f.write(f"  AUPRC (Macro):      {metrics['auprc_macro']:.6f}\n")

        if 'auroc_weighted' in metrics:
            f.write(f"  AUROC (Weighted):   {metrics['auroc_weighted']:.6f}\n")
            f.write(f"  AUPRC (Weighted):   {metrics['auprc_weighted']:.6f}\n")

        f.write("\n")

        f.write("Per-Class F1 Scores:\n")
        class_metrics = {k: v for k, v in metrics.items() if
                         k.startswith('f1_') and 'macro' not in k and 'weighted' not in k and 'micro' not in k}
        for metric_name, score in class_metrics.items():
            class_name = metric_name.replace('f1_', '').replace('_', ' ').title()
            f.write(f"  {class_name:15}: {score:.6f}\n")

    print(f"Detailed results saved to {filepath}")


def create_roc_curves(y_true, y_proba, class_names, save_path=None):
    """Create ROC curves for multiclass classification"""
    n_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(12, 10))

    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(class_names)]

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if n_classes == 2:
            fpr[i], tpr[i], _ = roc_curve(y_true, y_proba[:, 1])
            roc_auc[i] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_proba[:, i])

        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")

    plt.show()


def create_precision_recall_curves(y_true, y_proba, class_names, save_path=None):
    """Create Precision-Recall curves for multiclass classification"""
    n_classes = len(class_names)

    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    plt.figure(figsize=(12, 10))

    colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(class_names)]

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        if n_classes == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
            avg_precision = average_precision_score(y_true, y_proba[:, 1])
        else:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            avg_precision = average_precision_score(y_true_bin[:, i], y_proba[:, i])

        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{class_name} (AP = {avg_precision:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")

    plt.show()