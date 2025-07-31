import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import classification_report

from utils import (
    aggressive_memory_cleanup, monitor_memory, compute_comprehensive_metrics,
    print_comprehensive_results, create_confusion_matrix_plot,
    create_tsne_visualization, create_metrics_comparison_plot,
    analyze_embeddings_statistics, save_results_to_file
)
from data import create_data_splits, get_class_weights
from models import get_model_class, create_model


class GradientAccumulationTrainer:
    """Advanced trainer with gradient accumulation, comprehensive evaluation, and embedding visualization"""

    def __init__(self, model, data, device, num_classes=3, accumulation_steps=4,
                 class_names=None, save_dir='./results'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        self.num_classes = num_classes
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir

        if class_names is None:
            self.class_names = ['Short Stay', 'Medium Stay', 'Long Stay']
        else:
            self.class_names = class_names

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Calculate class weights using the data module function
        self.class_weights = get_class_weights(data, device, num_classes)

    def extract_embeddings_with_labels(self, mask_name='val'):
        """Extract embeddings and corresponding labels for visualization"""
        self.model.eval()
        with torch.no_grad():
            # Get embeddings from the model
            final_embeddings, layer_embeddings = self.model.get_embeddings(
                self.data.x_dict, self.data.edge_index_dict
            )

            # Get the mask and extract relevant embeddings and labels
            mask = getattr(self.data['stay'], f'{mask_name}_mask')
            embeddings = final_embeddings[mask].cpu().numpy()
            labels = self.data['stay'].y[mask].cpu().numpy()

            return embeddings, labels, layer_embeddings

    def create_initial_embeddings_visualization(self):
        """Create t-SNE visualization of initial (untrained) embeddings"""
        print("🎨 Creating initial embeddings visualization...")

        # Extract embeddings before training
        embeddings, labels, _ = self.extract_embeddings_with_labels('val')

        # Create visualization
        save_path = os.path.join(self.save_dir, 'tsne_initial_embeddings.png')
        create_tsne_visualization(
            embeddings, labels, self.class_names,
            save_path=save_path,
            title="t-SNE: Initial (Untrained) Node Embeddings"
        )

        print(f"✅ Initial embeddings visualization saved to {save_path}")

    def create_final_embeddings_visualization(self):
        """Create t-SNE visualization of final (trained) embeddings"""
        print("🎨 Creating final embeddings visualization...")

        # Extract embeddings after training
        embeddings, labels, layer_embeddings = self.extract_embeddings_with_labels('test')

        # Create final embeddings visualization
        save_path = os.path.join(self.save_dir, 'tsne_final_embeddings.png')
        create_tsne_visualization(
            embeddings, labels, self.class_names,
            save_path=save_path,
            title="t-SNE: Final (Trained) Node Embeddings"
        )

        # Create layer-wise embeddings visualization
        self.create_layer_wise_embeddings_visualization(layer_embeddings, labels)

        print(f"✅ Final embeddings visualization saved to {save_path}")

    def create_layer_wise_embeddings_visualization(self, layer_embeddings, labels):
        """Create t-SNE visualizations for each layer's embeddings"""
        print("🎨 Creating layer-wise embeddings visualizations...")

        import matplotlib.pyplot as plt

        # Create subplot for all layers
        num_layers = len(layer_embeddings)
        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))

        if num_layers == 1:
            axes = [axes]

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))

        from sklearn.manifold import TSNE

        for layer_idx, layer_emb in enumerate(layer_embeddings):
            # Get embeddings for the test mask
            test_mask = self.data['stay'].test_mask
            layer_emb_masked = layer_emb[test_mask].cpu().numpy()

            # Compute t-SNE for this layer
            if layer_emb_masked.shape[0] > 50:  # Only if we have enough samples
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, layer_emb_masked.shape[0] // 4))
                embeddings_2d = tsne.fit_transform(layer_emb_masked)

                ax = axes[layer_idx]

                for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                    mask = labels == i
                    if np.any(mask):
                        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                                   c=[color], label=class_name, alpha=0.7, s=20)

                ax.set_title(f'Layer {layer_idx} Embeddings')
                ax.set_xlabel('t-SNE Dimension 1')
                ax.set_ylabel('t-SNE Dimension 2')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        layer_save_path = os.path.join(self.save_dir, 'tsne_layer_wise_embeddings.png')
        plt.savefig(layer_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Layer-wise embeddings visualization saved to {layer_save_path}")

    def create_embedding_comparison_visualization(self):
        """Create side-by-side comparison of initial vs final embeddings"""
        print("🎨 Creating embedding comparison visualization...")

        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        # Get initial embeddings (reinitialize model temporarily)
        # Determine model type from the current model
        model_type = type(self.model).__name__.lower()
        if 'memory' in model_type:
            model_name = 'memory_efficient'
        elif 'rgcn' in model_type:
            model_name = 'rgcn'
        elif 'han' in model_type:
            model_name = 'han'
        elif 'hgt' in model_type:
            model_name = 'hgt'
        else:
            model_name = 'memory_efficient'  # fallback

        # Create fresh model with same architecture
        metadata = (list(self.data.x_dict.keys()), list(self.data.edge_index_dict.keys()))
        initial_model = create_model(
            model_name=model_name,
            metadata=metadata,
            hidden_dim=self.model.hidden_dim,
            num_layers=self.model.num_layers,
            num_classes=self.num_classes,
            dropout=getattr(self.model, 'dropout', 0.2)
        ).to(self.device)

        initial_embeddings, initial_labels, _ = self._extract_embeddings_from_model(
            initial_model, 'test'
        )

        # Get final embeddings
        final_embeddings, final_labels, _ = self.extract_embeddings_with_labels('test')

        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(self.class_names)))

        # Initial embeddings t-SNE
        if initial_embeddings.shape[0] > 50:
            tsne_initial = TSNE(n_components=2, random_state=42,
                                perplexity=min(30, initial_embeddings.shape[0] // 4))
            initial_2d = tsne_initial.fit_transform(initial_embeddings)

            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                mask = initial_labels == i
                if np.any(mask):
                    ax1.scatter(initial_2d[mask, 0], initial_2d[mask, 1],
                                c=[color], label=class_name, alpha=0.7, s=30)

            ax1.set_title('Initial (Untrained) Embeddings', fontsize=14, fontweight='bold')
            ax1.set_xlabel('t-SNE Dimension 1')
            ax1.set_ylabel('t-SNE Dimension 2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Final embeddings t-SNE
        if final_embeddings.shape[0] > 50:
            tsne_final = TSNE(n_components=2, random_state=42,
                              perplexity=min(30, final_embeddings.shape[0] // 4))
            final_2d = tsne_final.fit_transform(final_embeddings)

            for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
                mask = final_labels == i
                if np.any(mask):
                    ax2.scatter(final_2d[mask, 0], final_2d[mask, 1],
                                c=[color], label=class_name, alpha=0.7, s=30)

            ax2.set_title('Final (Trained) Embeddings', fontsize=14, fontweight='bold')
            ax2.set_xlabel('t-SNE Dimension 1')
            ax2.set_ylabel('t-SNE Dimension 2')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        comparison_save_path = os.path.join(self.save_dir, 'tsne_embedding_comparison.png')
        plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Embedding comparison visualization saved to {comparison_save_path}")

        # Clean up temporary model
        del initial_model
        aggressive_memory_cleanup()

    def _extract_embeddings_from_model(self, model, mask_name='val'):
        """Helper function to extract embeddings from any model"""
        model.eval()
        with torch.no_grad():
            final_embeddings, layer_embeddings = model.get_embeddings(
                self.data.x_dict, self.data.edge_index_dict
            )

            mask = getattr(self.data['stay'], f'{mask_name}_mask')
            embeddings = final_embeddings[mask].cpu().numpy()
            labels = self.data['stay'].y[mask].cpu().numpy()

            return embeddings, labels, layer_embeddings

    def train_epoch_with_accumulation(self, optimizer, criterion, scheduler=None):
        """Training epoch with gradient accumulation and advanced monitoring"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Get training indices
        train_mask = self.data['stay'].train_mask
        train_indices = torch.where(train_mask)[0]

        # Shuffle and split into mini-batches
        perm = torch.randperm(len(train_indices))
        train_indices = train_indices[perm]

        batch_size = len(train_indices) // self.accumulation_steps

        for step in range(self.accumulation_steps):
            # Get batch indices
            start_idx = step * batch_size
            end_idx = start_idx + batch_size if step < self.accumulation_steps - 1 else len(train_indices)
            batch_indices = train_indices[start_idx:end_idx]

            # Forward pass on full graph (required for GNNs)
            aggressive_memory_cleanup()

            logits = self.model(self.data.x_dict, self.data.edge_index_dict)

            # Calculate loss only on current batch
            batch_logits = logits[batch_indices]
            batch_targets = self.data['stay'].y[batch_indices]

            loss = criterion(batch_logits, batch_targets)

            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            total_loss += loss.item()
            num_batches += 1

            # Clean up
            del logits, batch_logits, batch_targets, loss
            aggressive_memory_cleanup()

        # Calculate gradient norm before clipping
        total_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # Update parameters after accumulating gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        return total_loss, total_grad_norm

    def evaluate_comprehensive(self, mask_name='val', return_predictions=False):
        """Comprehensive evaluation with all requested metrics"""
        self.model.eval()

        with torch.no_grad():
            aggressive_memory_cleanup()

            logits = self.model(self.data.x_dict, self.data.edge_index_dict)

            mask = getattr(self.data['stay'], f'{mask_name}_mask')
            y_true = self.data['stay'].y[mask].cpu().numpy()
            y_pred = logits[mask].argmax(dim=1).cpu().numpy()

            # Get probabilities for AUC calculation
            y_proba = torch.softmax(logits[mask], dim=1).cpu().numpy()

            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(y_true, y_pred, y_proba, self.class_names)

            del logits
            aggressive_memory_cleanup()

            result = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba
            }

            if return_predictions:
                result['predictions'] = {
                    'true_labels': y_true,
                    'predicted_labels': y_pred,
                    'probabilities': y_proba
                }

            return result

    def compute_validation_loss(self, criterion):
        """Compute validation loss for overfitting detection"""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x_dict, self.data.edge_index_dict)
            val_mask = self.data['stay'].val_mask
            val_loss = criterion(logits[val_mask], self.data['stay'].y[val_mask])
        return val_loss.item()

    def safe_save_checkpoint(self, epoch, optimizer, metrics, filename):
        """Safely save checkpoint without numpy objects that cause loading issues"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': float(metrics['f1_macro']),  # Convert numpy to native Python
            'val_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                            for k, v in metrics.items()}  # Convert all numpy types
        }
        torch.save(checkpoint, filename)

    def safe_load_checkpoint(self, filename):
        """Safely load checkpoint with fallback options"""
        try:
            # First try with weights_only=True (PyTorch 2.6+ default)
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            return checkpoint
        except Exception as e1:
            try:
                # Fallback: Add safe globals for numpy objects
                torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
                checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
                return checkpoint
            except Exception as e2:
                try:
                    # Final fallback: Use weights_only=False (less secure but works)
                    print(f"⚠️ Warning: Loading checkpoint with weights_only=False due to: {e1}")
                    checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                    return checkpoint
                except Exception as e3:
                    print(f"❌ All loading methods failed: {e1}, {e2}, {e3}")
                    raise e3

    def train(self, epochs=30, lr=0.001, weight_decay=1e-4, patience=10,
              use_scheduler=True, visualize_every=10):
        """
        Comprehensive training loop with early stopping and visualization

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            use_scheduler: Whether to use learning rate scheduler
            visualize_every: Create visualizations every N epochs
        """

        # Create initial embeddings visualization
        self.create_initial_embeddings_visualization()

        # Initialize optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=patience // 2, verbose=True
            )

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        # Training tracking
        best_val_f1 = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # History tracking
        train_losses = []
        val_losses = []
        val_f1_scores = []
        val_accuracies = []

        print("🚀 Starting comprehensive training...")
        monitor_memory()

        for epoch in range(epochs):
            # Train with gradient accumulation
            train_loss, grad_norm = self.train_epoch_with_accumulation(optimizer, criterion)
            train_losses.append(train_loss)

            # Compute validation loss for overfitting detection
            val_loss = self.compute_validation_loss(criterion)
            val_losses.append(val_loss)

            # Comprehensive evaluation
            val_results = self.evaluate_comprehensive('val')
            val_metrics = val_results['metrics']

            val_f1_scores.append(val_metrics['f1_macro'])
            val_accuracies.append(val_metrics['accuracy'])

            # Update scheduler
            if scheduler is not None:
                scheduler.step(val_metrics['f1_macro'])

            # Print progress
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f} | "
                  f"Val AUC: {val_metrics['auroc_macro']:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Detailed logging every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"  📊 Detailed metrics:")
                print(f"     Jaccard: {val_metrics['jaccard_macro']:.4f}")
                print(f"     Kappa: {val_metrics['cohen_kappa']:.4f}")
                print(f"     AUPRC: {val_metrics['auprc_macro']:.4f}")
                print(f"     Grad norm: {grad_norm:.6f}")

                # Weight statistics for key layers
                for name, param in self.model.named_parameters():
                    if 'classifier' in name and 'weight' in name:
                        print(f"     {name} - Mean: {param.data.mean().item():.6f}, "
                              f"Std: {param.data.std().item():.6f}")

            # Embedding analysis
            if (epoch + 1) % 20 == 0:
                analyze_embeddings_statistics(self.model, self.data, epoch, self.device)

            # Create intermediate embeddings visualizations
            if (epoch + 1) % visualize_every == 0:
                intermediate_embeddings, intermediate_labels, _ = self.extract_embeddings_with_labels('val')
                intermediate_path = os.path.join(self.save_dir, f'tsne_epoch_{epoch + 1}.png')
                create_tsne_visualization(
                    intermediate_embeddings, intermediate_labels, self.class_names,
                    save_path=intermediate_path,
                    title=f"t-SNE: Node Embeddings (Epoch {epoch + 1})"
                )

            # Early stopping and best model tracking
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                epochs_without_improvement = 0
                # Save best model using safe method
                best_model_path = os.path.join(self.save_dir, 'best_model.pth')
                self.safe_save_checkpoint(epoch, optimizer, val_metrics, best_model_path)
                print(f"  💾 New best model saved (F1: {best_val_f1:.4f})")
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"  ⏹️ Early stopping after {patience} epochs without improvement")
                break

            # Overfitting warning
            if val_loss > best_val_loss:
                if epoch > 20 and val_loss > best_val_loss * 1.1:
                    print(f"  ⚠️ WARNING: Validation loss increasing - potential overfitting!")
            else:
                best_val_loss = val_loss

            # Memory monitoring
            if epoch % 10 == 0:
                monitor_memory()

        # Load best model for final evaluation
        print("\n🔄 Loading best model for final evaluation...")
        best_model_path = os.path.join(self.save_dir, 'best_model.pth')
        try:
            checkpoint = self.safe_load_checkpoint(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Best model loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not load best model ({e}), using current model")

        # Final comprehensive evaluation
        test_results = self.evaluate_comprehensive('test', return_predictions=True)
        test_metrics = test_results['metrics']

        # Print final results
        print_comprehensive_results(test_metrics, "FINAL TEST RESULTS")

        # Create final visualizations and save results
        self._create_final_visualizations_and_reports(test_results, train_losses,
                                                      val_losses, val_f1_scores)

        # Create final embeddings visualization
        self.create_final_embeddings_visualization()

        # Create comparison visualization
        self.create_embedding_comparison_visualization()

        return {
            'test_metrics': test_metrics,
            'test_results': test_results,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_f1_scores': val_f1_scores,
                'val_accuracies': val_accuracies
            }
        }

    def _create_final_visualizations_and_reports(self, test_results, train_losses,
                                                 val_losses, val_f1_scores):
        """Create comprehensive final visualizations and reports"""

        # Confusion Matrix
        cm_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        create_confusion_matrix_plot(
            test_results['y_true'],
            test_results['y_pred'],
            self.class_names,
            save_path=cm_path
        )

        # Metrics Comparison Plot
        metrics_plot_path = os.path.join(self.save_dir, 'metrics_comparison.png')
        create_metrics_comparison_plot(
            test_results['metrics'],
            save_path=metrics_plot_path
        )

        # Training History Plot
        self._plot_training_history(train_losses, val_losses, val_f1_scores)

        # Save detailed results to file
        results_file = os.path.join(self.save_dir, 'evaluation_results.txt')
        save_results_to_file(test_results['metrics'], results_file)

        # Save classification report
        report_file = os.path.join(self.save_dir, 'classification_report.txt')
        with open(report_file, 'w') as f:
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(
                test_results['y_true'],
                test_results['y_pred'],
                target_names=self.class_names,
                digits=4
            ))

        print(f"\n📁 All results saved to: {self.save_dir}")

    def _plot_training_history(self, train_losses, val_losses, val_f1_scores):
        """Plot training history"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1 Score plot
        ax2.plot(epochs, val_f1_scores, 'g-', label='Validation F1', linewidth=2)
        ax2.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        history_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Training history plot saved to {history_path}")


class ModelEvaluator:
    """Standalone model evaluator for trained models"""

    def __init__(self, model_path, data, device, class_names=None):
        self.model_path = model_path
        self.data = data
        self.device = device
        self.class_names = class_names or ['Short Stay', 'Medium Stay', 'Long Stay']

    def safe_load_checkpoint(self, filename):
        """Safely load checkpoint with fallback options"""
        try:
            # First try with weights_only=True (PyTorch 2.6+ default)
            checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
            return checkpoint
        except Exception as e1:
            try:
                # Fallback: Add safe globals for numpy objects
                torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.ndarray])
                checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
                return checkpoint
            except Exception as e2:
                try:
                    # Final fallback: Use weights_only=False (less secure but works)
                    print(f"⚠️ Warning: Loading checkpoint with weights_only=False due to: {e1}")
                    checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
                    return checkpoint
                except Exception as e3:
                    print(f"❌ All loading methods failed: {e1}, {e2}, {e3}")
                    raise e3

    def load_and_evaluate(self, model_class, model_kwargs):
        """Load a trained model and evaluate it"""
        # Create model
        model = model_class(**model_kwargs)

        # Load trained weights
        checkpoint = self.safe_load_checkpoint(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        # Evaluate
        trainer = GradientAccumulationTrainer(
            model, self.data, self.device,
            class_names=self.class_names,
            save_dir='./evaluation_results'
        )

        results = trainer.evaluate_comprehensive('test', return_predictions=True)
        print_comprehensive_results(results['metrics'], "MODEL EVALUATION RESULTS")

        return results