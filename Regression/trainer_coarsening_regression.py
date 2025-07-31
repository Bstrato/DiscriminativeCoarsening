import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils_coarsening_regression import (
    aggressive_memory_cleanup, monitor_memory
)


class GradientAccumulationRegressionTrainer:
    """Trainer with gradient accumulation to reduce memory for regression tasks"""

    def __init__(self, model, data, coarsening_info, device, accumulation_steps=4,
                 huber_delta=1.0, save_dir='./results_regression_coarsening'):
        self.model = model.to(device)
        self.data = data.to(device)
        self.coarsening_info = coarsening_info
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.save_dir = save_dir
        self.huber_delta = huber_delta

        os.makedirs(save_dir, exist_ok=True)

        print(f"Using Huber Loss with delta={huber_delta}")
        print(f"Using gradient accumulation with {accumulation_steps} steps")

        self.edge_weight_dict = {}
        for edge_type in data.edge_types:
            if hasattr(data[edge_type], 'edge_weight'):
                self.edge_weight_dict[edge_type] = data[edge_type].edge_weight

    def create_data_splits(self, train_ratio=0.7, val_ratio=0.15):
        """Create train/validation/test splits"""
        num_stays = self.data['stay'].x.size(0)
        indices = torch.randperm(num_stays)

        train_size = int(train_ratio * num_stays)
        val_size = int(val_ratio * num_stays)

        train_mask = torch.zeros(num_stays, dtype=torch.bool)
        val_mask = torch.zeros(num_stays, dtype=torch.bool)
        test_mask = torch.zeros(num_stays, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

        self.data['stay'].train_mask = train_mask.to(self.device)
        self.data['stay'].val_mask = val_mask.to(self.device)
        self.data['stay'].test_mask = test_mask.to(self.device)

        print(f"Dataset splits - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    def train_epoch_with_accumulation(self, optimizer, criterion):
        """Training epoch with gradient accumulation for regression"""
        self.model.train()
        total_loss = 0

        train_mask = self.data['stay'].train_mask
        train_indices = torch.where(train_mask)[0]

        perm = torch.randperm(len(train_indices))
        train_indices = train_indices[perm]

        batch_size = len(train_indices) // self.accumulation_steps

        for step in range(self.accumulation_steps):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size if step < self.accumulation_steps - 1 else len(train_indices)
            batch_indices = train_indices[start_idx:end_idx]

            aggressive_memory_cleanup()

            predictions = self.model(self.data.x_dict, self.data.edge_index_dict,
                                   self.edge_weight_dict if self.edge_weight_dict else None)

            batch_predictions = predictions[batch_indices]
            batch_targets = self.data['stay'].y_continuous[batch_indices]

            finite_mask = torch.isfinite(batch_predictions) & torch.isfinite(batch_targets)
            if finite_mask.sum() == 0:
                print(f"Warning: No finite values in batch {step}")
                continue

            batch_predictions = batch_predictions[finite_mask]
            batch_targets = batch_targets[finite_mask]

            loss = criterion(batch_predictions, batch_targets)
            loss = loss / self.accumulation_steps

            loss.backward()
            total_loss += loss.item()

            del predictions, batch_predictions, batch_targets, loss
            aggressive_memory_cleanup()

        total_grad_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        return total_loss, total_grad_norm

    def evaluate(self, mask_name='val', detailed=False):
        """Memory-efficient evaluation for regression with comprehensive metrics"""
        self.model.eval()

        with torch.no_grad():
            aggressive_memory_cleanup()

            predictions = self.model(self.data.x_dict, self.data.edge_index_dict,
                                   self.edge_weight_dict if self.edge_weight_dict else None)

            mask = getattr(self.data['stay'], f'{mask_name}_mask')
            y_true = self.data['stay'].y_continuous[mask].cpu().numpy()
            y_pred = predictions[mask].cpu().numpy()

            finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if finite_mask.sum() == 0:
                print(f"Error: No finite values for {mask_name} evaluation")
                return {
                    'mae': float('inf'),
                    'rmse': float('inf'),
                    'r2': -1.0,
                    'mape': float('inf'),
                    'y_true': y_true,
                    'y_pred': y_pred
                }

            y_true_clean = y_true[finite_mask]
            y_pred_clean = y_pred[finite_mask]

            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            r2 = r2_score(y_true_clean, y_pred_clean)
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1e-8))) * 100

            if detailed:
                detailed_regression_analysis(y_true_clean, y_pred_clean, mask_name.title())

            del predictions
            aggressive_memory_cleanup()

            return {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'y_true': y_true,
                'y_pred': y_pred
            }

    def train(self, epochs=50, lr=0.005, weight_decay=1e-4, patience=10, visualize_every=10):
        """Training loop with memory optimization and debugging for regression"""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        criterion = nn.HuberLoss(delta=self.huber_delta)

        best_val_mae = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0

        print("Starting regression training...")
        monitor_memory()

        for epoch in range(epochs):
            train_loss, grad_norm = self.train_epoch_with_accumulation(optimizer, criterion)
            val_loss = compute_validation_loss(self.model, self.data, criterion, self.edge_weight_dict)

            detailed = (epoch + 1) % visualize_every == 0
            val_metrics = self.evaluate('val', detailed=detailed)

            print(f"Epoch {epoch + 1:2d}/{epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.3f} | Val RMSE: {val_metrics['rmse']:.3f} | "
                  f"Val R²: {val_metrics['r2']:.3f}")

            if (epoch + 1) % visualize_every == 0:
                print(f"  Grad norm: {grad_norm:.6f}")

                for name, param in self.model.named_parameters():
                    if 'regressor' in name and 'weight' in name:
                        print(f"  {name} - Mean: {param.data.mean().item():.6f}, Std: {param.data.std().item():.6f}")

            analyze_embeddings(self.model, self.data, epoch, self.edge_weight_dict)

            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model_regression_coarsened.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            elif val_loss > best_val_loss * 1.1 and epoch > 20:
                print(f"  WARNING: Validation loss increasing - potential overfitting!")

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            if epoch % 10 == 0:
                monitor_memory()

        best_model_path = os.path.join(self.save_dir, 'best_model_regression_coarsened.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        test_metrics = self.evaluate('test', detailed=True)

        print(f"\n{'=' * 50}")
        print(f"FINAL REGRESSION RESULTS (WITH SELECTIVE COARSENING)")
        print(f"{'=' * 50}")
        print(f"Test MAE: {test_metrics['mae']:.3f} days")
        print(f"Test RMSE: {test_metrics['rmse']:.3f} days")
        print(f"Test R²: {test_metrics['r2']:.3f}")
        print(f"Test MAPE: {test_metrics['mape']:.1f}%")

        return {'test_metrics': test_metrics}


def detailed_regression_analysis(y_true, y_pred, split_name="Validation"):
    """Analyze regression performance by LOS ranges"""
    print(f"  {split_name} Regression Analysis:")

    short_mask = y_true <= 3.0
    medium_mask = (y_true > 3.0) & (y_true <= 7.0)
    long_mask = y_true > 7.0

    if short_mask.sum() > 0:
        mae_short = mean_absolute_error(y_true[short_mask], y_pred[short_mask])
        print(f"    Short stays (≤3d): MAE {mae_short:.3f} ({short_mask.sum()} samples)")

    if medium_mask.sum() > 0:
        mae_medium = mean_absolute_error(y_true[medium_mask], y_pred[medium_mask])
        print(f"    Medium stays (3-7d): MAE {mae_medium:.3f} ({medium_mask.sum()} samples)")

    if long_mask.sum() > 0:
        mae_long = mean_absolute_error(y_true[long_mask], y_pred[long_mask])
        print(f"    Long stays (>7d): MAE {mae_long:.3f} ({long_mask.sum()} samples)")


def compute_validation_loss(model, data, criterion, edge_weight_dict=None):
    """Compute validation loss for overfitting detection"""
    model.eval()
    with torch.no_grad():
        predictions = model(data.x_dict, data.edge_index_dict, edge_weight_dict)
        val_mask = data['stay'].val_mask
        val_targets = data['stay'].y_continuous[val_mask]
        val_preds = predictions[val_mask]

        finite_mask = torch.isfinite(val_targets) & torch.isfinite(val_preds)
        if finite_mask.sum() == 0:
            return float('inf')

        val_loss = criterion(val_preds[finite_mask], val_targets[finite_mask])
    return val_loss.item()


def analyze_embeddings(model, data, epoch, edge_weight_dict=None):
    """Analyze the learned embeddings"""
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            stay_embeddings = model.input_projections['stay'](data['stay'].x)
            print(f"  Stay embedding - Mean: {stay_embeddings.mean():.4f}, Std: {stay_embeddings.std():.4f}")

            predictions = model(data.x_dict, data.edge_index_dict, edge_weight_dict)
            finite_preds = predictions[torch.isfinite(predictions)]
            if len(finite_preds) > 0:
                print(f"  Predictions - Mean: {finite_preds.mean():.4f}, Std: {finite_preds.std():.4f}")
                print(f"  Predictions - Range: {finite_preds.min():.4f} to {finite_preds.max():.4f}")
            else:
                print(f"  Predictions - All non-finite values!")
            del predictions, stay_embeddings