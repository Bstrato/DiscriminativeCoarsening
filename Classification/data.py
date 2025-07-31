import torch
import numpy as np
from torch_geometric.data import HeteroData
import os


def load_data_safely(filepath):
    """Load data safely with multiple fallback methods"""
    try:
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        print("✅ Data loaded successfully")
        return data
    except Exception as e1:
        try:
            from torch_geometric.data.storage import BaseStorage
            from torch_geometric.data import HeteroData, Data
            torch.serialization.add_safe_globals([BaseStorage, HeteroData, Data])
            data = torch.load(filepath, map_location='cpu', weights_only=True)
            print("✅ Data loaded with safe globals")
            return data
        except Exception as e2:
            print(f"❌ Loading failed: {e1}, {e2}")
            return None


def analyze_node_features(data):
    """Analyze the actual node features"""
    print("\n" + "=" * 50)
    print("NODE FEATURE ANALYSIS")
    print("=" * 50)
    for node_type, features in data.x_dict.items():
        print(f"\n{node_type.upper()} Features:")
        print(f"  Shape: {features.shape}")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")
        print(f"  NaN count: {torch.isnan(features).sum()}")
        print(f"  Zero values: {(features == 0).sum()}")


def create_data_splits(data, device, train_ratio=0.7, val_ratio=0.15):
    """Create train/validation/test splits"""
    num_stays = data['stay'].x.size(0)
    indices = torch.randperm(num_stays)

    train_size = int(train_ratio * num_stays)
    val_size = int(val_ratio * num_stays)

    train_mask = torch.zeros(num_stays, dtype=torch.bool)
    val_mask = torch.zeros(num_stays, dtype=torch.bool)
    test_mask = torch.zeros(num_stays, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    data['stay'].train_mask = train_mask.to(device)
    data['stay'].val_mask = val_mask.to(device)
    data['stay'].test_mask = test_mask.to(device)

    print(f"Dataset splits - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    return train_mask, val_mask, test_mask


def get_class_weights(data, device, num_classes=3):
    """Calculate class weights for handling imbalanced data"""
    y = data['stay'].y.cpu().numpy()
    class_counts = np.bincount(y)
    total_samples = len(y)

    class_weights = torch.FloatTensor([
        total_samples / (num_classes * count) for count in class_counts
    ]).to(device)

    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    return class_weights