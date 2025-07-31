#!/usr/bin/env python3

# ============================================================================
# Implementation is based on the paper: https://arxiv.org/pdf/2505.15842
# ============================================================================


import warnings

# Suppress transformers cache warning
warnings.filterwarnings("ignore", message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.datasets import IMDB, DBLP
import torch_geometric.transforms as T
from typing import Dict, List, Tuple, Optional
import os
import time
import argparse
import gc
import urllib.request
import zipfile
from pathlib import Path

# Import GNN models
try:
    from models import HeteroSageNet, HGTModel, HANModel, train_model, evaluate_model

    GNN_MODELS_AVAILABLE = True
    print("✅ GNN models imported successfully")
except ImportError as e:
    print(f"⚠️ Warning: GNN models not available: {e}")
    print("GNN training will be skipped. Please ensure models.py is in the same directory.")
    HeteroSageNet = HGTModel = HANModel = None
    train_model = evaluate_model = None
    GNN_MODELS_AVAILABLE = False


def clear_memory():
    """Clear memory efficiently"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Dataset configurations
DATASET_CONFIGS = {
    'imdb': {
        'coarsen_node_types': ['director'],
        'target_node_type': 'movie',
        'num_classes': 3,
        'class_names': ['Action', 'Comedy', 'Drama']
    },
    'dblp': {
        'coarsen_node_types': ['term', 'conference'],
        'target_node_type': 'author',
        'num_classes': 4,
        'class_names': ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    },
    'clinical': {
        'coarsen_node_types': ['careunit', 'inputevent'],
        'target_node_type': 'stay',
        'num_classes': 2,
        'class_names': ['Short Stay', 'Long Stay']
    }
}


def download_and_prepare_imdb(data_path):
    """Download and prepare IMDB dataset"""
    print("📥 Downloading IMDB dataset...")
    try:
        dataset = IMDB(root=os.path.join(data_path, 'imdb'))
        data = dataset[0]

        # Ensure all node types have features
        for node_type in data.node_types:
            if not hasattr(data[node_type], 'x') or data[node_type].x is None:
                # Create dummy features if missing
                num_nodes = data[node_type].num_nodes if hasattr(data[node_type], 'num_nodes') else 1000
                print(f"   Creating dummy features for {node_type}: {num_nodes} nodes")
                data[node_type].x = torch.randn(num_nodes, 128)  # 128-dim random features

        # Save as .pt file
        output_file = os.path.join(data_path, 'imdb_hetero.pt')
        torch.save(data, output_file)
        print(f"✅ IMDB dataset saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to download IMDB: {e}")
        return None


def download_and_prepare_dblp(data_path):
    """Download and prepare DBLP dataset"""
    print("📥 Downloading DBLP dataset...")
    try:
        dataset = DBLP(root=os.path.join(data_path, 'dblp'))
        data = dataset[0]

        # Ensure all node types have features
        for node_type in data.node_types:
            if not hasattr(data[node_type], 'x') or data[node_type].x is None:
                # Create dummy features if missing
                num_nodes = data[node_type].num_nodes if hasattr(data[node_type], 'num_nodes') else 1000
                print(f"   Creating dummy features for {node_type}: {num_nodes} nodes")
                data[node_type].x = torch.randn(num_nodes, 128)  # 128-dim random features

        # Save as .pt file
        output_file = os.path.join(data_path, 'dblp_hetero.pt')
        torch.save(data, output_file)
        print(f"✅ DBLP dataset saved to {output_file}")
        return output_file
    except Exception as e:
        print(f"❌ Failed to download DBLP: {e}")
        return None


def find_or_download_dataset(data_path, dataset_name):
    """Find existing dataset or download if not found"""
    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return None

    # Create data directory if it doesn't exist
    os.makedirs(data_path, exist_ok=True)

    # Look for existing dataset files
    if dataset_name == 'imdb':
        target_files = ['imdb_hetero.pt', 'imdb.pt']
    elif dataset_name == 'dblp':
        target_files = ['dblp_hetero.pt', 'dblp.pt']
    else:  # clinical
        target_files = [
            'classification_los_clinical_safe.pt',
            'clinical_safe.pt',
            'los_clinical.pt',
            'clinical_dataset.pt'
        ]

    # Search for existing files
    search_paths = [
        data_path,
        os.path.join(data_path, 'data'),
        os.path.join(data_path, 'datasets'),
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            for filename in target_files:
                full_path = os.path.join(search_path, filename)
                if os.path.exists(full_path):
                    print(f"✅ Found existing {dataset_name.upper()} dataset: {full_path}")
                    return full_path

    # Download if not found
    if dataset_name == 'imdb':
        return download_and_prepare_imdb(data_path)
    elif dataset_name == 'dblp':
        return download_and_prepare_dblp(data_path)
    else:
        print(f"❌ Clinical dataset not found and cannot be auto-downloaded.")
        print(f"Please provide the clinical dataset file in {data_path}")
        return None


class ProperAH_UGC:
    """
    Proper AH-UGC implementation following the paper methodology
    with memory optimizations
    """

    def __init__(self, num_projectors: int = 8, p_stable_alpha: float = 1.0, chunk_size: int = 1000):
        """
        Initialize AH-UGC framework following the paper

        Args:
            num_projectors: Number of LSH projectors (l in the paper)
            p_stable_alpha: Parameter for p-stable distribution (1.0 for Cauchy)
            chunk_size: Process data in chunks to reduce memory usage
        """
        self.num_projectors = num_projectors
        self.p_stable_alpha = p_stable_alpha
        self.projection_matrices = {}
        self.bias_vectors = {}
        self.chunk_size = chunk_size

    def compute_heterophily_factor(self, data: HeteroData, target_node_type: str) -> float:
        """
        Compute heterophily factor α = |{(v,u)∈E:y_v≠y_u}|/|E|
        As defined in the AH-UGC paper
        """
        if target_node_type not in data.node_types:
            return 0.5  # Default

        # Find edges involving the target node type
        relevant_edge_types = []
        for edge_type in data.edge_types:
            if isinstance(edge_type, tuple):
                src_type, relation, dst_type = edge_type
                if src_type == target_node_type and dst_type == target_node_type:
                    relevant_edge_types.append(edge_type)
            else:
                if target_node_type in str(edge_type):
                    relevant_edge_types.append(edge_type)

        if not relevant_edge_types:
            return 0.5

        try:
            edge_type = relevant_edge_types[0]
            edge_index = data[edge_type].edge_index
            labels = data[target_node_type].y

            # Count edges where nodes have DIFFERENT labels (heterophily)
            different_label_edges = 0
            total_edges = edge_index.size(1)

            for i in range(0, total_edges, self.chunk_size):
                end_i = min(i + self.chunk_size, total_edges)
                edge_chunk = edge_index[:, i:end_i]

                for j in range(edge_chunk.size(1)):
                    src, dst = edge_chunk[0, j].item(), edge_chunk[1, j].item()
                    if src < len(labels) and dst < len(labels):
                        if labels[src] != labels[dst]:  # Different labels = heterophily
                            different_label_edges += 1

            alpha = different_label_edges / total_edges if total_edges > 0 else 0.5
            return np.clip(alpha, 0.0, 1.0)

        except Exception as e:
            print(f"Warning: Could not compute heterophily factor: {e}")
            return 0.5

    def create_augmented_features(self, features: torch.Tensor,
                                  adjacency_vector: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Create augmented feature matrix F_i = (1-α)·X_i ⊕ α·A_i
        Where A_i is the adjacency vector (row of adjacency matrix)

        This is the core AH-UGC feature augmentation from the paper
        """
        # Normalize adjacency vector
        if adjacency_vector.sum() > 0:
            adj_normalized = adjacency_vector / adjacency_vector.sum()
        else:
            adj_normalized = adjacency_vector

        # Ensure adjacency vector is same length as number of nodes
        num_nodes = features.size(0)
        if adj_normalized.size(0) != num_nodes:
            # Pad or truncate adjacency vector to match number of nodes
            if adj_normalized.size(0) < num_nodes:
                padding = torch.zeros(num_nodes - adj_normalized.size(0))
                adj_normalized = torch.cat([adj_normalized, padding])
            else:
                adj_normalized = adj_normalized[:num_nodes]

        # Create augmented features by adding adjacency as additional features
        # Instead of concatenating with expanded adjacency, add it as a single feature per node
        adj_feature = adj_normalized.unsqueeze(1)  # [num_nodes, 1]

        # Create augmented features: original features + adjacency feature
        augmented = torch.cat([
            (1 - alpha) * features,
            alpha * adj_feature.expand(-1, features.size(1))  # Expand to match feature dim
        ], dim=1)

        return augmented

    def create_adjacency_vectors(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Create adjacency vectors for each node (memory efficient)
        Instead of full adjacency matrix, create vectors representing connections
        """
        adjacency_vectors = torch.zeros(num_nodes, num_nodes)

        # Process edges in chunks
        for i in range(0, edge_index.size(1), self.chunk_size):
            end_i = min(i + self.chunk_size, edge_index.size(1))
            edge_chunk = edge_index[:, i:end_i]

            for j in range(edge_chunk.size(1)):
                src, dst = edge_chunk[0, j].item(), edge_chunk[1, j].item()
                if src < num_nodes and dst < num_nodes:
                    adjacency_vectors[src, dst] = 1.0
                    adjacency_vectors[dst, src] = 1.0  # Assume undirected

        return adjacency_vectors

    def initialize_lsh_projections(self, feature_dim: int, node_type: str):
        """
        Initialize LSH projection matrices using p-stable distribution
        Following the AH-UGC paper methodology
        """
        if self.p_stable_alpha == 1.0:
            # Cauchy distribution for α=1 (as used in the paper)
            self.projection_matrices[node_type] = torch.from_numpy(
                np.random.standard_cauchy((feature_dim, self.num_projectors))
            ).float()
        else:
            # General p-stable distribution (more complex, not typically used)
            # For simplicity, use Gaussian approximation
            self.projection_matrices[node_type] = torch.from_numpy(
                np.random.normal(0, 1, (feature_dim, self.num_projectors))
            ).float()

        # Bias vectors uniformly sampled from [0, 2π] for LSH
        self.bias_vectors[node_type] = torch.from_numpy(
            np.random.uniform(0, 2 * np.pi, self.num_projectors)
        ).float()

    def compute_lsh_hash_scores(self, augmented_features: torch.Tensor, node_type: str) -> torch.Tensor:
        """
        Compute LSH hash scores using p-stable LSH
        This is the core LSH computation from AH-UGC paper
        """
        if node_type not in self.projection_matrices:
            self.initialize_lsh_projections(augmented_features.size(1), node_type)

        W = self.projection_matrices[node_type]
        b = self.bias_vectors[node_type]

        # Process in chunks to manage memory
        num_nodes = augmented_features.size(0)
        hash_scores = torch.zeros(num_nodes)

        for start_idx in range(0, num_nodes, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_nodes)
            feature_chunk = augmented_features[start_idx:end_idx]

            # LSH projection: s^k_i = W_k · F_i + b_k
            projections = torch.matmul(feature_chunk, W) + b  # [chunk_size, num_projectors]

            # For p-stable LSH with α=1 (Cauchy), use floor function
            if self.p_stable_alpha == 1.0:
                hash_values = torch.floor(projections / (2 * np.pi))
            else:
                hash_values = projections

            # Aggregate hash values across projectors (mean or sum)
            chunk_scores = torch.mean(hash_values, dim=1)
            hash_scores[start_idx:end_idx] = chunk_scores

            clear_memory()

        return hash_scores

    def consistent_hashing_coarsening(self, hash_scores: torch.Tensor, target_ratio: float) -> torch.Tensor:
        """
        Apply consistent hashing for adaptive coarsening
        Following the AH-UGC consistent hashing methodology
        """
        N = hash_scores.size(0)
        target_supernodes = max(1, int(N * target_ratio))

        # Sort nodes by hash scores for consistent assignment
        sorted_indices = torch.argsort(hash_scores)

        # Initialize supernode assignments
        assignment = torch.zeros(N, dtype=torch.long)

        # Create supernodes by grouping consecutive sorted nodes
        nodes_per_supernode = N // target_supernodes
        remainder = N % target_supernodes

        current_supernode = 0
        nodes_assigned = 0

        for i, node_idx in enumerate(sorted_indices):
            # Some supernodes get an extra node if there's a remainder
            supernode_size = nodes_per_supernode + (1 if current_supernode < remainder else 0)

            assignment[node_idx] = current_supernode
            nodes_assigned += 1

            # Move to next supernode when current one is full
            if nodes_assigned >= supernode_size and current_supernode < target_supernodes - 1:
                current_supernode += 1
                nodes_assigned = 0

        return assignment

    def create_coarsening_matrix(self, assignment: torch.Tensor, num_nodes: int,
                                 num_supernodes: int) -> torch.Tensor:
        """
        Create coarsening matrix C from assignment vector
        C[i,j] = 1/|S_i| if node j is in supernode i, 0 otherwise
        """
        # Count nodes in each supernode for normalization
        supernode_sizes = torch.bincount(assignment, minlength=num_supernodes).float()

        # Create coarsening matrix in chunks to save memory
        C = torch.zeros(num_supernodes, num_nodes)

        for node_idx, supernode_idx in enumerate(assignment):
            supernode_idx = supernode_idx.item()
            # Normalize by supernode size (averaging)
            C[supernode_idx, node_idx] = 1.0 / supernode_sizes[supernode_idx].item()

        return C

    def coarsen_single_node_type(self, features: torch.Tensor, edge_index: torch.Tensor,
                                 num_nodes: int, alpha: float, target_ratio: float,
                                 node_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarsen a single node type following AH-UGC methodology
        """
        print(f"  Processing {node_type}: {num_nodes:,} nodes")

        # Step 1: Create adjacency vectors (memory efficient)
        print(f"    Creating adjacency representation...")
        adjacency_vectors = self.create_adjacency_vectors(edge_index, num_nodes)

        # Step 2: Create augmented features for each node
        print(f"    Creating augmented features...")
        all_augmented_features = []

        # Process nodes in chunks to manage memory
        for start_idx in range(0, num_nodes, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_nodes)

            # Get adjacency vectors for this chunk
            adj_chunk = adjacency_vectors[start_idx:end_idx]
            feature_chunk = features[start_idx:end_idx]

            # Create augmented features for each node in chunk
            chunk_augmented = []
            for i in range(adj_chunk.size(0)):
                try:
                    augmented_node = self.create_augmented_features(
                        feature_chunk[i:i + 1], adj_chunk[i], alpha
                    )
                    chunk_augmented.append(augmented_node)
                except Exception as e:
                    print(f"      Warning: Error augmenting node {start_idx + i}: {e}")
                    # Use original features if augmentation fails
                    chunk_augmented.append(feature_chunk[i:i + 1])

            if chunk_augmented:
                all_augmented_features.append(torch.cat(chunk_augmented, dim=0))

            clear_memory()

        # Combine all augmented features
        if all_augmented_features:
            augmented_features = torch.cat(all_augmented_features, dim=0)
        else:
            # Fallback: use original features
            print(f"    Warning: Using original features (augmentation failed)")
            augmented_features = features

        del all_augmented_features, adjacency_vectors
        clear_memory()

        # Step 3: Compute LSH hash scores
        print(f"    Computing LSH hash scores...")
        hash_scores = self.compute_lsh_hash_scores(augmented_features, node_type)
        del augmented_features
        clear_memory()

        # Step 4: Apply consistent hashing coarsening
        print(f"    Applying consistent hashing...")
        assignment = self.consistent_hashing_coarsening(hash_scores, target_ratio)

        # Step 5: Create coarsening matrix and compute coarsened features
        print(f"    Creating coarsening matrix...")
        num_supernodes = len(torch.unique(assignment))
        C = self.create_coarsening_matrix(assignment, num_nodes, num_supernodes)

        print(f"    Computing coarsened features...")
        coarsened_features = torch.matmul(C, features)

        print(f"    Result: {num_supernodes:,} supernodes")
        clear_memory()

        return coarsened_features, C

    def coarsen_heterogeneous_graph(self, data: HeteroData, coarsen_node_types: List[str],
                                    coarsening_ratios: Dict[str, float] = None,
                                    target_node_type: str = None) -> Tuple[HeteroData, Dict[str, torch.Tensor]]:
        """
        Coarsen heterogeneous graph using AH-UGC methodology
        """
        print("\n🔄 Starting AH-UGC coarsening...")

        # Step 1: Compute heterophily factor
        alpha = 0.5  # Default
        if target_node_type:
            alpha = self.compute_heterophily_factor(data, target_node_type)
            print(f"Computed heterophily factor α = {alpha:.3f}")

        coarsened_data = HeteroData()
        coarsening_matrices = {}

        if coarsening_ratios is None:
            coarsening_ratios = {}

        default_ratio = 0.5

        # Step 2: Coarsen each specified node type
        for node_type in data.node_types:
            print(f"\n📊 Processing node type: {node_type}")
            features = data[node_type].x
            num_nodes = features.size(0)

            if node_type in coarsen_node_types:
                ratio = coarsening_ratios.get(node_type, default_ratio)

                # Find edges for this node type (self-edges)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                for edge_type in data.edge_types:
                    if isinstance(edge_type, tuple):
                        src_type, relation, dst_type = edge_type
                    else:
                        edge_parts = edge_type.split('__')
                        if len(edge_parts) >= 3:
                            src_type, relation, dst_type = edge_parts[0], edge_parts[1], edge_parts[2]
                        else:
                            continue

                    if src_type == node_type and dst_type == node_type:
                        edge_index = data[edge_type].edge_index
                        break

                # Apply AH-UGC coarsening
                coarsened_features, C = self.coarsen_single_node_type(
                    features, edge_index, num_nodes, alpha, ratio, node_type
                )

                coarsened_data[node_type].x = coarsened_features
                coarsening_matrices[node_type] = C

                # Handle labels with proper aggregation
                if hasattr(data[node_type], 'y') and data[node_type].y is not None:
                    labels = data[node_type].y
                    num_supernodes = coarsened_features.size(0)
                    coarsened_labels = torch.zeros(num_supernodes, dtype=labels.dtype)

                    # Majority voting for labels
                    assignment = torch.zeros(num_nodes, dtype=torch.long)
                    for supernode_idx in range(num_supernodes):
                        node_mask = (C[supernode_idx] > 0)
                        node_indices = torch.nonzero(node_mask).flatten()
                        for node_idx in node_indices:
                            assignment[node_idx] = supernode_idx

                    for supernode_idx in range(num_supernodes):
                        mask = (assignment == supernode_idx)
                        if mask.sum() > 0:
                            supernode_labels = labels[mask]
                            unique_labels, counts = torch.unique(supernode_labels, return_counts=True)
                            majority_label = unique_labels[torch.argmax(counts)]
                            coarsened_labels[supernode_idx] = majority_label

                    coarsened_data[node_type].y = coarsened_labels

            else:
                # Preserve this node type as-is
                coarsened_data[node_type].x = features.clone()
                coarsening_matrices[node_type] = torch.eye(num_nodes)

                if hasattr(data[node_type], 'y') and data[node_type].y is not None:
                    coarsened_data[node_type].y = data[node_type].y.clone()

            clear_memory()

        # Step 3: Coarsen edges between node types
        print("\n🔗 Processing inter-type edges...")
        for edge_type in data.edge_types:
            try:
                if isinstance(edge_type, tuple):
                    src_type, relation, dst_type = edge_type
                else:
                    edge_parts = edge_type.split('__')
                    if len(edge_parts) >= 3:
                        src_type, relation, dst_type = edge_parts[0], edge_parts[1], edge_parts[2]
                    else:
                        continue

                if src_type in coarsening_matrices and dst_type in coarsening_matrices:
                    edge_index = data[edge_type].edge_index
                    C_src = coarsening_matrices[src_type]
                    C_dst = coarsening_matrices[dst_type]

                    # Map edges to coarsened graph
                    coarsened_edges = set()

                    for i in range(0, edge_index.size(1), self.chunk_size):
                        end_i = min(i + self.chunk_size, edge_index.size(1))
                        edge_chunk = edge_index[:, i:end_i]

                        for j in range(edge_chunk.size(1)):
                            src_node, dst_node = edge_chunk[0, j].item(), edge_chunk[1, j].item()

                            # Find supernodes for these nodes
                            if src_node < C_src.size(1) and dst_node < C_dst.size(1):
                                src_supernode = torch.nonzero(C_src[:, src_node]).flatten()
                                dst_supernode = torch.nonzero(C_dst[:, dst_node]).flatten()

                                if len(src_supernode) > 0 and len(dst_supernode) > 0:
                                    coarsened_edges.add((src_supernode[0].item(), dst_supernode[0].item()))

                    if coarsened_edges:
                        edges_list = list(coarsened_edges)
                        coarsened_edge_index = torch.tensor(edges_list).T
                        coarsened_data[edge_type].edge_index = coarsened_edge_index

            except Exception as e:
                print(f"Warning: Could not process edge type {edge_type}: {e}")
                continue

            clear_memory()

        print("✅ AH-UGC coarsening completed!")
        return coarsened_data, coarsening_matrices


def load_data_safely(file_path):
    """Load data with error handling"""
    try:
        print(f"📂 Loading dataset from {file_path}")
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        print(f"✅ Dataset loaded successfully")
        return data
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None


def create_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
    """Create train/validation/test masks for node classification"""
    indices = torch.randperm(num_nodes)
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


def run_gnn_experiments(original_data, coarsened_data, target_node_type,
                        num_classes, device='cpu', reduced_epochs=200):
    """
    Run GNN experiments on both original and coarsened graphs
    """
    if not GNN_MODELS_AVAILABLE:
        print("⏭️ Skipping GNN experiments (models not available)")
        return {}

    print(f"\n🤖 Running GNN experiments...")
    print(f"   Target node type: {target_node_type}")
    print(f"   Number of classes: {num_classes}")

    results = {}

    # Create masks for original data
    num_nodes = original_data[target_node_type].x.size(0)
    train_mask, val_mask, test_mask = create_train_val_test_masks(num_nodes)
    print(f"   Original data: {train_mask.sum()} train, {val_mask.sum()} val, {test_mask.sum()} test")

    # Create masks for coarsened data
    coarsened_num_nodes = coarsened_data[target_node_type].x.size(0)
    coarsened_train_mask, coarsened_val_mask, coarsened_test_mask = create_train_val_test_masks(coarsened_num_nodes)
    print(
        f"   Coarsened data: {coarsened_train_mask.sum()} train, {coarsened_val_mask.sum()} val, {coarsened_test_mask.sum()} test")

    # Get metadata
    metadata = (original_data.node_types, original_data.edge_types)

    # Models to test - use the fixed models directly
    models_to_test = {
        'HeteroSageNet': HeteroSageNet,
        'HANModel': HANModel,
        'HGTModel': HGTModel
    }

    for model_name, model_class in models_to_test.items():
        print(f"\n=== Testing {model_name} ===")

        try:
            # Train on original graph
            print(f"Training {model_name} on Original Graph...")
            original_model = model_class(
                metadata,
                hidden_dim=32,  # Reduced for memory
                num_layers=2,
                num_classes=num_classes,
                dropout=0.3
            )

            trained_original = train_model(
                original_model, original_data, target_node_type,
                train_mask, val_mask, num_epochs=reduced_epochs, device=device
            )

            original_results = evaluate_model(
                trained_original, original_data, target_node_type, test_mask, device
            )

            print(f"Original Graph - Acc: {original_results['accuracy']:.4f}, "
                  f"F1: {original_results['f1_score']:.4f}, AUROC: {original_results['auroc']:.4f}")

            clear_memory()

            # Train on coarsened graph
            print(f"Training {model_name} on Coarsened Graph...")
            coarsened_model = model_class(
                metadata,
                hidden_dim=32,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.3
            )

            trained_coarsened = train_model(
                coarsened_model, coarsened_data, target_node_type,
                coarsened_train_mask, coarsened_val_mask, num_epochs=reduced_epochs, device=device
            )

            coarsened_results = evaluate_model(
                trained_coarsened, coarsened_data, target_node_type, coarsened_test_mask, device
            )

            print(f"Coarsened Graph - Acc: {coarsened_results['accuracy']:.4f}, "
                  f"F1: {coarsened_results['f1_score']:.4f}, AUROC: {coarsened_results['auroc']:.4f}")

            results[model_name] = {
                'original': original_results,
                'coarsened': coarsened_results
            }

            clear_memory()

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def run_ah_ugc_experiment(data_path, dataset_name, chunk_size=500):
    """
    Run AH-UGC experiment on specified dataset
    """
    print(f"🚀 Starting AH-UGC Experiment on {dataset_name.upper()} Dataset")

    if dataset_name not in DATASET_CONFIGS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return None

    config = DATASET_CONFIGS[dataset_name]

    # Initialize AH-UGC with proper parameters
    ah_ugc = ProperAH_UGC(num_projectors=8, p_stable_alpha=1.0, chunk_size=chunk_size)

    print("=" * 60)
    print(f"AH-UGC EXPERIMENT: {dataset_name.upper()} DATASET")
    print("=" * 60)

    # Find or download dataset
    dataset_file = find_or_download_dataset(data_path, dataset_name)
    if dataset_file is None:
        print(f"❌ {dataset_name.upper()} dataset not available!")
        return None

    # Load data
    data = load_data_safely(dataset_file)
    if data is None:
        print("❌ Failed to load dataset")
        return None

    # Display dataset information
    print(f"\n📊 {dataset_name.upper()} Dataset Information:")
    print(f"   Node types: {list(data.node_types)}")
    print(f"   Edge types: {list(data.edge_types)}")

    total_nodes = 0
    for node_type in data.node_types:
        # Check if node type has features
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            num_nodes = data[node_type].x.size(0)
            feature_dim = data[node_type].x.size(1)
        else:
            # Create dummy features if missing
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes = data[node_type].num_nodes
            else:
                # Estimate from edges
                num_nodes = 0
                for edge_type in data.edge_types:
                    if isinstance(edge_type, tuple):
                        src_type, _, dst_type = edge_type
                        if src_type == node_type or dst_type == node_type:
                            edge_index = data[edge_type].edge_index
                            if src_type == node_type:
                                max_node = edge_index[0].max().item() + 1
                            else:
                                max_node = edge_index[1].max().item() + 1
                            num_nodes = max(num_nodes, max_node)

                if num_nodes == 0:
                    num_nodes = 100  # Default fallback

            # Create dummy features
            feature_dim = 128
            print(f"   Creating features for {node_type}: {num_nodes} nodes")
            data[node_type].x = torch.randn(num_nodes, feature_dim)

        total_nodes += num_nodes
        print(f"   {node_type}: {num_nodes:,} nodes, {feature_dim} features")

        if hasattr(data[node_type], 'y') and data[node_type].y is not None:
            unique_labels = torch.unique(data[node_type].y)
            print(f"     Labels: {len(unique_labels)} classes {unique_labels.tolist()}")

    print(f"   Total nodes: {total_nodes:,}")

    # Get configuration
    coarsen_node_types = config['coarsen_node_types']
    target_node_type = config['target_node_type']
    num_classes = config['num_classes']
    class_names = config['class_names']

    # Validate configuration
    available_types = set(data.node_types)
    requested_types = set(coarsen_node_types)
    missing_types = requested_types - available_types

    if missing_types:
        print(f"⚠️ Warning: Missing node types in dataset: {missing_types}")
        coarsen_node_types = [nt for nt in coarsen_node_types if nt in available_types]

    if not coarsen_node_types:
        print(f"❌ No valid node types to coarsen!")
        return None

    if target_node_type not in available_types:
        print(f"⚠️ Warning: Target node type '{target_node_type}' not in dataset")
        target_node_type = list(data.node_types)[0]
        print(f"Using '{target_node_type}' as target node type")

    # Define coarsening ratios
    coarsening_ratios = {nt: 0.5 for nt in coarsen_node_types}

    print(f"\n🔧 AH-UGC Configuration:")
    print(f"   Dataset: {dataset_name.upper()}")
    print(f"   Target node type: {target_node_type}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Class names: {class_names}")
    print(f"   LSH projectors: {ah_ugc.num_projectors}")
    print(f"   p-stable α: {ah_ugc.p_stable_alpha}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Coarsen types: {coarsen_node_types}")
    print(f"   Preserve types: {[nt for nt in data.node_types if nt not in coarsen_node_types]}")
    print(f"   Coarsening ratio: 0.5")

    print("\n🔄 Applying AH-UGC coarsening...")
    start_time = time.time()

    try:
        coarsened_data, coarsening_matrices = ah_ugc.coarsen_heterogeneous_graph(
            data,
            coarsen_node_types=coarsen_node_types,
            coarsening_ratios=coarsening_ratios,
            target_node_type=target_node_type
        )

        coarsening_time = time.time() - start_time
        print(f"✅ AH-UGC coarsening completed in {coarsening_time:.2f} seconds")

    except Exception as e:
        print(f"❌ AH-UGC coarsening failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Calculate and display results
    print(f"\n📈 AH-UGC Coarsening Results:")
    total_original = 0
    total_coarsened = 0

    for node_type in data.node_types:
        original_count = data[node_type].x.size(0)
        coarsened_count = coarsened_data[node_type].x.size(0)

        total_original += original_count
        total_coarsened += coarsened_count

        if node_type in coarsen_node_types:
            reduction = (1 - coarsened_count / original_count) * 100
            print(f"   {node_type}: {original_count:,} → {coarsened_count:,} ({reduction:.1f}% reduction) [AH-UGC]")
        else:
            print(f"   {node_type}: {original_count:,} → {coarsened_count:,} (preserved)")

    overall_reduction = (1 - total_coarsened / total_original) * 100
    print(f"   Overall: {total_original:,} → {total_coarsened:,} ({overall_reduction:.1f}% reduction)")

    # Run GNN experiments
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    gnn_results = run_gnn_experiments(
        data, coarsened_data, target_node_type, num_classes, device, reduced_epochs=200
    )

    # Save results
    print(f"\n💾 Saving results...")

    experiment_results = {
        'dataset_name': dataset_name,
        'dataset_config': config,
        'ah_ugc_parameters': {
            'num_projectors': ah_ugc.num_projectors,
            'p_stable_alpha': ah_ugc.p_stable_alpha,
            'chunk_size': chunk_size
        },
        'original_data_info': {
            'node_types': list(data.node_types),
            'edge_types': list(data.edge_types),
            'node_counts': {nt: data[nt].x.size(0) for nt in data.node_types},
            'feature_dims': {nt: data[nt].x.size(1) for nt in data.node_types}
        },
        'coarsening_info': {
            'coarsened_node_types': coarsen_node_types,
            'preserved_node_types': [nt for nt in data.node_types if nt not in coarsen_node_types],
            'ratios': coarsening_ratios,
            'node_counts': {nt: coarsened_data[nt].x.size(0) for nt in coarsened_data.node_types},
            'overall_reduction': overall_reduction,
            'target_node_type': target_node_type,
            'runtime': coarsening_time
        },
        'gnn_results': gnn_results
    }

    try:
        results_file = f'ah_ugc_{dataset_name}_results.pt'
        torch.save(experiment_results, results_file)
        print(f"✅ Results saved to {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    print(f"\n🎉 AH-UGC {dataset_name.upper()} experiment completed!")
    print(f"   Total runtime: {coarsening_time:.2f} seconds")
    print(f"   Coarsened types: {coarsen_node_types}")
    print(f"   Preserved types: {[nt for nt in data.node_types if nt not in coarsen_node_types]}")
    print(f"   Overall reduction: {overall_reduction:.1f}%")

    return experiment_results


def run_multi_dataset_experiments(data_path, datasets, chunk_size=500):
    """
    Run AH-UGC experiments on multiple datasets
    """
    print("🎯 Running Multi-Dataset AH-UGC Experiments")
    print("=" * 70)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'=' * 20} {dataset_name.upper()} DATASET {'=' * 20}")

        try:
            result = run_ah_ugc_experiment(data_path, dataset_name, chunk_size)
            if result is not None:
                all_results[dataset_name] = result
                print(f"✅ {dataset_name.upper()} experiment completed successfully")
            else:
                print(f"❌ {dataset_name.upper()} experiment failed")
        except Exception as e:
            print(f"❌ Error in {dataset_name.upper()} experiment: {e}")
            continue

    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("MULTI-DATASET AH-UGC EXPERIMENT SUMMARY")
    print("=" * 70)

    if all_results:
        print(f"\n📊 Coarsening Results Summary:")
        print(f"{'Dataset':<10} {'Coarsened Types':<30} {'Reduction':<12} {'Runtime':<10}")
        print("-" * 65)

        for dataset_name, result in all_results.items():
            coarsened_types = ', '.join(result['coarsening_info']['coarsened_node_types'])
            reduction = result['coarsening_info']['overall_reduction']
            runtime = result['coarsening_info']['runtime']

            print(f"{dataset_name.upper():<10} {coarsened_types:<30} {reduction:>8.1f}%  {runtime:>8.2f}s")

        # Print GNN results if available
        if any('gnn_results' in result and result['gnn_results'] for result in all_results.values()):
            print(f"\n🤖 GNN Performance Results:")
            print(f"{'Dataset':<8} {'Model':<15} {'Original':<25} {'Coarsened':<25} {'Improvement':<15}")
            print(
                f"{'':8} {'':15} {'Acc':<8} {'F1':<8} {'AUROC':<8} {'Acc':<8} {'F1':<8} {'AUROC':<8} {'ΔAcc':<7} {'ΔF1':<7}")
            print("-" * 105)

            for dataset_name, result in all_results.items():
                if 'gnn_results' in result and result['gnn_results']:
                    for model_name, model_results in result['gnn_results'].items():
                        orig_acc = model_results['original']['accuracy']
                        orig_f1 = model_results['original']['f1_score']
                        orig_auc = model_results['original']['auroc']

                        coars_acc = model_results['coarsened']['accuracy']
                        coars_f1 = model_results['coarsened']['f1_score']
                        coars_auc = model_results['coarsened']['auroc']

                        print(f"{dataset_name.upper():<8} {model_name:<15} "
                              f"{orig_acc:.3f}    {orig_f1:.3f}    {orig_auc:.3f}    "
                              f"{coars_acc:.3f}    {coars_f1:.3f}    {coars_auc:.3f}    "
                              f"{coars_acc - orig_acc:+.3f}   {coars_f1 - orig_f1:+.3f}")

        print("\n📈 Detailed Results:")
        for dataset_name, result in all_results.items():
            config = result['dataset_config']
            coarsening_info = result['coarsening_info']

            print(f"\n{dataset_name.upper()} Dataset:")
            print(f"   Target node type: {coarsening_info['target_node_type']}")
            print(f"   Number of classes: {config['num_classes']}")
            print(f"   Class names: {config['class_names']}")
            print(f"   Coarsened types: {coarsening_info['coarsened_node_types']}")
            print(f"   Preserved types: {coarsening_info['preserved_node_types']}")
            print(f"   Overall reduction: {coarsening_info['overall_reduction']:.1f}%")
            print(f"   Runtime: {coarsening_info['runtime']:.2f} seconds")

            if 'gnn_results' in result and result['gnn_results']:
                print(f"   GNN Models tested: {list(result['gnn_results'].keys())}")

        # Save combined results
        try:
            combined_results_file = 'ah_ugc_multi_dataset_results.pt'
            torch.save(all_results, combined_results_file)
            print(f"\n💾 Combined results saved to {combined_results_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save combined results: {e}")

        print(f"\n🎉 Multi-dataset experiments completed!")
        print(f"   Successful experiments: {len(all_results)}/{len(datasets)}")

    else:
        print("\n❌ No experiments completed successfully!")

    return all_results


def main():
    """Main function for multi-dataset AH-UGC implementation"""
    parser = argparse.ArgumentParser(description='AH-UGC implementation with GNN training on multiple datasets')
    parser.add_argument('--data-path', type=str,
                        default='./data',
                        help='Path to dataset directory (default: ./data)')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['imdb', 'dblp'],
                        choices=['imdb', 'dblp', 'clinical'],
                        help='Datasets to run experiments on (default: imdb dblp)')
    parser.add_argument('--chunk-size', type=int, default=300,
                        help='Chunk size for memory-efficient processing (default: 300)')
    parser.add_argument('--num-projectors', type=int, default=8,
                        help='Number of LSH projectors (default: 8)')
    parser.add_argument('--p-stable-alpha', type=float, default=1.0,
                        help='p-stable distribution parameter (default: 1.0 for Cauchy)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs for GNN models (default: 30)')
    parser.add_argument('--skip-gnn', action='store_true',
                        help='Skip GNN training to save time and memory')

    args = parser.parse_args()

    print("=" * 70)
    print("AH-UGC: ADAPTIVE AND HETEROGENEOUS UNIVERSAL GRAPH COARSENING")
    print("Multi-Dataset Implementation with GNN Training")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Data path: {args.data_path}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  LSH projectors: {args.num_projectors}")
    print(f"  p-stable α: {args.p_stable_alpha}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Skip GNN: {args.skip_gnn}")
    print(f"  GNN models available: {GNN_MODELS_AVAILABLE}")
    print()

    # Display dataset configurations
    print("📋 Dataset Configurations:")
    for dataset_name in args.datasets:
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            print(f"  {dataset_name.upper()}:")
            print(f"    Coarsen: {config['coarsen_node_types']}")
            print(f"    Target: {config['target_node_type']}")
            print(f"    Classes: {config['num_classes']} {config['class_names']}")
    print()

    # Run experiments
    if len(args.datasets) == 1:
        # Single dataset experiment
        dataset_name = args.datasets[0]
        print(f"🚀 Running single dataset experiment: {dataset_name.upper()}")

        result = run_ah_ugc_experiment(
            data_path=args.data_path,
            dataset_name=dataset_name,
            chunk_size=args.chunk_size
        )

        if result is not None:
            print("\n✨ Experiment completed successfully!")
            reduction = result['coarsening_info']['overall_reduction']
            runtime = result['coarsening_info']['runtime']
            coarsened_types = result['coarsening_info']['coarsened_node_types']

            print(f"\n🏆 Final Results:")
            print(f"   Dataset: {dataset_name.upper()}")
            print(f"   Method: AH-UGC (Adaptive & Heterogeneous Universal Graph Coarsening)")
            print(f"   Coarsened node types: {coarsened_types}")
            print(f"   Node reduction: {reduction:.1f}%")
            print(f"   Runtime: {runtime:.2f} seconds")

            # Print GNN results if available
            if 'gnn_results' in result and result['gnn_results']:
                print(f"\n🤖 GNN Performance Summary:")
                for model_name, model_results in result['gnn_results'].items():
                    orig_acc = model_results['original']['accuracy']
                    coars_acc = model_results['coarsened']['accuracy']
                    improvement = coars_acc - orig_acc
                    print(f"   {model_name}: {orig_acc:.3f} → {coars_acc:.3f} ({improvement:+.3f})")
        else:
            print("\n❌ Experiment failed!")

    else:
        # Multi-dataset experiments
        print(f"🚀 Running multi-dataset experiments: {[d.upper() for d in args.datasets]}")

        results = run_multi_dataset_experiments(
            data_path=args.data_path,
            datasets=args.datasets,
            chunk_size=args.chunk_size
        )

        if results:
            print("\n✨ Multi-dataset experiments completed!")
            print(f"   Successful: {len(results)}/{len(args.datasets)} datasets")

            # Show best results
            best_reduction = max(results.values(),
                                 key=lambda x: x['coarsening_info']['overall_reduction'])
            best_dataset = [k for k, v in results.items()
                            if v['coarsening_info']['overall_reduction'] ==
                            best_reduction['coarsening_info']['overall_reduction']][0]

            print(f"\n🏆 Best Reduction:")
            print(f"   Dataset: {best_dataset.upper()}")
            print(f"   Reduction: {best_reduction['coarsening_info']['overall_reduction']:.1f}%")

            # Show best GNN improvement if available
            if any('gnn_results' in result and result['gnn_results'] for result in results.values()):
                best_improvements = []
                for dataset_name, result in results.items():
                    if 'gnn_results' in result and result['gnn_results']:
                        for model_name, model_results in result['gnn_results'].items():
                            improvement = (model_results['coarsened']['accuracy'] -
                                           model_results['original']['accuracy'])
                            best_improvements.append((dataset_name, model_name, improvement))

                if best_improvements:
                    best_improvement = max(best_improvements, key=lambda x: x[2])
                    print(f"\n🎯 Best GNN Improvement:")
                    print(f"   Dataset: {best_improvement[0].upper()}")
                    print(f"   Model: {best_improvement[1]}")
                    print(f"   Accuracy improvement: {best_improvement[2]:+.3f}")
        else:
            print("\n❌ All experiments failed!")


if __name__ == "__main__":
    main()