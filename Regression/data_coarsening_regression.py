import torch
import numpy as np
from torch_geometric.data import HeteroData
import os
from sklearn.cluster import KMeans
from collections import defaultdict


def load_data_safely(filepath):
    """Load data safely"""
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


def product_quantization_coarsening(features, m=8, k=16, random_state=42):
    """
    Coarsen node features using Product Quantization (PQ).

    Args:
        features: Node features tensor (n_nodes, feature_dim)
        m: Number of subvectors
        k: Number of centroids per subspace
        random_state: Random seed for reproducibility

    Returns:
        quantized_codes: Quantized feature codes for each node
        codebooks: Trained codebooks for reconstruction if needed
        supernode_mapping: Mapping from quantized codes to original node indices
    """
    features_np = features.cpu().numpy()
    n_nodes, d = features_np.shape

    # Ensure m divides d evenly
    if d % m != 0:
        # Pad features to make it divisible
        pad_size = m - (d % m)
        features_np = np.pad(features_np, ((0, 0), (0, pad_size)), mode='constant')
        d = features_np.shape[1]

    # Split features into m subvectors
    subvector_size = d // m
    subvectors = [features_np[:, i * subvector_size:(i + 1) * subvector_size] for i in range(m)]

    # Train codebooks for each subspace
    codebooks = []
    quantized_codes = []

    for i in range(m):
        kmeans = KMeans(n_clusters=min(k, len(np.unique(subvectors[i], axis=0))),
                        random_state=random_state, n_init=10)
        kmeans.fit(subvectors[i])
        labels = kmeans.predict(subvectors[i])
        codebooks.append(kmeans)
        quantized_codes.append(labels)

    # Combine quantized codes
    quantized_codes = np.column_stack(quantized_codes)

    # Create supernode mapping
    supernode_mapping = defaultdict(list)
    for idx in range(n_nodes):
        supernode_key = tuple(quantized_codes[idx])
        supernode_mapping[supernode_key].append(idx)

    return quantized_codes, codebooks, dict(supernode_mapping)


def apply_selective_coarsening(data, coarsen_node_types=['careunit', 'inputevent'],
                               m=8, k=16, random_state=42):
    """
    Apply selective coarsening to specific node types in heterogeneous graph data.
    For regression tasks, preserves continuous targets appropriately.

    Args:
        data: HeteroData object
        coarsen_node_types: List of node types to coarsen
        m: Number of subvectors for PQ
        k: Number of centroids per subspace

    Returns:
        coarsened_data: New HeteroData object with coarsened nodes
        coarsening_info: Information about the coarsening process
    """
    print(f"\n{'=' * 50}")
    print(f"APPLYING SELECTIVE GRAPH COARSENING FOR REGRESSION")
    print(f"{'=' * 50}")

    coarsened_data = HeteroData()
    coarsening_info = {}

    # Process each node type
    for node_type in data.node_types:
        print(f"\nProcessing {node_type} nodes...")
        original_features = data[node_type].x
        original_count = original_features.size(0)

        if node_type in coarsen_node_types:
            print(f"  Applying coarsening to {node_type} ({original_count} nodes)")

            # Apply product quantization coarsening
            quantized_codes, codebooks, supernode_mapping = product_quantization_coarsening(
                original_features, m=m, k=k, random_state=random_state
            )

            # Create coarsened features by averaging
            coarsened_features = []
            original_to_coarsened = {}
            coarsened_to_original = {}

            for coarsened_idx, (supernode_key, original_indices) in enumerate(supernode_mapping.items()):
                # Average features of nodes in the same supernode
                supernode_features = original_features[original_indices].mean(dim=0)
                coarsened_features.append(supernode_features)

                # Create mappings
                coarsened_to_original[coarsened_idx] = original_indices
                for orig_idx in original_indices:
                    original_to_coarsened[orig_idx] = coarsened_idx

            coarsened_features = torch.stack(coarsened_features)
            coarsened_count = coarsened_features.size(0)

            print(f"  Coarsened from {original_count} to {coarsened_count} nodes "
                  f"(reduction: {(1 - coarsened_count / original_count) * 100:.1f}%)")

            # Store coarsened data
            coarsened_data[node_type].x = coarsened_features

            # Store coarsening information
            coarsening_info[node_type] = {
                'original_count': original_count,
                'coarsened_count': coarsened_count,
                'original_to_coarsened': original_to_coarsened,
                'coarsened_to_original': coarsened_to_original,
                'codebooks': codebooks,
                'quantized_codes': quantized_codes
            }

            # Copy other attributes if they exist
            for attr_name in data[node_type].keys():
                if attr_name != 'x':
                    if attr_name in ['y', 'y_continuous', 'train_mask', 'val_mask', 'test_mask']:
                        # For target attributes and masks, we need to handle them specially
                        # For regression targets, skip them here - they'll be handled later for stay nodes
                        continue
                    else:
                        # For other attributes, we might need to aggregate or skip
                        print(f"    Skipping attribute {attr_name} for coarsened node type {node_type}")

        else:
            print(f"  Preserving {node_type} nodes ({original_count} nodes)")
            # Copy node data as-is for non-coarsened types
            for attr_name, attr_value in data[node_type].items():
                coarsened_data[node_type][attr_name] = attr_value

            coarsening_info[node_type] = {
                'original_count': original_count,
                'coarsened_count': original_count,
                'coarsened': False
            }

    # Process edges - this is the tricky part
    print(f"\nProcessing edges...")
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        original_edge_index = data[edge_type].edge_index

        print(f"  Processing edge type: {src_type} -> {relation} -> {dst_type}")
        print(f"    Original edges: {original_edge_index.size(1)}")

        # Get the edge mapping based on whether source/destination nodes are coarsened
        src_coarsened = src_type in coarsen_node_types
        dst_coarsened = dst_type in coarsen_node_types

        if not src_coarsened and not dst_coarsened:
            # Neither source nor destination is coarsened - copy as-is
            coarsened_data[edge_type].edge_index = original_edge_index
            print(f"    Preserved: {original_edge_index.size(1)} edges")

        else:
            # At least one end is coarsened - need to remap edges
            src_edges = original_edge_index[0].cpu().numpy()
            dst_edges = original_edge_index[1].cpu().numpy()

            new_src_edges = []
            new_dst_edges = []
            edge_weights = defaultdict(int)

            for i in range(len(src_edges)):
                src_idx = src_edges[i]
                dst_idx = dst_edges[i]

                # Map source index
                if src_coarsened:
                    new_src_idx = coarsening_info[src_type]['original_to_coarsened'].get(src_idx)
                    if new_src_idx is None:
                        continue  # Skip if mapping not found
                else:
                    new_src_idx = src_idx

                # Map destination index
                if dst_coarsened:
                    new_dst_idx = coarsening_info[dst_type]['original_to_coarsened'].get(dst_idx)
                    if new_dst_idx is None:
                        continue  # Skip if mapping not found
                else:
                    new_dst_idx = dst_idx

                # Count edge weights (multiple original edges may map to same coarsened edge)
                edge_key = (new_src_idx, new_dst_idx)
                edge_weights[edge_key] += 1

            # Create new edge index
            if edge_weights:
                new_edges = list(edge_weights.keys())
                new_src_edges = [edge[0] for edge in new_edges]
                new_dst_edges = [edge[1] for edge in new_edges]
                weights = [edge_weights[edge] for edge in new_edges]

                coarsened_data[edge_type].edge_index = torch.tensor(
                    [new_src_edges, new_dst_edges], dtype=torch.long
                )

                # Store edge weights if there are multiple edges between same supernodes
                if max(weights) > 1:
                    coarsened_data[edge_type].edge_weight = torch.tensor(weights, dtype=torch.float)

                print(f"    Coarsened: {len(new_edges)} edges (weights: min={min(weights)}, max={max(weights)})")
            else:
                # No valid edges after coarsening
                coarsened_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
                print(f"    No valid edges after coarsening")

        # Copy other edge attributes if they exist
        for attr_name, attr_value in data[edge_type].items():
            if attr_name not in ['edge_index', 'edge_weight']:
                # For now, skip other edge attributes as they may need special handling
                print(f"    Skipping edge attribute {attr_name}")

    print(f"\n{'=' * 50}")
    print(f"COARSENING SUMMARY")
    print(f"{'=' * 50}")
    total_original = 0
    total_coarsened = 0

    for node_type, info in coarsening_info.items():
        original_count = info['original_count']
        coarsened_count = info['coarsened_count']
        total_original += original_count
        total_coarsened += coarsened_count

        if info.get('coarsened', True):
            reduction = (1 - coarsened_count / original_count) * 100
            print(f"{node_type:12}: {original_count:6} -> {coarsened_count:6} ({reduction:5.1f}% reduction)")
        else:
            print(f"{node_type:12}: {original_count:6} -> {coarsened_count:6} (preserved)")

    total_reduction = (1 - total_coarsened / total_original) * 100
    print(f"{'TOTAL':12}: {total_original:6} -> {total_coarsened:6} ({total_reduction:5.1f}% reduction)")

    return coarsened_data, coarsening_info


def compare_with_baseline(original_data, coarsened_data, coarsening_info):
    """Compare original vs coarsened graph statistics"""
    print(f"\n{'=' * 50}")
    print(f"BASELINE VS COARSENED COMPARISON")
    print(f"{'=' * 50}")

    # Node counts
    print(f"Node Comparison:")
    for node_type in original_data.node_types:
        orig_count = original_data[node_type].x.size(0)
        coarse_count = coarsened_data[node_type].x.size(0)
        reduction = (1 - coarse_count / orig_count) * 100 if orig_count > 0 else 0
        print(f"  {node_type:12}: {orig_count:6} -> {coarse_count:6} ({reduction:5.1f}% reduction)")

    # Edge counts
    print(f"\nEdge Comparison:")
    for edge_type in original_data.edge_types:
        orig_edges = original_data[edge_type].edge_index.size(1)
        coarse_edges = coarsened_data[edge_type].edge_index.size(1)
        reduction = (1 - coarse_edges / orig_edges) * 100 if orig_edges > 0 else 0
        print(f"  {edge_type}: {orig_edges:6} -> {coarse_edges:6} ({reduction:5.1f}% reduction)")

    # Memory estimation
    original_params = sum(x.numel() for x in original_data.x_dict.values())
    coarsened_params = sum(x.numel() for x in coarsened_data.x_dict.values())
    memory_reduction = (1 - coarsened_params / original_params) * 100
    print(f"\nEstimated Memory Reduction: {memory_reduction:.1f}%")