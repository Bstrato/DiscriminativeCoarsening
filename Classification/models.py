import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, HANConv, HGTConv
from utils import aggressive_memory_cleanup


class MemoryEfficientGNN(nn.Module):
    """Ultra memory-efficient GNN with embedding extraction capability"""

    def __init__(self, metadata, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        node_types, edge_types = metadata

        # Minimal input projections
        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in node_types
        })

        # Lightweight convolutions
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    hidden_dim, hidden_dim,
                    normalize=True,
                    project=False  # Reduce memory
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Layer normalization for each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Simple classifier
        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, return_embeddings=False):
        """Memory-efficient forward pass with optional embedding extraction"""
        # Project inputs
        x_dict = {key: self.input_projections[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Store embeddings from each layer if requested
        layer_embeddings = []
        if return_embeddings:
            # Store initial embeddings after input projection
            layer_embeddings.append(x_dict['stay'].clone().detach())

        # Apply convolutions with immediate cleanup
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            x_dict_new = conv(x_dict, edge_index_dict)

            # Update and cleanup - handle None values from HeteroConv
            for key in x_dict.keys():
                if key in x_dict_new and x_dict_new[key] is not None:
                    x_dict[key] = F.relu(x_dict_new[key])
                    x_dict[key] = norm(x_dict[key])  # Apply layer normalization
                    x_dict[key] = self.dropout_layer(x_dict[key])
                # If x_dict_new[key] is None, keep the previous value

            # Store embeddings if requested
            if return_embeddings:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            # Clean intermediate results
            del x_dict_new
            if i < len(self.convs) - 1:  # Don't clean on last iteration
                aggressive_memory_cleanup()

        # Get final embeddings before classification
        stay_embeddings = x_dict['stay']

        # Classification
        logits = self.classifier(stay_embeddings)

        if return_embeddings:
            return logits, stay_embeddings.detach(), layer_embeddings
        else:
            return logits

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            logits, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


class HANModel(nn.Module):
    """Heterogeneous Attention Network (HAN) for heterogeneous graphs"""

    def __init__(self, metadata, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, heads=2):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in self.node_types
        })

        # HAN convolutions
        self.convs = nn.ModuleList([
            HANConv(hidden_dim, hidden_dim, metadata, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization and dropout
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

        # Classifier
        self.classifier = Linear(hidden_dim, num_classes)

    def _validate_x_dict(self, x_dict):
        """Ensure all values in x_dict are valid tensors"""
        validated_dict = {}
        for key, value in x_dict.items():
            if value is None or not isinstance(value, torch.Tensor):
                continue
            validated_dict[key] = value
        return validated_dict

    def _handle_hanconv_output(self, x_dict_prev, x_dict_new, layer_idx):
        """
        Handle HANConv output by preserving previous embeddings for None values
        """
        x_dict_processed = {}

        # Check which node types are in the new output
        if isinstance(x_dict_new, dict):
            for node_type in x_dict_prev.keys():
                if node_type in x_dict_new and x_dict_new[node_type] is not None:
                    # HANConv successfully processed this node type
                    x_dict_processed[node_type] = x_dict_new[node_type]
                else:
                    # HANConv returned None for this node type - preserve previous
                    x_dict_processed[node_type] = x_dict_prev[node_type]
        else:
            # If HANConv completely failed, return previous embeddings
            return x_dict_prev

        return x_dict_processed

    def forward(self, x_dict, edge_index_dict, return_embeddings=False):
        """Forward pass with proper None handling"""
        # Validate input dictionary
        x_dict = self._validate_x_dict(x_dict)

        # Project inputs - only for valid node types
        x_dict = {key: F.relu(self.input_projections[key](x))
                  for key, x in x_dict.items() if key in self.input_projections}

        layer_embeddings = []
        if return_embeddings and 'stay' in x_dict:
            layer_embeddings.append(x_dict['stay'].clone().detach())

        # Apply HAN layers with proper None handling
        for i in range(self.num_layers):
            # Store previous embeddings before HANConv
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}

            try:
                # Apply HANConv
                x_dict_new = self.convs[i](x_dict, edge_index_dict)

                # Handle HANConv output (this is where None handling happens)
                x_dict = self._handle_hanconv_output(x_dict_prev, x_dict_new, i)

                # Apply normalization and dropout to all valid tensors
                for key in x_dict.keys():
                    if x_dict[key] is not None and isinstance(x_dict[key], torch.Tensor):
                        x_dict[key] = self.dropout_layer(
                            self.norms[i](F.relu(x_dict[key]))
                        )

            except Exception as e:
                # On error, preserve previous embeddings
                x_dict = x_dict_prev
                break

            if return_embeddings and 'stay' in x_dict:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            aggressive_memory_cleanup()

        # Classification - ensure 'stay' exists
        if 'stay' not in x_dict:
            raise RuntimeError("'stay' node type not found in final embeddings")

        logits = self.classifier(x_dict['stay'])

        if return_embeddings:
            return logits, x_dict['stay'].detach(), layer_embeddings
        else:
            return logits

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            logits, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


class HGTModel(nn.Module):
    """Heterogeneous Graph Transformer (HGT) for heterogeneous graphs"""

    def __init__(self, metadata, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, num_heads=2):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        # Input projections for each node type
        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in self.node_types
        })

        # HGT convolutions - use 'heads' parameter instead of 'num_heads'
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

        # Layer normalization and dropout
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

        # Classifier
        self.classifier = Linear(hidden_dim, num_classes)

    def _validate_x_dict(self, x_dict):
        """Ensure all values in x_dict are valid tensors"""
        validated_dict = {}
        for key, value in x_dict.items():
            if value is None or not isinstance(value, torch.Tensor):
                continue
            validated_dict[key] = value
        return validated_dict

    def _handle_hgtconv_output(self, x_dict_prev, x_dict_new, layer_idx):
        """
        Handle HGTConv output by preserving previous embeddings for None values
        """
        x_dict_processed = {}

        # Check which node types are in the new output
        if isinstance(x_dict_new, dict):
            for node_type in x_dict_prev.keys():
                if node_type in x_dict_new and x_dict_new[node_type] is not None:
                    # HGTConv successfully processed this node type
                    x_dict_processed[node_type] = x_dict_new[node_type]
                else:
                    # HGTConv returned None for this node type - preserve previous
                    x_dict_processed[node_type] = x_dict_prev[node_type]
        else:
            # If HGTConv completely failed, return previous embeddings
            return x_dict_prev

        return x_dict_processed

    def forward(self, x_dict, edge_index_dict, return_embeddings=False):
        """Forward pass with proper None handling"""
        # Validate input dictionary
        x_dict = self._validate_x_dict(x_dict)

        # Project inputs - only for valid node types
        x_dict = {key: F.relu(self.input_projections[key](x))
                  for key, x in x_dict.items() if key in self.input_projections}

        layer_embeddings = []
        if return_embeddings and 'stay' in x_dict:
            layer_embeddings.append(x_dict['stay'].clone().detach())

        # Apply HGT layers with proper None handling
        for i in range(self.num_layers):
            # Store previous embeddings before HGTConv
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}

            try:
                # Apply HGTConv
                x_dict_new = self.convs[i](x_dict, edge_index_dict)

                # Handle HGTConv output (this is where None handling happens)
                x_dict = self._handle_hgtconv_output(x_dict_prev, x_dict_new, i)

                # Apply normalization and dropout to all valid tensors
                for key in x_dict.keys():
                    if x_dict[key] is not None and isinstance(x_dict[key], torch.Tensor):
                        x_dict[key] = self.dropout_layer(
                            self.norms[i](F.relu(x_dict[key]))
                        )

            except Exception as e:
                # On error, preserve previous embeddings
                x_dict = x_dict_prev
                break

            if return_embeddings and 'stay' in x_dict:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            aggressive_memory_cleanup()

        # Classification - ensure 'stay' exists
        if 'stay' not in x_dict:
            raise RuntimeError("'stay' node type not found in final embeddings")

        logits = self.classifier(x_dict['stay'])

        if return_embeddings:
            return logits, x_dict['stay'].detach(), layer_embeddings
        else:
            return logits

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            logits, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


# Model registry with clean, distinct models
MODEL_REGISTRY = {
    'memory_efficient': MemoryEfficientGNN,
    'han': HANModel,
    'hgt': HGTModel
}


def get_model_class(model_name):
    """Get model class by name"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]


def create_model(model_name, metadata, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.2, **kwargs):
    """Factory function to create models with consistent interface"""
    model_class = get_model_class(model_name)

    # Handle model-specific parameters
    model_kwargs = {
        'metadata': metadata,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': num_classes,
        'dropout': dropout
    }

    # Add model-specific parameters
    if model_name == 'han':
        model_kwargs['heads'] = kwargs.get('heads', 2)
    elif model_name == 'hgt':
        model_kwargs['num_heads'] = kwargs.get('num_heads', 2)

    return model_class(**model_kwargs)