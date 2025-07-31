import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, HANConv, HGTConv
from utils_coarsening_regression import aggressive_memory_cleanup


class MemoryEfficientGNNRegression(nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        node_types, edge_types = metadata

        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in node_types
        })

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    hidden_dim, hidden_dim,
                    normalize=True,
                    project=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.regressor = Linear(hidden_dim, 1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, return_embeddings=False):
        x_dict = {key: self.input_projections[key](x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        layer_embeddings = []
        if return_embeddings:
            layer_embeddings.append(x_dict['stay'].clone().detach())

        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            conv_edge_dict = {}
            for edge_type, edge_index in edge_index_dict.items():
                if edge_weight_dict and edge_type in edge_weight_dict:
                    conv_edge_dict[edge_type] = edge_index
                else:
                    conv_edge_dict[edge_type] = edge_index

            x_dict_new = conv(x_dict, conv_edge_dict)

            for key in x_dict.keys():
                if key in x_dict_new and x_dict_new[key] is not None:
                    x_dict[key] = F.relu(x_dict_new[key])
                    x_dict[key] = norm(x_dict[key])
                    x_dict[key] = self.dropout_layer(x_dict[key])

            if return_embeddings:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            del x_dict_new
            if i < len(self.convs) - 1:
                aggressive_memory_cleanup()

        stay_embeddings = x_dict['stay']
        los_prediction = self.regressor(stay_embeddings).squeeze()

        if return_embeddings:
            return los_prediction, stay_embeddings.detach(), layer_embeddings
        else:
            return los_prediction

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            los_prediction, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


class HANRegressionModel(nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_layers=2, dropout=0.2, heads=2):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads

        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in self.node_types
        })

        self.convs = nn.ModuleList([
            HANConv(hidden_dim, hidden_dim, metadata, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

        self.regressor = Linear(hidden_dim, 1)

    def _validate_x_dict(self, x_dict):
        """Ensure all values in x_dict are valid tensors"""
        validated_dict = {}
        for key, value in x_dict.items():
            if value is None or not isinstance(value, torch.Tensor):
                continue
            validated_dict[key] = value
        return validated_dict

    def _handle_hanconv_output(self, x_dict_prev, x_dict_new, layer_idx):
        """Handle HANConv output by preserving previous embeddings for None values"""
        x_dict_processed = {}

        if isinstance(x_dict_new, dict):
            for node_type in x_dict_prev.keys():
                if node_type in x_dict_new and x_dict_new[node_type] is not None:
                    x_dict_processed[node_type] = x_dict_new[node_type]
                else:
                    x_dict_processed[node_type] = x_dict_prev[node_type]
        else:
            return x_dict_prev

        return x_dict_processed

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, return_embeddings=False):
        x_dict = self._validate_x_dict(x_dict)

        x_dict = {key: F.relu(self.input_projections[key](x))
                  for key, x in x_dict.items() if key in self.input_projections}

        layer_embeddings = []
        if return_embeddings and 'stay' in x_dict:
            layer_embeddings.append(x_dict['stay'].clone().detach())

        for i in range(self.num_layers):
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}

            try:
                x_dict_new = self.convs[i](x_dict, edge_index_dict)
                x_dict = self._handle_hanconv_output(x_dict_prev, x_dict_new, i)

                for key in x_dict.keys():
                    if x_dict[key] is not None and isinstance(x_dict[key], torch.Tensor):
                        x_dict[key] = self.dropout_layer(
                            self.norms[i](F.relu(x_dict[key]))
                        )

            except Exception as e:
                print(f"HANConv layer {i} failed: {str(e)}")
                x_dict = x_dict_prev
                break

            if return_embeddings and 'stay' in x_dict:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            aggressive_memory_cleanup()

        if 'stay' not in x_dict:
            raise RuntimeError("'stay' node type not found in final embeddings")

        stay_embeddings = x_dict['stay']
        if stay_embeddings.dim() != 2:
            raise RuntimeError(f"Expected 2D stay embeddings, got {stay_embeddings.dim()}D")

        los_prediction = self.regressor(stay_embeddings).squeeze()

        if return_embeddings:
            return los_prediction, stay_embeddings.detach(), layer_embeddings
        else:
            return los_prediction

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            los_prediction, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


class HGTRegressionModel(nn.Module):
    def __init__(self, metadata, hidden_dim=64, num_layers=2, dropout=0.2, num_heads=2):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads

        self.input_projections = nn.ModuleDict({
            node_type: Linear(-1, hidden_dim) for node_type in self.node_types
        })

        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

        self.regressor = Linear(hidden_dim, 1)

    def _validate_x_dict(self, x_dict):
        """Ensure all values in x_dict are valid tensors"""
        validated_dict = {}
        for key, value in x_dict.items():
            if value is None or not isinstance(value, torch.Tensor):
                continue
            validated_dict[key] = value
        return validated_dict

    def _handle_hgtconv_output(self, x_dict_prev, x_dict_new, layer_idx):
        """Handle HGTConv output by preserving previous embeddings for None values"""
        x_dict_processed = {}

        if isinstance(x_dict_new, dict):
            for node_type in x_dict_prev.keys():
                if node_type in x_dict_new and x_dict_new[node_type] is not None:
                    x_dict_processed[node_type] = x_dict_new[node_type]
                else:
                    x_dict_processed[node_type] = x_dict_prev[node_type]
        else:
            return x_dict_prev

        return x_dict_processed

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, return_embeddings=False):
        x_dict = self._validate_x_dict(x_dict)

        x_dict = {key: F.relu(self.input_projections[key](x))
                  for key, x in x_dict.items() if key in self.input_projections}

        layer_embeddings = []
        if return_embeddings and 'stay' in x_dict:
            layer_embeddings.append(x_dict['stay'].clone().detach())

        for i in range(self.num_layers):
            x_dict_prev = {k: v.clone() for k, v in x_dict.items()}

            try:
                x_dict_new = self.convs[i](x_dict, edge_index_dict)
                x_dict = self._handle_hgtconv_output(x_dict_prev, x_dict_new, i)

                for key in x_dict.keys():
                    if x_dict[key] is not None and isinstance(x_dict[key], torch.Tensor):
                        x_dict[key] = self.dropout_layer(
                            self.norms[i](F.relu(x_dict[key]))
                        )

            except Exception as e:
                print(f"HGTConv layer {i} failed: {str(e)}")
                x_dict = x_dict_prev
                break

            if return_embeddings and 'stay' in x_dict:
                layer_embeddings.append(x_dict['stay'].clone().detach())

            aggressive_memory_cleanup()

        if 'stay' not in x_dict:
            raise RuntimeError("'stay' node type not found in final embeddings")

        stay_embeddings = x_dict['stay']
        if stay_embeddings.dim() != 2:
            raise RuntimeError(f"Expected 2D stay embeddings, got {stay_embeddings.dim()}D")

        los_prediction = self.regressor(stay_embeddings).squeeze()

        if return_embeddings:
            return los_prediction, stay_embeddings.detach(), layer_embeddings
        else:
            return los_prediction

    def get_embeddings(self, x_dict, edge_index_dict):
        """Extract embeddings without computing gradients"""
        self.eval()
        with torch.no_grad():
            los_prediction, final_embeddings, layer_embeddings = self.forward(
                x_dict, edge_index_dict, return_embeddings=True
            )
            return final_embeddings, layer_embeddings


MODEL_REGISTRY_REGRESSION = {
    'memory_efficient': MemoryEfficientGNNRegression,
    'han': HANRegressionModel,
    'hgt': HGTRegressionModel
}


def get_regression_model_class(model_name):
    """Get regression model class by name"""
    if model_name not in MODEL_REGISTRY_REGRESSION:
        raise ValueError(
            f"Unknown regression model: {model_name}. Available models: {list(MODEL_REGISTRY_REGRESSION.keys())}")
    return MODEL_REGISTRY_REGRESSION[model_name]


def create_regression_model(model_name, metadata, hidden_dim=64, num_layers=2, dropout=0.2, **kwargs):
    """Factory function to create regression models with consistent interface"""
    model_class = get_regression_model_class(model_name)

    model_kwargs = {
        'metadata': metadata,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout
    }

    if model_name == 'han':
        model_kwargs['heads'] = kwargs.get('heads', 2)
    elif model_name == 'hgt':
        model_kwargs['num_heads'] = kwargs.get('num_heads', 2)

    return model_class(**model_kwargs)


MemoryEfficientGNN = MemoryEfficientGNNRegression