# ============================================================================
# codes adapted from:
# https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10423255
# https://github.com/jianjianGJ/hgcond/tree/main
# ============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_
from torch_scatter import scatter_mean, scatter_sum
from sklearn.cluster import BisectingKMeans
from torch_sparse import sum as sparsesum, mul, SparseTensor
from torch_geometric.nn import Linear
from tqdm import tqdm
import time
from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP, IMDB
from sklearn.metrics import f1_score
import prettytable as pt

def asymmetric_gcn_norm(adj_t):
    """Normalize adjacency matrix for heterogeneous graphs."""
    if isinstance(adj_t, SparseTensor):
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        deg_src = sparsesum(adj_t, dim=0) + 0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = sparsesum(adj_t, dim=1) + 0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)

        adj_t = mul(adj_t, deg_dst_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_src_inv_sqrt.view(1, -1))
    else:
        deg_src = adj_t.sum(0) + 0.00001
        deg_src_inv_sqrt = deg_src.pow_(-0.5)
        deg_src_inv_sqrt.masked_fill_(deg_src_inv_sqrt == float('inf'), 0.)
        deg_dst = adj_t.sum(1) + 0.00001
        deg_dst_inv_sqrt = deg_dst.pow_(-0.5)
        deg_dst_inv_sqrt.masked_fill_(deg_dst_inv_sqrt == float('inf'), 0.)
        adj_t = adj_t * deg_dst_inv_sqrt.view(-1, 1)
        adj_t = adj_t * deg_src_inv_sqrt.view(1, -1)
    return adj_t


def related_parameters(basic_model, x_dict, adj_t_dict, y, train_mask):
    """Get model parameters that contribute to gradients."""
    output = basic_model(x_dict, adj_t_dict)
    loss_real = F.nll_loss(output[train_mask], y[train_mask])
    gw_reals = torch.autograd.grad(loss_real, basic_model.parameters(), allow_unused=True)
    parameters = []
    for i, p in enumerate(basic_model.parameters()):
        if gw_reals[i] is not None and p.data.dim() == 2:
            parameters.append(p)
    return parameters


def train_model(model, opt_parameter, optimizer, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn):
    """Train model on synthetic data for several epochs."""
    for epoch in range(1, opt_parameter['epochs_basic_model'] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x_syn_dict, adj_t_syn_dict)[mask_syn]
        loss = F.nll_loss(out, y_syn[mask_syn])
        loss.backward()
        optimizer.step()


def train_model_ealystop(model, opt_parameter, x_dict, adj_t_dict, y, train_mask, val_mask):
    """Train model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_parameter['lr'], weight_decay=opt_parameter['weight_decay'])

    best_val_acc = 0
    for epoch in tqdm(range(1, opt_parameter['epochs'] + 1), desc='Training', ncols=80):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, adj_t_dict)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(x_dict, adj_t_dict).argmax(dim=-1)
        val_acc = (pred[val_mask] == y[val_mask]).sum() / val_mask.sum()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights = deepcopy(model.state_dict())

    model.load_state_dict(weights)
    return best_val_acc


def lazy_initialize(model, x_dict, adj_t_dict):
    """Initialize model with a forward pass."""
    with torch.no_grad():
        model(x_dict, adj_t_dict)


def evalue_model(model, x_dict, adj_t_dict, y, test_mask):
    """Evaluate model performance."""
    model.eval()
    logits = model(x_dict, adj_t_dict)[test_mask]
    labels = y[test_mask].cpu()
    preds = logits.argmax(1).cpu()
    acc = (labels == preds).sum() / test_mask.sum()
    acc = acc.item()
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    return acc, f1_micro, f1_macro


def reset_adj_t_dict_to_mean(adj_t_dict):
    """Convert adjacency matrices to mean aggregation format."""
    adj_t_dict_mean = {}
    for edge_type, adj_t in adj_t_dict.items():
        if isinstance(adj_t, SparseTensor):
            adj_t_mean = adj_t.set_value(torch.ones(adj_t.nnz(), device=adj_t.device()), layout='coo')
            degree_inv = 1 / (adj_t_mean.sum(1).view(adj_t_mean.size(0), 1) + 0.0001)
            adj_t_mean = adj_t_mean * degree_inv
        else:
            adj_t_mean = adj_t.clone()
            adj_t_mean[adj_t_mean > 0] = 1
            adj_t_mean = adj_t_mean / (adj_t_mean.sum(1, keepdim=True) + 0.0001)
        adj_t_dict_mean[edge_type] = adj_t_mean
    return adj_t_dict_mean


class HeteroSAGE(torch.nn.Module):
    """HeteroSAGE model for heterogeneous graphs."""

    def __init__(self, hidden_channels, out_channels, num_layers,
                 edge_types, node_types, target_node_type, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.num_layers = num_layers
        self.target_node_type = target_node_type

        self.lins = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = torch.nn.ModuleList()
            self.lins[node_type].append(Linear(-1, hidden_channels, bias=False))
            for _ in range(num_layers - 1):
                self.lins[node_type].append(Linear(hidden_channels, hidden_channels, bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self, init_list=None):
        """Reset model parameters."""
        if init_list is None:
            for node_type in self.lins.keys():
                for lin in self.lins[node_type]:
                    lin.reset_parameters()
            self.out_lin.reset_parameters()
        else:
            i = 0
            for self_p in self.parameters():
                if self_p.dim() == 2:
                    self_p.data.copy_(init_list[i])
                    i += 1

    def forward(self, x_dict, adj_t_dict, get_embeddings=False):
        """Forward pass."""
        adj_t_dict_mean = reset_adj_t_dict_to_mean(adj_t_dict)
        h_dict = {}

        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l == 0:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](x_dict[node_type]))
                else:
                    h_dict[node_type] = F.relu_(self.lins[node_type][l](h_dict[node_type]))

            out_dict = {node_type: [self.alpha * x] for node_type, x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict_mean.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])

            for node_type in x_dict.keys():
                h_dict[node_type] = torch.mean(torch.stack(out_dict[node_type], dim=0), dim=0)

        target_logits = self.out_lin(h_dict[self.target_node_type])
        if get_embeddings:
            h_dict = {node_type: h for node_type, h in h_dict.items()}
            return target_logits, h_dict
        else:
            return target_logits


def get_DBLP(root):
    """Load DBLP dataset."""
    dataset = DBLP(root + '/DBLP', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    data['conference'].x = torch.ones(data['conference'].num_nodes, 1)
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type = 'author'
    data.num_classes = int(data['author'].y.max() + 1)
    data.name = 'dblp'
    return data


def get_IMDB(root):
    """Load IMDB dataset."""
    dataset = IMDB(root + '/IMDB', transform=T.ToSparseTensor(remove_edge_index=False))
    data = dataset[0]
    for edge_type in data.edge_types:
        data[edge_type]['adj_t'] = asymmetric_gcn_norm(data[edge_type]['adj_t'])
    data.target_node_type = 'movie'
    data.num_classes = int(data['movie'].y.max() + 1)
    data.name = 'imdb'
    return data


def get_data(name, root='./datahetero'):
    """Get dataset by name."""
    if name.lower() == 'dblp':
        return get_DBLP(root)
    elif name.lower() == 'imdb':
        return get_IMDB(root)
    else:
        raise NotImplementedError


def match_loss(gw_syns, gw_reals):
    """Compute matching loss between synthetic and real gradients."""
    dis = 0
    for ig in range(len(gw_reals)):
        gw_real = gw_reals[ig]
        gw_syn = gw_syns[ig]
        if gw_syn.dim() == 2:
            dis += distance_w(gw_real, gw_syn)
    return dis


def distance_w(gwr, gws):
    """Compute cosine distance between gradient matrices."""
    gwr = gwr.T
    gws = gws.T
    dis_weight = torch.mean(1 - torch.sum(gwr * gws, dim=-1) /
                            (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    return dis_weight


def gcond_initialize(data, num_syn_dict):
    """Random initialization for graph condensation."""
    node_types, edge_type = data.node_types, data.edge_types
    target_node_type = data.target_node_type
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y.clone()
    train_mask = data[target_node_type].train_mask

    x_initial_dict = {}
    for node_type in node_types:
        num_full = x_dict[node_type].shape[0]
        num_syn = num_syn_dict[node_type]
        if node_type != target_node_type:
            random_selected = torch.randperm(num_full)[:num_syn]
            x_initial_dict[node_type] = x_dict[node_type][random_selected]
        else:
            labeled_index = torch.arange(num_full)[train_mask]
            num_labeld = labeled_index.shape[0]
            random_selected = labeled_index[torch.randint(num_labeld, (num_syn,))]
            x_initial_dict[node_type] = x_dict[node_type][random_selected]
            y_syn = y[random_selected]
            mask_syn = y_syn > -99

    indices_dict = {}
    for edge_type, adj_t in adj_t_dict.items():
        src_type, _, dst_type = edge_type
        dst_num, src_num = num_syn_dict[dst_type], num_syn_dict[src_type]
        edge_cout = torch.ones(dst_num, src_num)
        indices_dict[edge_type] = torch.LongTensor(edge_cout.to_sparse().indices())

    return x_initial_dict, indices_dict, y_syn, mask_syn


def cluster_initialize(data, num_syn_dict, model_name, architecture, opt_parameter, file_path):
    """Cluster-based initialization for graph condensation."""
    device = data[data.target_node_type].y.device
    target_node_type, num_classes = data.target_node_type, data.num_classes
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    val_mask = data[target_node_type].val_mask

    if os.path.exists(file_path):
        cluster_dict, cluster_adi_dict, cluster_y_count = torch.load(file_path)
    else:
        print('Getting embedding for clustering:')
        model = HeteroSAGE(**architecture,
                           out_channels=num_classes,
                           node_types=data.node_types,
                           edge_types=data.edge_types,
                           target_node_type=target_node_type).to(device)
        lazy_initialize(model, x_dict, adj_t_dict)
        time_start = time.time()
        acc = train_model_ealystop(model, opt_parameter, x_dict, adj_t_dict, y, train_mask, val_mask)
        time_end = time.time()
        time_used = time_end - time_start
        print(f'Embedding obtained: acc:{acc:.4f} time for epochs:{time_used:.4f}')

        _, h_dict = model(x_dict, adj_t_dict, get_embeddings=True)
        h_dict = {node_type: h.detach().cpu() for node_type, h in h_dict.items()}
        cluster_dict = {}
        print('Clustering for initialization.')

        for node_type, h in h_dict.items():
            k_means = BisectingKMeans(n_clusters=num_syn_dict[node_type], random_state=0)
            k_means.fit(h)
            cluster_dict[node_type] = torch.LongTensor(k_means.predict(h))
            if node_type == target_node_type:
                y_train = y.clone()
                y_train[~train_mask] = -1
                y_onehot = F.one_hot(y_train + 1, num_classes=num_classes + 1)[:, 1:].cpu()
                cluster_y_count = scatter_sum(y_onehot, cluster_dict[node_type], dim=0)

        cluster_adi_dict = {}
        for edge_type, adj_t in adj_t_dict.items():
            src_type, _, dst_type = edge_type
            src_num, dst_num = cluster_dict[src_type].max() + 1, cluster_dict[dst_type].max() + 1
            cluster_adi_dict[edge_type] = torch.zeros(dst_num, src_num)
            row, col, v = adj_t.coo()
            c_src, c_dst = cluster_dict[src_type], cluster_dict[dst_type]
            c_dst_row = c_dst[row]
            c_src_col = c_src[col]
            for i in range(dst_num):
                mask_i = c_dst_row == i
                connected = c_src_col[mask_i]
                connected = connected[connected > -1]
                cluster_adi_dict[edge_type][i] = F.one_hot(connected, num_classes=src_num).sum(0)

        torch.save((cluster_dict, cluster_adi_dict, cluster_y_count), file_path)

    x_initial_dict = {}
    for node_type in data.node_types:
        clusters = cluster_dict[node_type]
        x_initial_dict[node_type] = scatter_mean(x_dict[node_type].cpu(), clusters, dim=0)
        if node_type == target_node_type:
            y_train = y.clone()
            y_train[~train_mask] = -1
            y_onehot = F.one_hot(y_train + 1, num_classes=num_classes + 1)[:, 1:].cpu()
            count = scatter_sum(y_onehot, clusters, dim=0)
            y_syn = count.argmax(dim=1)
            mask_syn = count.sum(1) > 0
            y_syn[~mask_syn] = -1

    indices_dict = {}
    for edge_type in data.edge_types:
        edge_cout = cluster_adi_dict[edge_type]
        indices_dict[edge_type] = torch.LongTensor(edge_cout.to_sparse().indices())

    print('Initialization obtained.')
    return x_initial_dict, indices_dict, y_syn, mask_syn


class GraphSynthesizer(nn.Module):
    """Graph synthesizer for heterogeneous graph condensation."""

    def __init__(self, data, cond_rate, feat_init='cluster', edge_hidden_channels=None):
        super(GraphSynthesizer, self).__init__()
        self.device = data[data.target_node_type].x.device
        self.name = data.name
        self.cond_rate = cond_rate
        self.target_node_type = data.target_node_type
        self.edge_hidden_channels = edge_hidden_channels
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        self.num_classes = data.num_classes

        self.num_syn_dict = {}
        for node_type, x in data.x_dict.items():
            num_syn = max(int(x.shape[0] * cond_rate), 1)
            self.num_syn_dict[node_type] = num_syn

        if feat_init == 'cluster':
            architecture = {'hidden_channels': 64, 'num_layers': 3}
            opt_parameter = {'epochs': 100, 'lr': 0.005, 'weight_decay': 0.001}
            save_path = './clusters/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_path = save_path + f'/{data.name}-{cond_rate}.clusters'
            x_initial_dict, indices_dict, y_syn, mask_syn = cluster_initialize(
                data, self.num_syn_dict, 'HeteroSAGE', architecture, opt_parameter, file_path)
        elif feat_init == 'sample':
            x_initial_dict, indices_dict, y_syn, mask_syn = gcond_initialize(data, self.num_syn_dict)

        self.x_initial_dict = {k: v.to(self.device) for k, v in x_initial_dict.items()}
        self.indices_dict = {k: v.to(self.device) for k, v in indices_dict.items()}
        self.y_syn, self.mask_syn = y_syn.to(self.device), mask_syn.to(self.device)

        self.x_syn_dict = {}
        for node_type, x in data.x_dict.items():
            num_syn = self.num_syn_dict[node_type]
            self.x_syn_dict[node_type] = nn.Parameter(torch.FloatTensor(num_syn, x.shape[1]).to(self.device))
            self.x_syn_dict[node_type].data.copy_(self.x_initial_dict[node_type])

        if self.edge_hidden_channels is not None:
            self.edge_mlp_dict = {}
            for edge_type in data.edge_types:
                src_type, _, dst_type = edge_type
                in_channels = data.x_dict[src_type].shape[1] + data.x_dict[dst_type].shape[1]
                self.edge_mlp_dict[edge_type] = nn.Sequential(
                    nn.Linear(in_channels, edge_hidden_channels),
                    nn.BatchNorm1d(edge_hidden_channels),
                    nn.ReLU(),
                    nn.Linear(edge_hidden_channels, 1)
                ).to(self.device)

        self.get_adj_t_syn_dict()

    def x_parameters(self):
        """Get trainable node feature parameters."""
        return list(self.x_syn_dict.values())

    def adj_parameters(self):
        """Get trainable adjacency parameters."""
        parameters = []
        if self.edge_hidden_channels is not None:
            for edge_mlp in self.edge_mlp_dict.values():
                parameters.extend(edge_mlp.parameters())
        return parameters

    def get_x_syn_dict(self):
        """Get current synthetic node features."""
        return self.x_syn_dict

    def get_adj_t_syn_dict(self):
        """Compute current synthetic adjacency matrices."""
        self.adj_t_syn_dict = {}
        for edge_type in self.edge_types:
            src_type, _, dst_type = edge_type
            num_src, num_dst = self.num_syn_dict[src_type], self.num_syn_dict[dst_type]

            if isinstance(num_src, torch.Tensor):
                num_src = int(num_src.sum().item())
            if isinstance(num_dst, torch.Tensor):
                num_dst = int(num_dst.sum().item())

            adj_t_syn = torch.zeros(size=(num_dst, num_src), device=self.device)
            indices = self.indices_dict[edge_type]

            if indices.shape[1] <= 1:
                continue

            row, col = indices[0], indices[1]

            if self.edge_hidden_channels is None:
                adj_t_syn[row, col] = 1.
            else:
                adj_t_syn[row, col] = torch.sigmoid(self.edge_mlp_dict[edge_type](
                    torch.cat([self.x_syn_dict[dst_type][row], self.x_syn_dict[src_type][col]], dim=1)).flatten())

            adj_t_syn = asymmetric_gcn_norm(adj_t_syn)
            self.adj_t_syn_dict[edge_type] = adj_t_syn

        return self.adj_t_syn_dict

    def get_x_syn_detached_dict(self):
        """Get detached node features."""
        return {node_type: x.detach() for node_type, x in self.x_syn_dict.items()}

    def get_adj_t_syn_detached_dict(self):
        """Get detached adjacency matrices."""
        adj_dict = self.get_adj_t_syn_dict()
        return {edge_type: adj_t.detach() for edge_type, adj_t in adj_dict.items()}


class Orth_Initializer:
    """Orthogonal parameter initializer."""

    def __init__(self, model):
        self.cach_list = []
        self.index_list = []
        for p in model.parameters():
            if p.dim() == 2:
                n_row, n_col = p.shape[0], p.shape[1]
                base = torch.empty(n_row, n_row, n_col)
                for i in range(n_col):
                    orthogonal_(base[:, :, i])
                self.cach_list.append(base)
                self.index_list.append(0)

    def next_init(self):
        """Get next orthogonal initialization."""
        init_list = []
        for i, cach in enumerate(self.cach_list):
            if self.index_list[i] < cach.shape[0]:
                init_list.append(cach[self.index_list[i]])
                self.index_list[i] += 1
            else:
                for j in range(cach.shape[2]):
                    orthogonal_(self.cach_list[i][:, :, j])
                init_list.append(self.cach_list[i][0])
                self.index_list[i] = 1
        return init_list


def hgcond(data, cond_rate, feat_init, para_init, basicmodel, model_architecture, cond_train):
    """Main HGCond function for heterogeneous graph condensation."""
    target_node_type, num_classes = data.target_node_type, data.num_classes
    node_types, edge_types = data.node_types, data.edge_types
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask

    basic_model = HeteroSAGE(**model_architecture,
                             out_channels=num_classes,
                             node_types=node_types,
                             edge_types=edge_types,
                             target_node_type=target_node_type).to(data[target_node_type].y.device)
    lazy_initialize(basic_model, x_dict, adj_t_dict)

    if para_init == 'orth':
        orth_initi = Orth_Initializer(basic_model)

    graphsyner = GraphSynthesizer(data, cond_rate, feat_init, edge_hidden_channels=64)
    y_syn = graphsyner.y_syn
    mask_syn = graphsyner.mask_syn
    optimizer_cond = torch.optim.Adam(graphsyner.x_parameters() + graphsyner.adj_parameters(),
                                      lr=cond_train['lr'])
    parameters = related_parameters(basic_model, x_dict, adj_t_dict, y, train_mask)

    losses_log = []
    smallest_loss = 99999.
    for initial_i in tqdm(range(cond_train['epochs_initial']),
                          desc='Condensation', ncols=80):
        if para_init == 'orth':
            basic_model.reset_parameters(orth_initi.next_init())
        else:
            basic_model.reset_parameters()
        optimizer_basic_model = torch.optim.Adam(basic_model.parameters(),
                                                 lr=cond_train['lr_basic_model'])

        loss_avg = 0
        for step_syn in range(cond_train['epochs_deep']):
            basic_model.eval()

            x_syn_dict = graphsyner.get_x_syn_dict()
            adj_t_syn_dict = graphsyner.get_adj_t_syn_dict()

            output = basic_model(x_dict, adj_t_dict)
            loss_real = F.nll_loss(output[train_mask], y[train_mask])
            gw_reals = torch.autograd.grad(loss_real, parameters)
            gw_reals = list((_.detach().clone() for _ in gw_reals))

            output_syn = basic_model(x_syn_dict, adj_t_syn_dict)[mask_syn]
            loss_syn = F.nll_loss(output_syn, y_syn[mask_syn])
            gw_syns = torch.autograd.grad(loss_syn, parameters, create_graph=True)

            loss = match_loss(gw_syns, gw_reals)
            optimizer_cond.zero_grad()
            loss.backward()
            optimizer_cond.step()

            loss_avg += loss.item() / cond_train['epochs_deep']

            if step_syn < cond_train['epochs_deep'] - 1:
                x_syn_dict = graphsyner.get_x_syn_detached_dict()
                adj_t_syn_dict = graphsyner.get_adj_t_syn_detached_dict()
                train_model(basic_model, cond_train, optimizer_basic_model,
                            x_syn_dict, adj_t_syn_dict, y_syn, mask_syn)

        losses_log.append(loss_avg)
        if loss_avg < smallest_loss:
            smallest_loss = loss_avg
            graphsyner.best_x_syn_dict = graphsyner.get_x_syn_detached_dict()
            graphsyner.best_adj_t_syn_dict = graphsyner.get_adj_t_syn_detached_dict()

    return graphsyner, losses_log


def evalue_hgcond(num_evalue, data, x_syn_dict, adj_t_syn_dict, y_syn, mask_syn,
                  model_name, model_architecture, model_train, loss_fn=F.cross_entropy):
    """Evaluate condensed graph performance."""
    node_types = data.node_types
    edge_types = data.edge_types
    target_node_type, num_classes = data.target_node_type, data.num_classes
    x_dict, adj_t_dict, y = data.x_dict, data.adj_t_dict, data[target_node_type].y
    train_mask = data[target_node_type].train_mask
    val_mask = data[target_node_type].val_mask
    test_mask = data[target_node_type].test_mask
    device = train_mask.device

    x_syn_dict = {k: v.to(device) for k, v in x_syn_dict.items()}
    adj_t_syn_dict = {k: v.to(device) for k, v in adj_t_syn_dict.items()}
    y_syn = y_syn.to(device)
    mask_syn = mask_syn.to(device)

    val_model = HeteroSAGE(**model_architecture,
                           out_channels=num_classes,
                           node_types=node_types,
                           edge_types=edge_types,
                           target_node_type=target_node_type).to(device)
    lazy_initialize(val_model, x_dict, adj_t_dict)

    max_patience = 10
    trig_early_stop = True
    accs, f1_micros, f1_macros = [], [], []

    for i in range(num_evalue):
        best_acc = 0.
        for j in tqdm(range(model_train['epochs']), desc='Training', ncols=80):
            if trig_early_stop:
                patience = 0
                trig_early_stop = False
                val_model.reset_parameters()
                optimizer_val_model = torch.optim.Adam(val_model.parameters(), lr=model_train['lr'])

            val_model.train()
            optimizer_val_model.zero_grad()
            logits_train = val_model(x_syn_dict, adj_t_syn_dict)[mask_syn]
            loss = loss_fn(logits_train, y_syn[mask_syn])
            loss.backward()
            optimizer_val_model.step()

            with torch.no_grad():
                val_model.eval()
                logits_val = val_model(x_dict, adj_t_dict)[val_mask]
                acc = (logits_val.argmax(1) == y[val_mask]).sum() / logits_val.shape[0]
                if acc > best_acc:
                    best_acc = acc
                    patience = 0
                    weights = deepcopy(val_model.state_dict())
                else:
                    patience += 1
                if patience == max_patience:
                    trig_early_stop = True

        val_model.load_state_dict(weights)
        acc, f1_micro, f1_macro = evalue_model(val_model, x_dict, adj_t_dict, y, test_mask)
        accs.append(acc)
        f1_micros.append(f1_micro)
        f1_macros.append(f1_macro)

    return accs, f1_micros, f1_macros


def main():
    """Main execution function for HGCond with HeteroSAGE on IMDB and DBLP."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_architecture = {
        'hidden_channels': 64,
        'num_layers': 2,
        'alpha': 0.1
    }

    cond_train = {
        'epochs_initial': 20,
        'epochs_deep': 10,
        'lr': 0.01,
        'lr_basic_model': 0.01,
        'epochs_basic_model': 5
    }

    model_train = {
        'epochs': 1000,
        'lr': 0.01
    }

    datasets = ['dblp', 'imdb']
    condensation_rates = [0.001, 0.005, 0.01]

    for dataset_name in datasets:
        print(f"\n=== Processing {dataset_name.upper()} Dataset ===")

        data = get_data(dataset_name, root='./datahetero')
        data = data.to(device)

        print(f"Target node type: {data.target_node_type}")
        print(f"Number of classes: {data.num_classes}")
        print(f"Node types: {data.node_types}")
        print(f"Edge types: {data.edge_types}")

        for cond_rate in condensation_rates:
            print(f"\n--- Condensation Rate: {cond_rate} ---")

            graphsyner, losses_log = hgcond(
                data=data,
                cond_rate=cond_rate,
                feat_init='cluster',
                para_init='orth',
                basicmodel='HeteroSAGE',
                model_architecture=model_architecture,
                cond_train=cond_train
            )

            print(f"Condensation completed. Final loss: {losses_log[-1]:.4f}")

            best_x_syn = graphsyner.best_x_syn_dict
            best_adj_syn = graphsyner.best_adj_t_syn_dict

            print(f"Condensed graph info:")
            for node_type, x_syn in best_x_syn.items():
                original_size = data.x_dict[node_type].shape[0]
                condensed_size = x_syn.shape[0]
                print(f"  {node_type}: {original_size} -> {condensed_size} nodes")

            print("Evaluating condensed graph...")
            accs, f1_micros, f1_macros = evalue_hgcond(
                num_evalue=3,
                data=data,
                x_syn_dict=best_x_syn,
                adj_t_syn_dict=best_adj_syn,
                y_syn=graphsyner.y_syn,
                mask_syn=graphsyner.mask_syn,
                model_name='HeteroSAGE',
                model_architecture=model_architecture,
                model_train=model_train
            )

            acc_mean, acc_std = torch.tensor(accs).mean().item(), torch.tensor(accs).std().item()
            f1_micro_mean = torch.tensor(f1_micros).mean().item()
            f1_macro_mean = torch.tensor(f1_macros).mean().item()

            print(f"Results for {dataset_name} with rate {cond_rate}:")
            print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
            print(f"  F1-Micro: {f1_micro_mean:.4f}")
            print(f"  F1-Macro: {f1_macro_mean:.4f}")


if __name__ == "__main__":
    main()