"""
cgcnn_encoder.py
CO2RR 团簇特化版 CGCNN 编码器 (稀疏图构建 + 纯笛卡尔坐标)
"""
import torch
import torch.nn as nn

try:
    from torch_cluster import knn_graph as _knn_graph
    _HAS_TORCH_CLUSTER = True
except ImportError:
    _HAS_TORCH_CLUSTER = False

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=8.0, num_gaussians=64):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(self.coeff * torch.pow(dist - self.offset.view(1, -1), 2))

class CGCNNLayer(nn.Module):
    def __init__(self, atom_fea_len=64, nbr_fea_len=64):
        super().__init__()
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, edge_src, edge_dst):
        atom_src, atom_dst = atom_in_fea[edge_src], atom_in_fea[edge_dst]
        total_fea = self.bn1(self.fc_full(torch.cat([atom_src, atom_dst, nbr_fea], dim=1)))
        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        nbr_msg = self.sigmoid(filter_fea) * self.softplus1(core_fea)

        atom_update = torch.zeros_like(atom_in_fea).index_add_(0, edge_dst, nbr_msg)
        return self.softplus2(atom_in_fea + self.bn2(atom_update))

class CGCNNEncoder(nn.Module):
    def __init__(self, latent_dim=128, atom_fea_len=64, nbr_fea_len=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(119, atom_fea_len, padding_idx=0)
        self.distance_expansion = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=nbr_fea_len)
        self.convs = nn.ModuleList([CGCNNLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Sequential(nn.Linear(atom_fea_len, 128), nn.SiLU(), nn.Linear(128, latent_dim * 2))

    def build_graph_vectorized(self, cart_coords, batch_indices, k_neighbors=12):
        N = cart_coords.size(0)
        device = cart_coords.device
        if N == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 1, device=device)

        max_n = torch.bincount(batch_indices).max().item()
        k = min(k_neighbors, max_n - 1)
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 1, device=device)

        if _HAS_TORCH_CLUSTER:
            edge_index = _knn_graph(cart_coords, k=k, batch=batch_indices, loop=False)
            edge_src, edge_dst = edge_index[0], edge_index[1]
            diff = cart_coords[edge_src] - cart_coords[edge_dst]
            dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
            return edge_src, edge_dst, torch.sqrt(dist_sq + 1e-8)

        # Fallback: original dense O(N^2) implementation
        mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)
        diff_dense = cart_coords.unsqueeze(1) - cart_coords.unsqueeze(0)
        dist_sq_dense = torch.sum(diff_dense**2, dim=-1)

        dist_sq_dense.masked_fill_(~mask, float('inf'))
        dist_sq_dense.fill_diagonal_(float('inf'))

        topk_dist_sq, topk_idx = torch.topk(dist_sq_dense, k, dim=-1, largest=False)
        edge_dst_fb = torch.arange(N, device=device).unsqueeze(1).expand(N, k).flatten()
        edge_src_fb = topk_idx.flatten()
        edge_dist_sq = topk_dist_sq.flatten()

        valid = ~torch.isinf(edge_dist_sq)
        return edge_src_fb[valid], edge_dst_fb[valid], torch.sqrt(edge_dist_sq[valid].unsqueeze(-1) + 1e-8)

    def forward(self, cart_coords, species, batch_indices, num_graphs):
        atom_fea = self.embedding(species)
        edge_src, edge_dst, edge_dist = self.build_graph_vectorized(cart_coords, batch_indices)
        if len(edge_src) > 0:
            nbr_fea = self.distance_expansion(edge_dist)
            for conv_func in self.convs:
                atom_fea = conv_func(atom_fea, nbr_fea, edge_src, edge_dst)

        crys_fea = torch.zeros(num_graphs, atom_fea.size(1), device=cart_coords.device).index_add_(0, batch_indices, atom_fea)
        bincount = torch.bincount(batch_indices, minlength=num_graphs).clamp(min=1).view(-1, 1).float()
        crys_fea = crys_fea / bincount
        return torch.chunk(self.conv_to_fc(crys_fea), 2, dim=-1)
