"""
dynamics.py
纯笛卡尔坐标空间 E(n)-等变动力学引擎 (全张量并行版)
"""
import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        return torch.cat(((time[:, None] * embeddings[None, :]).sin(), (time[:, None] * embeddings[None, :]).cos()), dim=-1)

class EGNNLayer(nn.Module):
    def __init__(self, node_dim, time_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(node_dim * 2 + 1 + time_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim), nn.SiLU())
        self.coord_mlp = nn.Sequential(nn.Linear(node_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, 1, bias=False))
        self.node_mlp = nn.Sequential(nn.Linear(node_dim * 2 + time_dim, node_dim), nn.SiLU(), nn.Linear(node_dim, node_dim))

    def forward(self, h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes):
        m_ij = self.edge_mlp(torch.cat([h[edge_src], h[edge_dst], dist_sq, t_emb_edges], dim=-1))
        
        # 🌟 修复 P10: 归一化方向向量
        dist = torch.sqrt(dist_sq + 1e-8)
        coord_shift = (diff_cart / dist) * self.coord_mlp(m_ij)
        coord_update = torch.zeros(h.size(0), 3, device=h.device).index_add_(0, edge_dst, coord_shift) 
        
        m_i = torch.zeros_like(h).index_add_(0, edge_dst, m_ij)
        h_update = h + self.node_mlp(torch.cat([h, m_i, t_emb_nodes], dim=-1))
        return h_update, coord_update

class CrystalDynamics(nn.Module):
    def __init__(self, node_dim=64, time_dim=64, num_layers=4):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbeddings(time_dim), nn.Linear(time_dim, time_dim * 2), nn.SiLU(), nn.Linear(time_dim * 2, time_dim))
        self.node_embedding = nn.Linear(128, node_dim) 
        self.layers = nn.ModuleList([EGNNLayer(node_dim, time_dim) for _ in range(num_layers)])

    def build_cluster_graph_vectorized(self, cart_coords, batch_indices, k_neighbors=12):
        """🌟 修复 P13：采样/去噪时的批量图重建，消灭 Python for 循环"""
        N = cart_coords.size(0)
        device = cart_coords.device
        mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)
        diff = cart_coords.unsqueeze(1) - cart_coords.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=-1)
        
        dist_sq.masked_fill_(~mask, float('inf'))
        dist_sq.fill_diagonal_(float('inf'))
        
        max_n = torch.bincount(batch_indices).max().item()
        k = min(k_neighbors, max_n - 1)
        if k <= 0: return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, 3, device=device), torch.empty(0, 1, device=device)

        topk_dist_sq, topk_idx = torch.topk(dist_sq, k, dim=-1, largest=False)
        edge_dst = torch.arange(N, device=device).unsqueeze(1).expand(N, k).flatten()
        edge_src = topk_idx.flatten()
        
        valid = ~torch.isinf(topk_dist_sq.flatten())
        edge_src, edge_dst = edge_src[valid], edge_dst[valid]
        return edge_src, edge_dst, diff[edge_src, edge_dst], topk_dist_sq.flatten()[valid].unsqueeze(-1)

    def forward(self, z_nodes, t, cart_coords, batch_indices):
        # 🌟 修复 P1: 现在完全抛弃了 Lattice，输入和输出全都是 Cartesian 坐标
        t_emb_nodes = self.time_mlp(t)[batch_indices]
        h = self.node_embedding(z_nodes)
        
        edge_src, edge_dst, diff_cart, dist_sq = self.build_cluster_graph_vectorized(cart_coords, batch_indices)
        total_coord_shift = torch.zeros_like(cart_coords)
        
        if len(edge_src) > 0:
            t_emb_edges = t_emb_nodes[edge_src] 
            for layer in self.layers:
                h, coord_update = layer(h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes)
                total_coord_shift += coord_update
                
        return total_coord_shift, h