"""
dynamics.py
核心去噪引擎：基于 E(n)-等变图神经网络 (EGNN) 预测坐标位移与局部几何特征
"""
import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class EGNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, time_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1 + time_dim, node_dim), nn.SiLU(),
            nn.Linear(node_dim, node_dim), nn.SiLU()
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(node_dim, node_dim), nn.SiLU(),
            nn.Linear(node_dim, 1, bias=False) 
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + time_dim, node_dim), nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes):
        edge_input = torch.cat([h[edge_src], h[edge_dst], dist_sq, t_emb_edges], dim=-1)
        m_ij = self.edge_mlp(edge_input) 
        
        coord_weights = self.coord_mlp(m_ij) 
        coord_shift = diff_cart * coord_weights 
        coord_update = torch.zeros(h.size(0), 3, device=h.device).index_add_(0, edge_src, coord_shift) 
        
        m_i = torch.zeros_like(h).index_add_(0, edge_dst, m_ij)
        h_update = h + self.node_mlp(torch.cat([h, m_i, t_emb_nodes], dim=-1)) 
        
        return h_update, coord_update

class CrystalDynamics(nn.Module):
    def __init__(self, node_dim=64, time_dim=64, num_layers=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2), nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        self.node_embedding = nn.Linear(128, node_dim) 
        self.layers = nn.ModuleList([EGNNLayer(node_dim, 1, time_dim) for _ in range(num_layers)])

    def build_pbc_graph(self, frac_coords, lattice, num_atoms_list, k_neighbors=12):
        edge_src, edge_dst, diff_cart_list, dist_sq_list = [], [], [], []
        start_idx = 0
        device = frac_coords.device
        
        for i, n in enumerate(num_atoms_list):
            f, lat = frac_coords[start_idx : start_idx + n], lattice[i]
            diff_f = f.unsqueeze(1) - f.unsqueeze(0)
            diff_f = diff_f - torch.round(diff_f)
            diff_c = torch.matmul(diff_f, lat) 
            dist_sq = torch.sum(diff_c ** 2, dim=-1) 
            dist_sq.fill_diagonal_(float('inf'))
            
            k = min(k_neighbors, n - 1)
            if k > 0:
                topk_dist_sq, topk_idx = torch.topk(dist_sq, k, dim=-1, largest=False)
                edge_src.append(topk_idx.flatten() + start_idx)
                edge_dst.append(torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx)
                diff_cart_list.append(diff_c[torch.arange(n).unsqueeze(1), topk_idx].view(-1, 3))
                dist_sq_list.append(topk_dist_sq.flatten().unsqueeze(-1))
                
            start_idx += n
        return torch.cat(edge_src), torch.cat(edge_dst), torch.cat(diff_cart_list), torch.cat(dist_sq_list)

    def forward(self, z_nodes, t, frac_coords, lattice, num_atoms_list, batch_indices):
        t_emb_nodes = self.time_mlp(t)[batch_indices]
        h = self.node_embedding(z_nodes)
        edge_src, edge_dst, diff_cart, dist_sq = self.build_pbc_graph(frac_coords, lattice, num_atoms_list)
        t_emb_edges = t_emb_nodes[edge_src] 
        total_coord_shift_cart = torch.zeros_like(frac_coords)
        
        for layer in self.layers:
            h, coord_update = layer(h, diff_cart, dist_sq, edge_src, edge_dst, t_emb_edges, t_emb_nodes)
            total_coord_shift_cart += coord_update
            
        inv_lattice = torch.linalg.inv(lattice)[batch_indices] 
        shift_frac = torch.bmm(total_coord_shift_cart.unsqueeze(1), inv_lattice).squeeze(1)
        return shift_frac, h