"""
cgcnn_encoder.py
正宗的 CGCNN 编码器，精准提取 CO2RR 团簇的局部化学环境特征
"""
import torch
import torch.nn as nn

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
        atom_src = atom_in_fea[edge_src]
        atom_dst = atom_in_fea[edge_dst]
        total_fea = torch.cat([atom_src, atom_dst, nbr_fea], dim=1)
        
        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)
        
        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.softplus1(core_fea)
        
        nbr_msg = filter_fea * core_fea
        atom_update = torch.zeros_like(atom_in_fea)
        atom_update.index_add_(0, edge_dst, nbr_msg)
        
        atom_update = self.bn2(atom_update)
        return self.softplus2(atom_in_fea + atom_update)

class CGCNNEncoder(nn.Module):
    def __init__(self, latent_dim=128, atom_fea_len=64, nbr_fea_len=64, n_conv=3):
        super().__init__()
        self.embedding = nn.Embedding(100, atom_fea_len, padding_idx=0)
        self.distance_expansion = GaussianSmearing(start=0.0, stop=8.0, num_gaussians=nbr_fea_len)
        self.convs = nn.ModuleList([CGCNNLayer(atom_fea_len, nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Sequential(
            nn.Linear(atom_fea_len + 9, 128), 
            nn.Softplus(),
            nn.Linear(128, latent_dim * 2)
        )

    def build_graph(self, lattice, fracs, num_atoms_list, k_neighbors=12):
        edge_src, edge_dst, edge_dist = [], [], []
        start_idx = 0
        device = lattice.device
        
        for i, n in enumerate(num_atoms_list):
            f, lat = fracs[start_idx : start_idx + n], lattice[i]
            diff = f.unsqueeze(1) - f.unsqueeze(0)
            diff = diff - torch.round(diff)
            dist_matrix = torch.norm(torch.matmul(diff, lat), dim=-1)
            dist_matrix.fill_diagonal_(float('inf'))
            
            k = min(k_neighbors, n - 1)
            if k > 0:
                topk_dist, topk_idx = torch.topk(dist_matrix, k, dim=-1, largest=False)
                edge_src.append(topk_idx.flatten() + start_idx)
                edge_dst.append(torch.arange(n, device=device).unsqueeze(1).expand(-1, k).flatten() + start_idx)
                edge_dist.append(topk_dist.flatten())
                
            start_idx += n
        return torch.cat(edge_src), torch.cat(edge_dst), torch.cat(edge_dist).unsqueeze(-1)

    def forward(self, lattice, fracs, species, batch_indices, num_atoms_list):
        atom_fea = self.embedding(species)
        edge_src, edge_dst, edge_dist = self.build_graph(lattice, fracs, num_atoms_list)
        nbr_fea = self.distance_expansion(edge_dist)
        
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, edge_src, edge_dst)
            
        num_graphs = lattice.size(0)
        crys_fea = torch.zeros(num_graphs, atom_fea.size(1), device=lattice.device)
        crys_fea.index_add_(0, batch_indices, atom_fea)
        crys_fea = crys_fea / torch.bincount(batch_indices).view(-1, 1).float()
        
        crys_fea = torch.cat([crys_fea, lattice.view(num_graphs, 9)], dim=1)
        return torch.chunk(self.conv_to_fc(crys_fea), 2, dim=-1)