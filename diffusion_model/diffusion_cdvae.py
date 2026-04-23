"""
diffusion_cdvae.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from schedules import ContinuousScheduler
from dynamics import CrystalDynamics
from cgcnn_encoder import CGCNNEncoder

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, cond_dim))
    def forward(self, z): return self.net(z)

class LatticePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 9))
    def forward(self, z):
        L = self.net(z).view(-1, 3, 3)
        diag_mask = torch.eye(3, device=z.device).bool().unsqueeze(0)
        # 增加容差至 1e-2，极大程度避免极度扁平的奇异矩阵导致的 linalg.inv NaN
        return torch.where(diag_mask, F.softplus(L) + 1e-2, L)

class LengthPredictor(nn.Module):
    def __init__(self, latent_dim): 
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.SiLU(), nn.Linear(256, 1))
    def forward(self, z): return self.net(z).squeeze(-1)

class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=128, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.node_dim = node_dim # 🌟 动态记录 node_dim，杜绝硬编码
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim)
        self.species_predictor = nn.Sequential(nn.Linear(node_dim, 128), nn.SiLU(), nn.Linear(128, 119))

    def forward_training(self, z_nodes, cart_coords, batch_indices, num_atoms_list, species, cond_drop_prob=0.1):
        device = cart_coords.device
        batch_size = len(num_atoms_list) # 🌟 安全推断 batch_size
        
        if cond_drop_prob > 0 and torch.rand(1).item() < cond_drop_prob:
            z_nodes = torch.zeros_like(z_nodes)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise_cart = torch.randn_like(cart_coords)
        cart_t = self.scheduler.q_sample(cart_coords, t[batch_indices], noise=noise_cart)
        
        pred_noise_cart, h_final = self.dynamics(z_nodes, t, cart_t, batch_indices)
        loss_diff = F.mse_loss(pred_noise_cart, noise_cart)
        
        # O(N^2) 排斥力计算。为防巨型团簇 OOM，利用 num_atoms_list 进行分块保护
        t_nodes = t[batch_indices]
        sqrt_alphas_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_nodes, cart_t.shape)
        sqrt_1m_alphas_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, cart_t.shape)
        pred_x0 = (cart_t - sqrt_1m_alphas_t * pred_noise_cart) / sqrt_alphas_t
        
        l_repulsion = 0.0
        start_idx = 0
        for n in num_atoms_list:
            if n > 1:
                # 只有数百原子的大块才会被切片，保证内存安全，同时兼容 O-H 键 (阈值降至 0.6)
                f_chunk = pred_x0[start_idx : start_idx + n]
                diff = f_chunk.unsqueeze(1) - f_chunk.unsqueeze(0)
                dist_sq = torch.sum(diff ** 2, dim=-1)
                dist_sq.fill_diagonal_(float('inf'))
                dist = torch.sqrt(dist_sq + 1e-8)
                l_repulsion += (torch.relu(0.6 - dist) ** 2).sum() / n 
            start_idx += n
            
        l_repulsion = l_repulsion / batch_size

        # 🌟 修复 fallback 维度错误
        if h_final is None or h_final.size(0) == 0: 
            h_final = torch.zeros(z_nodes.size(0), self.node_dim, device=device)
            
        loss_species = F.cross_entropy(self.species_predictor(h_final), species)
        return loss_diff, loss_species, l_repulsion

    @torch.no_grad()
    def sample(self, z_nodes, num_atoms_list, batch_indices, guidance_scale=3.0, temperature=0.5):
        device = z_nodes.device
        batch_size = len(num_atoms_list)
        cart_t = torch.randn(sum(num_atoms_list), 3, device=device)
        z_nodes_uncond = torch.zeros_like(z_nodes)
        h_final = torch.zeros(sum(num_atoms_list), self.node_dim, device=device) # 使用 self.node_dim
        
        for time_step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            
            if guidance_scale > 1.0:
                pred_cond, h_cond = self.dynamics(z_nodes, t, cart_t, batch_indices)
                pred_uncond, _ = self.dynamics(z_nodes_uncond, t, cart_t, batch_indices)
                pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred_noise, h_cond = self.dynamics(z_nodes, t, cart_t, batch_indices)

            if time_step == 0 and h_cond is not None: h_final = h_cond
            
            t_nodes = t[batch_indices]
            alphas_t = self.scheduler._extract(1.0 - self.scheduler.betas, t_nodes, cart_t.shape)
            sqrt_1m_alphas_cumprod = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, cart_t.shape)
            
            cart_t_prev_mean = (1.0 / torch.sqrt(alphas_t)) * (cart_t - (1.0 - alphas_t) / sqrt_1m_alphas_cumprod * pred_noise)
            if time_step > 0:
                post_var_t = self.scheduler._extract(self.scheduler.posterior_variance, t_nodes, cart_t.shape)
                cart_t = cart_t_prev_mean + torch.sqrt(post_var_t) * (torch.randn_like(cart_t) * temperature)
            else:
                cart_t = cart_t_prev_mean
            
        return cart_t, self.species_predictor(h_final)

class DiffusionCDVAE(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CGCNNEncoder(latent_dim=latent_dim) 
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim)
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def compute_loss(self, batch, device, epoch=0):
        lattice, fracs, species, props = batch['lattice'].to(device), batch['fracs'].to(device), batch['species'].to(device), batch['props'].to(device)
        batch_indices, num_atoms_list = batch['batch_indices'].to(device), batch['num_atoms']

        mu, logvar = self.encoder(lattice, fracs, species, batch_indices)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        
        kl_weight = min(1.0, epoch / 50.0) * 0.05
        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        loss_prop = F.mse_loss(self.property_predictor(z), props)
        loss_lattice = F.mse_loss(self.lattice_predictor(z), lattice)
        target_lengths = torch.tensor(num_atoms_list, device=device, dtype=torch.float32)
        
        # 🌟 修复 MSE 掩盖扩散问题，权重从 0.1 降至 0.001
        loss_length = F.mse_loss(self.length_predictor(z), target_lengths)

        cart_coords = torch.bmm(fracs.unsqueeze(1), lattice[batch_indices]).squeeze(1)
        
        loss_diff_pure, loss_species, l_rep = self.decoder.forward_training(
            z[batch_indices], cart_coords, batch_indices, num_atoms_list, species, cond_drop_prob=0.1
        )

        total_loss = loss_diff_pure + 5.0 * l_rep + 0.5 * loss_prop + 0.1 * loss_lattice + 0.001 * loss_length + 0.5 * loss_species + kl_weight * loss_kl

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_diff": float(loss_diff_pure.detach().item()),
            "loss_prop": float(loss_prop.detach().item()),
            "loss_rep": float(l_rep.detach().item()) if torch.is_tensor(l_rep) else float(l_rep),
            "loss_kl": float(loss_kl.detach().item()),
        }
        return total_loss, loss_dict