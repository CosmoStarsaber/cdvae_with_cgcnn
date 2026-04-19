"""
diffusion_cdvae.py
主网络架构：包含隐式排斥先验防坍缩设计与局部感知的化学元素分配
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
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, cond_dim)
        )
    def forward(self, z): return self.net(z)

class LatticePredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 9)
        )
    def forward(self, z): return self.net(z).view(-1, 3, 3)

class LengthPredictor(nn.Module):
    def __init__(self, latent_dim, max_atoms=50): # 放宽团簇的原子数上限
        super().__init__()
        self.max_atoms = max_atoms
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.SiLU(),
            nn.Linear(128, max_atoms + 1) 
        )
    def forward(self, z): return self.net(z)

class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=128, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim)
        self.species_predictor = nn.Sequential(nn.Linear(node_dim, 128), nn.SiLU(), nn.Linear(128, 100))

    def forward_training(self, z_nodes, frac_coords, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1):
        device = frac_coords.device
        batch_size = lattice.size(0)
        
        if cond_drop_prob > 0 and torch.rand(1).item() < cond_drop_prob:
            z_nodes = torch.zeros_like(z_nodes)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(frac_coords)
        t_nodes = t[batch_indices] 
        
        x_t = self.scheduler.q_sample(frac_coords, t_nodes, noise=noise)
        x_t = x_t - torch.floor(x_t)
        
        pred_noise, h_final = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
        loss_diffusion = F.mse_loss(pred_noise, noise)
        
        # --- 隐含 x_0 排斥惩罚，严防团簇引力坍缩 ---
        sqrt_alphas_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_nodes, x_t.shape)
        sqrt_one_minus_alphas_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
        pred_x0 = (x_t - sqrt_one_minus_alphas_t * pred_noise) / sqrt_alphas_t
        pred_x0 = pred_x0 - torch.floor(pred_x0) 
        
        l_repulsion = 0.0
        start_idx = 0
        for i, n in enumerate(num_atoms_list):
            if n > 1:
                f, lat = pred_x0[start_idx : start_idx + n], lattice[i]
                diff = f.unsqueeze(1) - f.unsqueeze(0)
                diff = diff - torch.round(diff)
                dist_sq = torch.sum(torch.matmul(diff, lat) ** 2, dim=-1)
                dist_sq.fill_diagonal_(float('inf'))
                dist = torch.sqrt(dist_sq + 1e-8)
                l_repulsion += (torch.relu(0.8 - dist) ** 2).sum() / n
            start_idx += n
            
        l_repulsion = l_repulsion / batch_size
        loss_diffusion = loss_diffusion + 5.0 * l_repulsion

        loss_species = F.cross_entropy(self.species_predictor(h_final), species)
        return loss_diffusion, loss_species, l_repulsion

    @torch.no_grad()
    def sample(self, z_nodes, lattice, num_atoms_list, batch_indices, guidance_scale=8.0, temperature=0.05):
        device = lattice.device
        batch_size = lattice.size(0)
        x_t = torch.randn(sum(num_atoms_list), 3, device=device)
        x_t = x_t - torch.floor(x_t) 
        z_nodes_uncond = torch.zeros_like(z_nodes)
        h_final = None 
        
        for time_step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)
            
            if guidance_scale > 1.0:
                pred_noise_cond, h_cond = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)
                pred_noise_uncond, _ = self.dynamics(z_nodes_uncond, t, x_t, lattice, num_atoms_list, batch_indices)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
                h_current = h_cond
            else:
                pred_noise, h_current = self.dynamics(z_nodes, t, x_t, lattice, num_atoms_list, batch_indices)

            if time_step == 0: h_final = h_current
            
            t_nodes = t[batch_indices]
            alphas_t = self.scheduler._extract(1.0 - self.scheduler.betas, t_nodes, x_t.shape)
            sqrt_1m_alphas_cumprod = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, x_t.shape)
            
            x_t_prev_mean = (1.0 / torch.sqrt(alphas_t)) * (x_t - (1.0 - alphas_t) / sqrt_1m_alphas_cumprod * pred_noise)
            
            if time_step > 0:
                post_var_t = self.scheduler._extract(self.scheduler.posterior_variance, t_nodes, x_t.shape)
                x_t = x_t_prev_mean + torch.sqrt(post_var_t) * (torch.randn_like(x_t) * temperature)
            else:
                x_t = x_t_prev_mean
            x_t = x_t - torch.floor(x_t)
            
        return x_t, self.species_predictor(h_final)

class DiffusionCDVAE(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, timesteps=1000, max_atoms=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CGCNNEncoder(latent_dim=latent_dim) 
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.lattice_predictor = LatticePredictor(latent_dim)
        self.length_predictor = LengthPredictor(latent_dim, max_atoms)
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def compute_loss(self, batch, device):
        lattice, fracs, species, props = batch['lattice'].to(device), batch['fracs'].to(device), batch['species'].to(device), batch['props'].to(device)
        batch_indices, num_atoms_list = batch['batch_indices'].to(device), batch['num_atoms']

        mu, logvar = self.encoder(lattice, fracs, species, batch_indices, num_atoms_list)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        loss_prop = F.mse_loss(self.property_predictor(z), props)
        loss_lattice = F.mse_loss(self.lattice_predictor(z), lattice)
        target_lengths = torch.tensor(num_atoms_list, device=device).clamp(max=self.length_predictor.max_atoms)
        loss_length = F.cross_entropy(self.length_predictor(z), target_lengths)

        loss_diff, loss_species, l_rep = self.decoder.forward_training(
            z[batch_indices], fracs, lattice, num_atoms_list, batch_indices, species, cond_drop_prob=0.1
        )

        total_loss = loss_diff + 0.5 * loss_prop + 0.1 * loss_lattice + 0.1 * loss_length + 0.5 * loss_species + 0.01 * loss_kl
        loss_dict = {"loss_total": total_loss, "loss_diff": loss_diff.item(), "loss_prop": loss_prop.item(), 
                     "loss_species": loss_species.item(), "loss_kl": loss_kl.item(), "loss_rep": l_rep.item() if isinstance(l_rep, torch.Tensor) else l_rep}
        return total_loss, loss_dict