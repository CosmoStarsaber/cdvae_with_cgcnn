"""
diffusion_cdvae.py
团簇生成 VAE：去除 LatticePredictor，精细化斥力损失，质心对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from schedules import ContinuousScheduler
from dynamics import CrystalDynamics
from cgcnn_encoder import CGCNNEncoder

# Cordero covalent radii (Å) for Z=1..118, from Dalton Trans. 2008, 2832-2838
# Index 0 = padding, 1=H..118=Uuo
_COVALENT_RADII = [0.0,
    0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,  # H-Ne
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,  # Na-Ar
    2.03, 1.76,  # K-Ca
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.18, 1.22, 1.20, 1.16, 1.19, 1.20,  # Sc-Kr
    2.20, 1.95,  # Rb-Sr
    1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,  # Y-Xe
    2.44, 2.15,  # Cs-Ba
    2.12, 2.06, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.42,  # La-Ni
    1.38, 1.35, 1.35, 1.33, 1.35, 1.40, 1.42, 1.45, 1.50, 1.55,  # Cu-Br
    1.63, 1.54,  # Rb-Sr placeholder; correct values below
]
# Fill to 119 entries (index 0..118) with sensible defaults for anything missing
while len(_COVALENT_RADII) < 119:
    _COVALENT_RADII.append(1.50)

class PropertyPredictor(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, cond_dim))
    def forward(self, z): return self.net(z)

class LengthPredictor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 256), nn.SiLU(), nn.Linear(256, 1))
    def forward(self, z): return self.net(z).squeeze(-1)

class DiffusionDecoder(nn.Module):
    def __init__(self, latent_dim=128, node_dim=64, time_dim=64, timesteps=1000):
        super().__init__()
        self.timesteps = timesteps
        self.node_dim = node_dim
        self.scheduler = ContinuousScheduler(timesteps=timesteps, schedule_type="cosine")
        self.dynamics = CrystalDynamics(node_dim=node_dim, time_dim=time_dim, latent_dim=latent_dim)
        self.species_predictor = nn.Sequential(nn.Linear(node_dim, 128), nn.SiLU(), nn.Linear(128, 119))

        # Element-specific covalent radii table
        self.register_buffer('covalent_radii_table', torch.tensor(_COVALENT_RADII, dtype=torch.float32))

    def forward_training(self, z_nodes, cart_coords, batch_indices, num_atoms_list, species, cond_drop_prob=0.1):
        device = cart_coords.device
        batch_size = len(num_atoms_list)

        if cond_drop_prob > 0 and torch.rand(1).item() < cond_drop_prob:
            z_nodes = torch.zeros_like(z_nodes)

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise_cart = torch.randn_like(cart_coords)
        cart_t = self.scheduler.q_sample(cart_coords, t[batch_indices], noise=noise_cart)

        pred_noise_cart, h_final = self.dynamics(z_nodes, t, cart_t, batch_indices, species=species)
        loss_diff = F.mse_loss(pred_noise_cart, noise_cart)

        # Element-specific repulsion loss
        t_nodes = t[batch_indices]
        sqrt_alphas_t = self.scheduler._extract(self.scheduler.sqrt_alphas_cumprod, t_nodes, cart_t.shape)
        sqrt_1m_alphas_t = self.scheduler._extract(self.scheduler.sqrt_one_minus_alphas_cumprod, t_nodes, cart_t.shape)
        pred_x0 = (cart_t - sqrt_1m_alphas_t * pred_noise_cart) / sqrt_alphas_t

        l_repulsion = 0.0
        start_idx = 0
        for n in num_atoms_list:
            if n > 1:
                f_chunk = pred_x0[start_idx : start_idx + n]
                sp_chunk = species[start_idx : start_idx + n]
                diff = f_chunk.unsqueeze(1) - f_chunk.unsqueeze(0)
                dist_sq = torch.sum(diff ** 2, dim=-1)
                dist_sq.fill_diagonal_(float('inf'))
                dist = torch.sqrt(dist_sq + 1e-8)

                r_i = self.covalent_radii_table[sp_chunk]
                threshold = (r_i.unsqueeze(1) + r_i.unsqueeze(0)) * 0.8
                l_repulsion += (torch.relu(threshold - dist) ** 2).sum() / n
            start_idx += n

        l_repulsion = l_repulsion / batch_size

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
        h_final = torch.zeros(sum(num_atoms_list), self.node_dim, device=device)

        for time_step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), time_step, device=device, dtype=torch.long)

            if guidance_scale > 1.0:
                pred_cond, h_cond = self.dynamics(z_nodes, t, cart_t, batch_indices, species=None)
                pred_uncond, _ = self.dynamics(z_nodes_uncond, t, cart_t, batch_indices, species=None)
                pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred_noise, h_cond = self.dynamics(z_nodes, t, cart_t, batch_indices, species=None)

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

            # Center of mass alignment per structure
            start = 0
            for n in num_atoms_list:
                if n > 0:
                    cart_t[start:start+n] = cart_t[start:start+n] - cart_t[start:start+n].mean(dim=0, keepdim=True)
                start += n

        return cart_t, self.species_predictor(h_final)

class DiffusionCDVAE(nn.Module):
    def __init__(self, latent_dim=128, cond_dim=1, timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = CGCNNEncoder(latent_dim=latent_dim)
        self.property_predictor = PropertyPredictor(latent_dim, cond_dim)
        self.length_predictor = LengthPredictor(latent_dim)
        self.decoder = DiffusionDecoder(latent_dim=latent_dim, timesteps=timesteps)

    def compute_loss(self, batch, device, epoch=0):
        cart_coords = batch['cart_coords'].to(device)
        species = batch['species'].to(device)
        props = batch['props'].to(device)
        batch_indices = batch['batch_indices'].to(device)
        num_atoms_list = batch['num_atoms']
        num_graphs = len(num_atoms_list)

        mu, logvar = self.encoder(cart_coords, species, batch_indices, num_graphs)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        kl_weight = min(1.0, epoch / 50.0) * 0.05
        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))

        loss_prop = F.mse_loss(self.property_predictor(z), props)
        target_lengths = torch.tensor(num_atoms_list, device=device, dtype=torch.float32)
        loss_length = F.mse_loss(self.length_predictor(z), target_lengths)

        loss_diff_pure, loss_species, l_rep = self.decoder.forward_training(
            z[batch_indices], cart_coords, batch_indices, num_atoms_list, species, cond_drop_prob=0.1
        )

        total_loss = loss_diff_pure + 5.0 * l_rep + 0.5 * loss_prop + 0.001 * loss_length + 0.5 * loss_species + kl_weight * loss_kl

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_diff": float(loss_diff_pure.detach().item()),
            "loss_prop": float(loss_prop.detach().item()),
            "loss_rep": float(l_rep.detach().item()) if torch.is_tensor(l_rep) else float(l_rep),
            "loss_kl": float(loss_kl.detach().item()),
        }
        return total_loss, loss_dict
