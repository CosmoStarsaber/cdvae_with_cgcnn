"""
train.py
集成内存缓存、安全碰撞检测、回归优化与 OOM 超大团簇过滤护盾的 CO2RR 生产级流水线
"""
import os
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from diffusion_cdvae import DiffusionCDVAE

warnings.filterwarnings("ignore")

class CO2RRCatalystDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        self.cache = {} 
        csv_path = os.path.join(root_dir, id_prop_csv)
        if not os.path.exists(csv_path): raise FileNotFoundError(f"找不到 {csv_path}")

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or not line[0].isalnum() or "id" in line.lower() or "filename" in line.lower(): continue
                parts = [p.strip() for p in line.split(",")]
                self.entries.append((parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)))
        
        all_props = np.array([e[1] for e in self.entries])
        self.cond_dim = all_props.shape[1] 
        self.prop_mean = all_props.mean(axis=0) 
        self.prop_std = all_props.std(axis=0) + 1e-6 

    def __len__(self): return len(self.entries)

    def __getitem__(self, idx):
        cid, props = self.entries[idx]
        norm_props = (props - self.prop_mean) / self.prop_std 
        
        if cid not in self.cache:
            struct = Structure.from_file(os.path.join(self.root_dir, f"{cid}.cif"))
            lat = struct.lattice.matrix.astype(np.float32)
            fracs = np.array([s.frac_coords for s in struct]).astype(np.float32)
            species = np.array([s.specie.Z for s in struct], dtype=np.int64)
            self.cache[cid] = (lat, fracs, species)
        else:
            lat, fracs, species = self.cache[cid]
            
        return {"lattice": torch.tensor(lat), "fracs": torch.tensor(fracs), "species": torch.tensor(species), "props": torch.tensor(norm_props), "num_atoms": len(species)}

def collate_fn(batch):
    # 🌟 终极护盾：过滤掉原子数超过 500 的巨型团簇，彻底杜绝张量建图 OOM
    batch = [b for b in batch if b['num_atoms'] <= 500]
    if not batch: return None # 如果这个 batch 全是巨型怪物，返回 None

    batch_indices = []
    for i, b in enumerate(batch): batch_indices.extend([i] * b['num_atoms'])
    return {
        "lattice": torch.stack([b['lattice'] for b in batch]), "fracs": torch.cat([b['fracs'] for b in batch], dim=0), 
        "species": torch.cat([b['species'] for b in batch], dim=0), "props": torch.stack([b['props'] for b in batch]), 
        "batch_indices": torch.tensor(batch_indices, dtype=torch.long), "num_atoms": [b['num_atoms'] for b in batch]
    }

@torch.no_grad()
def generate_co2rr_catalysts(model, optimal_props_norm, out_dir, n_samples=5, device="cpu", guidance_scale=3.0, temperature=0.5, epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    prefix = f"ep{epoch}_" if epoch is not None else "final_"
    print(f"\n⚡ [{prefix.strip('_')}] 启动 CO2RR 潜空间活性飘移...")
    
    z = torch.randn(n_samples, model.latent_dim, device=device, requires_grad=True)
    drift_target = torch.tensor([optimal_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    
    optimizer_z = torch.optim.Adam([z], lr=0.01)
    for step in range(200):
        with torch.enable_grad():
            drift_loss = F.mse_loss(model.property_predictor(z), drift_target)
            if drift_loss.item() < 1e-4: break
            optimizer_z.zero_grad()
            drift_loss.backward()
            torch.nn.utils.clip_grad_norm_([z], 1.0)
            optimizer_z.step()
    z = z.detach()
    
    num_atoms_list = model.length_predictor(z).round().clamp(min=1, max=500).long().tolist()
    lattice = model.lattice_predictor(z)
    batch_indices = torch.tensor([i for i, n in enumerate(num_atoms_list) for _ in range(n)], device=device)
    
    cart_coords, species_logits = model.decoder.sample(z[batch_indices], num_atoms_list, batch_indices, guidance_scale, temperature)
    
    species_logits[:, 0] = -float('inf') 
    species_logits[:, 84:] = -float('inf')
    species = torch.argmax(species_logits, dim=-1).cpu().numpy()
    
    lat_nodes = lattice[batch_indices]
    inv_lat = torch.linalg.pinv(lat_nodes)
    fracs_np = torch.bmm(cart_coords.unsqueeze(1), inv_lat).squeeze(1).cpu().numpy()
    lattice_np = lattice.cpu().numpy()
    
    start_idx, valid_count = 0, 0
    for i, n in enumerate(num_atoms_list):
        f, s, l = fracs_np[start_idx : start_idx + n], species[start_idx : start_idx + n], lattice_np[i]
        start_idx += n
        try:
            symbols = [Element.from_Z(int(s[j])).symbol for j in range(len(s))]
            struct = Structure(Lattice(l), symbols, f.tolist())
            if len(struct) > 1 and any(len(neigh) > 0 for neigh in struct.get_all_neighbors(r=0.5)): continue
            
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                if "P1" not in sga.get_space_group_symbol(): 
                    struct = sga.get_refined_structure()
            except: pass 
            
            struct.to(filename=os.path.join(out_dir, f"{prefix}co2rr_{i}.cif"))
            valid_count += 1
        except: pass
    print(f"🎯 存活率: {valid_count}/{n_samples}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints_co2rr")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sample_every", type=int, default=50)
    
    parser.add_argument("--optimal_co2rr_props", type=float, nargs='+', default=[0.8, 0.8, 0.8]) 
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 CO2RR 团簇生成平台！计算设备: {device}")
    
    dataset = CO2RRCatalystDataset(args.data)
    
    target_props = (args.optimal_co2rr_props + [0.5]*dataset.cond_dim)[:dataset.cond_dim]
    norm_targets = (np.array(target_props, dtype=np.float32) - dataset.prop_mean) / dataset.prop_std

    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = DiffusionCDVAE(latent_dim=128, cond_dim=dataset.cond_dim, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch, best_val_loss = 0, float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch, best_val_loss = checkpoint['epoch'] + 1, checkpoint.get('best_val_loss', float('inf'))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {k: 0 for k in ["loss_total", "loss_diff", "loss_prop", "loss_kl", "loss_rep"]}
        valid_batches = 0 # 记录有效批次
        
        for batch in train_loader:
            if batch is None: continue # 🌟 核心防线：跳过包含全量巨型团簇的空 batch
            
            loss, logs = model.compute_loss(batch, device, epoch) 
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            valid_batches += 1
            for k in train_metrics: train_metrics[k] += logs.get(k, 0)
            
        if valid_batches > 0:
            for k in train_metrics: train_metrics[k] /= valid_batches
            
        scheduler.step()

        model.eval()
        val_metrics = {k: 0 for k in ["loss_total", "loss_diff", "loss_prop", "loss_kl", "loss_rep"]}
        valid_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue # 🌟 核心防线
                
                loss, logs = model.compute_loss(batch, device, epoch)
                valid_val_batches += 1
                for k in val_metrics: val_metrics[k] += logs.get(k, 0)
                
        if valid_val_batches > 0:
            for k in val_metrics: val_metrics[k] /= valid_val_batches

        print(f"Ep [{epoch+1:03d}/{args.epochs}] LR: {scheduler.get_last_lr()[0]:.1e} | Tr Diff: {train_metrics['loss_diff']:.3f} | Val Diff: {val_metrics['loss_diff']:.3f} (Rep: {val_metrics['loss_rep']:.3f}, KL: {val_metrics['loss_kl']:.3f})")

        checkpoint_data = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(checkpoint_data, os.path.join(args.save_dir, "latest_checkpoint.pt"))
        if val_metrics['loss_total'] < best_val_loss:
            best_val_loss = val_metrics['loss_total']
            torch.save(checkpoint_data, os.path.join(args.save_dir, "best_checkpoint.pt"))
            
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            generate_co2rr_catalysts(model, norm_targets.tolist(), "generated_co2rr_cifs", n_samples=3, device=device, guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=epoch+1)

    print("\n" + "="*50 + "\n🔥 训练完毕，执行大批量生成\n" + "="*50)
    if os.path.exists(os.path.join(args.save_dir, "best_checkpoint.pt")):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_checkpoint.pt"), map_location=device, weights_only=False)['model_state_dict'])
    generate_co2rr_catalysts(model, norm_targets.tolist(), "generated_co2rr_cifs", n_samples=20, device=device, guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=None)