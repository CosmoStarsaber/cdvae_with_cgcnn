"""
train.py
CO2RR 潜空间活性飘移框架 (Latent Activity Drift)
从纯净环境加载 CO2RR 团簇数据，利用梯度锚定高活性极值点生成新结构。
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

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

class CO2RRCatalystDataset(Dataset):
    def __init__(self, root_dir: str, id_prop_csv: str = "id_prop.csv"):
        self.root_dir = root_dir
        self.entries = []
        csv_path = os.path.join(root_dir, id_prop_csv)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 {csv_path}。请确保数据位于上一级的 data 文件夹中。")

        with open(csv_path, "r") as f:
            for line in f:
                if not line.strip(): continue
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
        struct = Structure.from_file(os.path.join(self.root_dir, f"{cid}.cif"))
        fracs = np.array([s.frac_coords for s in struct])
        fracs = (fracs - np.floor(fracs)).astype(np.float32)
        species = np.array([s.specie.Z for s in struct], dtype=np.int64)
        return {
            "lattice": torch.tensor(struct.lattice.matrix, dtype=torch.float32),
            "fracs": torch.tensor(fracs, dtype=torch.float32),
            "species": torch.tensor(species, dtype=torch.long),
            "props": torch.tensor(norm_props, dtype=torch.float32),
            "num_atoms": len(species)
        }

def collate_fn(batch):
    batch_lattice = torch.stack([b['lattice'] for b in batch])
    batch_props = torch.stack([b['props'] for b in batch])
    all_fracs = torch.cat([b['fracs'] for b in batch], dim=0)
    all_species = torch.cat([b['species'] for b in batch], dim=0)
    batch_indices = []
    for i, b in enumerate(batch): batch_indices.extend([i] * b['num_atoms'])
    return {
        "lattice": batch_lattice, "fracs": all_fracs, "species": all_species,
        "props": batch_props, "batch_indices": torch.tensor(batch_indices, dtype=torch.long),
        "num_atoms": [b['num_atoms'] for b in batch]
    }

@torch.no_grad()
def generate_co2rr_catalysts(model, optimal_props_norm, out_dir, n_samples=5, device="cpu", guidance_scale=8.0, temperature=0.05, epoch=None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    prefix = f"ep{epoch}_" if epoch is not None else "final_"
    print(f"\n⚡ [{prefix.strip('_')}] 启动 CO2RR 潜空间活性飘移 (Drift)...")
    
    z = torch.randn(n_samples, model.latent_dim, device=device, requires_grad=True)
    drift_target = torch.tensor([optimal_props_norm], dtype=torch.float32).expand(n_samples, -1).to(device)
    
    optimizer_z = torch.optim.Adam([z], lr=0.08)
    for step in range(150):
        with torch.enable_grad():
            drift_loss = F.mse_loss(model.property_predictor(z), drift_target)
            optimizer_z.zero_grad()
            drift_loss.backward()
            optimizer_z.step()
    
    z = z.detach()
    print(f"   => 飘移完成！已锚定高活性火山曲线顶点。")
    
    num_atoms_list = torch.argmax(model.length_predictor(z), dim=-1).clamp(min=1).tolist()
    lattice = model.lattice_predictor(z)
    batch_indices = torch.tensor([i for i, n in enumerate(num_atoms_list) for _ in range(n)], device=device)
    
    print(f"🌀 团簇结构朗之万去噪 (CFG={guidance_scale}, Temp={temperature})...")
    frac_coords, species_logits = model.decoder.sample(z[batch_indices], lattice, num_atoms_list, batch_indices, guidance_scale, temperature)
    
    species = torch.argmax(species_logits, dim=-1).cpu().numpy()
    fracs_np, lattice_np = frac_coords.cpu().numpy(), lattice.cpu().numpy()
    
    start_idx, valid_count = 0, 0
    for i, n in enumerate(num_atoms_list):
        f, s, l = fracs_np[start_idx : start_idx + n], species[start_idx : start_idx + n], lattice_np[i]
        start_idx += n
        
        valid_idx = [j for j, z_num in enumerate(s) if 0 < z_num <= 118]
        if not valid_idx: continue
            
        try:
            symbols = [Element.from_Z(int(s[j])).symbol for j in valid_idx]
            struct = Structure(Lattice(l), symbols, f[valid_idx].tolist())
            
            if len(struct) > 1:
                if any(len(neigh) > 0 for neigh in struct.get_all_neighbors(r=0.8)):
                    print(f"   🚫 废片拦截: 样本 {i} 原子碰撞。")
                    continue
            
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                sg_symbol = sga.get_space_group_symbol()
                if "P1" not in sg_symbol: struct = sga.get_refined_structure()
            except: pass 
            
            out_path = os.path.join(out_dir, f"{prefix}co2rr_{i}.cif")
            struct.to(filename=out_path)
            valid_count += 1
            print(f"   ✅ 生成成功: {out_path} (原子数: {len(struct)})")
        except: pass
    print(f"🎯 存活率: {valid_count}/{n_samples}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🌟 默认读取上一级的 data 目录
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints_co2rr")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sample_every", type=int, default=50)
    
    # 假设你的 csv 里是 ΔG_CO，默认向理论极值 -0.67 飘移
    parser.add_argument("--optimal_co2rr_props", type=float, nargs='+', default=[-0.67])
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 CO2RR 团簇生成平台！计算设备: {device}")
    
    dataset = CO2RRCatalystDataset(args.data)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = DiffusionCDVAE(latent_dim=128, cond_dim=dataset.cond_dim, timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    os.makedirs(args.save_dir, exist_ok=True)
    
    start_epoch, best_val_loss = 0, float('inf')
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch, best_val_loss = checkpoint['epoch'] + 1, checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ 断点恢复！Epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_metrics = {k: 0 for k in ["loss_total", "loss_diff", "loss_prop", "loss_species", "loss_rep"]}
        for batch in train_loader:
            loss, logs = model.compute_loss(batch, device) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in train_metrics: train_metrics[k] += logs.get(k, 0)
        for k in train_metrics: train_metrics[k] /= len(train_loader)

        model.eval()
        val_metrics = {k: 0 for k in ["loss_total", "loss_diff", "loss_prop", "loss_species", "loss_rep"]}
        with torch.no_grad():
            for batch in val_loader:
                loss, logs = model.compute_loss(batch, device)
                for k in val_metrics: val_metrics[k] += logs.get(k, 0)
        for k in val_metrics: val_metrics[k] /= len(val_loader)

        print(f"Ep [{epoch+1:03d}/{args.epochs}] | Tr Diff: {train_metrics['loss_diff']:.3f} | Val Diff: {val_metrics['loss_diff']:.3f} (Rep: {val_metrics['loss_rep']:.3f}) | Total: {val_metrics['loss_total']:.3f}")

        checkpoint_data = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_loss': best_val_loss}
        torch.save(checkpoint_data, os.path.join(args.save_dir, "latest_checkpoint.pt"))
        if val_metrics['loss_total'] < best_val_loss:
            best_val_loss = val_metrics['loss_total']
            torch.save(checkpoint_data, os.path.join(args.save_dir, "best_checkpoint.pt"))
            
        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            norm_targets = (np.array((args.optimal_co2rr_props + [0.0] * dataset.cond_dim)[:dataset.cond_dim], dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
            generate_co2rr_catalysts(model, norm_targets.tolist(), "generated_co2rr_cifs", n_samples=3, device=device, guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=epoch+1)

    print("\n" + "="*50 + "\n🔥 训练完毕，执行大批量高活性候选材料生成\n" + "="*50)
    if os.path.exists(os.path.join(args.save_dir, "best_checkpoint.pt")):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_checkpoint.pt"), map_location=device)['model_state_dict'])
        
    norm_targets = (np.array((args.optimal_co2rr_props + [0.0] * dataset.cond_dim)[:dataset.cond_dim], dtype=np.float32) - dataset.prop_mean) / dataset.prop_std
    generate_co2rr_catalysts(model, norm_targets.tolist(), "generated_co2rr_cifs", n_samples=20, device=device, guidance_scale=args.guidance_scale, temperature=args.temperature, epoch=None)