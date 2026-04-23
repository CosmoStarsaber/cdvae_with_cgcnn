"""
verify_and_generate_labels.py

读取 data/ 下所有 CIF 文件，计算多维 CO2RR 结构/化学描述符，
输出带列名的 data/id_prop.csv，并同步写出 data/id_prop_dimensions.csv。
"""

import csv
import glob
import os
import warnings

import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

warnings.filterwarnings("ignore", category=UserWarning, module="ase")

ELECTRONEG = {
    "Cu": 1.90, "Ag": 1.93, "Au": 2.54, "Zn": 1.65, "Ni": 1.91,
    "Pd": 2.20, "Co": 1.88, "Fe": 1.83, "Sn": 1.96, "Bi": 2.02,
    "In": 1.78, "Pb": 2.33, "Ru": 2.20, "Ti": 1.54, "Mo": 2.16,
    "Rh": 2.28, "Pt": 2.28, "Ir": 2.20, "Cd": 1.69, "Cr": 1.66,
    "Mn": 1.55, "Mg": 1.31, "Al": 1.61,
}

CO2RR_METALS = {"Cu", "Ag", "Au", "Zn"}
ALLOY_METALS = {"Ni", "Pd", "Co", "Fe", "Sn", "Bi", "In", "Pb", "Ru", "Ti", "Mo"}
METAL_ELEMENTS = set(ELECTRONEG.keys())

BULK_CN = {
    "Cu": 12, "Ag": 12, "Au": 12, "Zn": 12, "Ni": 12,
    "Pd": 12, "Co": 12, "Fe": 8, "Sn": 6, "Bi": 3,
    "In": 4, "Pb": 12, "Ru": 12, "Ti": 12, "Mo": 8,
}

PROPERTY_COLUMNS = [
    ("total_score", "加权 CO2RR 活性代理总分，综合金属成分、电子结构/位点和尺寸，范围 0-1"),
    ("metal_score", "金属成分得分：活性金属占比越高越接近 1，范围 0-1"),
    ("d_band_score", "d-band 代理得分：越接近经验吸附窗口越高，范围 0-1"),
    ("undercoord_fraction", "低配位金属原子比例，表示潜在暴露活性位点密度，范围 0-1"),
    ("size_score", "团簇尺寸适宜性得分：越接近经验活性窗口越高，范围 0-1"),
    ("active_metal_fraction", "核心活性金属 Cu/Ag/Au/Zn 在金属原子中的占比，范围 0-1"),
    ("alloy_metal_fraction", "合金辅助活性金属在金属原子中的占比，范围 0-1"),
    ("metal_atom_fraction", "全结构中金属原子占比，反映团簇金属核相对有机配体的占比，范围 0-1"),
    ("mean_metal_en", "金属原子平均电负性（Pauling 标度），非归一化实数"),
]


def get_metal_info(atoms):
    syms = np.array(atoms.get_chemical_symbols())
    metal_mask = np.array([s in METAL_ELEMENTS for s in syms])
    return metal_mask, syms[metal_mask]


def _coordination_numbers(atoms, metal_mask):
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    cn = np.array([nl.get_neighbors(i)[0].shape[0] for i in range(len(atoms))])
    return cn[metal_mask]


def score_metal_composition(metal_syms):
    if len(metal_syms) == 0:
        return 0.0, 0.0, 0.0

    core_active = sum(1 for s in metal_syms if s in CO2RR_METALS)
    alloy_active = sum(1 for s in metal_syms if s in ALLOY_METALS)
    active_frac = core_active / len(metal_syms)
    alloy_frac = alloy_active / len(metal_syms)
    metal_score = min(1.0, active_frac * 0.8 + alloy_frac * 0.2)
    return metal_score, active_frac, alloy_frac


def score_d_band_and_sites(metal_syms, metal_cn):
    if len(metal_syms) == 0:
        return 0.0, 0.0

    scores = []
    n_under = 0
    for i, sym in enumerate(metal_syms):
        en = ELECTRONEG.get(sym, 2.0)
        bulk = BULK_CN.get(sym, 12)
        cn = metal_cn[i] if i < len(metal_cn) else bulk
        if cn < bulk * 0.8:
            n_under += 1
        d_band_est = -(en * 1.2) + (1 - cn / max(bulk, 1)) * 0.8
        score = max(0.0, 1 - abs(d_band_est - (-2.0)) / 1.5)
        scores.append(score)

    return float(np.mean(scores)), n_under / len(metal_syms)


def score_size(n_atoms):
    if n_atoms <= 0:
        return 0.0
    log_n = np.log10(n_atoms)
    return float(np.exp(-0.5 * ((log_n - 1.7) / 0.8) ** 2))


def mean_metal_electronegativity(metal_syms):
    if len(metal_syms) == 0:
        return 0.0
    values = [ELECTRONEG.get(sym, 2.0) for sym in metal_syms]
    return float(np.mean(values))


def compute_properties(fpath):
    basename = os.path.basename(fpath)
    atoms = read(fpath, format="cif")
    metal_mask, metal_syms = get_metal_info(atoms)
    if not metal_mask.any():
        return basename, None

    metal_cn = _coordination_numbers(atoms, metal_mask)
    metal_score, active_frac, alloy_frac = score_metal_composition(metal_syms)
    d_band_score, undercoord_fraction = score_d_band_and_sites(metal_syms, metal_cn)
    size_score = score_size(len(atoms))
    metal_atom_fraction = float(np.sum(metal_mask) / len(atoms))
    mean_en = mean_metal_electronegativity(metal_syms)
    site_score = d_band_score * 0.6 + undercoord_fraction * 0.4
    total_score = metal_score * 0.40 + site_score * 0.45 + size_score * 0.15

    props = {
        "total_score": round(float(total_score), 4),
        "metal_score": round(float(metal_score), 4),
        "d_band_score": round(float(d_band_score), 4),
        "undercoord_fraction": round(float(undercoord_fraction), 4),
        "size_score": round(float(size_score), 4),
        "active_metal_fraction": round(float(active_frac), 4),
        "alloy_metal_fraction": round(float(alloy_frac), 4),
        "metal_atom_fraction": round(float(metal_atom_fraction), 4),
        "mean_metal_en": round(float(mean_en), 4),
    }
    return basename, props


def verify_single(fpath):
    basename = os.path.basename(fpath)
    try:
        _, props = compute_properties(fpath)
        if props is None:
            return basename, 0.0
        return basename, props["total_score"]
    except Exception:
        return basename, 0.0


def write_dimension_file(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["property", "meaning"])
        for name, meaning in PROPERTY_COLUMNS:
            writer.writerow([name, meaning])


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        print(f"❌ 找不到 data 文件夹: {data_dir}")
        return

    csv_out = os.path.join(data_dir, "id_prop.csv")
    dim_out = os.path.join(data_dir, "id_prop_dimensions.csv")
    cif_files = glob.glob(os.path.join(data_dir, "*.cif"))
    print(f"开始对 {len(cif_files)} 个团簇进行多维 CO2RR 描述符评估...")

    results = []
    for i, fpath in enumerate(cif_files):
        try:
            basename, props = compute_properties(fpath)
            if props is not None and props["total_score"] > 0.01:
                results.append([basename] + [props[name] for name, _ in PROPERTY_COLUMNS])
        except Exception:
            pass
        if (i + 1) % 500 == 0 or i == len(cif_files) - 1:
            print(f"  [{i + 1}/{len(cif_files)}] 已处理")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [name for name, _ in PROPERTY_COLUMNS])
        writer.writerows(results)

    write_dimension_file(dim_out)

    total_scores = [row[1] for row in results]
    print("\n" + "=" * 40)
    print(f"筛选完毕！生成有效数据: {len(results)} 条")
    print(f"多维属性已保存为: {csv_out}")
    print(f"维度说明已保存为: {dim_out}")
    if total_scores:
        print(f"total_score 范围: {min(total_scores):.3f} - {max(total_scores):.3f}, 平均分: {np.mean(total_scores):.3f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
