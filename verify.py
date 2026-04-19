"""
CO2RR activity verification via descriptor-based screening.

Reads all CIF files from data/, computes 7 descriptors correlated with
CO2 electroreduction activity, scores each structure, and writes results
to data/verify_results.csv.

Descriptors (each scored 0-1):
  1. metal_score      – presence of known CO2RR-active metals
  2. d_band_proxy     – estimated d-band centre from coordination + electronegativity
  3. active_site_score – fraction of under-coordinated metal atoms
  4. size_score       – cluster size within known active range
  5. en_score         – average metal electronegativity in optimal window
  6. intermediate_score – presence of *CO, *COOH, *CHO, *OH fragments
  7. surface_ratio    – surface-to-total atom ratio

Total = weighted sum (0-100).
Label: likely_active (>=60), uncertain (30-59), unlikely (<30).
"""

import csv
import os
import sys
import warnings
from collections import Counter
from math import erf, sqrt

import numpy as np
from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList

warnings.filterwarnings("ignore", category=UserWarning, module="ase")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pauling electronegativity
ELECTRONEG = {
    "Cu": 1.90, "Ag": 1.93, "Au": 2.54, "Zn": 1.65, "Ni": 1.91,
    "Pd": 2.20, "Co": 1.88, "Fe": 1.83, "Sn": 1.96, "Bi": 2.02,
    "In": 1.78, "Pb": 2.33, "Ru": 2.20, "Ti": 1.54, "Mo": 2.16,
    "Rh": 2.28, "Pt": 2.28, "Ir": 2.20, "Cd": 1.69, "Cr": 1.66,
    "Mn": 1.55, "Mg": 1.31, "Al": 1.61, "Se": 2.55, "C": 2.55,
    "O": 3.44, "H": 2.20, "N": 3.04, "S": 2.58, "P": 2.19,
    "Cl": 3.16, "F": 3.98, "Br": 2.96, "I": 2.66, "B": 2.04,
    "Si": 1.90, "K": 0.82, "Na": 0.93, "Ca": 1.00, "Ba": 0.89,
    "Cs": 0.79, "Sr": 0.95, "La": 1.10, "Ce": 1.12, "V": 1.63,
    "W": 2.36, "Nb": 1.60, "Ta": 1.50, "Re": 1.90, "Os": 2.20,
    "Hf": 1.30, "Zr": 1.33, "Y": 1.22, "Sc": 1.36, "Ga": 1.81,
    "Ge": 2.01, "As": 2.18, "Sb": 2.05, "Te": 2.10, "Li": 0.98,
    "Be": 1.57,
}

# Metals known to be active for CO2RR
CO2RR_METALS = {
    "Cu", "Ag", "Au", "Zn", "Ni", "Pd", "Co", "Fe", "Sn", "Bi",
    "In", "Pb", "Ru", "Ti", "Mo", "Rh", "Pt", "Ir",
}

# Broad metal set for atom typing (active + inactive metals)
NONMETAL_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "As", "Te",
}
METAL_ELEMENTS = {el for el in ELECTRONEG if el not in NONMETAL_ELEMENTS}

# Bulk reference coordination numbers (nearest neighbours in metal)
BULK_CN = {
    "Cu": 12, "Ag": 12, "Au": 12, "Zn": 12, "Ni": 12,
    "Pd": 12, "Co": 12, "Fe": 8, "Sn": 6, "Bi": 3,
    "In": 4, "Pb": 12, "Ru": 12, "Ti": 12, "Mo": 8,
    "Rh": 12, "Pt": 12, "Ir": 12, "Mg": 12,
}

# Weights for final score (must sum to 1)
WEIGHTS = {
    "metal_score": 0.20,
    "d_band_proxy": 0.15,
    "active_site_score": 0.15,
    "size_score": 0.10,
    "en_score": 0.10,
    "intermediate_score": 0.20,
    "surface_ratio": 0.10,
}

# ---------------------------------------------------------------------------
# Descriptor functions
# ---------------------------------------------------------------------------


def get_metal_info(atoms):
    """Return (metal_mask, metal_symbols, nonmetal_symbols)."""
    syms = np.array(atoms.get_chemical_symbols())
    metal_mask = np.array([s in METAL_ELEMENTS for s in syms])
    return metal_mask, syms[metal_mask], syms[~metal_mask]


def score_metal(metal_syms):
    """Fraction of atoms that are CO2RR-active metals, with bonus for diversity."""
    if len(metal_syms) == 0:
        return 0.0
    n_active = sum(1 for s in metal_syms if s in CO2RR_METALS)
    frac = n_active / len(metal_syms)
    if frac == 0:
        return 0.0
    diversity = len(set(metal_syms) & CO2RR_METALS) / len(CO2RR_METALS)
    return min(1.0, 0.7 * frac + 0.3 * min(diversity * 5, 1.0))


def _coordination_numbers(atoms, metal_mask=None):
    """Compute coordination number for each atom using natural cutoffs."""
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    cn = np.array([nl.get_neighbors(i)[0].shape[0] for i in range(len(atoms))])
    if metal_mask is not None:
        return cn[metal_mask]
    return cn


def score_d_band(metal_syms, metal_cn):
    """Proxy for d-band centre: under-coordinated metals near e.g. Cu/Au
    have higher d-band centres -> stronger CO2 adsorption.
    Uses electronegativity + under-coordination as proxy."""
    if len(metal_syms) == 0:
        return 0.0
    scores = []
    for i, sym in enumerate(metal_syms):
        en = ELECTRONEG.get(sym, 2.0)
        bulk = BULK_CN.get(sym, 12)
        cn = metal_cn[i] if i < len(metal_cn) else bulk
        # Under-coordination raises d-band centre; optimal window around -1.5 to -2.5 eV
        # Higher EN pushes d-band down; under-coordination pushes it up
        d_band_est = -(en * 1.2) + (1 - cn / max(bulk, 1)) * 0.8
        # Score peaks around d_band = -1.5 to -2.5
        s = max(0, 1 - abs(d_band_est - (-2.0)) / 1.5)
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def score_active_sites(metal_syms, metal_cn):
    """Fraction of under-coordinated metal atoms (< 80% of bulk CN)."""
    if len(metal_syms) == 0:
        return 0.0
    n_under = 0
    for i, sym in enumerate(metal_syms):
        bulk = BULK_CN.get(sym, 12)
        cn = metal_cn[i] if i < len(metal_cn) else bulk
        if cn < bulk * 0.8:
            n_under += 1
    return n_under / len(metal_syms)


def score_size(n_atoms):
    """Cluster size suitability: peak activity typically 4-500 metal atoms.
    Gaussian-like score centred at ~50 atoms."""
    if n_atoms <= 0:
        return 0.0
    log_n = np.log10(max(n_atoms, 1))
    # Peak at log10(50) ~ 1.7, width covering 4-500
    s = np.exp(-0.5 * ((log_n - 1.7) / 0.7) ** 2)
    return float(s)


def score_electronegativity(metal_syms):
    """Average metal electronegativity in the CO2RR optimal window (~1.8-2.3)."""
    if len(metal_syms) == 0:
        return 0.0
    ens = [ELECTRONEG.get(s, 2.0) for s in metal_syms]
    avg_en = np.mean(ens)
    # Optimal window 1.8-2.3 (Cu=1.9, Au=2.54, Ag=1.93, Zn=1.65)
    if 1.8 <= avg_en <= 2.3:
        return 1.0
    dist = min(abs(avg_en - 1.8), abs(avg_en - 2.3))
    return max(0, 1 - dist / 0.5)


def _detect_intermediates(atoms):
    """Detect CO2RR intermediate fragments (*CO, *COOH, *CHO, *OH, *O, *H)
    by looking for C/O/H atoms bonded to metal atoms."""
    syms = atoms.get_chemical_symbols()
    n = len(atoms)
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    metal_idx = {i for i, s in enumerate(syms) if s in METAL_ELEMENTS}
    if not metal_idx:
        return {"CO": 0, "COOH": 0, "CHO": 0, "OH": 0, "COH": 0, "O": 0, "H_metal": 0}

    # Find non-metal atoms bonded to metals
    ads_C, ads_O, ads_H = set(), set(), set()
    for i in range(n):
        if i in metal_idx:
            continue
        neighbors, _ = nl.get_neighbors(i)
        if any(nb in metal_idx for nb in neighbors):
            if syms[i] == "C":
                ads_C.add(i)
            elif syms[i] == "O":
                ads_O.add(i)
            elif syms[i] == "H":
                ads_H.add(i)

    # Detect fragments via connectivity among adsorbates
    counts = {"CO": 0, "COOH": 0, "CHO": 0, "OH": 0, "COH": 0, "O": 0, "H_metal": 0}

    # CO: C bonded to O, both adsorbed on metal
    for c in ads_C:
        nb_c, _ = nl.get_neighbors(c)
        o_neighbors = [j for j in nb_c if j in ads_O]
        if o_neighbors:
            counts["CO"] += 1

    # CHO: C bonded to H and O
    for c in ads_C:
        nb_c, _ = nl.get_neighbors(c)
        has_h = any(j in ads_H for j in nb_c)
        has_o = any(j in ads_O for j in nb_c)
        if has_h and has_o:
            counts["CHO"] += 1

    # COOH: C bonded to two O (one OH)
    for c in ads_C:
        nb_c, _ = nl.get_neighbors(c)
        o_nb = [j for j in nb_c if j in ads_O]
        if len(o_nb) >= 2:
            counts["COOH"] += 1

    # OH: O bonded to H, on metal
    for o in ads_O:
        nb_o, _ = nl.get_neighbors(o)
        if any(j in ads_H for j in nb_o):
            counts["OH"] += 1

    # COH: C bonded to O and H
    for c in ads_C:
        nb_c, _ = nl.get_neighbors(c)
        has_o = any(j in ads_O for j in nb_c)
        has_h = any(j in ads_H for j in nb_c)
        if has_o and has_h:
            counts["COH"] += 1

    # Bare O on metal
    counts["O"] = len(ads_O)

    # H on metal
    counts["H_metal"] = len(ads_H)

    return counts


def score_intermediate(atoms):
    """Score based on detected CO2RR intermediates."""
    frags = _detect_intermediates(atoms)
    total = sum(frags.values())
    if total == 0:
        return 0.0
    # Weight intermediates by relevance
    weights = {"CO": 1.0, "COOH": 1.0, "CHO": 1.0, "OH": 0.8,
               "COH": 0.9, "O": 0.5, "H_metal": 0.3}
    weighted = sum(frags.get(k, 0) * w for k, w in weights.items())
    return min(1.0, weighted / 3.0)


def score_surface_ratio(metal_mask, metal_cn, metal_syms):
    """Fraction of surface (under-coordinated) metal atoms relative to all atoms."""
    if len(metal_mask) == 0 or metal_mask.sum() == 0:
        return 0.0
    n_total = len(metal_mask)
    # Surface metal atoms: CN < 80% of bulk reference
    n_surface = 0
    for i, sym in enumerate(metal_syms):
        bulk = BULK_CN.get(sym, 12)
        cn = metal_cn[i] if i < len(metal_cn) else bulk
        if cn < bulk * 0.8:
            n_surface += 1
    if n_total == 0:
        return 0.0
    ratio = n_surface / n_total
    return min(1.0, ratio * 2.0)


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def verify_single(fpath):
    """Verify a single CIF file. Returns dict of descriptor scores + total."""
    basename = os.path.basename(fpath)
    result = {
        "filename": basename,
        "metal_score": 0, "d_band_proxy": 0, "active_site_score": 0,
        "size_score": 0, "en_score": 0, "intermediate_score": 0,
        "surface_ratio": 0, "total_score": 0, "label": "error",
        "n_atoms": 0, "composition": "", "error": "",
    }
    try:
        atoms = read(fpath, format="cif")
        syms = atoms.get_chemical_symbols()
        comp = dict(Counter(syms))
        comp_str = " ".join(f"{k}{v}" for k, v in sorted(comp.items()))
        result["n_atoms"] = len(atoms)
        result["composition"] = comp_str

        metal_mask, metal_syms, nonmetal_syms = get_metal_info(atoms)
        metal_cn = _coordination_numbers(atoms, metal_mask) if metal_mask.any() else np.array([])

        result["metal_score"] = round(score_metal(metal_syms), 4)
        result["d_band_proxy"] = round(score_d_band(metal_syms, metal_cn), 4)
        result["active_site_score"] = round(score_active_sites(metal_syms, metal_cn), 4)
        result["size_score"] = round(score_size(int(metal_mask.sum())), 4)
        result["en_score"] = round(score_electronegativity(metal_syms), 4)
        result["intermediate_score"] = round(score_intermediate(atoms), 4)
        result["surface_ratio"] = round(score_surface_ratio(metal_mask, metal_cn, metal_syms), 4)

        total = sum(
            WEIGHTS[k] * result[k]
            for k in WEIGHTS
        )
        result["total_score"] = round(total * 100, 2)

        if total * 100 >= 60:
            result["label"] = "likely_active"
        elif total * 100 >= 30:
            result["label"] = "uncertain"
        else:
            result["label"] = "unlikely"

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    csv_in = os.path.join(data_dir, "_init_.csv")
    csv_out = os.path.join(data_dir, "verify_results.csv")

    # Collect CIF files from _init_.csv
    cif_files = []
    with open(csv_in, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("local_filename", "")
            if fname:
                fpath = os.path.join(data_dir, fname)
                if os.path.exists(fpath):
                    cif_files.append(fpath)

    print(f"Verifying {len(cif_files)} CIF files ...")

    fields = [
        "filename", "n_atoms", "composition",
        "metal_score", "d_band_proxy", "active_site_score",
        "size_score", "en_score", "intermediate_score", "surface_ratio",
        "total_score", "label", "error",
    ]

    results = []
    for i, fpath in enumerate(cif_files):
        r = verify_single(fpath)
        results.append(r)
        if (i + 1) % 100 == 0 or i == len(cif_files) - 1:
            print(f"  [{i+1}/{len(cif_files)}] processed")

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # Summary
    from collections import Counter as C2
    labels = C2(r["label"] for r in results)
    print(f"\nResults written to {csv_out}")
    print(f"Total: {len(results)}")
    for label in ["likely_active", "uncertain", "unlikely", "error"]:
        print(f"  {label}: {labels.get(label, 0)}")

    scores = [r["total_score"] for r in results if r["label"] != "error"]
    if scores:
        print(f"  Score range: {min(scores):.1f} - {max(scores):.1f}, "
              f"mean: {np.mean(scores):.1f}")


if __name__ == "__main__":
    main()
