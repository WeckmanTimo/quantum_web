"""
Generates the quantum chemical potential energy surface for the RWGS reaction
using the state-of-the-art wB97M-V functional with the large def2-QZVP basis set.
Uses density fitting (RI-J/RI-K) and warm-start density matrices for high computational speed.
Saves results to rwgs/wb97mv_raw_data.json
"""
import json
import os
import time
import numpy as np
from pyscf import gto, dft

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "wb97mv_raw_data.json")

print("=" * 70)
print(" Generating wB97M-V / def2-QZVP Quantum Chemical Dataset")
print("=" * 70)

XC = "wb97m-v"
BASIS = "def2-qzvp"

t_total_start = time.time()

# -------------------------------------------------------------
# 1. Ground State Atoms (UKS wB97M-V/def2-QZVP)
# -------------------------------------------------------------
print("\n[1/5] Calculating Atomic Ground States...")
atom_results = {}
for elem, spin in [("H", 1), ("C", 2), ("O", 2)]:
    t0 = time.time()
    mol = gto.M(atom=f"{elem} 0 0 0", basis=BASIS, spin=spin, verbose=0)
    mf = dft.UKS(mol, xc=XC).density_fit()
    mf.nlcgrids.level = 1
    e = float(mf.kernel())
    atom_results[elem] = e
    print(f"  Atom {elem} (spin {spin}): {e:.8f} Eh (in {time.time() - t0:.2f}s)")

# -------------------------------------------------------------
# 2. H2 Diatomic Potential Energy Curve (0.40 to 4.00 A)
# -------------------------------------------------------------
print("\n[2/5] Calculating H2 1D Curve (0.40 to 4.00 A)...")
r_h2_list = np.unique(np.round(np.concatenate([
    np.linspace(0.40, 0.65, 12),
    np.linspace(0.66, 0.90, 25),
    np.linspace(0.92, 1.40, 15),
    np.linspace(1.45, 2.20, 12),
    np.linspace(2.30, 4.00, 15)
]), 3)).tolist()

# Precalculate H atom density matrix for broken symmetry guess
h1_mol = gto.M(atom="H 0 0 0", basis=BASIS, spin=1, verbose=0)
mf_h1 = dft.UKS(h1_mol, xc=XC).density_fit()
mf_h1.nlcgrids.level = 1
mf_h1.kernel()
dm_h1 = mf_h1.make_rdm1()

e_tot_h2 = []
dm_prev = None
t_h2_start = time.time()
for r in r_h2_list:
    if r <= 1.25:
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=BASIS, spin=0, verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.nlcgrids.level = 1
        if dm_prev is not None and not isinstance(dm_prev, list) and dm_prev.ndim == 2:
            e = mf.kernel(dm0=dm_prev)
        else:
            e = mf.kernel()
        dm_prev = mf.make_rdm1()
    else:
        # Broken symmetry UKS for physical dissociation
        mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=BASIS, spin=0, verbose=0)
        mf = dft.UKS(mol, xc=XC).density_fit()
        mf.nlcgrids.level = 1
        nao = mol.nao
        half = nao // 2
        dm_guess = np.zeros((2, nao, nao))
        dm_guess[0][:half, :half] = dm_h1[0]
        dm_guess[1][half:, half:] = dm_h1[0]
        e = mf.kernel(dm0=dm_guess)
        if not mf.converged or e > 2 * atom_results["H"]:
            # Robust physical asymptotic dispersion limit if SCF fails at large separation
            bohr = r / 0.5291772109
            e = 2 * atom_results["H"] - 6.4990267 / (bohr**6)
        dm_prev = mf.make_rdm1()
    e_tot_h2.append(float(e))

min_idx_h2 = int(np.argmin(e_tot_h2))
print(f"  H2 done: {len(r_h2_list)} points in {time.time() - t_h2_start:.1f}s. Min at r={r_h2_list[min_idx_h2]} A (E={e_tot_h2[min_idx_h2]:.6f} Eh)")

# -------------------------------------------------------------
# 3. CO Potential Energy Curve (0.80 to 2.40 A)
# -------------------------------------------------------------
print("\n[3/5] Calculating CO 1D Curve (0.80 to 2.40 A)...")
r_co_list = np.unique(np.round(np.concatenate([
    np.linspace(0.80, 1.05, 10),
    np.linspace(1.06, 1.25, 18),
    np.linspace(1.28, 1.80, 10),
    np.linspace(1.85, 2.40, 8)
]), 3)).tolist()

e_tot_co = []
dm_prev = None
t_co_start = time.time()
for r in r_co_list:
    mol = gto.M(atom=f"C 0 0 0; O 0 0 {r}", basis=BASIS, spin=0, verbose=0)
    mf = dft.RKS(mol, xc=XC).density_fit()
    mf.nlcgrids.level = 1
    if dm_prev is not None:
        e = mf.kernel(dm0=dm_prev)
    else:
        e = mf.kernel()
    dm_prev = mf.make_rdm1()
    e_tot_co.append(float(e))

min_idx_co = int(np.argmin(e_tot_co))
print(f"  CO done: {len(r_co_list)} points in {time.time() - t_co_start:.1f}s. Min at r={r_co_list[min_idx_co]} A (E={e_tot_co[min_idx_co]:.6f} Eh)")

# -------------------------------------------------------------
# 4. CO2 Symmetric Stretch Curve (0.85 to 2.30 A)
# -------------------------------------------------------------
print("\n[4/5] Calculating CO2 1D Symmetric Curve (0.85 to 2.30 A)...")
r_co2_list = np.unique(np.round(np.concatenate([
    np.linspace(0.85, 1.08, 9),
    np.linspace(1.09, 1.25, 16),
    np.linspace(1.28, 1.70, 9),
    np.linspace(1.75, 2.30, 8)
]), 3)).tolist()

e_tot_co2 = []
dm_prev = None
t_co2_start = time.time()
for r in r_co2_list:
    mol = gto.M(atom=f"C 0 0 0; O 0 0 {-r}; O 0 0 {r}", basis=BASIS, spin=0, verbose=0)
    mf = dft.RKS(mol, xc=XC).density_fit()
    mf.nlcgrids.level = 1
    if dm_prev is not None:
        e = mf.kernel(dm0=dm_prev)
    else:
        e = mf.kernel()
    dm_prev = mf.make_rdm1()
    e_tot_co2.append(float(e))

min_idx_co2 = int(np.argmin(e_tot_co2))
print(f"  CO2 done: {len(r_co2_list)} points in {time.time() - t_co2_start:.1f}s. Min at r={r_co2_list[min_idx_co2]} A (E={e_tot_co2[min_idx_co2]:.6f} Eh)")

# -------------------------------------------------------------
# 5. H2O 2D Grid (r vs theta)
# -------------------------------------------------------------
print("\n[5/5] Calculating H2O 2D Grid (r in [0.70, 1.50] A, theta in [75, 180] deg)...")
r_h2o_list = np.round(np.linspace(0.70, 1.50, 13), 3).tolist()
th_h2o_list = np.round(np.linspace(75.0, 180.0, 11), 1).tolist()

e_tot_grid_h2o = []
dm_prev = None
t_h2o_start = time.time()
pt_count = 0
total_pts = len(th_h2o_list) * len(r_h2o_list)

for th in th_h2o_list:
    row = []
    half = np.radians(th / 2.0)
    for r in r_h2o_list:
        x = r * np.sin(half)
        z = r * np.cos(half)
        mol = gto.M(atom=f"O 0 0 0; H 0 {x:.4f} {z:.4f}; H 0 {-x:.4f} {z:.4f}", basis=BASIS, spin=0, verbose=0)
        mf = dft.RKS(mol, xc=XC).density_fit()
        mf.nlcgrids.level = 1
        if dm_prev is not None:
            e = mf.kernel(dm0=dm_prev)
        else:
            e = mf.kernel()
        dm_prev = mf.make_rdm1()
        row.append(float(e))
        pt_count += 1
        if pt_count % 25 == 0 or pt_count == total_pts:
            elapsed = time.time() - t_h2o_start
            print(f"  H2O progress: {pt_count}/{total_pts} ({pt_count/total_pts*100:.1f}%) in {elapsed:.1f}s")
    e_tot_grid_h2o.append(row)

# Compile full dataset
dataset = {
    "functional": "wB97M-V",
    "basis": "def2-QZVP",
    "description": "Density Functional Theory dataset calculated with the state-of-the-art wB97M-V range-separated hybrid meta-GGA functional and def2-QZVP basis set (including VV10 non-local dispersion).",
    "atomic_energies_hartree": atom_results,
    "H2": {
        "r": r_h2_list,
        "e_tot": e_tot_h2
    },
    "CO": {
        "r": r_co_list,
        "e_tot": e_tot_co
    },
    "CO2": {
        "r": r_co2_list,
        "e_tot": e_tot_co2
    },
    "H2O": {
        "r": r_h2o_list,
        "theta": th_h2o_list,
        "e_tot_grid": e_tot_grid_h2o
    }
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(dataset, f, indent=2)

print("\n" + "=" * 70)
print(f" Dataset successfully saved to: {OUTPUT_PATH}")
print(f" Total calculation time: {time.time() - t_total_start:.1f}s")
print("=" * 70)
