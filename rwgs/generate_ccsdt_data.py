"""
Calculates the quantum chemical potential energy surfaces for RWGS
using Coupled Cluster theory: CCSD(T) with def2-TZVP basis set.
Saves results to rwgs/ccsdt_raw_data.json
"""
import json
import os
import time
import numpy as np
from pyscf import gto, scf, cc

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "ccsdt_raw_data.json")

print("=" * 65)
print(" Generating CCSD(T) / def2-TZVP Quantum Chemical PES Dataset")
print("=" * 65)

# 1. Ground State Atoms (UHF + UCCSD(T))
print("\n[1/5] Calculating Atomic Ground States...")
atom_results = {}

# H atom: 1 electron (no correlation, UHF = exact within basis)
mol_h = gto.M(atom="H 0 0 0", basis="def2-tzvp", spin=1, verbose=0)
mf_h = scf.UHF(mol_h).run()
atom_results["H"] = float(mf_h.e_tot)
print(f"  Atom H (spin 1): {atom_results['H']:.8f} Eh")

# C atom: triplet (spin 2)
mol_c = gto.M(atom="C 0 0 0", basis="def2-tzvp", spin=2, verbose=0)
mf_c = scf.UHF(mol_c).run()
mycc_c = cc.UCCSD(mf_c).run()
et_c = mycc_c.ccsd_t()
atom_results["C"] = float(mycc_c.e_tot + et_c)
print(f"  Atom C (spin 2): {atom_results['C']:.8f} Eh (UCCSD={mycc_c.e_tot:.6f}, (T)={et_c:.6f})")

# O atom: triplet (spin 2)
mol_o = gto.M(atom="O 0 0 0", basis="def2-tzvp", spin=2, verbose=0)
mf_o = scf.UHF(mol_o).run()
mycc_o = cc.UCCSD(mf_o).run()
et_o = mycc_o.ccsd_t()
atom_results["O"] = float(mycc_o.e_tot + et_o)
print(f"  Atom O (spin 2): {atom_results['O']:.8f} Eh (UCCSD={mycc_o.e_tot:.6f}, (T)={et_o:.6f})")

# 2. H2 Diatomic Potential Energy Curve (0.40 to 4.00 A)
print("\n[2/5] Calculating H2 1D Curve (0.40 to 4.00 A)...")
r_h2_list = np.unique(np.round(np.concatenate([
    np.linspace(0.40, 0.65, 8),
    np.linspace(0.66, 0.85, 14),
    np.linspace(0.88, 1.30, 10),
    np.linspace(1.40, 2.20, 8),
    np.linspace(2.40, 4.00, 8)
]), 3)).tolist()

e_tot_h2 = []
for r in r_h2_list:
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="def2-tzvp", spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    e_tot_h2.append(float(mycc.e_tot))

print(f"  H2 done: {len(r_h2_list)} points. Min at r={r_h2_list[np.argmin(e_tot_h2)]} A (E={min(e_tot_h2):.6f} Eh)")

# 3. CO Potential Energy Curve (0.80 to 2.40 A)
print("\n[3/5] Calculating CO 1D Curve (0.80 to 2.40 A)...")
r_co_list = np.unique(np.round(np.concatenate([
    np.linspace(0.80, 1.05, 7),
    np.linspace(1.07, 1.25, 12),
    np.linspace(1.28, 1.70, 7),
    np.linspace(1.80, 2.40, 6)
]), 3)).tolist()

e_tot_co = []
for r in r_co_list:
    mol = gto.M(atom=f"C 0 0 0; O 0 0 {r}", basis="def2-tzvp", spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    et = mycc.ccsd_t()
    e_tot_co.append(float(mycc.e_tot + et))

print(f"  CO done: {len(r_co_list)} points. Min at r={r_co_list[np.argmin(e_tot_co)]} A (E={min(e_tot_co):.6f} Eh)")

# 4. CO2 Symmetric Stretch Curve (0.85 to 2.30 A)
print("\n[4/5] Calculating CO2 1D Symmetric Curve (0.85 to 2.30 A)...")
r_co2_list = np.unique(np.round(np.concatenate([
    np.linspace(0.85, 1.08, 6),
    np.linspace(1.10, 1.25, 11),
    np.linspace(1.28, 1.65, 5),
    np.linspace(1.75, 2.30, 4)
]), 3)).tolist()

e_tot_co2 = []
dm_prev = None
t_co2_start = time.time()
for idx, r in enumerate(r_co2_list):
    mol = gto.M(atom=f"C 0 0 0; O 0 0 {-r}; O 0 0 {r}", basis="def2-tzvp", spin=0, verbose=0)
    mf = scf.RHF(mol)
    if dm_prev is not None:
        mf.kernel(dm0=dm_prev)
    else:
        mf.kernel()
    dm_prev = mf.make_rdm1()
    mycc = cc.CCSD(mf).run()
    et = mycc.ccsd_t()
    e = float(mycc.e_tot + et)
    e_tot_co2.append(e)
    print(f"  CO2 point {idx+1}/{len(r_co2_list)}: r={r:.3f} A -> E={e:.6f} Eh ({time.time()-t_co2_start:.1f}s)")

print(f"  CO2 done: {len(r_co2_list)} points. Min at r={r_co2_list[np.argmin(e_tot_co2)]} A (E={min(e_tot_co2):.6f} Eh)")

# 5. H2O 2D Grid (r vs theta)
print("\n[5/5] Calculating H2O 2D Grid (r in [0.70, 1.50] A, theta in [75, 180] deg)...")
r_h2o_list = np.round(np.linspace(0.70, 1.50, 11), 3).tolist()
th_h2o_list = np.round(np.linspace(75.0, 180.0, 10), 1).tolist()

e_tot_grid_h2o = []
pt_count = 0
total_pts = len(th_h2o_list) * len(r_h2o_list)
t_h2o_start = time.time()

for th in th_h2o_list:
    row = []
    half = np.radians(th / 2.0)
    for r in r_h2o_list:
        x = r * np.sin(half)
        z = r * np.cos(half)
        mol = gto.M(atom=f"O 0 0 0; H 0 {x:.4f} {z:.4f}; H 0 {-x:.4f} {z:.4f}", basis="def2-tzvp", spin=0, verbose=0)
        mf = scf.RHF(mol).run()
        mycc = cc.CCSD(mf).run()
        et = mycc.ccsd_t()
        row.append(float(mycc.e_tot + et))
        pt_count += 1
        if pt_count % 25 == 0 or pt_count == total_pts:
            print(f"  H2O progress: {pt_count}/{total_pts} ({pt_count/total_pts*100:.1f}%) in {time.time()-t_h2o_start:.1f}s")
    e_tot_grid_h2o.append(row)

# Compile full CCSD(T) dataset
dataset = {
    "method": "CCSD(T)",
    "basis": "def2-TZVP",
    "description": "Gold-standard Coupled Cluster dataset calculated with CCSD(T) and def2-TZVP basis set.",
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

print("\n" + "=" * 65)
print(f" CCSD(T) dataset successfully saved to: {OUTPUT_PATH}")
print("=" * 65)
