"""
Luo 3D Gaussian .cube -tiedostot formaldehydille (H2CO) ja bentseenille (Benzene / C6H6)
RHF/def2-SVP -tasolla PySCF:llä (HOMO-1, HOMO, LUMO, LUMO+1).
"""
import os
import sys
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
CUBES_DIR = os.path.join(SCRIPT_DIR, "cubes")
os.makedirs(CUBES_DIR, exist_ok=True)

def generate_h2co_cubes():
    print("-> Generoidaan H2CO kuutiot...")
    geom_h2co = """
    C  0.0000  0.0000  0.0000
    O  0.0000  1.2100  0.0000
    H -0.9400 -0.5800  0.0000
    H  0.9400 -0.5800  0.0000
    """
    mol = gto.M(atom=geom_h2co, basis="def2-SVP", verbose=0)
    mf = scf.RHF(mol).run()

    occ = mf.mo_occ
    nocc = int(np.sum(occ > 0))
    homo = nocc - 1

    targets = {
        "HOMO-1": homo - 1,
        "HOMO": homo,
        "LUMO": homo + 1,
        "LUMO+1": homo + 2
    }

    for key, idx in targets.items():
        fname = os.path.join(CUBES_DIR, f"H2CO_{key}.cube")
        cubegen.orbital(mol, fname, mf.mo_coeff[:, idx], nx=46, ny=46, nz=46, margin=3.2)
        print(f"   Tallennettu: H2CO_{key}.cube (MO {idx}, E = {mf.mo_energy[idx]*27.2114:.2f} eV)")

def generate_benzene_cubes():
    print("-> Generoidaan Benzene (C6H6) kuutiot...")
    r_cc = 1.397
    r_ch = 1.397 + 1.084
    geom_c6h6 = []
    for i in range(6):
        ang = i * np.pi / 3.0
        geom_c6h6.append(f"C {r_cc * np.cos(ang):.6f} {r_cc * np.sin(ang):.6f} 0.0000")
    for i in range(6):
        ang = i * np.pi / 3.0
        geom_c6h6.append(f"H {r_ch * np.cos(ang):.6f} {r_ch * np.sin(ang):.6f} 0.0000")

    geom_str = "; ".join(geom_c6h6)
    mol = gto.M(atom=geom_str, basis="def2-SVP", verbose=0)
    mf = scf.RHF(mol).run()

    occ = mf.mo_occ
    nocc = int(np.sum(occ > 0))
    homo = nocc - 1

    targets = {
        "HOMO-1": homo - 1,
        "HOMO": homo,
        "LUMO": homo + 1,
        "LUMO+1": homo + 2
    }

    for key, idx in targets.items():
        fname = os.path.join(CUBES_DIR, f"Benzene_{key}.cube")
        cubegen.orbital(mol, fname, mf.mo_coeff[:, idx], nx=46, ny=46, nz=46, margin=3.5)
        print(f"   Tallennettu: Benzene_{key}.cube (MO {idx}, E = {mf.mo_energy[idx]*27.2114:.2f} eV)")

if __name__ == "__main__":
    generate_h2co_cubes()
    generate_benzene_cubes()
    print("✅ Kaikki H2CO ja Benzene kuutiot luotu onnistuneesti!")
