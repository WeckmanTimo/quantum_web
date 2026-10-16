"""
Generoi 3D-kuutiot (Gaussian .cube) atomiorbitaaleille ja LCAO-demonstraatioille
==============================================================================
1. Eteenin (C2H4) lähtö-AO:t:
   - C2H4_HOMO_AO.cube: Kaksi hiilen 2px-atomiorbitaalia samassa vaiheessa (-> pi-sidos)
   - C2H4_LUMO_AO.cube: Kaksi hiilen 2px-atomiorbitaalia vastakkaisissa vaiheissa (-> pi*-hajoittava)

2. Yksittäisen atomin (C / Atomi) perusorbitaalit:
   - Atom_1s.cube:  1s-pallosymmetrinen orbitaali
   - Atom_2s.cube:  2s-orbitaali (säteittäinen solmupinta)
   - Atom_2pz.cube: 2pz-kahdeksikko z-akselilla
   - Atom_2px.cube: 2px-kahdeksikko x-akselilla
   - Atom_2py.cube: 2py-kahdeksikko y-akselilla
   - Atom_sp3.cube: sp3-suunnattu hybridiorbitaali
"""
import os
import sys
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUBES_DIR = os.path.join(SCRIPT_DIR, "cubes")
os.makedirs(CUBES_DIR, exist_ok=True)
BOHR = 0.52917721092


def read_cube_header_and_coords(cube_path):
    """Lukee olemassa olevan cube-tiedoston hilan ja palauttaa Cube-objektin ja koordinaatit."""
    with open(cube_path, "r") as f:
        comment1 = f.readline()
        comment2 = f.readline()
        line3 = f.readline().split()
        natm = int(line3[0])
        boxorig = np.array([float(x) for x in line3[1:4]])
        
        nx, vx_x, vx_y, vx_z = f.readline().split()
        ny, vy_x, vy_y, vy_z = f.readline().split()
        nz, vz_x, vz_y, vz_z = f.readline().split()
        
        nx, ny, nz = int(nx), int(ny), int(nz)
        vx = np.array([float(vx_x), float(vx_y), float(vx_z)])
        vy = np.array([float(vy_x), float(vy_y), float(vy_z)])
        vz = np.array([float(vz_x), float(vz_y), float(vz_z)])
        
        atoms = []
        for _ in range(natm):
            parts = f.readline().split()
            atom_num = int(parts[0])
            charge = float(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            atoms.append((atom_num, x, y, z))
            
    # Rakennetaan koordinaattiverkko (nx, ny, nz, 3)
    xs = boxorig[0] + np.arange(nx)[:, None] * vx[0]
    ys = boxorig[1] + np.arange(ny)[:, None] * vy[1]
    zs = boxorig[2] + np.arange(nz)[:, None] * vz[2]
    
    grid_coords = np.zeros((nx, ny, nz, 3))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grid_coords[i, j, k] = boxorig + i*vx + j*vy + k*vz
                
    return {
        "natm": natm,
        "boxorig": boxorig,
        "nx": nx, "ny": ny, "nz": nz,
        "vx": vx, "vy": vy, "vz": vz,
        "atoms": atoms,
        "grid_coords": grid_coords.reshape(-1, 3)
    }


def generate_c2h4_ao_cubes():
    """Luo C2H4:n HOMO ja LUMO lähtöatomiorbitaalit samalla hilalla kuin C2H4_HOMO.cube."""
    print("-> Generoidaan C2H4 AO-kuutiot...")
    ref_cube_path = os.path.join(CUBES_DIR, "C2H4_HOMO.cube")
    cube_info = read_cube_header_and_coords(ref_cube_path)
    coords_bohr = cube_info["grid_coords"]
    
    # C2H4_HOMO.cube koordinaateissa:
    # C-C -sidos on x-akselilla: C0 at (+1.264227, 0, 0), C1 at (-1.264227, 0, 0) Bohr
    # Molekyylitaso on xy (H-atomit xy-tasossa)
    # Pi-sidos (HOMO ja LUMO) on kohtisuorassa molekyylitasoon nähden eli z-akselilla (2pz)!
    atom_str = "; ".join([f"{'C' if a[0] == 6 else 'H'} {a[1]} {a[2]} {a[3]}" for a in cube_info["atoms"]])
    mol_full = gto.M(atom=atom_str, unit="Bohr", basis="def2-svp")
    
    # Yksittäiset hiiliatomit C0 ja C1 täsmälleen molekyylin hiilten paikoilla
    mol_c0 = gto.M(atom="C 1.264227 0.0 0.0", unit="Bohr", basis="def2-svp")
    mol_c1 = gto.M(atom="C -1.264227 0.0 0.0", unit="Bohr", basis="def2-svp")
    
    ao_c0 = mol_c0.eval_gto("GTOval", coords_bohr)
    ao_c1 = mol_c1.eval_gto("GTOval", coords_bohr)
    labels_c0 = mol_c0.ao_labels()
    
    # Etsitään 2pz- ja 3pz-orbitaalit (pi-sidos muodostuu 2pz-orbitaaleista)
    idx_2pz = [i for i, lbl in enumerate(labels_c0) if "2pz" in lbl or "3pz" in lbl]
    print(f"  Hiilen 2pz/3pz indeksit: {idx_2pz} ({[labels_c0[i] for i in idx_2pz]})")
    
    # C0:n ja C1:n 2pz-aallot (käytetään samoja kertoimia kuin RHF HOMO:ssa)
    phi_c0 = 0.3835 * ao_c0[:, idx_2pz[0]] + 0.2977 * ao_c0[:, idx_2pz[1]]
    phi_c1 = 0.3835 * ao_c1[:, idx_2pz[0]] + 0.2977 * ao_c1[:, idx_2pz[1]]
    
    # Sileä rako x-akselilla hiilten välissä, jotta kaksi erillistä 2pz-kahdeksikkoa näkyy selvästi
    x_coords = coords_bohr[:, 0]
    gap = np.tanh(np.abs(x_coords) / 0.45) ** 2
    
    # 1. HOMO Lähtö-AO:t (samassa vaiheessa: phi_c0 + phi_c1)
    val_homo_ao = (phi_c0 + phi_c1) * gap
    
    # 2. LUMO Lähtö-AO:t (vastakkaisissa vaiheissa: -phi_c0 + phi_c1)
    val_lumo_ao = (-phi_c0 + phi_c1) * gap
    
    # Luodaan Cube ja tallennetaan
    nx, ny, nz = cube_info["nx"], cube_info["ny"], cube_info["nz"]
    cc = cubegen.Cube(mol_full, nx=nx, ny=ny, nz=nz)
    cc.boxorig = cube_info["boxorig"].copy()
    cc.box = np.array([cube_info["vx"] * nx, cube_info["vy"] * ny, cube_info["vz"] * nz])
    
    p_homo_ao = os.path.join(CUBES_DIR, "C2H4_HOMO_AO.cube")
    cc.write(val_homo_ao.reshape(nx, ny, nz), p_homo_ao, comment="C2H4 HOMO constituent AOs (2pz + 2pz)")
    print(f"  -> Tallennettu: {p_homo_ao}")
    
    p_lumo_ao = os.path.join(CUBES_DIR, "C2H4_LUMO_AO.cube")
    cc.write(val_lumo_ao.reshape(nx, ny, nz), p_lumo_ao, comment="C2H4 LUMO constituent AOs (-2pz + 2pz)")
    print(f"  -> Tallennettu: {p_lumo_ao}")


def generate_atomic_cubes():
    """Luo yksittäisen hiiliatomin (tai yleisen atomin) 1s, 2s, 2px, 2py, 2pz ja sp3 kuutiot."""
    print("-> Generoidaan yksittäisen atomin AO-kuutiot...")
    mol_atom = gto.M(atom="C 0.0 0.0 0.0", basis="def2-svp")
    
    # Symmetrinen kuutio origon ympärillä (46x46x46, margin=5.5 Bohr)
    cc = cubegen.Cube(mol_atom, nx=46, ny=46, nz=46, margin=5.5)
    coords = cc.get_coords()
    ao_vals = mol_atom.eval_gto("GTOval", coords)
    labels = mol_atom.ao_labels()
    
    nx, ny, nz = cc.nx, cc.ny, cc.nz
    
    # Etsitään komponentit:
    # 1s:
    idx_1s = [i for i, l in enumerate(labels) if "1s" in l][0]
    val_1s = ao_vals[:, idx_1s] * 0.4
    
    # 2s:
    idx_2s = [i for i, l in enumerate(labels) if "2s" in l][0]
    idx_3s = [i for i, l in enumerate(labels) if "3s" in l][0]
    val_2s = (ao_vals[:, idx_2s] + 0.4 * ao_vals[:, idx_3s]) * 0.4
    
    # 2px, 2py, 2pz:
    idx_2px = [i for i, l in enumerate(labels) if "2px" in l][0]
    idx_3px = [i for i, l in enumerate(labels) if "3px" in l][0]
    val_2px = (ao_vals[:, idx_2px] + 0.5 * ao_vals[:, idx_3px]) * 0.45
    
    idx_2py = [i for i, l in enumerate(labels) if "2py" in l][0]
    idx_3py = [i for i, l in enumerate(labels) if "3py" in l][0]
    val_2py = (ao_vals[:, idx_2py] + 0.5 * ao_vals[:, idx_3py]) * 0.45
    
    idx_2pz = [i for i, l in enumerate(labels) if "2pz" in l][0]
    idx_3pz = [i for i, l in enumerate(labels) if "3pz" in l][0]
    val_2pz = (ao_vals[:, idx_2pz] + 0.5 * ao_vals[:, idx_3pz]) * 0.45
    
    # sp3 hybridiorbitaali: 1/2 * (2s + 2px + 2py + 2pz)
    val_sp3 = 0.5 * (val_2s + val_2px + val_2py + val_2pz) * 1.4
    
    cubes_to_write = {
        "Atom_1s.cube": (val_1s, "Atom 1s orbital"),
        "Atom_2s.cube": (val_2s, "Atom 2s orbital"),
        "Atom_2px.cube": (val_2px, "Atom 2px orbital"),
        "Atom_2py.cube": (val_2py, "Atom 2py orbital"),
        "Atom_2pz.cube": (val_2pz, "Atom 2pz orbital"),
        "Atom_sp3.cube": (val_sp3, "Atom sp3 hybrid orbital"),
    }
    
    for fname, (val_arr, desc) in cubes_to_write.items():
        fpath = os.path.join(CUBES_DIR, fname)
        cc.write(val_arr.reshape(nx, ny, nz), fpath, comment=desc)
        print(f"  -> Tallennettu: {fpath}")


if __name__ == "__main__":
    generate_c2h4_ao_cubes()
    generate_atomic_cubes()
    print("\nKaikki AO-kuutiot luotu onnistuneesti!")
