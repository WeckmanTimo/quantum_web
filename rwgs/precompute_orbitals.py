"""
Precompute Molecular Orbital 3D Grids using PySCF
=================================================
Laskee ja tallentaa 3D-hila-arvot (24x24x24) etulinjan molekyyliorbitaaleille
(HOMO-1, HOMO, LUMO, LUMO+1) neljälle mallimolekyylille:
H2O, CO2, CO, C2H4.
Tallentaa tulokset pakattuna tiedostoon: rwgs/mo_cache.npz
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
CACHE_PATH = os.path.join(SCRIPT_DIR, "mo_cache.npz")
META_PATH = os.path.join(SCRIPT_DIR, "mo_meta.json")

BOHR_TO_ANG = 0.529177210903

CPK_COLORS = {
    "H": "#F8FAFC",
    "C": "#334155",
    "N": "#2563EB",
    "O": "#EF4444"
}

CPK_SIZES = {
    "H": 10,
    "C": 15,
    "N": 15,
    "O": 16
}

MOLECULES_DEF = {
    "H2O": {
        "name": "Vesi (H₂O)",
        "formula": "H₂O",
        "desc": "Taipunut polaarinen molekyyli & elämän liuotin",
        "geom": "O 0.0 0.0 0.117; H 0.0 0.757 -0.469; H 0.0 -0.757 -0.469",
        "bonds": [(0, 1), (0, 2)],
        "orbitals": [
            {
                "key": "HOMO-1",
                "rel_idx": -1,
                "label": "HOMO-1 (2a₁)",
                "type": "O-H sitova / hapen sp²-hybridin kaltainen",
                "desc": "Elektronitiheys keskittyy O-H -sidoksiin ja hapen taakse muodostaen vahvan kovalenttisen rungon.",
                "role": "Sitova perusrunko"
            },
            {
                "key": "HOMO",
                "rel_idx": 0,
                "label": "HOMO (1b₁)",
                "type": "Hapen vapaa elektronipari (puhdas p_x)",
                "desc": "Kohtisuorassa molekyylitasoon (xy). Määrittää veden emäksisyyden, nukleofiilisyyden ja vetysidokset!",
                "role": "Elektroniparin luovuttaja (nukleofiili)"
            },
            {
                "key": "LUMO",
                "rel_idx": 1,
                "label": "LUMO (4a₁)",
                "type": "O-H hajoittava orbitaali (σ*)",
                "desc": "Ensimmäinen tyhjä orbitaali. Sisältää solmutason happi- ja vetyatomien välissä.",
                "role": "Elektronin vastaanottaja (hajoittava)"
            },
            {
                "key": "LUMO+1",
                "rel_idx": 2,
                "label": "LUMO+1 (2b₂)",
                "type": "O-H hajoittava orbitaali (σ*)",
                "desc": "Korkeampi viritystila solmutasolla O-atomien symmetriatasossa.",
                "role": "Korkea viritystila"
            }
        ]
    },
    "CO2": {
        "name": "Hiilidioksidi (CO₂)",
        "formula": "CO₂",
        "desc": "Lineaarinen kasvihuonekaasu & RWGS-lähtöaine",
        "geom": "C 0.0 0.0 0.0; O 0.0 0.0 1.162; O 0.0 0.0 -1.162",
        "bonds": [(0, 1), (0, 2)],
        "orbitals": [
            {
                "key": "HOMO-1",
                "rel_idx": -1,
                "label": "HOMO-1 (1πᵤ)",
                "type": "Sitova π-orbitaali (C=O)",
                "desc": "Elektronitiheys jakautuu symmetrisesti hiilen ja molempien happien välille vahvistaen kaksoissidosta.",
                "role": "Sitova π-sidos"
            },
            {
                "key": "HOMO",
                "rel_idx": 0,
                "label": "HOMO (1π_g)",
                "type": "Hapen vapaat elektroniparit (ei-sitova)",
                "desc": "Elektronitiheys sijaitsee vain happiatomeissa. Hiiliatomin kohdalla on tarkka solmutaso.",
                "role": "Ei-sitova elektronipari"
            },
            {
                "key": "LUMO",
                "rel_idx": 1,
                "label": "LUMO (2πᵤ*)",
                "type": "Hajoittava π*-orbitaali (tyhjä)",
                "desc": "Keskeinen RWGS-katalyysille: elektroni siirtyy tälle orbitaalille, mikä heikentää C=O -sidosta ja mahdollistaa pelkistymisen häkäksi (CO)!",
                "role": "Aktiivinen reaktiokeskus pelkistyksessä"
            },
            {
                "key": "LUMO+1",
                "rel_idx": 2,
                "label": "LUMO+1 (2πᵤ*)",
                "type": "Hajoittava π*-orbitaali (degeneroitunut)",
                "desc": "LUMO:n kohtisuora degeneroitunut vastinpari samalla energialla.",
                "role": "Degeneroitunut LUMO-pari"
            }
        ]
    },
    "CO": {
        "name": "Häkä / Hiilimonoksidi (CO)",
        "formula": "CO",
        "desc": "RWGS-reaktiotuote & teollinen synteesikaasu",
        "geom": "C 0.0 0.0 -0.649; O 0.0 0.0 0.487",
        "bonds": [(0, 1)],
        "orbitals": [
            {
                "key": "HOMO-1",
                "rel_idx": -1,
                "label": "HOMO-1 (1π)",
                "type": "Sitova π-orbitaali (C≡O)",
                "desc": "Vahva sitova π-vuorovaikutus hiilen ja hapen välillä kolmoissidoksessa.",
                "role": "Sitova π-sidos"
            },
            {
                "key": "HOMO",
                "rel_idx": 0,
                "label": "HOMO (5σ)",
                "type": "Hiilen vapaa elektronipari (sp-hybridi)",
                "desc": "Suuri elektronilohko ulkonee hiiliatomin päästä! Selittää, miksi CO sitoutuu katalyytteihin ja hemoglobiiniin nimenomaan hiilen kautta.",
                "role": "Vahva σ-luovuttaja (hiilipää)"
            },
            {
                "key": "LUMO",
                "rel_idx": 1,
                "label": "LUMO (2π*)",
                "type": "Hajoittava π*-orbitaali (tyhjä)",
                "desc": "Vastaanottaa metallin d-elektroneja katalyyttipinnoilla (π-backbonding), mikä tekee CO:sta erinomaisen ligandin.",
                "role": "π-vastaanottaja (backbonding)"
            },
            {
                "key": "LUMO+1",
                "rel_idx": 2,
                "label": "LUMO+1 (2π*)",
                "type": "Hajoittava π*-orbitaali (degeneroitunut)",
                "desc": "LUMO:n kohtisuora degeneroitunut vastinpari.",
                "role": "Degeneroitunut LUMO-pari"
            }
        ]
    },
    "C2H4": {
        "name": "Eteeni (C₂H₄)",
        "formula": "C₂H₄",
        "desc": "Orgaanisen kemian mallimolekyyli & kaksoissidos",
        "geom": "C 0.0 0.0 0.669; C 0.0 0.0 -0.669; H 0.0 0.923 1.232; H 0.0 -0.923 1.232; H 0.0 0.923 -1.232; H 0.0 -0.923 -1.232",
        "bonds": [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
        "orbitals": [
            {
                "key": "HOMO-1",
                "rel_idx": -1,
                "label": "HOMO-1 (1b₂ᵤ)",
                "type": "Sitova σ-kehys (C-C ja C-H)",
                "desc": "Tasossa oleva vahva kovalenttinen σ-runko, joka pitää molekyylin litteänä.",
                "role": "Tasomainen σ-runko"
            },
            {
                "key": "HOMO",
                "rel_idx": 0,
                "label": "HOMO (1b₃ᵤ)",
                "type": "C=C kaksoissidoksen π-orbitaali",
                "desc": "Klassinen π-sidos! Elektronipilvet molekyylitason ylä- ja alapuolella. Reagoi elektrofiilien (esim. Br₂) kanssa additioreaktioissa.",
                "role": "Reaktiivinen π-elektronipari"
            },
            {
                "key": "LUMO",
                "rel_idx": 1,
                "label": "LUMO (1b₂g*)",
                "type": "C=C hajoittava π*-orbitaali",
                "desc": "Solmutaso kahden hiiliatomin välissä. Valoabsorptio virittää elektronin HOMO -> LUMO, mikä mahdollistaa kaksoissidoksen kiertymisen.",
                "role": "Hajoittava π*-viritystila"
            },
            {
                "key": "LUMO+1",
                "rel_idx": 2,
                "label": "LUMO+1 (4a_g*)",
                "type": "Hajoittava σ*-orbitaali",
                "desc": "Korkea tyhjä viritystila kovalenttisen rungon ulkopuolella.",
                "role": "Korkea viritystila"
            }
        ]
    }
}


def precompute_all_orbitals(n_grid: int = 24) -> None:
    """Suorittaa PySCF-laskennan ja tallentaa 3D-hilat ja metatiedot."""
    from pyscf import gto, scf
    from pyscf.tools import cubegen

    grids_dict = {}
    meta_dict = {}

    print(f"Precomputing molecular orbitals on {n_grid}x{n_grid}x{n_grid} grid...")

    for mol_key, mol_def in MOLECULES_DEF.items():
        print(f"-> Calculating {mol_def['name']}...")
        mol = gto.M(atom=mol_def["geom"], basis="def2-SVP", verbose=0)
        mf = scf.RHF(mol).run()

        occ = mf.mo_occ
        nocc = int(np.sum(occ > 0))
        homo_abs = nocc - 1
        e_ev_all = mf.mo_energy * 27.211386

        # Atomikoordinaatit Ångströmeinä
        atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        atom_coords = mol.atom_coords() * BOHR_TO_ANG

        cube = cubegen.Cube(mol, nx=n_grid, ny=n_grid, nz=n_grid, margin=2.6)
        coords_bohr = cube.get_coords()
        coords_ang = coords_bohr * BOHR_TO_ANG
        ao_vals = mol.eval_gto("GTOval", coords_bohr)

        # Hilakoordinaatit
        xs = coords_ang[:, 0].reshape((n_grid, n_grid, n_grid))
        ys = coords_ang[:, 1].reshape((n_grid, n_grid, n_grid))
        zs = coords_ang[:, 2].reshape((n_grid, n_grid, n_grid))

        grids_dict[f"{mol_key}_grid_x"] = xs.astype(np.float32)
        grids_dict[f"{mol_key}_grid_y"] = ys.astype(np.float32)
        grids_dict[f"{mol_key}_grid_z"] = zs.astype(np.float32)

        mol_meta = {
            "name": mol_def["name"],
            "formula": mol_def["formula"],
            "desc": mol_def["desc"],
            "atoms": [
                {
                    "symbol": sym,
                    "x": float(atom_coords[i, 0]),
                    "y": float(atom_coords[i, 1]),
                    "z": float(atom_coords[i, 2]),
                    "color": CPK_COLORS.get(sym, "#A855F7"),
                    "size": CPK_SIZES.get(sym, 14)
                }
                for i, sym in enumerate(atom_symbols)
            ],
            "bonds": mol_def["bonds"],
            "homo_idx": homo_abs,
            "orbitals": []
        }

        for orb in mol_def["orbitals"]:
            mo_idx = homo_abs + orb["rel_idx"]
            mo_energy = float(e_ev_all[mo_idx])
            mo_val = (ao_vals @ mf.mo_coeff[:, mo_idx]).reshape((n_grid, n_grid, n_grid))

            # Tallennetaan hila
            grid_key = f"{mol_key}_{orb['key']}"
            grids_dict[grid_key] = mo_val.astype(np.float32)

            mol_meta["orbitals"].append({
                "key": orb["key"],
                "mo_idx": mo_idx,
                "label": orb["label"],
                "energy_ev": mo_energy,
                "is_occupied": bool(orb["rel_idx"] <= 0),
                "type": orb["type"],
                "desc": orb["desc"],
                "role": orb["role"]
            })

        # HOMO-LUMO gap
        homo_e = e_ev_all[homo_abs]
        lumo_e = e_ev_all[homo_abs + 1]
        mol_meta["gap_ev"] = float(lumo_e - homo_e)

        meta_dict[mol_key] = mol_meta

    # Tallennetaan pakattu npz ja json
    np.savez_compressed(CACHE_PATH, **grids_dict)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)

    print(f"Precomputation complete! Saved to:\n- {CACHE_PATH} ({os.path.getsize(CACHE_PATH)/1024:.1f} KB)\n- {META_PATH}")


if __name__ == "__main__":
    precompute_all_orbitals(n_grid=24)
