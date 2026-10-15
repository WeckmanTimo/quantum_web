"""
Comp-Chem-Lab: Propyylibentseenin empiirinen voimakenttä ja optimoija (MM)
========================================================================
Malli aromaattiselle bentseenirenkaalle ja siihen liittyvälle alkyyliketjulle:
C6H5-CH2-CH2-CH3 (21 atomia: 9 C, 12 H).

Sisältää:
- Aromaattisen renkaan tasomaisen jousimallin (sidokset, kulmat, improper-tasotorsiot)
- Alkyyliketjun joustavan sidos- ja kulmamallin
- Renkaan ja ketjun välisen torsiopotentiaalin (suosii kohtisuoraa 90° asentoa)
- Alifaattisen ketjun torsiopotentiaalin (Anti 180° / Gauche 60°)
- Lennard-Jones (12-6) dispersio- ja steeriset repulsiot (orto-H este)
- L-BFGS-geometriaoptimoinnin trajektoriseurannalla
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

ELEMENTS = ["C"] * 9 + ["H"] * 12

ATOM_NAMES = [
    "C1 (ipso-rengashiili)", "C2 (orto-rengashiili)", "C3 (meta-rengashiili)",
    "C4 (para-rengashiili)", "C5 (meta-rengashiili)", "C6 (orto-rengashiili)",
    "C7 (Cα-ketjuhiili)", "C8 (Cβ-ketjuhiili)", "C9 (Cγ-ketjuhiili)",
    "H2 (orto-vety)", "H3 (meta-vety)", "H4 (para-vety)",
    "H5 (meta-vety)", "H6 (orto-vety)",
    "H7a (ketjuvety)", "H7b (ketjuvety)",
    "H8a (ketjuvety)", "H8b (ketjuvety)",
    "H9a (ketjuvety)", "H9b (ketjuvety)", "H9c (ketjuvety)"
]

# Sidokset: (i, j, r0 [Å], kb [kJ/(mol*Å^2)])
BONDS = [
    # Aromaattinen rengas C-C
    (0, 1, 1.399, 3600.0), (1, 2, 1.399, 3600.0), (2, 3, 1.399, 3600.0),
    (3, 4, 1.399, 3600.0), (4, 5, 1.399, 3600.0), (5, 0, 1.399, 3600.0),
    # Rengas C-H
    (1, 9, 1.085, 2600.0), (2, 10, 1.085, 2600.0), (3, 11, 1.085, 2600.0),
    (4, 12, 1.085, 2600.0), (5, 13, 1.085, 2600.0),
    # Renkaan ja ketjun liitos C1-C7
    (0, 6, 1.510, 2400.0),
    # Alkyyliketjun C-C
    (6, 7, 1.530, 2200.0), (7, 8, 1.530, 2200.0),
    # Ketjun C-H
    (6, 14, 1.090, 2500.0), (6, 15, 1.090, 2500.0),
    (7, 16, 1.090, 2500.0), (7, 17, 1.090, 2500.0),
    (8, 18, 1.090, 2500.0), (8, 19, 1.090, 2500.0), (8, 20, 1.090, 2500.0)
]

TH_120 = np.radians(120.0)
TH_TET = np.radians(109.471)

# Kulmat: (i, j, k, theta0 [rad], kth [kJ/(mol*rad^2)])
ANGLES = [
    # Renkaan C-C-C (120°)
    (5, 0, 1, TH_120, 600.0), (0, 1, 2, TH_120, 600.0), (1, 2, 3, TH_120, 600.0),
    (2, 3, 4, TH_120, 600.0), (3, 4, 5, TH_120, 600.0), (4, 5, 0, TH_120, 600.0),
    # Renkaan C-C-H (120°)
    (0, 1, 9, TH_120, 400.0), (2, 1, 9, TH_120, 400.0),
    (1, 2, 10, TH_120, 400.0), (3, 2, 10, TH_120, 400.0),
    (2, 3, 11, TH_120, 400.0), (4, 3, 11, TH_120, 400.0),
    (3, 4, 12, TH_120, 400.0), (5, 4, 12, TH_120, 400.0),
    (4, 5, 13, TH_120, 400.0), (0, 5, 13, TH_120, 400.0),
    # Rengas-ketju C-C-C7 (120°)
    (1, 0, 6, TH_120, 450.0), (5, 0, 6, TH_120, 450.0),
    # Ketjun kulmat (tetraedriset ~109.5°)
    (0, 6, 7, TH_TET, 400.0), (6, 7, 8, TH_TET, 400.0),
    (0, 6, 14, TH_TET, 320.0), (0, 6, 15, TH_TET, 320.0), (7, 6, 14, TH_TET, 320.0), (7, 6, 15, TH_TET, 320.0), (14, 6, 15, TH_TET, 280.0),
    (6, 7, 16, TH_TET, 320.0), (6, 7, 17, TH_TET, 320.0), (8, 7, 16, TH_TET, 320.0), (8, 7, 17, TH_TET, 320.0), (16, 7, 17, TH_TET, 280.0),
    (7, 8, 18, TH_TET, 320.0), (7, 8, 19, TH_TET, 320.0), (7, 8, 20, TH_TET, 320.0),
    (18, 8, 19, TH_TET, 280.0), (18, 8, 20, TH_TET, 280.0), (19, 8, 20, TH_TET, 280.0)
]

# Bentseenirenkaan tasomaisuutta ylläpitävät out-of-plane / improper dihedraalit (i, j, k, l, koop)
# Pitää renkaan 6 hiiltä ja orto/meta/para-substituutiot tismalleen tasossa
IMPROPERS = [
    (0, 1, 2, 3, 350.0), (1, 2, 3, 4, 350.0), (2, 3, 4, 5, 350.0),
    (3, 4, 5, 0, 350.0), (4, 5, 0, 1, 350.0), (5, 0, 1, 2, 350.0),
    # C7 ja orto-vedyt pysyvät renkaan tasossa
    (9, 0, 2, 1, 250.0), (10, 1, 3, 2, 250.0), (11, 2, 4, 3, 250.0),
    (12, 3, 5, 4, 250.0), (13, 4, 0, 5, 250.0), (6, 5, 1, 0, 300.0)
]

# Torsioparametrit
# 1. Renkaan ja ketjun välinen torsio (C2-C1-C7-C8):
# Minimiasento on 90° (C7-C8 kohtisuorassa rengasta vastaan), maksimi 0° ja 180° (steerinen törmäys orto-vetyyn)
V2_RING = 8.5  # kJ/mol

# 2. Ketjun sisäinen torsio C1-C7-C8-C9:
V1_CHAIN = 5.5
V2_CHAIN = -1.5
V3_CHAIN = 13.5

LJ_PARAMS = {
    "C": (3.40, 0.36),
    "H": (2.50, 0.15),
}

# 1-2 ja 1-3 parit suljetaan pois vdW-laskennasta
EXCLUDED_PAIRS = set()
for i, j, _, _ in BONDS:
    EXCLUDED_PAIRS.add((min(i, j), max(i, j)))
for i, j, k, _, _ in ANGLES:
    EXCLUDED_PAIRS.add((min(i, k), max(i, k)))

LJ_PAIRS = []
for i in range(21):
    for j in range(i + 1, 21):
        if (i, j) not in EXCLUDED_PAIRS:
            sig_i, eps_i = LJ_PARAMS[ELEMENTS[i]]
            sig_j, eps_j = LJ_PARAMS[ELEMENTS[j]]
            sigma = 0.5 * (sig_i + sig_j)
            epsilon = math.sqrt(eps_i * eps_j)
            LJ_PAIRS.append((i, j, sigma, epsilon))


def _place_atom(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, r: float, theta_rad: float, phi_rad: float) -> np.ndarray:
    """Asettaa atomin p4 annettujen p3 (emo), p2 ja p1 suhteen (NeRF)."""
    v23 = p3 - p2
    norm_v23 = np.linalg.norm(v23)
    v23_u = v23 / norm_v23 if norm_v23 > 1e-6 else np.array([1.0, 0.0, 0.0])

    v12 = p2 - p1
    n = np.cross(v12, v23_u)
    norm_n = np.linalg.norm(n)
    n = n / norm_n if norm_n > 1e-6 else np.array([0.0, 0.0, 1.0])

    u = np.cross(n, v23_u)
    d = -v23_u * np.cos(theta_rad) + u * np.sin(theta_rad) * np.cos(phi_rad) + n * np.sin(theta_rad) * np.sin(phi_rad)
    return p3 + r * d


def place_ch2_hydrogens(c_prev: np.ndarray, c_curr: np.ndarray, c_next: np.ndarray, r_ch: float = 1.090) -> Tuple[np.ndarray, np.ndarray]:
    """Sijoittaa kaksi vetyä -CH2- hiileen tetraedripuolittajan avulla."""
    u1 = (c_prev - c_curr) / np.linalg.norm(c_prev - c_curr)
    u3 = (c_next - c_curr) / np.linalg.norm(c_next - c_curr)
    bis = u1 + u3
    norm_bis = np.linalg.norm(bis)
    bis = bis / norm_bis if norm_bis > 1e-6 else np.array([0.0, 1.0, 0.0])
    n = np.cross(u1, u3)
    norm_n = np.linalg.norm(n)
    n = n / norm_n if norm_n > 1e-6 else np.array([0.0, 0.0, 1.0])

    half_th = np.radians(109.471 / 2.0)
    d1 = -bis * np.cos(half_th) + n * np.sin(half_th)
    d2 = -bis * np.cos(half_th) - n * np.sin(half_th)
    return c_curr + r_ch * d1, c_curr + r_ch * d2


def calc_dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Laskee dihedraalikulman neljän pisteen välillä asteina [-180, 180]."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    norm_b1 = np.linalg.norm(b1)
    if norm_b1 < 1e-6:
        return 0.0
    b1_u = b1 / norm_b1
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def build_propylbenzene(phi_ring_deg: float = 90.0, phi_chain_deg: float = 180.0) -> np.ndarray:
    """
    Rakentaa propyylibentseenin koordinaatit (21, 3) Ångströmeinä.
    - phi_ring_deg: C2-C1-C7-C8 torsiokulma (90° = kohtisuora, 0° = tasossa)
    - phi_chain_deg: C1-C7-C8-C9 torsiokulma (180° = anti, 60° = gauche)
    """
    r_cc_ring = 1.399
    r_ch_ring = 1.085
    r_c1_c7 = 1.510
    r_cc = 1.530
    r_ch = 1.090
    th_120 = np.radians(120.0)
    th_tet = np.radians(109.471)

    # 1. Rengashiilet C1..C6 tasossa xy
    angles = np.radians([270, 210, 150, 90, 30, 330])
    R = r_cc_ring
    ring_c = [np.array([R * np.cos(a), R * np.sin(a), 0.0]) for a in angles]

    # Rengasvedyt H2..H6
    ring_h = []
    for i in range(1, 6):
        a = angles[i]
        ring_h.append(ring_c[i] + r_ch_ring * np.array([np.cos(a), np.sin(a), 0.0]))

    c1 = ring_c[0]
    c2 = ring_c[1]
    c6 = ring_c[5]

    # C7: kiinnittyy C1:een, kulma 120° C2 ja C6 suhteen renkaan tasossa
    c7 = _place_atom(c6, c2, c1, r_c1_c7, th_120, np.radians(180.0))

    # C8: kiinnittyy C7:ään, kulma 109.5°, dihedraali C2-C1-C7-C8 = phi_ring_deg
    c8 = _place_atom(c2, c1, c7, r_cc, th_tet, np.radians(phi_ring_deg))

    # H7a, H7b C7:lle
    h7a, h7b = place_ch2_hydrogens(c1, c7, c8, r_ch)

    # C9: kiinnittyy C8:aan, kulma 109.5°, dihedraali C1-C7-C8-C9 = phi_chain_deg
    c9 = _place_atom(c1, c7, c8, r_cc, th_tet, np.radians(phi_chain_deg))

    # H8a, H8b C8:lle
    h8a, h8b = place_ch2_hydrogens(c7, c8, c9, r_ch)

    # H9a, H9b, H9c C9:lle
    h9a = _place_atom(c7, c8, c9, r_ch, th_tet, np.radians(180.0))
    h9b = _place_atom(c7, c8, c9, r_ch, th_tet, np.radians(60.0))
    h9c = _place_atom(c7, c8, c9, r_ch, th_tet, np.radians(-60.0))

    coords = np.array([
        ring_c[0], ring_c[1], ring_c[2], ring_c[3], ring_c[4], ring_c[5],  # 0..5: C1..C6
        c7, c8, c9,                                                        # 6..8: C7..C9
        ring_h[0], ring_h[1], ring_h[2], ring_h[3], ring_h[4],             # 9..13: H2..H6
        h7a, h7b,                                                          # 14, 15: H7a, H7b
        h8a, h8b,                                                          # 16, 17: H8a, H8b
        h9a, h9b, h9c                                                      # 18..20: H9a..H9c
    ], dtype=float)

    coords -= np.mean(coords[:6], axis=0)
    return coords


def calc_energy_and_components(coords: np.ndarray) -> Dict[str, float]:
    """Laskee propyylibentseenin voimakenttäenergian ja sen osakomponentit [kJ/mol]."""
    # 1. Sidokset
    e_bond = 0.0
    for i, j, r0, kb in BONDS:
        r = np.linalg.norm(coords[i] - coords[j])
        e_bond += 0.5 * kb * (r - r0) ** 2

    # 2. Kulmat
    e_angle = 0.0
    for i, j, k, th0, kth in ANGLES:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 1e-6:
            cos_th = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            th = np.arccos(cos_th)
            e_angle += 0.5 * kth * (th - th0) ** 2

    # 3. Impropers (renkaan litteys)
    e_oop = 0.0
    for i, j, k, l, koop in IMPROPERS:
        phi = calc_dihedral_angle(coords[i], coords[j], coords[k], coords[l])
        phi_rad = np.radians(phi)
        # Ihannetila on 0° tai 180°
        e_oop += 0.5 * koop * (1.0 - np.cos(2.0 * phi_rad))

    # 4. Torsiot
    phi_ring = calc_dihedral_angle(coords[1], coords[0], coords[6], coords[7])  # C2-C1-C7-C8
    phi_chain = calc_dihedral_angle(coords[0], coords[6], coords[7], coords[8]) # C1-C7-C8-C9

    phi_r_rad = np.radians(phi_ring)
    # Torsiopotentiaali renkaan ja ketjun välillä: minimit 90° ja -90° (cos(2*90°) = -1 -> E=0)
    e_tors_ring = 0.5 * V2_RING * (1.0 + np.cos(2.0 * phi_r_rad))

    phi_c_rad = np.radians(phi_chain)
    e_tors_chain = (0.5 * V1_CHAIN * (1.0 + np.cos(phi_c_rad)) +
                    0.5 * V2_CHAIN * (1.0 - np.cos(2.0 * phi_c_rad)) +
                    0.5 * V3_CHAIN * (1.0 + np.cos(3.0 * phi_c_rad)))

    e_torsion = e_tors_ring + e_tors_chain

    # 5. Lennard-Jones (12-6)
    e_vdw = 0.0
    for i, j, sigma, epsilon in LJ_PAIRS:
        r = np.linalg.norm(coords[i] - coords[j])
        if r < 0.6:
            r = 0.6
        sr6 = (sigma / r) ** 6
        sr12 = sr6 * sr6
        e_vdw += 0.5 * 4.0 * epsilon * (sr12 - sr6)

    e_tot = e_bond + e_angle + e_oop + e_torsion + e_vdw

    return {
        "E_tot": float(e_tot),
        "E_bond": float(e_bond),
        "E_angle": float(e_angle + e_oop),
        "E_torsion": float(e_torsion),
        "E_vdw": float(e_vdw),
        "phi_ring": float(phi_ring),
        "phi_chain": float(phi_chain)
    }


def optimize_propylbenzene(initial_coords: np.ndarray, max_iter: int = 16) -> Tuple[np.ndarray, List[Dict]]:
    """Minimoi geometrian L-BFGS-menetelmällä ja palauttaa trajektorin."""
    trajectory = []
    step_counter = [0]

    init_res = calc_energy_and_components(initial_coords)
    trajectory.append({
        "step": 0,
        "energy": init_res["E_tot"],
        "e_bond": init_res["E_bond"],
        "e_angle": init_res["E_angle"],
        "e_torsion": init_res["E_torsion"],
        "e_vdw": init_res["E_vdw"],
        "phi_ring": init_res["phi_ring"],
        "phi_chain": init_res["phi_chain"],
        "coords": initial_coords.copy()
    })

    def obj_fn(x_flat: np.ndarray) -> float:
        c = x_flat.reshape((-1, 3))
        res = calc_energy_and_components(c)
        return res["E_tot"]

    def callback(xk: np.ndarray) -> None:
        step_counter[0] += 1
        c = xk.reshape((-1, 3))
        res = calc_energy_and_components(c)
        trajectory.append({
            "step": step_counter[0],
            "energy": res["E_tot"],
            "e_bond": res["E_bond"],
            "e_angle": res["E_angle"],
            "e_torsion": res["E_torsion"],
            "e_vdw": res["E_vdw"],
            "phi_ring": res["phi_ring"],
            "phi_chain": res["phi_chain"],
            "coords": c.copy()
        })

    res = minimize(
        obj_fn,
        initial_coords.flatten(),
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": max_iter, "gtol": 1e-3, "ftol": 1e-4}
    )

    opt_coords = res.x.reshape((-1, 3))
    final_res = calc_energy_and_components(opt_coords)
    if not trajectory or np.abs(trajectory[-1]["energy"] - final_res["E_tot"]) > 1e-3:
        trajectory.append({
            "step": len(trajectory),
            "energy": final_res["E_tot"],
            "e_bond": final_res["E_bond"],
            "e_angle": final_res["E_angle"],
            "e_torsion": final_res["E_torsion"],
            "e_vdw": final_res["E_vdw"],
            "phi_ring": final_res["phi_ring"],
            "phi_chain": final_res["phi_chain"],
            "coords": opt_coords.copy()
        })

    return opt_coords, trajectory


def perturb_coords(coords: np.ndarray, scale: float = 0.20) -> np.ndarray:
    """Lisää satunnaisen häiriön atomien koordinaatteihin."""
    rng = np.random.default_rng()
    noise = rng.normal(0.0, scale, size=coords.shape)
    perturbed = coords + noise
    perturbed -= np.mean(perturbed[:6], axis=0)
    return perturbed
