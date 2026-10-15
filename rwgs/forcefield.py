"""
Molecular Mechanics (MM) Force Field Engine for Butane (C4H10)
==============================================================
Klassinen empiirinen voimakenttämoottori pienen joustavan molekyylin (butaani)
konformaatioanalyysiin, vääntämiseen ja geometriaoptimointiin.

Voimakenttäpotentiaali:
  E_MM = E_bond + E_angle + E_torsion + E_vdW

- E_bond: Harmoninen sidosvenytys (C-C, C-H)
- E_angle: Harmoninen kulmataivutus (C-C-C, C-C-H, H-C-H)
- E_torsion: 3-terminen Fourier-torsiopotentiaali C1-C2-C3-C4 dihedraalille
             (Anti 180° = 0.0 kJ/mol, Gauche 60° ≈ 3.8 kJ/mol, Syn 0° ≈ 19 kJ/mol)
- E_vdW: Lennard-Jones 12-6 ei-sitovat vuorovaikutukset (1-4 ja etäisemmät)
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize

ELEMENTS = [
    "C", "C", "C", "C",                     # C1, C2, C3, C4
    "H", "H", "H",                         # H1a, H1b, H1c (C1)
    "H", "H",                              # H2a, H2b (C2)
    "H", "H",                              # H3a, H3b (C3)
    "H", "H", "H"                          # H4a, H4b, H4c (C4)
]

# Sidokset: (i, j, r0 [Å], k_b [kJ/(mol*Å^2)])
BONDS = [
    # C-C
    (0, 1, 1.53, 2200.0),
    (1, 2, 1.53, 2200.0),
    (2, 3, 1.53, 2200.0),
    # C1-H
    (0, 4, 1.09, 2500.0),
    (0, 5, 1.09, 2500.0),
    (0, 6, 1.09, 2500.0),
    # C2-H
    (1, 7, 1.09, 2500.0),
    (1, 8, 1.09, 2500.0),
    # C3-H
    (2, 9, 1.09, 2500.0),
    (2, 10, 1.09, 2500.0),
    # C4-H
    (3, 11, 1.09, 2500.0),
    (3, 12, 1.09, 2500.0),
    (3, 13, 1.09, 2500.0),
]

# Kulmat: (i, j, k, theta0 [rad], k_theta [kJ/(mol*rad^2)])
TH0 = np.radians(109.471)  # tetraedrinen kulma
ANGLES = [
    # C-C-C
    (0, 1, 2, TH0, 420.0),
    (1, 2, 3, TH0, 420.0),
    # C-C-H
    (1, 0, 4, TH0, 320.0), (1, 0, 5, TH0, 320.0), (1, 0, 6, TH0, 320.0),
    (0, 1, 7, TH0, 320.0), (0, 1, 8, TH0, 320.0),
    (2, 1, 7, TH0, 320.0), (2, 1, 8, TH0, 320.0),
    (1, 2, 9, TH0, 320.0), (1, 2, 10, TH0, 320.0),
    (3, 2, 9, TH0, 320.0), (3, 2, 10, TH0, 320.0),
    (2, 3, 11, TH0, 320.0), (2, 3, 12, TH0, 320.0), (2, 3, 13, TH0, 320.0),
    # H-C-H
    (4, 0, 5, TH0, 280.0), (4, 0, 6, TH0, 280.0), (5, 0, 6, TH0, 280.0),
    (7, 1, 8, TH0, 280.0),
    (9, 2, 10, TH0, 280.0),
    (11, 3, 12, TH0, 280.0), (11, 3, 13, TH0, 280.0), (12, 3, 13, TH0, 280.0),
]

# Torsioparametrit C1-C2-C3-C4 (kJ/mol)
# Anti (180°) = 0, Gauche (60°) ≈ 3.8, Eclipsed (0°) ≈ 19.0
V1 = 5.5
V2 = -1.5
V3 = 13.5

# Lennard-Jones parametrit: sigma [Å], epsilon [kJ/mol]
LJ_PARAMS = {
    "C": (3.40, 0.36),
    "H": (2.50, 0.15),
}

# 1-2 ja 1-3 parit suljetaan pois
EXCLUDED_PAIRS = set()
for i, j, _, _ in BONDS:
    EXCLUDED_PAIRS.add((min(i, j), max(i, j)))
for i, j, k, _, _ in ANGLES:
    EXCLUDED_PAIRS.add((min(i, k), max(i, k)))


def _place_atom(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, r: float, theta_rad: float, phi_rad: float) -> np.ndarray:
    """Asettaa atomin p4 annettujen p3 (emo), p2 ja p1 suhteen (NeRF-menetelmä)."""
    v23 = p3 - p2
    norm_v23 = np.linalg.norm(v23)
    if norm_v23 < 1e-6:
        v23_u = np.array([1.0, 0.0, 0.0])
    else:
        v23_u = v23 / norm_v23

    v12 = p2 - p1
    n = np.cross(v12, v23_u)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-6:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / norm_n

    u = np.cross(n, v23_u)
    d = -v23_u * np.cos(theta_rad) + u * np.sin(theta_rad) * np.cos(phi_rad) + n * np.sin(theta_rad) * np.sin(phi_rad)
    return p3 + r * d


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


def build_butane(phi_deg: float = 180.0, r_cc: float = 1.53, ccc_deg: float = 109.471) -> np.ndarray:
    """
    Rakentaa butaanimolekyylin (C4H10) koordinaatit (14, 3) Ångströmeinä.
    - phi_deg: C1-C2-C3-C4 dihedraalikulma [°]
    - r_cc: C2-C3 keskisidoksen pituus [Å]
    - ccc_deg: C-C-C sidoskulmat [°]
    """
    th_ccc = np.radians(ccc_deg)
    r_ch = 1.09
    th_tet = np.radians(109.471)

    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([1.53, 0.0, 0.0])
    ref = np.array([0.0, 1.0, 0.0])
    c3 = _place_atom(ref, c1, c2, r_cc, th_ccc, 0.0)
    c4 = _place_atom(c1, c2, c3, 1.53, th_ccc, np.radians(phi_deg))

    # H C1:llä (H4, H5, H6)
    h1a = _place_atom(c3, c2, c1, r_ch, th_tet, np.radians(180.0))
    h1b = _place_atom(c3, c2, c1, r_ch, th_tet, np.radians(60.0))
    h1c = _place_atom(c3, c2, c1, r_ch, th_tet, np.radians(-60.0))

    # H C2:lla (H7, H8) - tarkka tetraedrinen puolittaja
    u1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    u3 = (c3 - c2) / np.linalg.norm(c3 - c2)
    bis2 = u1 + u3
    bis2 /= np.linalg.norm(bis2)
    n2 = np.cross(u1, u3)
    n2 /= np.linalg.norm(n2)
    half_th = np.radians(109.471 / 2.0)
    h2a = c2 + r_ch * (-bis2 * np.cos(half_th) + n2 * np.sin(half_th))
    h2b = c2 + r_ch * (-bis2 * np.cos(half_th) - n2 * np.sin(half_th))

    # H C3:lla (H9, H10) - tarkka tetraedrinen puolittaja
    u2 = (c2 - c3) / np.linalg.norm(c2 - c3)
    u4 = (c4 - c3) / np.linalg.norm(c4 - c3)
    bis3 = u2 + u4
    bis3 /= np.linalg.norm(bis3)
    n3 = np.cross(u2, u4)
    n3 /= np.linalg.norm(n3)
    h3a = c3 + r_ch * (-bis3 * np.cos(half_th) + n3 * np.sin(half_th))
    h3b = c3 + r_ch * (-bis3 * np.cos(half_th) - n3 * np.sin(half_th))

    # H C4:llä (H11, H12, H13)
    h4a = _place_atom(c2, c3, c4, r_ch, th_tet, np.radians(180.0))
    h4b = _place_atom(c2, c3, c4, r_ch, th_tet, np.radians(60.0))
    h4c = _place_atom(c2, c3, c4, r_ch, th_tet, np.radians(-60.0))

    coords = np.array([
        c1, c2, c3, c4,
        h1a, h1b, h1c,
        h2a, h2b,
        h3a, h3b,
        h4a, h4b, h4c
    ], dtype=float)

    # Keskitetään koordinaatit
    coords -= np.mean(coords[:4], axis=0)
    return coords


def calc_energy_and_components(coords: np.ndarray) -> Dict[str, float]:
    """
    Laskee butaanin voimakenttäenergian ja sen osakomponentit [kJ/mol].
    """
    e_bond = 0.0
    for i, j, r0, kb in BONDS:
        rij = np.linalg.norm(coords[i] - coords[j])
        e_bond += 0.5 * kb * (rij - r0)**2

    e_angle = 0.0
    for i, j, k, th0, kth in ANGLES:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 1e-6:
            cos_th = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
            th = np.arccos(cos_th)
            e_angle += 0.5 * kth * (th - th0)**2

    # Torsio C1-C2-C3-C4
    phi_deg = calc_dihedral_angle(coords[0], coords[1], coords[2], coords[3])
    phi_rad = np.radians(phi_deg)
    e_torsion = (0.5 * V1 * (1.0 + np.cos(phi_rad)) +
                 0.5 * V2 * (1.0 - np.cos(2.0 * phi_rad)) +
                 0.5 * V3 * (1.0 + np.cos(3.0 * phi_rad)))

    # Lennard-Jones van der Waals (1-4 skaalattu 0.5 tekijällä standardin MM mukaisesti)
    e_vdw = 0.0
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in EXCLUDED_PAIRS:
                continue
            r = np.linalg.norm(coords[i] - coords[j])
            if r < 0.6:
                r = 0.6
            sig_i, eps_i = LJ_PARAMS[ELEMENTS[i]]
            sig_j, eps_j = LJ_PARAMS[ELEMENTS[j]]
            sigma = 0.5 * (sig_i + sig_j)
            epsilon = np.sqrt(eps_i * eps_j)
            sr6 = (sigma / r)**6
            sr12 = sr6**2
            e_vdw += 0.5 * 4.0 * epsilon * (sr12 - sr6)

    # Normalisoidaan anti-konformaation energia nollatasolle oppimiskäyttöä varten
    # Anti-butaani (phi=180°) = 0.0 kJ/mol
    e_tot = e_bond + e_angle + e_torsion + e_vdw

    r_cc = float(np.linalg.norm(coords[1] - coords[2]))
    v1 = coords[0] - coords[1]
    v2 = coords[2] - coords[1]
    ccc_deg = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))))

    return {
        "E_tot": float(e_tot),
        "E_bond": float(e_bond),
        "E_angle": float(e_angle),
        "E_torsion": float(e_torsion),
        "E_vdw": float(e_vdw),
        "phi": float(phi_deg),
        "r_cc": float(r_cc),
        "ccc_deg": float(ccc_deg)
    }


def coords_to_xyz(coords: np.ndarray, title: str = "Butane") -> str:
    """Muuntaa koordinaatit XYZ-muotoon py3Dmolia varten."""
    lines = [f"{len(coords)}", title]
    for elem, (x, y, z) in zip(ELEMENTS, coords):
        lines.append(f"{elem:2s} {x:10.5f} {y:10.5f} {z:10.5f}")
    return "\n".join(lines)


def optimize_butane(initial_coords: np.ndarray, max_iter: int = 40) -> Tuple[np.ndarray, List[Dict]]:
    """
    Optimoi butaanin geometrian L-BFGS-B -menetelmällä.
    Palauttaa: (opt_coords, trajectory)
    """
    trajectory: List[Dict] = []
    step_counter = [0]

    init_res = calc_energy_and_components(initial_coords)
    trajectory.append({
        "step": 0,
        "energy": init_res["E_tot"],
        "e_bond": init_res["E_bond"],
        "e_angle": init_res["E_angle"],
        "e_torsion": init_res["E_torsion"],
        "e_vdw": init_res["E_vdw"],
        "phi": init_res["phi"],
        "r_cc": init_res["r_cc"],
        "ccc_deg": init_res["ccc_deg"],
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
            "phi": res["phi"],
            "r_cc": res["r_cc"],
            "ccc_deg": res["ccc_deg"],
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
            "phi": final_res["phi"],
            "r_cc": final_res["r_cc"],
            "ccc_deg": final_res["ccc_deg"],
            "coords": opt_coords.copy()
        })

    return opt_coords, trajectory


def perturb_coords(coords: np.ndarray, scale: float = 0.20) -> np.ndarray:
    """Lisää satunnaisen häiriön atomien koordinaatteihin."""
    rng = np.random.default_rng()
    noise = rng.normal(0.0, scale, size=coords.shape)
    perturbed = coords + noise
    perturbed -= np.mean(perturbed[:4], axis=0)
    return perturbed


def rotate_dihedral(coords: np.ndarray, target_phi_deg: float) -> np.ndarray:
    """Kiertää C1-C2-C3-C4 dihedraalikulmaa annettuun arvoon pyörittämällä C3-päätä C2-C3-akselin ympäri."""
    c = coords.copy()
    current_phi = calc_dihedral_angle(c[0], c[1], c[2], c[3])
    delta_phi = np.radians(target_phi_deg - current_phi)

    axis = c[2] - c[1]
    norm_ax = np.linalg.norm(axis)
    if norm_ax < 1e-6:
        return c
    axis /= norm_ax

    # C4 ja siihen liittyvät atomit
    rot_indices = [3, 9, 10, 11, 12, 13]
    origin = c[2]

    cos_a = np.cos(delta_phi)
    sin_a = np.sin(delta_phi)
    for idx in rot_indices:
        p = c[idx] - origin
        p_rot = p * cos_a + np.cross(axis, p) * sin_a + axis * np.dot(axis, p) * (1.0 - cos_a)
        c[idx] = origin + p_rot

    c -= np.mean(c[:4], axis=0)
    return c


def stretch_central_bond(coords: np.ndarray, target_r_cc: float) -> np.ndarray:
    """Muuttaa C2-C3 sidoksen pituutta säilyttäen molekyylin suuntauksen."""
    c = coords.copy()
    current_r = np.linalg.norm(c[2] - c[1])
    if current_r < 1e-4:
        return c
    delta_r = target_r_cc - current_r
    axis = (c[2] - c[1]) / current_r

    c3_group = [2, 3, 9, 10, 11, 12, 13]
    c2_group = [0, 1, 4, 5, 6, 7, 8]

    for idx in c3_group:
        c[idx] += 0.5 * delta_r * axis
    for idx in c2_group:
        c[idx] -= 0.5 * delta_r * axis

    c -= np.mean(c[:4], axis=0)
    return c
