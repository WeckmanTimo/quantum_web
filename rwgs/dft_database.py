"""
Kvanttikemiallinen tietokanta ja interpolointi (Laskennallinen malli / wB97M-V)
================================================================================
Tietokanta perustuu kvanttikemialliseen tiheysfunktionaalilaskentaan (wB97M-V / def2-QZVP).
Lukiolaisille suunnattu selkeä laskennallinen referenssimalli ilman teknistä jargonia.

Energiakonventio:
Erilliset neutraalit atomit perustilassaan = 0.0 kJ/mol:
  - H (2S)  : E = -0.494601 Eh
  - C (3P)  : E = -37.848232 Eh
  - O (3P)  : E = -75.080778 Eh

Molekyylien sidosenergiat (atomisaatioenergiat, -De) ovat negatiivisia suhteessa erillisiin atomeihin.
"""
import json
import os
from typing import Dict, Tuple
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

HARTREE_TO_KJMOL = 2625.4996

# Polku laskennalliseen wB97M-V -tietokantaan
DATA_PATH = os.path.join(os.path.dirname(__file__), "wb97mv_raw_data.json")

# Fallback-polku tarvittaessa
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "dft_raw_data.json")

with open(DATA_PATH, "r") as f:
    RAW = json.load(f)

RAW_WB97MV = RAW
METHOD_NAME = "Laskennallinen malli"

ATOMS = RAW["atomic_energies_hartree"]
E_ATOM = {
    "H": float(ATOMS["H"]),
    "C": float(ATOMS["C"]),
    "O": float(ATOMS["O"]),
}

# 1D Splinet: H2, CO, CO2
_splines_tot = {}
_splines_atom = {}

# H2: 2 x H
r_h2 = np.array(RAW["H2"]["r"])
e_tot_h2 = np.array(RAW["H2"]["e_tot"])
e_atom_h2 = (e_tot_h2 - 2 * E_ATOM["H"]) * HARTREE_TO_KJMOL
_splines_tot["H2"] = CubicSpline(r_h2, e_tot_h2)
_splines_atom["H2"] = CubicSpline(r_h2, e_atom_h2)

# CO: C + O
r_co = np.array(RAW["CO"]["r"])
e_tot_co = np.array(RAW["CO"]["e_tot"])
e_atom_co = (e_tot_co - (E_ATOM["C"] + E_ATOM["O"])) * HARTREE_TO_KJMOL
_splines_tot["CO"] = CubicSpline(r_co, e_tot_co)
_splines_atom["CO"] = CubicSpline(r_co, e_atom_co)

# CO2: C + 2 x O (symmetrinen venytys)
r_co2 = np.array(RAW["CO2"]["r"])
e_tot_co2 = np.array(RAW["CO2"]["e_tot"])
e_atom_co2 = (e_tot_co2 - (E_ATOM["C"] + 2 * E_ATOM["O"])) * HARTREE_TO_KJMOL
_splines_tot["CO2"] = CubicSpline(r_co2, e_tot_co2)
_splines_atom["CO2"] = CubicSpline(r_co2, e_atom_co2)

# H2O: 2 x H + O (2D-pinta: r vs theta)
r_h2o = np.array(RAW["H2O"]["r"])
th_h2o = np.array(RAW["H2O"]["theta"])
z_tot_h2o = np.array(RAW["H2O"]["e_tot_grid"])
z_atom_h2o = (z_tot_h2o - (2 * E_ATOM["H"] + E_ATOM["O"])) * HARTREE_TO_KJMOL

_spline_tot_h2o = RectBivariateSpline(th_h2o, r_h2o, z_tot_h2o)
_spline_atom_h2o = RectBivariateSpline(th_h2o, r_h2o, z_atom_h2o)


# Laskennalliset tasapainominimit (wB97M-V tulokset)
EQUILIBRIUM = {
    "H2": {"r": 0.740, "E_tot": float(_splines_tot["H2"](0.740)), "E_atom": float(_splines_atom["H2"](0.740))},
    "CO": {"r": 1.127, "E_tot": float(_splines_tot["CO"](1.127)), "E_atom": float(_splines_atom["CO"](1.127))},
    "CO2": {"r": 1.154, "E_tot": float(_splines_tot["CO2"](1.154)), "E_atom": float(_splines_atom["CO2"](1.154))},
    "H2O": {
        "r": 0.958,
        "theta": 104.9,
        "E_tot": float(_spline_tot_h2o(104.9, 0.958)[0, 0]),
        "E_atom": float(_spline_atom_h2o(104.9, 0.958)[0, 0]),
    },
}

# RWGS-reaktioenergia laskennallisessa tasapainossa
RWGS_DELTA_E = (EQUILIBRIUM["CO"]["E_atom"] + EQUILIBRIUM["H2O"]["E_atom"]) - (
    EQUILIBRIUM["CO2"]["E_atom"] + EQUILIBRIUM["H2"]["E_atom"]
)


def calc_1d(key: str, r: float) -> Tuple[float, float, float]:
    """
    Palauttaa (E_total [Eh], E_atom [kJ/mol], 0.0) molekyyleille H2, CO tai CO2.
    """
    e_tot = float(_splines_tot[key](r))
    e_atom = float(_splines_atom[key](r))
    force = -float(_splines_atom[key].derivative()(r))
    return e_tot, e_atom, force


def calc_h2o(r: float, theta_deg: float) -> Tuple[float, float, float, float]:
    """
    Palauttaa (E_total [Eh], E_atom [kJ/mol], Force_r, Force_theta) vedelle H2O.
    """
    e_tot = float(_spline_tot_h2o(theta_deg, r)[0, 0])
    e_atom = float(_spline_atom_h2o(theta_deg, r)[0, 0])
    f_th = -float(_spline_atom_h2o(theta_deg, r, dx=1, dy=0)[0, 0])
    f_r = -float(_spline_atom_h2o(theta_deg, r, dx=0, dy=1)[0, 0])
    return e_tot, e_atom, f_r, f_th
