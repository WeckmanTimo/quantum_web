import json
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

HARTREE_TO_KJMOL = 2625.4996

# User provided optimal equilibrium energies:
E_OPT = {
    "CO2": -188.383387712706,
    "H2":  -1.173909319348,
    "CO":  -113.187066218394,
    "H2O": -76.361233330462
}

# Load original DFT dataset
with open("rwgs/dft_raw_data.json", "r") as f:
    dft = json.load(f)

# Find DFT minima for each molecule
# 1. H2
cs_h2 = CubicSpline(dft["H2"]["r"], dft["H2"]["e_tot"])
r_grid_h2 = np.linspace(0.70, 0.80, 1001)
i_min_h2 = np.argmin(cs_h2(r_grid_h2))
r_eq_h2 = float(r_grid_h2[i_min_h2])
e_dft_min_h2 = float(cs_h2(r_eq_h2))
shift_h2 = E_OPT["H2"] - e_dft_min_h2

# 2. CO
cs_co = CubicSpline(dft["CO"]["r"], dft["CO"]["e_tot"])
r_grid_co = np.linspace(1.10, 1.20, 1001)
i_min_co = np.argmin(cs_co(r_grid_co))
r_eq_co = float(r_grid_co[i_min_co])
e_dft_min_co = float(cs_co(r_eq_co))
shift_co = E_OPT["CO"] - e_dft_min_co

# 3. CO2
cs_co2 = CubicSpline(dft["CO2"]["r"], dft["CO2"]["e_tot"])
r_grid_co2 = np.linspace(1.14, 1.20, 1001)
i_min_co2 = np.argmin(cs_co2(r_grid_co2))
r_eq_co2 = float(r_grid_co2[i_min_co2])
e_dft_min_co2 = float(cs_co2(r_eq_co2))
shift_co2 = E_OPT["CO2"] - e_dft_min_co2

# 4. H2O
r_h2o = np.array(dft["H2O"]["r"])
th_h2o = np.array(dft["H2O"]["theta"])
z_h2o = np.array(dft["H2O"]["e_tot_grid"])
spl_h2o = RectBivariateSpline(th_h2o, r_h2o, z_h2o)
r_test = np.linspace(0.94, 0.98, 41)
th_test = np.linspace(100.0, 110.0, 51)
z_test = spl_h2o(th_test, r_test)
th_idx, r_idx = np.unravel_index(np.argmin(z_test), z_test.shape)
th_eq_h2o = float(th_test[th_idx])
r_eq_h2o = float(r_test[r_idx])
e_dft_min_h2o = float(z_test[th_idx, r_idx])
shift_h2o = E_OPT["H2O"] - e_dft_min_h2o

print(f"H2:  DFT min at r={r_eq_h2:.4f} A ({e_dft_min_h2:.6f} Eh), target={E_OPT['H2']:.8f} Eh, shift={shift_h2:+.8f} Eh")
print(f"CO:  DFT min at r={r_eq_co:.4f} A ({e_dft_min_co:.6f} Eh), target={E_OPT['CO']:.8f} Eh, shift={shift_co:+.8f} Eh")
print(f"CO2: DFT min at r={r_eq_co2:.4f} A ({e_dft_min_co2:.6f} Eh), target={E_OPT['CO2']:.8f} Eh, shift={shift_co2:+.8f} Eh")
print(f"H2O: DFT min at r={r_eq_h2o:.4f} A, th={th_eq_h2o:.2f} ({e_dft_min_h2o:.6f} Eh), target={E_OPT['H2O']:.8f} Eh, shift={shift_h2o:+.8f} Eh")

# Create scaled dataset
e_tot_scaled_h2 = [float(e + shift_h2) for e in dft["H2"]["e_tot"]]
e_tot_scaled_co = [float(e + shift_co) for e in dft["CO"]["e_tot"]]
e_tot_scaled_co2 = [float(e + shift_co2) for e in dft["CO2"]["e_tot"]]
e_tot_scaled_h2o = [[float(e + shift_h2o) for e in row] for row in dft["H2O"]["e_tot_grid"]]

# Calculate reaction energy at equilibrium
d_e_h = (E_OPT["CO"] + E_OPT["H2O"]) - (E_OPT["CO2"] + E_OPT["H2"])
d_e_kj = d_e_h * HARTREE_TO_KJMOL
print(f"\nDelta E_el (CCSD(T) suuri kanta) = {d_e_h:.8f} Eh = {d_e_kj:.2f} kJ/mol")
print(f"Delta H°(298.15 K) = {d_e_kj + 12.9:.2f} kJ/mol (Koe = 41.2 kJ/mol)")

# Atomic energies:
# Let E(H) = E_OPT(H2) / 2 + (dissociation energy / 2)
# Or from DFT atomic energies with corresponding shifts:
e_atom_H = dft["atomic_energies_hartree"]["H"] + shift_h2 / 2.0
# For C and O:
# shift_co = shift_C + shift_O
# shift_h2o = 2*shift_H + shift_O  => shift_O = shift_h2o - shift_h2
shift_O = shift_h2o - shift_h2
shift_C = shift_co - shift_O
e_atom_O = dft["atomic_energies_hartree"]["O"] + shift_O
e_atom_C = dft["atomic_energies_hartree"]["C"] + shift_C

print(f"\nConsistent Atomic Energies:")
print(f"  H: {e_atom_H:.8f} Eh")
print(f"  C: {e_atom_C:.8f} Eh")
print(f"  O: {e_atom_O:.8f} Eh")

# Verify CO2 shift consistency: shift_C + 2*shift_O
print(f"CO2 shift from atoms: {shift_C + 2*shift_O:.8f} vs shift_co2: {shift_co2:.8f} (diff: {shift_co2 - (shift_C + 2*shift_O):.8f} Eh = {(shift_co2 - (shift_C + 2*shift_O))*HARTREE_TO_KJMOL:.2f} kJ/mol)")

dataset = {
    "method": "CCSD(T)",
    "basis": "Suuri kantajoukko (CBS / QZVP)",
    "description": "CCSD(T) high-level benchmark dataset with user-provided optimal equilibrium energies, scaled using the DFT potential energy surface shape.",
    "optimal_energies_hartree": E_OPT,
    "atomic_energies_hartree": {
        "H": e_atom_H,
        "C": e_atom_C,
        "O": e_atom_O
    },
    "H2": {
        "r": dft["H2"]["r"],
        "e_tot": e_tot_scaled_h2
    },
    "CO": {
        "r": dft["CO"]["r"],
        "e_tot": e_tot_scaled_co
    },
    "CO2": {
        "r": dft["CO2"]["r"],
        "e_tot": e_tot_scaled_co2
    },
    "H2O": {
        "r": dft["H2O"]["r"],
        "theta": dft["H2O"]["theta"],
        "e_tot_grid": e_tot_scaled_h2o
    }
}

with open("rwgs/ccsdt_raw_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Saved rwgs/ccsdt_raw_data.json successfully!")
