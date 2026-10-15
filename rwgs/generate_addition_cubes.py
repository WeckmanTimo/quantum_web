"""
Comp-Chem-Lab: Generoi 3D-kuutiot nukleofiilisen addition demonstraatiolle (Formaldehydi + Syanidi CN-)
======================================================================================================
Karbonyyli: Formaldehydi (H2CO)
Nukleofiilimolekyyli: Syanidi-ioni (CN-)
Reaktio: H2C=O + CN- ➔ NC–CH2–O- (Syanohydriinin muodostuminen, uusi C-C sidos)

Kuutiot (46x46x46 hila, margin=5.5 Bohr):
1. addition_carbonyl_lumo.cube         : H2CO LUMO yksin
2. addition_nucleophile_homo.cube      : CN- HOMO yksin (hiilen sp-vapaa elektronipari)
3. addition_carb_in_comp.cube          : H2CO LUMO hyökkäysasennossa (kaikki 6 atomia)
4. addition_nuc_in_comp.cube           : CN- HOMO hyökkäysasennossa (kaikki 6 atomia, eri sävyllä)
5. addition_hybridized_bonding.cube    : Vuorovaikuttanut kompleksi: muodostunut sigma(C-C)-sidos
6. addition_hybridized_homo.cube       : Vuorovaikuttaneen kompleksin HOMO (elektronitiheyden siirtyminen hapelle)
7. addition_hybridized_antibonding.cube: Vuorovaikuttaneen kompleksin hajottava sigma*-orbitaali
"""
import os
import sys
import numpy as np
from pyscf import gto, scf
from pyscf.tools import cubegen

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUBES_DIR = os.path.join(SCRIPT_DIR, "cubes")
os.makedirs(CUBES_DIR, exist_ok=True)

# 1. Geometriat:
# Formaldehydi XY-tasossa, C origossa, O +X-akselilla
H2CO_GEOM = """
C  0.0000  0.0000  0.0000
O  1.2100  0.0000  0.0000
H -0.5800  0.9300  0.0000
H -0.5800 -0.9300  0.0000
"""

# Nukleofiili CN- lähestyy Bürgi-Dunitz -kulmassa (~107° suhteessa C=O-sidokseen)
# d(C_carb - C_cn) = 2.05 A, d(C_cn - N) = 1.17 A
# C(cyanide): x = 2.05 * cos(107°) = -0.599 A, z = 2.05 * sin(107°) = +1.960 A
# N(cyanide): x = 3.22 * cos(107°) = -0.941 A, z = 3.22 * sin(107°) = +3.079 A
CN_GEOM = """
C -0.5990  0.0000  1.9600
N -0.9410  0.0000  3.0790
"""

COMPLEX_GEOM = H2CO_GEOM.strip() + "\n" + CN_GEOM.strip()


def run_calculations():
    print("1. Lasketaan erillinen formaldehydi (H2CO)...")
    mol_h2co = gto.M(atom=H2CO_GEOM, basis="def2-svp")
    mf_h2co = scf.RHF(mol_h2co).run(verbose=0)
    lumo_idx_h2co = mol_h2co.nelectron // 2
    coeff_lumo_h2co = mf_h2co.mo_coeff[:, lumo_idx_h2co]

    print("2. Lasketaan erillinen syanidi-ioni (CN-)...")
    mol_cn = gto.M(atom=CN_GEOM, charge=-1, spin=0, basis="def2-svp")
    mf_cn = scf.RHF(mol_cn).run(verbose=0)
    homo_idx_cn = mol_cn.nelectron // 2 - 1
    coeff_homo_cn = mf_cn.mo_coeff[:, homo_idx_cn]

    print("3. Lasketaan vuorovaikuttanut kompleksi (H2CO ... CN-)...")
    mol_comp = gto.M(atom=COMPLEX_GEOM, charge=-1, spin=0, basis="def2-svp")
    mf_comp = scf.RHF(mol_comp).run(verbose=0)
    nocc_comp = mol_comp.nelectron // 2

    # Luodaan yhteinen laatikko (Cube 46x46x46, margin=5.5 Bohr)
    print("4. Määritetään yhteinen 3D-hila...")
    cc_comp = cubegen.Cube(mol_comp, nx=46, ny=46, nz=46, margin=5.5)
    coords = cc_comp.get_coords()

    # Arvioidaan H2CO LUMO yhteisellä hilalla
    ao_h2co = mol_h2co.eval_gto("GTOval", coords)
    val_h2co = np.dot(ao_h2co, coeff_lumo_h2co)

    # Arvioidaan CN- HOMO yhteisellä hilalla
    ao_cn = mol_cn.eval_gto("GTOval", coords)
    val_cn = np.dot(ao_cn, coeff_homo_cn)

    # Varmistetaan vaihesopivuus hyökkäysvyöhykkeellä (rakentava kohtaaminen)
    # C(carb) ja C(cn) välinen keskipiste:
    midpoint = np.array([[-0.2995, 0.0, 0.9800]])
    v_mid_h2co = np.dot(mol_h2co.eval_gto("GTOval", midpoint), coeff_lumo_h2co)[0]
    v_mid_cn = np.dot(mol_cn.eval_gto("GTOval", midpoint), coeff_homo_cn)[0]

    if v_mid_h2co < 0:
        val_h2co = -val_h2co
        v_mid_h2co = -v_mid_h2co
    if v_mid_cn < 0:
        val_cn = -val_cn
        v_mid_cn = -v_mid_cn

    print(f"Vaiheet hyökkäysvälissä: H2CO LUMO = {v_mid_h2co:+.3f}, CN- HOMO = {v_mid_cn:+.3f} (rakentava interferenssi!)")

    # Kompleksin vuorovaikuttaneet orbitaalit
    ao_comp = mol_comp.eval_gto("GTOval", coords)

    # MO 9 on muodostuva sigma(C-C)-sidos
    val_bonding = np.dot(ao_comp, mf_comp.mo_coeff[:, 9])
    if np.dot(mol_comp.eval_gto("GTOval", midpoint), mf_comp.mo_coeff[:, 9])[0] < 0:
        val_bonding = -val_bonding

    # MO 14 on HOMO (elektronitiheyden siirtyminen hapelle)
    val_homo = np.dot(ao_comp, mf_comp.mo_coeff[:, 14])

    # MO 16 on hajoittava sigma*(C-C)-orbitaali
    val_antibonding = np.dot(ao_comp, mf_comp.mo_coeff[:, 16])

    print("5. Kirjoitetaan .cube-tiedostot...")

    # A. Karbonyyli yksin (vain H2CO atomit)
    cc_h2co = cubegen.Cube(mol_h2co, nx=cc_comp.nx, ny=cc_comp.ny, nz=cc_comp.nz)
    cc_h2co.box = cc_comp.box.copy()
    cc_h2co.boxorig = cc_comp.boxorig.copy()
    cc_h2co.xs = cc_comp.xs.copy()
    cc_h2co.ys = cc_comp.ys.copy()
    cc_h2co.zs = cc_comp.zs.copy()
    f_carb = os.path.join(CUBES_DIR, "addition_carbonyl_lumo.cube")
    cc_h2co.write(val_h2co.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_carb, comment="H2CO LUMO (pi*)")
    print(f"  -> {f_carb}")

    # B. Syanidi yksin (vain CN- atomit)
    cc_cn = cubegen.Cube(mol_cn, nx=cc_comp.nx, ny=cc_comp.ny, nz=cc_comp.nz)
    cc_cn.box = cc_comp.box.copy()
    cc_cn.boxorig = cc_comp.boxorig.copy()
    cc_cn.xs = cc_comp.xs.copy()
    cc_cn.ys = cc_comp.ys.copy()
    cc_cn.zs = cc_comp.zs.copy()
    f_cn = os.path.join(CUBES_DIR, "addition_nucleophile_homo.cube")
    cc_cn.write(val_cn.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_cn, comment="CN- HOMO (C sp lone pair)")
    print(f"  -> {f_cn}")

    # C. Karbonyylin LUMO hyökkäysasennossa (kaikki 6 atomia)
    f_carb_comp = os.path.join(CUBES_DIR, "addition_carb_in_comp.cube")
    cc_comp.write(val_h2co.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_carb_comp, comment="H2CO LUMO in complex geometry")
    print(f"  -> {f_carb_comp}")

    # D. Syanidin HOMO hyökkäysasennossa (kaikki 6 atomia)
    f_nuc_comp = os.path.join(CUBES_DIR, "addition_nuc_in_comp.cube")
    cc_comp.write(val_cn.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_nuc_comp, comment="CN- HOMO in complex geometry")
    print(f"  -> {f_nuc_comp}")

    # E. Vuorovaikuttanut kompleksi: Muodostunut sigma(C-C)-sidos
    f_bonding = os.path.join(CUBES_DIR, "addition_hybridized_bonding.cube")
    cc_comp.write(val_bonding.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_bonding, comment="H2CO...CN- forming sigma(C-C)")
    print(f"  -> {f_bonding}")

    # F. Vuorovaikuttanut kompleksi: HOMO
    f_homo = os.path.join(CUBES_DIR, "addition_hybridized_homo.cube")
    cc_comp.write(val_homo.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_homo, comment="H2CO...CN- complex HOMO")
    print(f"  -> {f_homo}")

    # G. Vuorovaikuttanut kompleksi: Hajottava sigma*(C-C)-orbitaali
    f_anti = os.path.join(CUBES_DIR, "addition_hybridized_antibonding.cube")
    cc_comp.write(val_antibonding.reshape(cc_comp.nx, cc_comp.ny, cc_comp.nz), f_anti, comment="H2CO...CN- complex sigma*(C-C)")
    print(f"  -> {f_anti}")

    print("Kaikki kuutiot luotu onnistuneesti!")


if __name__ == "__main__":
    run_calculations()
