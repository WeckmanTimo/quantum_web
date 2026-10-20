"""
Comp-Chem-Lab: Pääsovellus ja Välilehtinäkymä
=============================================
1. Välilehti: Käänteinen vesikaasuntasausreaktio (RWGS, kvanttikemiallinen malli)
2. Välilehti: Voimakenttä & Molekyylimekaniikka (Butaani & Propyylibentseeni)
3. Välilehti: Molekyyliorbitaalit (HOMO / LUMO 3D & LCAO)
4. Välilehti: Nukleofiili-additio (LUMO 3D)
"""
from __future__ import annotations
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import ipywidgets as widgets
from IPython.display import display

import rwgs_studio
import mm_studio
import benzene_studio
import orbital_studio
import nucleophilic_addition_studio

# Voimakenttä- ja molekyylimekaniikkastudio (Butaani ja Propyylibentseeni yhdessä välilehdessä)
ff_tabs = widgets.Tab(children=[
    mm_studio.create_mm_studio(),
    benzene_studio.create_benzene_studio()
], layout=widgets.Layout(width="100%"))

ff_tabs.set_title(0, "⚙️ Butaani (C₄H₁₀)")
ff_tabs.set_title(1, "⬡ Propyylibentseeni (C₉H₁₂)")

# Luodaan päävälilehdet laboratorioille
main_tabs = widgets.Tab(children=[
    rwgs_studio.dashboard,
    ff_tabs,
    orbital_studio.create_orbital_studio(),
    nucleophilic_addition_studio.create_nucleophilic_addition_studio()
])

main_tabs.set_title(0, "🧪 RWGS-reaktio")
main_tabs.set_title(1, "⚙️ Molekyylimekaniikka (MM)")
main_tabs.set_title(2, "⚛️ Molekyyliorbitaalit (HOMO / LUMO 3D)")
main_tabs.set_title(3, "🎯 Nukleofiili-additio")

app_container = widgets.VBox([
    main_tabs
], layout=widgets.Layout(width="100%", max_width="1320px", margin="0 auto"))

if __name__ == "__main__":
    display(app_container)
