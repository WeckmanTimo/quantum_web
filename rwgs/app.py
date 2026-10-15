"""
Comp-Chem-Lab: Pääsovellus ja Välilehtinäkymä
=============================================
1. Välilehti: Käänteinen vesikaasuntasausreaktio (RWGS, kvanttikemiallinen malli)
2. Välilehti: Butaanin voimakenttä ja 3D-hiiriohjaus (Molekyylimekaniikka)
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

# Luodaan välilehdet laboratorioille
main_tabs = widgets.Tab(children=[
    rwgs_studio.dashboard,
    mm_studio.create_mm_studio(),
    benzene_studio.create_benzene_studio(),
    orbital_studio.create_orbital_studio(),
    nucleophilic_addition_studio.create_nucleophilic_addition_studio()
])

main_tabs.set_title(0, "🧪 RWGS-reaktiostudio (Kvanttikemia)")
main_tabs.set_title(1, "⚙️ Voimakenttä (Butaani)")
main_tabs.set_title(2, "⬡ Bentseenirengas & ketju (Propyylibentseeni)")
main_tabs.set_title(3, "⚛️ Molekyyliorbitaalit (HOMO / LUMO 3D)")
main_tabs.set_title(4, "🎯 Nukleofiili-additio (LUMO 3D)")

app_container = widgets.VBox([
    main_tabs
], layout=widgets.Layout(width="100%", max_width="1320px", margin="0 auto"))

if __name__ == "__main__":
    display(app_container)
