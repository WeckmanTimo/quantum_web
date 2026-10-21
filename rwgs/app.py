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
main_tabs.add_class("main-nav-tabs")

main_tabs.set_title(0, "🧪 RWGS-reaktio")
main_tabs.set_title(1, "⚙️ Molekyylimekaniikka (MM)")
main_tabs.set_title(2, "⚛️ Molekyyliorbitaalit (MO)")
main_tabs.set_title(3, "🎯 Nukleofiili-additio")

nav_style = widgets.HTML("""
<style>
  /* Päävalikon välilehtien tyylitys: kaikki 4 välilehteä samalle riville */
  .main-nav-tabs > .lm-TabBar,
  .main-nav-tabs > .p-TabBar,
  .main-nav-tabs .widget-tab-bar,
  .main-nav-tabs > ul.nav-tabs {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    height: auto !important;
    min-height: 54px !important;
    border-bottom: 3px solid #cbd5e1 !important;
    margin-bottom: 10px !important;
  }

  .main-nav-tabs > .lm-TabBar > .lm-TabBar-content,
  .main-nav-tabs > .p-TabBar > .p-TabBar-content {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 6px !important;
    width: 100% !important;
  }

  .main-nav-tabs > .lm-TabBar .lm-TabBar-tab,
  .main-nav-tabs > .p-TabBar .p-TabBar-tab,
  .main-nav-tabs .widget-tab-bar .lm-TabBar-tab,
  .main-nav-tabs .widget-tab-bar .p-TabBar-tab,
  .main-nav-tabs > ul.nav-tabs > li,
  .main-nav-tabs > ul.nav-tabs > li > a {
    min-height: 52px !important;
    height: auto !important;
    padding: 12px 14px !important;
    min-width: 0 !important;
    flex: 1 1 0px !important;
    font-size: 17.5px !important;
    font-size: clamp(14px, 1.25vw, 18px) !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.15s ease-in-out !important;
    letter-spacing: -0.01em !important;
    box-sizing: border-box !important;
  }

  .main-nav-tabs .lm-TabBar-tabLabel,
  .main-nav-tabs .p-TabBar-tabLabel,
  .main-nav-tabs > ul.nav-tabs > li > a {
    font-size: 17.5px !important;
    font-size: clamp(14px, 1.25vw, 18px) !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }

  /* Aktiivisen välilehden erottuva teema */
  .main-nav-tabs > .lm-TabBar .lm-TabBar-tab.lm-mod-current,
  .main-nav-tabs > .p-TabBar .p-TabBar-tab.p-mod-current,
  .main-nav-tabs > ul.nav-tabs > li.active > a {
    border-bottom: 4px solid #2563eb !important;
    color: #1d4ed8 !important;
    background-color: #eff6ff !important;
  }

  .main-nav-tabs > .lm-TabBar .lm-TabBar-tab:hover,
  .main-nav-tabs > .p-TabBar .p-TabBar-tab:hover,
  .main-nav-tabs > ul.nav-tabs > li > a:hover {
    background-color: #f8fafc !important;
  }
</style>
""")

app_container = widgets.VBox([
    nav_style,
    main_tabs
], layout=widgets.Layout(width="100%", max_width="1320px", margin="0 auto"))

if __name__ == "__main__":
    display(app_container)

