"""
Comp-Chem-Lab: Käänteisen vesikaasuntasausreaktion studio (RWGS)
================================================================================
Interaktiivinen laskennallisen kemian oppimisympäristö lukio-opetukseen:
CO2 + H2 ⇌ CO + H2O

Kaikki molekyylit ja reaaliaikainen Hessin lain reaktiotasapaino yhdellä sivulla.
Potentiaalienergiat perustuvat kvanttikemialliseen laskennalliseen malliin (wB97M-V).
"""

from __future__ import annotations
import html
import os
import sys
import time
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import py3Dmol
from scipy.interpolate import PchipInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import dft_database as dft_db

EXPERIMENTAL_DH = 41.2    # kokeellinen RWGS ΔH°(298 K) [kJ/mol]
LABELS = {"H2": "H₂", "CO2": "CO₂", "CO": "CO", "H2O": "H₂O"}
ROLES_FI = {"H2": "Lähtöaine", "CO2": "Lähtöaine", "CO": "Tuote", "H2O": "Tuote"}


# ==============================================================================
# 1. 3D-molekyylivisualisointi (py3Dmol upotettuna iframe srcdoc -rakenteeseen)
# ==============================================================================
def render_3dmol_iframe(xyz: str, width: int = 240, height: int = 155) -> str:
    """Palauttaa iframe-elementin py3Dmol-näkymällä varustettuna (lähietäisyydeltä)."""
    try:
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz, "xyz")
        view.setStyle({}, {"stick": {"radius": 0.20, "color": "#CBD5E0"}})
        view.addStyle({"elem": "H"}, {"sphere": {"radius": 0.58, "color": "#FFFFFF"}})
        view.addStyle({"elem": "C"}, {"sphere": {"radius": 0.78, "color": "#4A5568"}})
        view.addStyle({"elem": "O"}, {"sphere": {"radius": 0.74, "color": "#E53E3E"}})
        view.zoomTo()
        view.zoom(2.2)
        raw_html = view._make_html()
        escaped_html = html.escape(raw_html, quote=True)
        return (f'<iframe srcdoc="{escaped_html}" '
                f'style="width:{width+2}px; height:{height+2}px; border:1px solid #CBD5E0; border-radius:6px; background:#FFFFFF;" '
                f'frameborder="0"></iframe>')
    except Exception as e:
        return f"<div style='color:#718096;padding:4px;'><i>3D-näkymä ei saatavilla: {e}</i></div>"


def xyz_h2(r: float) -> str:
    return f"2\nH2\nH {-r/2:.4f} 0.0 0.0\nH {r/2:.4f} 0.0 0.0"


def xyz_co2(r: float) -> str:
    return f"3\nCO2\nC 0.0 0.0 0.0\nO {-r:.4f} 0.0 0.0\nO {r:.4f} 0.0 0.0"


def xyz_co(r: float) -> str:
    return f"2\nCO\nC {-r/2:.4f} 0.0 0.0\nO {r/2:.4f} 0.0 0.0"


def xyz_h2o(r: float, theta_deg: float) -> str:
    half = np.radians(theta_deg / 2.0)
    x = r * np.sin(half)
    y = r * np.cos(half)
    return f"3\nH2O\nO 0.0 {-y/3:.4f} 0.0\nH {-x:.4f} {2*y/3:.4f} 0.0\nH {x:.4f} {2*y/3:.4f} 0.0"


# ==============================================================================
# 2. Nykyiset geometriat ja reaaliaikainen Hessin lain reaktiotasapaino
# ==============================================================================
current_geoms: Dict[str, str] = {
    "H2": "r = 4.00 Å",
    "CO2": "r = 4.00 Å",
    "CO": "r = 4.00 Å",
    "H2O": "r = 1.50 Å, ∠ = 90°"
}
current_energies: Dict[str, float] = {
    "H2": dft_db.calc_1d("H2", 4.00)[1],
    "CO2": dft_db.calc_1d("CO2", 4.00)[1],
    "CO": dft_db.calc_1d("CO", 4.00)[1],
    "H2O": dft_db.calc_h2o(1.50, 90.0)[1]
}

summary_table = widgets.HTML()
rxn_analysis = widgets.HTML()

fig_rwgs = go.FigureWidget(
    go.Waterfall(
        x=["Lähtöaineet<br>(CO₂ + H₂)", "Sidoskatkokset<br>(+ΔE_sidos)", "Tuotesidosten muodostus<br>(−ΔE_sidos)", "Tuotteet<br>(CO + H₂O)"],
        measure=["absolute", "relative", "relative", "total"],
        y=[0.0, 0.0, 0.0, 0.0],
        text=["", "", "", ""],
        textposition="outside",
        increasing=dict(marker=dict(color="#DD6B20")),
        decreasing=dict(marker=dict(color="#2B6CB0")),
        totals=dict(marker=dict(color="#2D3748")),
        connector=dict(line=dict(color="#A0AEC0")),
    )
)
fig_rwgs.update_layout(
    title="Hessin lain termodynaaminen sykli ja reaktioentalpia",
    yaxis_title="Energia [kJ/mol] (suhteessa erillisiin atomeihin)",
    width=520, height=360,
    margin=dict(l=65, r=25, t=50, b=60),
    showlegend=False,
)

# Erillinen "Oikeat tulokset" -laatikko lukio-opiskelijoille
is_results_unlocked: bool = False
calculated_molecules: set[str] = set()
results_status_badge = widgets.HTML()
results_content = widgets.HTML()
results_accordion = widgets.Accordion(children=[results_content], selected_index=None)


def _guard_accordion(change):
    global is_results_unlocked
    if not is_results_unlocked and change["new"] is not None:
        results_accordion.selected_index = None

results_accordion.observe(_guard_accordion, names="selected_index")


def _render_correct_results_html() -> str:
    """Palauttaa selkeän, lukio-opiskelijalle suunnatun tulosyhteenvedon ja vertailun kokeelliseen arvoon."""
    return f"""
    <div style="background:#FFFFFF; padding:12px 14px; border-radius:6px; font-size:12.5px; line-height:1.5; color:#2D3748;">
      <h4 style="margin:0 0 8px 0; font-size:14px; color:#2B6CB0;">
        📊 Laskennalliset tulokset ja vertailu kokeelliseen arvoon (298.15 K)
      </h4>
      <p style="margin:0 0 10px 0; color:#4A5568;">
        Kvanttikemiallinen laskennallinen malli ratkaisee molekyylien elektronirakenteen ja kokonaisenergian atomien eri etäisyyksillä.
        Tasapainotilassa molekyylillä on alin mahdollinen energia, jolloin sidospituudet ja sidoskulmat asettuvat luonnollisiin arvoihinsa.
      </p>

      <div style="background:#EDF2F7; border:1px solid #CBD5E0; border-radius:6px; padding:10px 12px; margin-bottom:12px;">
        <h5 style="margin:0 0 6px 0; font-size:13px; color:#2D3748;">Molekyylien laskennalliset tasapainogeometriat ja sidosenergiat:</h5>
        <table style="width:100%; border-collapse:collapse; font-size:12px; background:white; border-radius:4px; overflow:hidden;">
          <thead>
            <tr style="background:#E2E8F0; color:#4A5568;">
              <th style="padding:6px 10px; text-align:left;">Molekyyli</th>
              <th style="padding:6px 10px; text-align:center;">Laskennallinen geometria</th>
              <th style="padding:6px 10px; text-align:center;">Kokeellinen geometria</th>
              <th style="padding:6px 10px; text-align:right;">Sidosenergia (E_sidos)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7;"><b>Vety (H₂)</b></td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">r = 0.74 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">0.74 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:right; font-weight:600; color:#2B6CB0;">−454.2 kJ/mol</td>
            </tr>
            <tr>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7;"><b>Hiilidioksidi (CO₂)</b></td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">r = 1.15 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">1.16 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:right; font-weight:600; color:#2B6CB0;">−1634.2 kJ/mol</td>
            </tr>
            <tr>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7;"><b>Hiilimonoksidi (CO)</b></td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">r = 1.13 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">1.13 Å</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:right; font-weight:600; color:#2B6CB0;">−1090.0 kJ/mol</td>
            </tr>
            <tr>
              <td style="padding:5px 10px;"><b>Vesi (H₂O)</b></td>
              <td style="padding:5px 10px; text-align:center;">r = 0.96 Å, ∠ = 104.9°</td>
              <td style="padding:5px 10px; text-align:center;">0.96 Å, 104.5°</td>
              <td style="padding:5px 10px; text-align:right; font-weight:600; color:#2B6CB0;">−969.3 kJ/mol</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="background:#F7FAFC; border:1px solid #CBD5E0; border-radius:6px; padding:10px 12px; margin-bottom:12px;">
        <h5 style="margin:0 0 6px 0; font-size:13px; color:#2D3748;">Reaktioentalpia ja kokeellinen tarkkuus:</h5>
        <table style="width:100%; border-collapse:collapse; font-size:12px; background:white; border-radius:4px; overflow:hidden;">
          <thead>
            <tr style="background:#E2E8F0; color:#4A5568;">
              <th style="padding:6px 10px; text-align:left;">Suure / Malli</th>
              <th style="padding:6px 10px; text-align:center;">Laskennallinen tulos</th>
              <th style="padding:6px 10px; text-align:center;">Kokeellinen arvo</th>
              <th style="padding:6px 10px; text-align:right;">Erotus</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7;">Elektroninen reaktioenergia (ΔE)</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center; font-weight:600;">+29.1 kJ/mol</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">—</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:right;">—</td>
            </tr>
            <tr>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7;">Värähdys- ja lämpötilakorjaus (298.15 K)</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center; font-weight:600;">+12.9 kJ/mol</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:center;">—</td>
              <td style="padding:5px 10px; border-bottom:1px solid #EDF2F7; text-align:right;">—</td>
            </tr>
            <tr style="background:#EBF8FF; font-weight:700;">
              <td style="padding:6px 10px; color:#2B6CB0;">Reaktioentalpia ΔH°(298.15 K)</td>
              <td style="padding:6px 10px; text-align:center; color:#2B6CB0; font-size:13px;">+42.0 kJ/mol</td>
              <td style="padding:6px 10px; text-align:center; color:#B7791F; font-size:13px;">+{EXPERIMENTAL_DH:.1f} kJ/mol</td>
              <td style="padding:6px 10px; text-align:right; color:#276749; font-size:13px;">+0.8 kJ/mol</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div style="background:#EDF2F7; border-left:4px solid #276749; padding:8px 12px; border-radius:4px; font-size:12px; line-height:1.5;">
        <b>Kemiallinen johtopäätös lukiolaiselle:</b><br>
        • <b>Endoterminen reaktio:</b> Koska reaktioentalpia on positiivinen (ΔH° &gt; 0), reaktio vaatii lämpöenergiaa tapahtuakseen (lähtöaineiden vahvojen sidosten rikkominen vaatii enemmän energiaa kuin tuotteiden muodostumisessa vapautuu).<br>
        • <b>Hämmästyttävä tarkkuus:</b> Laskennallinen malli ennustaa reaktioentalpian <b>+42.0 kJ/mol</b>, mikä osuu alle 1 kJ/mol päähän laboratoriossa mitatusta arvosta <b>+{EXPERIMENTAL_DH:.1f} kJ/mol</b>. Kvanttimekaniikan ja tietokonesimulaatioiden avulla kemiallisten reaktioiden energiat voidaan siis ennustaa erittäin tarkasti jo ennen kokeiden tekemistä!
      </div>
    </div>
    """


def _render_ledger_table() -> str:
    cell = "padding:6px 10px; border-bottom:1px solid #E2E8F0; text-align:left;"
    cell_r = "padding:6px 10px; border-bottom:1px solid #E2E8F0; text-align:right;"
    rows = ""
    for key in ("H2", "CO2", "CO", "H2O"):
        e = current_energies[key]
        geom = current_geoms[key]
        rows += (f"<tr><td style='{cell}'><b>{LABELS[key]}</b></td>"
                 f"<td style='{cell}'>{ROLES_FI[key]}</td>"
                 f"<td style='{cell}'>{geom}</td>"
                 f"<td style='{cell_r}'><b>{e:.1f}</b></td></tr>")
    head = "".join(f"<th style='padding:6px 10px; background:#EDF2F7; font-size:12px; text-transform:uppercase; color:#4A5568;'>{h}</th>"
                   for h in ("Molekyyli", "Rooli", "Nykyinen geometria", "E_sidos [kJ/mol]"))
    return (f"<table style='border-collapse:collapse; width:100%; font-size:13px; border:1px solid #E2E8F0; border-radius:6px; overflow:hidden;'>"
            f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")


def update_reaction_balance() -> None:
    global is_results_unlocked
    summary_table.value = _render_ledger_table()

    e_r = current_energies["CO2"] + current_energies["H2"]
    e_p = current_energies["CO"] + current_energies["H2O"]
    d_e = e_p - e_r

    with fig_rwgs.batch_update():
        fig_rwgs.data[0].y = [e_r, -e_r, e_p, 0.0]
        fig_rwgs.data[0].text = [f"{e_r:.1f}", f"+{-e_r:.1f}", f"{e_p:.1f}", f"{e_p:.1f}"]
        fig_rwgs.layout.title = f"Hessin lain sykli: ΔE_reaktio = {d_e:+.1f} kJ/mol"

    color = "#276749" if d_e > 0 else "#C53030"
    rxn_type = "Endoterminen reaktio (ΔE > 0)" if d_e > 0 else "Eksoterminen reaktio (ΔE < 0)"
    explanation = (
        "Koska lähtöaineiden sidosten rikkominen vaatii enemmän energiaa kuin tuotteiden muodostuminen vapauttaa, reaktio on nykyisillä geometrioilla <b>endoterminen</b> (ΔE &gt; 0)."
        if d_e > 0 else
        "Nykyisillä venytetyillä tai vääristyneillä geometrioilla reaktio näyttäytyy <b>eksotermisenä</b> (ΔE &lt; 0). Etsi molekyylien tasapainogeometriat siirtymällä minimiin."
    )

    rxn_analysis.value = f"""
    <div style="background:#F7FAFC; border:1px solid #CBD5E0; border-left:4px solid {color}; padding:12px 14px; border-radius:6px; margin-top:10px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span style="font-size:13px; color:#4A5568;">Lähtöaineet (CO₂ + H₂): <b>{e_r:.1f} kJ/mol</b> &nbsp;|&nbsp; Tuotteet (CO + H₂O): <b>{e_p:.1f} kJ/mol</b></span>
          <h3 style="color:{color}; margin:4px 0 2px 0;">ΔE_reaktio = {d_e:+.1f} kJ/mol &nbsp; ({rxn_type})</h3>
        </div>
      </div>
      <p style="font-size:12.5px; line-height:1.45; color:#4A5568; margin:6px 0 0 0;">
        <b>Termodynaaminen tarkastelu:</b> Lähtöaineiden kaikkien sidosten rikkominen vapaiksi erillisiksi atomeiksi vaatii <b>+{-e_r:.1f} kJ/mol</b>.
        Tuotteiden sidosten muodostuminen vapauttaa <b>{e_p:.1f} kJ/mol</b>. {explanation}
      </p>
    </div>
    """

    # Tarkistetaan ehto oikeiden tulosten avaamiselle:
    # 1. Käyttäjä on laskenut vähintään kerran kaikki 4 molekyyliä
    # 2. Endoterminen tulos: d_e > 0
    # 3. Enintään 40 kJ/mol päässä oikeasta tasapainoenergiasta: abs(d_e - d_e_target) <= 40.0
    d_e_target = dft_db.RWGS_DELTA_E  # +29.13 kJ/mol
    diff = abs(d_e - d_e_target)
    has_all_calculated = len(calculated_molecules) >= 4
    new_unlocked = has_all_calculated and (d_e > 0) and (diff <= 40.0)

    if not new_unlocked:
        is_results_unlocked = False
        results_accordion.selected_index = None
        if not has_all_calculated:
            missing = [LABELS[m] for m in ("H2", "CO2", "CO", "H2O") if m not in calculated_molecules]
            hint = f"Laske vielä puuttuvien molekyylien energiat: {', '.join(missing)} ({len(calculated_molecules)}/4 laskettu)"
        elif d_e <= 0:
            hint = "Reaktion oltava endoterminen (ΔE > 0)"
        else:
            hint = f"Vaatii vielä {diff - 40.0:.1f} kJ/mol lähemmäs oikeaa tasapainoa"

        results_status_badge.value = f"""
        <div style="background:#FFF5F5; border:1px solid #FEB2B2; border-left:4px solid #E53E3E; padding:10px 14px; border-radius:6px; margin-top:12px; margin-bottom:6px;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
              <span style="font-weight:700; color:#C53030; font-size:13.5px;">🔒 Oikeat laskennalliset tulokset ovat lukittuina</span>
              <p style="margin:4px 0 0 0; font-size:12px; color:#4A5568; line-height:1.4;">
                Saat oikeat laskennalliset tulokset ja kokeellisen vertailun näkyviin, kun löydät molekyyleille sellaiset geometriat,
                joilla reaktio on <b>endoterminen (ΔE > 0)</b> ja <b>enintään 40 kJ/mol päässä oikeasta tasapainotuloksesta (+29.1 kJ/mol)</b>.<br>
                <i>Nykyinen tila: <b>ΔE = {d_e:+.1f} kJ/mol</b> &nbsp;|&nbsp; Etäisyys tavoitteesta: <b>{diff:.1f} kJ/mol</b> &bull; <b>{hint}</b>.</i>
              </p>
            </div>
          </div>
        </div>
        """
        results_accordion.set_title(0, f"🔒 Oikeat laskennalliset tulokset & kokeellinen vertailu (Lukittu: ΔE = {d_e:+.1f} kJ/mol)")
        results_content.value = f"""
        <div style="padding:15px; text-align:center; color:#718096; font-size:13px;">
          <i>🔒 Tämä osio on lukittu. Säädä molekyylien geometrioita niin, että reaktio on endoterminen (ΔE > 0) ja enintään 40 kJ/mol päässä oikeasta tasapainoenergiasta (+29.1 kJ/mol).</i>
        </div>
        """
    else:
        was_locked = not is_results_unlocked
        is_results_unlocked = True
        if was_locked:
            results_accordion.selected_index = None

        results_status_badge.value = f"""
        <div style="background:#F0FFF4; border:1px solid #9AE6B4; border-left:4px solid #38A169; padding:10px 14px; border-radius:6px; margin-top:12px; margin-bottom:6px;">
          <span style="font-weight:700; color:#276749; font-size:13.5px;">🔓 Hienoa! Oikeat laskennalliset tulokset ovat nyt avattavissa</span>
          <p style="margin:4px 0 0 0; font-size:12px; color:#4A5568; line-height:1.4;">
            Laskemasi reaktioenergia (<b>ΔE = {d_e:+.1f} kJ/mol</b>) on endoterminen ja alle 40 kJ/mol päässä oikeasta tasapainoenergiasta (+29.1 kJ/mol).<br>
            Voit nyt avata alla olevan laatikon klikkaamalla nähdäksesi laskennallisen mallin tulokset, lämpötilakorjaukset ja kokeellisen vertailun.
          </p>
        </div>
        """

        results_accordion.set_title(0, "🔓 Oikeat laskennalliset tulokset & kokeellinen vertailu (Klikkaa avataksesi)")
        results_content.value = _render_correct_results_html()


# ==============================================================================
# 3. Taulukkomuotoilija (näyttää kaikki pisteet selattavassa laatikossa, ilman voimia)
# ==============================================================================
def _format_table(records: List[Dict]) -> str:
    if not records:
        return "<div style='color:#A0AEC0; font-style:italic; font-size:12px; padding:6px 0;'>Ei vielä laskettuja pisteitä. Säädä arvoa ja paina 'Laske energia'.</div>"

    rows = ""
    for idx, rec in enumerate(reversed(records)):
        is_latest = (idx == 0)
        bg = "#FEFCBF" if is_latest else ("#F7FAFC" if idx % 2 == 1 else "#FFFFFF")
        badge = " <span style='background:#DD6B20; color:white; font-size:9.5px; font-weight:700; padding:1px 5px; border-radius:4px;'>UUSIN</span>" if is_latest else ""

        cells = ""
        for k, v in rec.items():
            if k == "Piste #":
                cells += f"<td style='padding:4px 6px; text-align:center; font-weight:{'700' if is_latest else '400'};'>{v}{badge}</td>"
            elif "E_sidos" in k:
                cells += f"<td style='padding:4px 6px; text-align:center; font-weight:600; color:#2B6CB0;'>{v}</td>"
            else:
                cells += f"<td style='padding:4px 6px; text-align:center;'>{v}</td>"

        rows += f"<tr style='background:{bg};'>{cells}</tr>"

    headers = "".join(f"<th style='background:#EDF2F7; color:#4A5568; padding:5px 6px; font-weight:600; text-align:center; border:1px solid #CBD5E0; font-size:11px; position:sticky; top:0; z-index:1;'>{k}</th>" for k in records[0].keys())

    return f"""
    <div style="font-size:11.5px; margin-top:4px;">
      <div style="font-weight:600; color:#4A5568; font-size:11px; margin-bottom:2px; display:flex; justify-content:space-between;">
        <span>Mitatut pisteet ({len(records)} kpl)</span>
        <span style="color:#718096; font-size:10px;">(Uusin aina ylimpänä)</span>
      </div>
      <div style="max-height:135px; overflow-y:auto; border:1px solid #CBD5E0; border-radius:4px; box-shadow:inset 0 1px 2px rgba(0,0,0,0.04);">
        <table style="width:100%; border-collapse:collapse; font-size:11px;">
          <thead><tr>{headers}</tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
    """


# ==============================================================================
# 4. Epälineaarinen liukusäädinkuvaus ja 1D-molekyylikortit (H2, CO, CO2)
# ==============================================================================
def build_nonlinear_mapper(r_min: float, r_max: float, r_eq: float, strength: float = 0.5) -> Tuple[PchipInterpolator, PchipInterpolator]:
    """
    Muodostaa pehmeästi epälineaarisen (PCHIP) kuvauksen liukusäätimen normeeratun
    sijainnin u in [0, 1] ja fysikaalisen suureen r in [r_min, r_max] välille.
    Tasapainopisteen r_eq ympäristössä on maltillisesti tiheämpi resoluutio.
    """
    u_eq_lin = (r_eq - r_min) / (r_max - r_min)
    u_eq_target = u_eq_lin + (0.30 - u_eq_lin) * strength
    avg_slope = (r_max - r_min)
    slope_near_eq = avg_slope * (1.0 - 0.45 * strength)

    u_nodes = np.array([0.0, max(0.01, u_eq_target - 0.15), u_eq_target, min(0.99, u_eq_target + 0.20), 1.0])
    r_nodes = np.array([
        r_min,
        max(r_min + 0.05, r_eq - 0.15 * slope_near_eq),
        r_eq,
        min(r_max - 0.1, r_eq + 0.20 * slope_near_eq),
        r_max
    ])
    for i in range(1, len(r_nodes)):
        if r_nodes[i] <= r_nodes[i-1]:
            r_nodes[i] = r_nodes[i-1] + 0.01

    u_to_r = PchipInterpolator(u_nodes, r_nodes)
    r_to_u = PchipInterpolator(r_nodes, u_nodes)
    return u_to_r, r_to_u


def make_1d_card(
    key: str,
    title: str,
    role_badge: str,
    slider_label: str,
    xyz_fn: Callable[[float], str],
    r_range: Tuple[float, float],
    default_r: float,
    y_range: Tuple[float, float],
) -> widgets.VBox:
    history_records: List[Dict] = []
    r_eq = dft_db.EQUILIBRIUM[key]["r"]
    u_to_r, r_to_u = build_nonlinear_mapper(r_range[0], r_range[1], r_eq)

    slider = widgets.FloatSlider(
        value=float(np.clip(r_to_u(default_r), 0.0, 1.0)),
        min=0.0, max=1.0, step=0.001,
        readout=False, continuous_update=False,
        layout=widgets.Layout(width="180px")
    )
    num_input = widgets.BoundedFloatText(
        value=round(default_r, 2), min=r_range[0], max=r_range[1], step=0.01,
        layout=widgets.Layout(width="75px")
    )

    is_syncing = False

    btn_calc = widgets.Button(description="Laske energia", button_style="success", icon="calculator", layout=widgets.Layout(width="135px"))
    btn_clear = widgets.Button(description="Tyhjennä pisteet", icon="trash", layout=widgets.Layout(width="125px"))

    readout = widgets.HTML()
    viewer_box = widgets.HTML()

    # Alustetaan ensimmäinen mittauspiste ääriasentoon (venytetty sidos)
    init_r = round(default_r, 2)
    init_e_tot, init_e_atom, _ = dft_db.calc_1d(key, init_r)
    last_calc = {
        "r": init_r,
        "e_tot": init_e_tot,
        "e_atom": init_e_atom
    }

    history_records.append({
        "Piste #": 1,
        "r [Å]": f"{init_r:.2f}",
        "E_tot [Eh]": f"{init_e_tot:.5f}",
        "E_sidos [kJ/mol]": f"{init_e_atom:.1f}"
    })
    table_box = widgets.HTML(value=_format_table(history_records))

    fig = go.FigureWidget(data=[
        go.Scatter(
            x=[init_r], y=[init_e_atom], mode="lines+markers", name="Mitatut pisteet",
            line=dict(color="#2B6CB0", width=2),
            marker=dict(color="#2B6CB0", size=8, symbol="circle", line=dict(color="white", width=1))
        ),
        go.Scatter(
            x=[init_r], y=[init_e_atom], mode="markers", name="Viimeisin piste",
            marker=dict(color="#DD6B20", size=13, symbol="diamond", line=dict(color="black", width=1.5))
        )
    ])
    fig.update_layout(
        title=dict(text=f"{LABELS[key]}: Potentiaalienergiakäyrä (Laskennallinen malli)", font=dict(size=12.5)),
        xaxis=dict(title=f"{slider_label} [Å]", range=[r_range[0]-0.05, r_range[1]+0.05]),
        yaxis=dict(title="E_sidos [kJ/mol]", range=[y_range[0], y_range[1]]),
        width=330, height=260,
        margin=dict(l=55, r=15, t=35, b=35),
        showlegend=False
    )

    def refresh_view(_=None) -> None:
        r = round(num_input.value, 2)
        viewer_box.value = render_3dmol_iframe(xyz_fn(r), width=230, height=140)

        if r == last_calc["r"]:
            e_tot = last_calc["e_tot"]
            e_atom = last_calc["e_atom"]
            readout.value = f"""
            <div style="font-family:sans-serif; font-size:12px; background:#F7FAFC; border:1px solid #CBD5E0; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
              <b>Geometria:</b> r = {r:.2f} Å<br>
              <b>Kokonaisenergia:</b> {e_tot:.5f} Eh<br>
              <b>Sidosenergia:</b> <span style="color:#2B6CB0;font-weight:700;">{e_atom:.1f} kJ/mol</span>
            </div>
            """
        else:
            readout.value = f"""
            <div style="font-family:sans-serif; font-size:12px; background:#FFF5F5; border:1px solid #FEB2B2; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
              <b>Asetettu r:</b> {r:.2f} Å &nbsp;<span style="background:#E53E3E; color:white; font-size:9.5px; font-weight:700; padding:1px 5px; border-radius:3px;">EI LASKETTU</span><br>
              <span style="color:#718096; font-size:11px;">Paina <b>Laske energia</b> päivittääksesi energian.</span><br>
              <b>Kokonaisenergia:</b> <span style="color:#A0AEC0;">—</span> &nbsp;|&nbsp; <b>Sidosenergia:</b> <span style="color:#A0AEC0;">—</span>
            </div>
            """

    def on_slider_change(change) -> None:
        nonlocal is_syncing
        if is_syncing:
            return
        is_syncing = True
        r = round(float(u_to_r(change["new"])), 2)
        num_input.value = r
        refresh_view()
        is_syncing = False

    def on_num_change(change) -> None:
        nonlocal is_syncing
        if is_syncing:
            return
        is_syncing = True
        r = change["new"]
        slider.value = float(np.clip(r_to_u(r), 0.0, 1.0))
        refresh_view()
        is_syncing = False

    def on_calculate_clicked(_) -> None:
        r = round(num_input.value, 2)
        btn_calc.description = "Lasketaan..."
        btn_calc.icon = "hourglass-half"
        btn_calc.disabled = True

        time.sleep(1.0)

        e_tot, e_atom, _ = dft_db.calc_1d(key, r)
        last_calc["r"] = r
        last_calc["e_tot"] = e_tot
        last_calc["e_atom"] = e_atom

        current_geoms[key] = f"r = {r:.2f} Å"
        current_energies[key] = e_atom
        calculated_molecules.add(key)
        update_reaction_balance()

        history_records.append({
            "Piste #": len(history_records) + 1,
            "r [Å]": f"{r:.2f}",
            "E_tot [Eh]": f"{e_tot:.5f}",
            "E_sidos [kJ/mol]": f"{e_atom:.1f}"
        })
        table_box.value = _format_table(history_records)

        pts = sorted([(float(rec["r [Å]"]), float(rec["E_sidos [kJ/mol]"])) for rec in history_records], key=lambda p: p[0])
        x_pts = [p[0] for p in pts]
        y_pts = [p[1] for p in pts]

        with fig.batch_update():
            fig.data[0].x = x_pts
            fig.data[0].y = y_pts
            fig.data[1].x = [r]
            fig.data[1].y = [e_atom]

        readout.value = f"""
        <div style="font-family:sans-serif; font-size:12px; background:#F7FAFC; border:1px solid #CBD5E0; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
          <b>Geometria:</b> r = {r:.2f} Å<br>
          <b>Kokonaisenergia:</b> {e_tot:.5f} Eh<br>
          <b>Sidosenergia:</b> <span style="color:#2B6CB0;font-weight:700;">{e_atom:.1f} kJ/mol</span>
        </div>
        """

        btn_calc.description = "Laske energia"
        btn_calc.icon = "calculator"
        btn_calc.disabled = False

    def on_clear_clicked(_) -> None:
        history_records.clear()
        table_box.value = _format_table([])
        with fig.batch_update():
            fig.data[0].x, fig.data[0].y = [], []
            fig.data[1].x, fig.data[1].y = [], []

    slider.observe(on_slider_change, names="value")
    num_input.observe(on_num_change, names="value")
    btn_calc.on_click(on_calculate_clicked)
    btn_clear.on_click(on_clear_clicked)

    refresh_view()

    badge_bg = "#EBF8FF" if "Lähtöaine" in role_badge else "#FEFCBF"
    badge_fg = "#2B6CB0" if "Lähtöaine" in role_badge else "#B7791F"

    card_header = widgets.HTML(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
      <h3 style="margin:0; font-size:15px; color:#2D3748;">{title}</h3>
      <span style="background:{badge_bg}; color:{badge_fg}; font-size:11px; font-weight:700; padding:2px 8px; border-radius:12px;">{role_badge}</span>
    </div>
    """)

    ctrl_row = widgets.VBox([
        widgets.HBox([widgets.Label(slider_label, layout=widgets.Layout(width="70px")), slider, num_input]),
        widgets.HBox([btn_calc, btn_clear])
    ])

    left_sub = widgets.VBox([
        ctrl_row,
        viewer_box,
        readout
    ], layout=widgets.Layout(width="280px"))

    right_sub = widgets.VBox([
        fig,
        table_box
    ], layout=widgets.Layout(width="340px"))

    card_content = widgets.HBox([left_sub, right_sub])

    return widgets.VBox([
        card_header,
        card_content
    ], layout=widgets.Layout(border="1px solid #CBD5E0", border_radius="8px", padding="10px", margin="6px", background="#FFFFFF"))


# ==============================================================================
# 5. Molekyylikortti: 2D/3D H2O (Pyöritettävä 3D-potentiaalipinta)
# ==============================================================================
def make_h2o_card() -> widgets.VBox:
    history_records: List[Dict] = []

    r_eq = dft_db.EQUILIBRIUM["H2O"]["r"]
    th_eq = dft_db.EQUILIBRIUM["H2O"]["theta"]
    u_to_r, r_to_u = build_nonlinear_mapper(0.70, 1.50, r_eq)
    u_to_th, th_to_u = build_nonlinear_mapper(75.0, 180.0, th_eq)

    s_r = widgets.FloatSlider(
        value=float(np.clip(r_to_u(1.50), 0.0, 1.0)),
        min=0.0, max=1.0, step=0.001,
        readout=False, continuous_update=False,
        layout=widgets.Layout(width="180px")
    )
    num_r = widgets.BoundedFloatText(
        value=1.50, min=0.70, max=1.50, step=0.01,
        layout=widgets.Layout(width="75px")
    )

    s_th = widgets.FloatSlider(
        value=float(np.clip(th_to_u(90.0), 0.0, 1.0)),
        min=0.0, max=1.0, step=0.001,
        readout=False, continuous_update=False,
        layout=widgets.Layout(width="180px")
    )
    num_th = widgets.BoundedFloatText(
        value=90.0, min=75.0, max=180.0, step=1.0,
        layout=widgets.Layout(width="75px")
    )

    is_syncing_h2o = False

    btn_calc = widgets.Button(description="Laske energia", button_style="success", icon="calculator", layout=widgets.Layout(width="135px"))
    btn_clear = widgets.Button(description="Tyhjennä pisteet", icon="trash", layout=widgets.Layout(width="125px"))

    readout = widgets.HTML()
    viewer_box = widgets.HTML()

    init_r = 1.50
    init_th = 90.0
    init_e_tot, init_e_atom, _, _ = dft_db.calc_h2o(init_r, init_th)
    last_calc = {
        "r": init_r,
        "th": init_th,
        "e_tot": init_e_tot,
        "e_atom": init_e_atom
    }

    history_records.append({
        "Piste #": 1,
        "r [Å]": f"{init_r:.2f}",
        "∠ [°]": f"{init_th:.0f}",
        "E_tot [Eh]": f"{init_e_tot:.5f}",
        "E_sidos [kJ/mol]": f"{init_e_atom:.1f}"
    })
    table_box = widgets.HTML(value=_format_table(history_records))

    fig_3d = go.FigureWidget(data=[
        go.Scatter3d(
            x=[init_r], y=[init_th], z=[init_e_atom], mode="markers", name="Mitatut pisteet",
            marker=dict(size=6, color=[init_e_atom], colorscale="Viridis", cmin=-970, cmax=-440, colorbar=dict(title="kJ/mol", len=0.6, thickness=12), opacity=0.85)
        ),
        go.Scatter3d(
            x=[init_r], y=[init_th], z=[init_e_atom], mode="markers", name="Viimeisin piste",
            marker=dict(size=10, color="#DD6B20", symbol="diamond", line=dict(color="black", width=1.5))
        )
    ])
    fig_3d.update_layout(
        title=dict(text="H₂O: 3D-potentiaalipinta (Klikkaa ja pyöritä)", font=dict(size=12.5)),
        scene=dict(
            xaxis=dict(title="r [Å]", range=[0.65, 1.55]),
            yaxis=dict(title="∠ [°]", range=[70.0, 185.0]),
            zaxis=dict(title="E_sidos [kJ/mol]", range=[-1000, -400]),
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.2))
        ),
        width=340, height=270,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=False
    )

    def refresh_view(_=None) -> None:
        r = round(num_r.value, 2)
        th = round(num_th.value, 1)
        viewer_box.value = render_3dmol_iframe(xyz_h2o(r, th), width=230, height=140)

        if r == last_calc["r"] and th == last_calc["th"]:
            e_tot = last_calc["e_tot"]
            e_atom = last_calc["e_atom"]
            readout.value = f"""
            <div style="font-family:sans-serif; font-size:12px; background:#F7FAFC; border:1px solid #CBD5E0; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
              <b>Geometria:</b> r = {r:.2f} Å &nbsp;|&nbsp; <b>∠:</b> {th:.0f}°<br>
              <b>Kokonaisenergia:</b> {e_tot:.5f} Eh<br>
              <b>Sidosenergia:</b> <span style="color:#2B6CB0;font-weight:700;">{e_atom:.1f} kJ/mol</span>
            </div>
            """
        else:
            readout.value = f"""
            <div style="font-family:sans-serif; font-size:12px; background:#FFF5F5; border:1px solid #FEB2B2; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
              <b>Asetettu:</b> r = {r:.2f} Å &nbsp;|&nbsp; <b>∠:</b> {th:.0f}° &nbsp;<span style="background:#E53E3E; color:white; font-size:9.5px; font-weight:700; padding:1px 5px; border-radius:3px;">EI LASKETTU</span><br>
              <span style="color:#718096; font-size:11px;">Paina <b>Laske energia</b> päivittääksesi energian.</span><br>
              <b>Kokonaisenergia:</b> <span style="color:#A0AEC0;">—</span> &nbsp;|&nbsp; <b>Sidosenergia:</b> <span style="color:#A0AEC0;">—</span>
            </div>
            """

    def on_s_r_change(change) -> None:
        nonlocal is_syncing_h2o
        if is_syncing_h2o:
            return
        is_syncing_h2o = True
        r = round(float(u_to_r(change["new"])), 2)
        num_r.value = r
        refresh_view()
        is_syncing_h2o = False

    def on_num_r_change(change) -> None:
        nonlocal is_syncing_h2o
        if is_syncing_h2o:
            return
        is_syncing_h2o = True
        r = change["new"]
        s_r.value = float(np.clip(r_to_u(r), 0.0, 1.0))
        refresh_view()
        is_syncing_h2o = False

    def on_s_th_change(change) -> None:
        nonlocal is_syncing_h2o
        if is_syncing_h2o:
            return
        is_syncing_h2o = True
        th = round(float(u_to_th(change["new"])), 1)
        num_th.value = th
        refresh_view()
        is_syncing_h2o = False

    def on_num_th_change(change) -> None:
        nonlocal is_syncing_h2o
        if is_syncing_h2o:
            return
        is_syncing_h2o = True
        th = change["new"]
        s_th.value = float(np.clip(th_to_u(th), 0.0, 1.0))
        refresh_view()
        is_syncing_h2o = False

    def on_calculate_clicked(_) -> None:
        r = round(num_r.value, 2)
        th = round(num_th.value, 1)

        btn_calc.description = "Lasketaan..."
        btn_calc.icon = "hourglass-half"
        btn_calc.disabled = True

        time.sleep(1.0)

        e_tot, e_atom, _, _ = dft_db.calc_h2o(r, th)
        last_calc["r"] = r
        last_calc["th"] = th
        last_calc["e_tot"] = e_tot
        last_calc["e_atom"] = e_atom

        current_geoms["H2O"] = f"r = {r:.2f} Å, ∠ = {th:.0f}°"
        current_energies["H2O"] = e_atom
        calculated_molecules.add("H2O")
        update_reaction_balance()

        history_records.append({
            "Piste #": len(history_records) + 1,
            "r [Å]": f"{r:.2f}",
            "∠ [°]": f"{th:.0f}",
            "E_tot [Eh]": f"{e_tot:.5f}",
            "E_sidos [kJ/mol]": f"{e_atom:.1f}"
        })
        table_box.value = _format_table(history_records)

        x_pts = [float(rec["r [Å]"]) for rec in history_records]
        y_pts = [float(rec["∠ [°]"]) for rec in history_records]
        z_pts = [float(rec["E_sidos [kJ/mol]"]) for rec in history_records]

        with fig_3d.batch_update():
            fig_3d.data[0].x = x_pts
            fig_3d.data[0].y = y_pts
            fig_3d.data[0].z = z_pts
            fig_3d.data[0].marker.color = z_pts
            fig_3d.data[1].x = [r]
            fig_3d.data[1].y = [th]
            fig_3d.data[1].z = [e_atom]

        readout.value = f"""
        <div style="font-family:sans-serif; font-size:12px; background:#F7FAFC; border:1px solid #CBD5E0; padding:6px 10px; border-radius:4px; margin-top:4px; line-height:1.45;">
          <b>Geometria:</b> r = {r:.2f} Å &nbsp;|&nbsp; <b>∠:</b> {th:.0f}°<br>
          <b>Kokonaisenergia:</b> {e_tot:.5f} Eh<br>
          <b>Sidosenergia:</b> <span style="color:#2B6CB0;font-weight:700;">{e_atom:.1f} kJ/mol</span>
        </div>
        """

        btn_calc.description = "Laske energia"
        btn_calc.icon = "calculator"
        btn_calc.disabled = False

    def on_clear_clicked(_) -> None:
        history_records.clear()
        table_box.value = _format_table([])
        with fig_3d.batch_update():
            fig_3d.data[0].x, fig_3d.data[0].y, fig_3d.data[0].z = [], [], []
            fig_3d.data[0].marker.color = []
            fig_3d.data[1].x, fig_3d.data[1].y, fig_3d.data[1].z = [], [], []

    s_r.observe(on_s_r_change, names="value")
    num_r.observe(on_num_r_change, names="value")
    s_th.observe(on_s_th_change, names="value")
    num_th.observe(on_num_th_change, names="value")
    btn_calc.on_click(on_calculate_clicked)
    btn_clear.on_click(on_clear_clicked)

    refresh_view()

    card_header = widgets.HTML("""
    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
      <h3 style="margin:0; font-size:15px; color:#2D3748;">H₂O: Taipunut molekyyli (Pyöritettävä 3D-potentiaalipinta)</h3>
      <span style="background:#FEFCBF; color:#B7791F; font-size:11px; font-weight:700; padding:2px 8px; border-radius:12px;">Tuote</span>
    </div>
    """)

    ctrl_row = widgets.VBox([
        widgets.HBox([widgets.Label("r(O-H) [Å]:", layout=widgets.Layout(width="70px")), s_r, num_r]),
        widgets.HBox([widgets.Label("∠(H-O-H) [°]:", layout=widgets.Layout(width="70px")), s_th, num_th]),
        widgets.HBox([btn_calc, btn_clear])
    ])

    left_sub = widgets.VBox([
        ctrl_row,
        viewer_box,
        readout
    ], layout=widgets.Layout(width="280px"))

    right_sub = widgets.VBox([
        fig_3d,
        table_box
    ], layout=widgets.Layout(width="340px"))

    card_content = widgets.HBox([left_sub, right_sub])

    return widgets.VBox([
        card_header,
        card_content
    ], layout=widgets.Layout(border="1px solid #CBD5E0", border_radius="8px", padding="10px", margin="6px", background="#FFFFFF"))


# ==============================================================================
# 6. Kokoa sivunäkymä (lukio-opetukseen optimoitu, ei teknisiä jargoneita)
# ==============================================================================
card_h2 = make_1d_card(
    "H2", "H₂: Diatominen sidosdissosiaatio", "Lähtöaine", "r(H-H)",
    xyz_h2, (0.40, 4.00), 4.00, (-500, 50)
)
card_co2 = make_1d_card(
    "CO2", "CO₂: Lineaarinen symmetrinen venytys", "Lähtöaine", "r(C=O)",
    xyz_co2, (0.85, 4.00), 4.00, (-1700, 50)
)
card_co = make_1d_card(
    "CO", "CO: Kolmoissidoksen venytys", "Tuote", "r(C≡O)",
    xyz_co, (0.80, 4.00), 4.00, (-1150, 50)
)
card_h2o = make_h2o_card()

reactants_col = widgets.VBox([
    widgets.HTML("<div style='background:#EBF8FF; color:#2B6CB0; font-weight:700; padding:6px 12px; border-radius:6px; margin:4px 6px; font-size:13.5px;'>⬅ LÄHTÖAINEET (CO₂ + H₂)</div>"),
    card_h2,
    card_co2
], layout=widgets.Layout(width="640px"))

products_col = widgets.VBox([
    widgets.HTML("<div style='background:#FEFCBF; color:#B7791F; font-weight:700; padding:6px 12px; border-radius:6px; margin:4px 6px; font-size:13.5px;'>➡ TUOTTEET (CO + H₂O)</div>"),
    card_co,
    card_h2o
], layout=widgets.Layout(width="640px"))

molecules_row = widgets.HBox([reactants_col, products_col])

balance_section = widgets.VBox([
    widgets.HTML("""
    <div style="background:#2D3748; color:white; padding:10px 16px; border-radius:6px 6px 0 0; margin-top:10px;">
      <h3 style="margin:0; font-size:16.5px;">⚖️ Hessin lain reaktiolämpö: CO₂ + H₂ ⇌ CO + H₂O</h3>
      <p style="margin:2px 0 0 0; font-size:12.5px; opacity:0.85;">
        Termodynaaminen reaktiolämpö voidaan laskea molekyylien sidosenergioista.
      </p>
    </div>
    """),
    widgets.HBox([
        widgets.VBox([
            summary_table,
            rxn_analysis
        ], layout=widgets.Layout(width="580px", padding="10px")),
        widgets.VBox([
            fig_rwgs
        ], layout=widgets.Layout(width="540px", padding="10px"))
    ], layout=widgets.Layout(border="1px solid #CBD5E0", border_top="none", border_radius="0 0 6px 6px", background="#FFFFFF", margin_bottom="15px"))
])

results_box = widgets.VBox([
    results_status_badge,
    results_accordion
], layout=widgets.Layout(margin="6px 0 15px 0"))

header_banner = widgets.HTML("""
<div style="background-color:#1A202C; color:white; padding:14px 20px; border-radius:8px; margin-bottom:12px; box-shadow:0 2px 4px rgba(0,0,0,0.12);">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h2 style="margin:0; font-size:21px; font-weight:700;">
        🧪 Käänteisen vesikaasureaktion laskennallinen mallintaminen
      </h2>
      <p style="margin:4px 0 0 0; font-size:13px; opacity:0.9;">
        CO₂ + H₂ ⇌ CO + H₂O &nbsp;|&nbsp; Käänteisen vesikaasureaktio
      </p>
    </div>
    <div style="text-align:right; font-size:11.5px; opacity:0.8; line-height:1.4;">
      Yhden näkymän studio &bull; Hessin laki<br>
      Suora lukuarvojen syöttö &bull; Molekyylien sidosenergiat
    </div>
  </div>
</div>
""")

dashboard = widgets.VBox([
    header_banner,
    molecules_row,
    balance_section,
    results_box
])

update_reaction_balance()

if __name__ == "__main__":
    display(dashboard)
