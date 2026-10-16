"""
Comp-Chem-Lab: Molekyyliorbitaalien 3D-visualisointistudio & MO-kaavio
======================================================================
Interaktiivinen työkalu molekyylien rajaorbitaalien (HOMO & LUMO +/- 1)
3D-tarkasteluun puhtaalla WebGL-molekyylivisualisoijalla (3Dmol.js) valkoisella
taustalla, sekä oppikirjamaisella MO-kaaviolla, joka mallintaa s-p-hybridisoitumista
ja korrelaatioviivoja (vrt. GPAW / Chemogenesis co_bonding.jpg).

Ominaisuudet:
- Korkea hilaresoluutio ja laajennettu tila (ei leikkautumista)
- Kaikki atomit kuvataan palloina (CPK)
- Puhdas valkoinen tausta molemmissa kuvaajissa
- Fiksattu isopinta 0.03 ilman ylimääräisiä säätimiä
- Pelkät kemialliset kaavat (CO, H₂O, CO₂, C₂H₄)
- 'Vaihe'-terminologia (+ vaihe ja - vaihe)
- Molekyyli asettuu aina poikittain kameraan nähden
- Orbitaalia vaihdettaessa molekyylin orientaatio ja zoomaus säilyvät täysin
"""
from __future__ import annotations
import html
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import ipywidgets as widgets
import py3Dmol

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
CUBES_DIR = os.path.join(SCRIPT_DIR, "cubes")
META_PATH = os.path.join(SCRIPT_DIR, "mo_meta.json")

FIXED_ISOVAL = 0.03
FIXED_OPACITY = 0.78


# ==============================================================================
# 1. 3D-molekyyli- ja orbitaalivisualisointi (py3Dmol valkoisella taustalla)
# ==============================================================================
def render_molecule_3d_iframe(
    mol_key: str,
    initial_orb_key: str = "HOMO",
    width: int = 580,
    height: int = 540
) -> str:
    """
    Palauttaa 3Dmol.js iframe -näkymän, joka sisältää kaikki 4 orbitaalia
    kyseiselle molekyylille. Orbitaalin vaihto tapahtuu JavaScript-tasolla
    ilman iframen uudelleenlatausta, jolloin käyttäjän kääntämä orientaatio säilyy!
    """
    # Ladataan kaikki 4 orbitaalia JavaScriptin muistiin
    cubes_data = {}
    for orb in ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]:
        fn = os.path.join(CUBES_DIR, f"{mol_key}_{orb}.cube")
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8") as f:
                cubes_data[orb] = f.read()

    cubes_json = json.dumps(cubes_data)
    initial_cube = cubes_data.get(initial_orb_key, "")

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body, html {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    #viewer-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      background: #ffffff;
    }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"></script>
</head>
<body>
  <div id="viewer-container"></div>
  <script>
    const CUBES = {cubes_json};
    let currentOrb = "{initial_orb_key}";
    let viewer = null;
    let orbitalShapes = [];

    function initViewer() {{
      const container = document.getElementById("viewer-container");
      viewer = $3Dmol.createViewer(container, {{
        backgroundColor: "#ffffff",
        antialias: true
      }});

      // Ladataan molekyylimalli (kaikki atomit palloina)
      const baseCube = CUBES[currentOrb];
      if (baseCube) {{
        viewer.addModel(baseCube, "cube");

        // Kaikki atomit kuvataan selkeinä palloina ja sidokset sauvoina
        viewer.setStyle({{}}, {{
          stick: {{ radius: 0.14, color: "#64748b" }}
        }});
        viewer.addStyle({{ elem: "H" }}, {{ sphere: {{ radius: 0.42, color: "#FFFFFF" }} }});
        viewer.addStyle({{ elem: "C" }}, {{ sphere: {{ radius: 0.62, color: "#334155" }} }});
        viewer.addStyle({{ elem: "O" }}, {{ sphere: {{ radius: 0.58, color: "#DC2626" }} }});

        // Lisätään orbitaalipinnat (isopinta 0.03)
        addOrbitalSurfaces(baseCube);

        // Asetetaan molekyyli poikittain kameraan nähden ja sovitetaan näkymään
        viewer.zoomTo();
        viewer.render();
      }}
    }}

    function clearOrbitalSurfaces() {{
      if (!viewer) return;
      if (orbitalShapes && orbitalShapes.length > 0) {{
        for (let i = 0; i < orbitalShapes.length; i++) {{
          try {{
            if (orbitalShapes[i]) viewer.removeShape(orbitalShapes[i]);
          }} catch (e) {{}}
        }}
        orbitalShapes = [];
      }}
      try {{
        viewer.removeAllShapes();
      }} catch (e) {{}}
      try {{
        viewer.removeAllSurfaces();
      }} catch (e) {{}}
    }}

    function addOrbitalSurfaces(cubeData) {{
      clearOrbitalSurfaces();
      const voldata = new $3Dmol.VolumeData(cubeData, "cube");
      // Positiivinen vaihe (+0.03)
      const sPos = viewer.addIsosurface(voldata, {{
        isoval: {FIXED_ISOVAL},
        color: "#0ea5e9",
        opacity: {FIXED_OPACITY},
        smoothness: 3
      }});
      if (sPos) orbitalShapes.push(sPos);

      // Negatiivinen vaihe (-0.03)
      const sNeg = viewer.addIsosurface(voldata, {{
        isoval: -{FIXED_ISOVAL},
        color: "#f43f5e",
        opacity: {FIXED_OPACITY},
        smoothness: 3
      }});
      if (sNeg) orbitalShapes.push(sNeg);
    }}

    // Vaihtaa orbitaalin poistamatta ja resetoimatta kameraa!
    function setOrbital(newOrb) {{
      if (newOrb === currentOrb && orbitalShapes.length > 0) return;
      currentOrb = newOrb;
      const cubeData = CUBES[newOrb];
      if (!cubeData || !viewer) return;

      // Lisätään uuden orbitaalin pinnat (addOrbitalSurfaces poistaa vanhat pinnat)
      addOrbitalSurfaces(cubeData);

      // Renderöidään ilman kameran nollausta (zoomTo ei kutsuta!)
      viewer.render();
    }}

    // Kuunnellaan komentoja Python-isännältä
    let lastCmd = "";
    function checkParentCommand() {{
      try {{
        const inp = window.parent.document.querySelector(".orbital-cmd-to-iframe input");
        if (inp && inp.value && inp.value !== lastCmd) {{
          lastCmd = inp.value;
          const cmd = JSON.parse(inp.value);
          if (cmd && cmd.orb) {{
            setOrbital(cmd.orb);
          }}
        }}
      }} catch(err) {{}}
    }}
    setInterval(checkParentCommand, 40);

    window.addEventListener("message", (e) => {{
      if (e.data && e.data.orb) {{
        setOrbital(e.data.orb);
      }}
    }});

    window.addEventListener("DOMContentLoaded", initViewer);
  </script>
</body>
</html>
"""
    escaped_html = html.escape(inner_html, quote=True)
    return (
        f'<iframe class="orbital-3d-iframe" srcdoc="{escaped_html}" '
        f'style="width:100%; height:{height}px; border:none; border-radius:8px; background:#ffffff;" '
        f'frameborder="0"></iframe>'
    )


# ==============================================================================
# 2. Oppikirjamaiset SVG MO-kaaviot (Valkoinen tausta, mallina co_bonding.jpg)
# ==============================================================================
def draw_pair(x: float, y: float, color: str = "#0284c7") -> str:
    """Piirtää elektroniparin perinteisillä oppikirjan nuolilla (single-barbed arrows)."""
    return (
        f'<path d="M {x-5:.1f} {y+11:.1f} L {x-5:.1f} {y-11:.1f} M {x-5:.1f} {y-11:.1f} L {x-9:.1f} {y-5:.1f}" '
        f'stroke="{color}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="M {x+5:.1f} {y-11:.1f} L {x+5:.1f} {y+11:.1f} M {x+5:.1f} {y+11:.1f} L {x+9:.1f} {y+5:.1f}" '
        f'stroke="{color}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def draw_single(x: float, y: float, color: str = "#0284c7") -> str:
    """Piirtää yhden parittoman elektronin nuolen."""
    return (
        f'<path d="M {x:.1f} {y+11:.1f} L {x:.1f} {y-11:.1f} M {x:.1f} {y-11:.1f} L {x-4:.1f} {y-5:.1f}" '
        f'stroke="{color}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def render_mo_diagram_svg(mol_key: str, selected_orb_key: str) -> str:
    """Luo ja palauttaa selkeän, oppikirjamaisen SVG-diagrammin valkoisella taustalla."""
    bg = "#ffffff"
    corr_col = "rgba(71, 85, 105, 0.45)"
    lvl_norm = "#1e293b"
    lvl_hi_occ = "#0284c7"
    lvl_hi_virt = "#d97706"
    text_main = "#0f172a"
    text_sub = "#475569"

    svg_lines = [
        f'<svg viewBox="0 0 540 560" width="100%" height="540" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:{bg}; border-radius:8px; font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
    ]

    def line(x1, y1, x2, y2, color=lvl_norm, width=2.5, dash=""):
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d_attr}/>'

    def corr(x1, y1, x2, y2):
        return line(x1, y1, x2, y2, color=corr_col, width=1.3, dash="3,3")

    def text(x, y, s, anchor="middle", size=12, weight="normal", color=text_main, italic=False):
        style = f'font-style:italic;' if italic else ''
        return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}" style="{style}">{s}</text>'

    # --------------------------------------------------------------------------
    # 2.1. CO – Täsmälleen GPAW / Chemogenesis co_bonding.jpg mukainen malli
    # --------------------------------------------------------------------------
    if mol_key == "CO":
        svg_lines.append(text(270, 24, "CO – Molekyyliorbitaalikaavio", size=14, weight="bold"))
        svg_lines.append(text(85, 525, "C", size=13, weight="bold", color=text_sub))
        svg_lines.append(text(270, 525, "CO", size=14, weight="bold"))
        svg_lines.append(text(455, 525, "O", size=13, weight="bold", color=text_sub))

        # Korrelaatioviivat (osoittavat s-p-hybridisoitumisen)
        svg_lines.append(corr(115, 385, 240, 475))
        svg_lines.append(corr(425, 440, 300, 475))

        svg_lines.append(corr(115, 385, 240, 350))
        svg_lines.append(corr(425, 440, 300, 350))
        svg_lines.append(corr(395, 185, 300, 350))

        svg_lines.append(corr(145, 150, 225, 230))
        svg_lines.append(corr(395, 185, 315, 230))

        svg_lines.append(corr(115, 385, 240, 190))
        svg_lines.append(corr(145, 150, 240, 190))
        svg_lines.append(corr(395, 185, 300, 190))

        svg_lines.append(corr(145, 150, 225, 95))
        svg_lines.append(corr(395, 185, 315, 95))

        svg_lines.append(corr(115, 385, 240, 42))
        svg_lines.append(corr(145, 150, 240, 42))
        svg_lines.append(corr(395, 185, 300, 42))

        # Vasen: C 2s & 2p
        svg_lines.append(line(60, 385, 115, 385))
        svg_lines.append(draw_pair(87.5, 385))
        svg_lines.append(text(40, 390, "2s", anchor="end", size=13, italic=True))

        for idx, (x1, x2) in enumerate([(50, 75), (85, 110), (120, 145)]):
            svg_lines.append(line(x1, 150, x2, 150))
            if idx < 2:
                svg_lines.append(draw_single((x1 + x2) / 2.0, 150))
        svg_lines.append(text(35, 155, "2p", anchor="end", size=13, italic=True))

        # Oikea: O 2s & 2p
        svg_lines.append(line(425, 440, 480, 440))
        svg_lines.append(draw_pair(452.5, 440))
        svg_lines.append(text(500, 445, "2s", anchor="start", size=13, italic=True))

        for idx, (x1, x2) in enumerate([(395, 420), (430, 455), (465, 490)]):
            svg_lines.append(line(x1, 185, x2, 185))
            if idx == 0:
                svg_lines.append(draw_pair((x1 + x2) / 2.0, 185))
            else:
                svg_lines.append(draw_single((x1 + x2) / 2.0, 185))
        svg_lines.append(text(505, 190, "2p", anchor="start", size=13, italic=True))

        # Keskellä: MO-tasot
        svg_lines.append(line(240, 475, 300, 475))
        svg_lines.append(draw_pair(270, 475))
        svg_lines.append(text(225, 480, "σ₁", anchor="end", size=13, italic=True))

        svg_lines.append(line(240, 350, 300, 350))
        svg_lines.append(draw_pair(270, 350))
        svg_lines.append(text(225, 355, "σ₂", anchor="end", size=13, italic=True))

        is_h_m1 = (selected_orb_key == "HOMO-1")
        pi_col = lvl_hi_occ if is_h_m1 else lvl_norm
        pi_w = 4.5 if is_h_m1 else 2.5
        svg_lines.append(line(225, 230, 255, 230, color=pi_col, width=pi_w))
        svg_lines.append(line(285, 230, 315, 230, color=pi_col, width=pi_w))
        svg_lines.append(draw_pair(240, 230))
        svg_lines.append(draw_pair(300, 230))
        svg_lines.append(text(270, 252, "π (HOMO-1)", anchor="middle", size=12, italic=True, color=pi_col))
        if is_h_m1:
            svg_lines.append(text(330, 235, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_homo = (selected_orb_key == "HOMO")
        s3_col = lvl_hi_occ if is_homo else lvl_norm
        s3_w = 4.5 if is_homo else 2.5
        svg_lines.append(line(240, 190, 300, 190, color=s3_col, width=s3_w))
        svg_lines.append(draw_pair(270, 190))
        svg_lines.append(text(225, 195, "σ₃ (HOMO)", anchor="end", size=12, italic=True, color=s3_col))
        if is_homo:
            svg_lines.append(text(315, 195, "◄ 3D-malli (C-vapaa pari)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_lumo = (selected_orb_key == "LUMO")
        pistar_col = lvl_hi_virt if is_lumo else lvl_norm
        pistar_w = 4.5 if is_lumo else 2.5
        svg_lines.append(line(225, 95, 255, 95, color=pistar_col, width=pistar_w))
        svg_lines.append(line(285, 95, 315, 95, color=pistar_col, width=pistar_w))
        svg_lines.append(text(270, 118, "π* (LUMO)", anchor="middle", size=12, italic=True, color=pistar_col))
        if is_lumo:
            svg_lines.append(text(330, 100, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        is_lumo_p1 = (selected_orb_key == "LUMO+1")
        s4_col = lvl_hi_virt if is_lumo_p1 else lvl_norm
        s4_w = 4.5 if is_lumo_p1 else 2.5
        svg_lines.append(line(240, 42, 300, 42, color=s4_col, width=s4_w))
        svg_lines.append(text(225, 47, "σ₄ (LUMO+1)", anchor="end", size=12, italic=True, color=s4_col))
        if is_lumo_p1:
            svg_lines.append(text(315, 47, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

    # --------------------------------------------------------------------------
    # 2.2. H2O – Taipunut molekyyli & sp-hybridisaatio
    # --------------------------------------------------------------------------
    elif mol_key == "H2O":
        svg_lines.append(text(270, 24, "H₂O – Molekyyliorbitaalikaavio", size=14, weight="bold"))
        svg_lines.append(text(95, 525, "2 × H (1s)", size=13, weight="bold", color=text_sub))
        svg_lines.append(text(270, 525, "H₂O", size=14, weight="bold"))
        svg_lines.append(text(455, 525, "O", size=13, weight="bold", color=text_sub))

        # Korrelaatiot
        svg_lines.append(corr(130, 230, 240, 475))
        svg_lines.append(corr(425, 440, 300, 475))

        svg_lines.append(corr(130, 230, 240, 280))
        svg_lines.append(corr(395, 185, 300, 280))

        svg_lines.append(corr(130, 230, 240, 200))
        svg_lines.append(corr(425, 440, 300, 200))
        svg_lines.append(corr(440, 185, 300, 200))

        svg_lines.append(corr(475, 185, 300, 165))

        svg_lines.append(corr(130, 230, 240, 95))
        svg_lines.append(corr(440, 185, 300, 95))

        svg_lines.append(corr(130, 230, 240, 42))
        svg_lines.append(corr(395, 185, 300, 42))

        # Vasen: 2x H (1s)
        svg_lines.append(line(65, 230, 95, 230))
        svg_lines.append(line(105, 230, 135, 230))
        svg_lines.append(draw_single(80, 230))
        svg_lines.append(draw_single(120, 230))
        svg_lines.append(text(50, 235, "1s", anchor="end", size=13, italic=True))

        # Oikea: O 2s & 2p
        svg_lines.append(line(425, 440, 480, 440))
        svg_lines.append(draw_pair(452.5, 440))
        svg_lines.append(text(500, 445, "2s", anchor="start", size=13, italic=True))

        for idx, (x1, x2) in enumerate([(395, 420), (430, 455), (465, 490)]):
            svg_lines.append(line(x1, 185, x2, 185))
            if idx == 0:
                svg_lines.append(draw_pair((x1 + x2) / 2.0, 185))
            else:
                svg_lines.append(draw_single((x1 + x2) / 2.0, 185))
        svg_lines.append(text(505, 190, "2p", anchor="start", size=13, italic=True))

        # Keskellä: MOt
        svg_lines.append(line(240, 475, 300, 475))
        svg_lines.append(draw_pair(270, 475))
        svg_lines.append(text(225, 480, "2a₁ (σ sitova)", anchor="end", size=12))

        svg_lines.append(line(240, 280, 300, 280))
        svg_lines.append(draw_pair(270, 280))
        svg_lines.append(text(225, 285, "1b₂ (σ sitova)", anchor="end", size=12))

        is_h_m1 = (selected_orb_key == "HOMO-1")
        c_hm1 = lvl_hi_occ if is_h_m1 else lvl_norm
        svg_lines.append(line(240, 200, 300, 200, color=c_hm1, width=4.5 if is_h_m1 else 2.5))
        svg_lines.append(draw_pair(270, 200))
        svg_lines.append(text(225, 205, "3a₁ (HOMO-1)", anchor="end", size=12, color=c_hm1))
        if is_h_m1:
            svg_lines.append(text(315, 205, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_homo = (selected_orb_key == "HOMO")
        c_h = lvl_hi_occ if is_homo else lvl_norm
        svg_lines.append(line(240, 165, 300, 165, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(270, 165))
        svg_lines.append(text(225, 170, "1b₁ (HOMO)", anchor="end", size=12, color=c_h))
        if is_homo:
            svg_lines.append(text(315, 170, "◄ 3D-malli (vapaa 2p-pari)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_lumo = (selected_orb_key == "LUMO")
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        svg_lines.append(line(240, 95, 300, 95, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(225, 100, "4a₁ (LUMO, σ*)", anchor="end", size=12, color=c_l))
        if is_lumo:
            svg_lines.append(text(315, 100, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        is_lp1 = (selected_orb_key == "LUMO+1")
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm
        svg_lines.append(line(240, 42, 300, 42, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(225, 47, "2b₂ (LUMO+1, σ*)", anchor="end", size=12, color=c_lp1))
        if is_lp1:
            svg_lines.append(text(315, 47, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

    # --------------------------------------------------------------------------
    # 2.3. CO2 – Lineaarinen molekyyli
    # --------------------------------------------------------------------------
    elif mol_key == "CO2":
        svg_lines.append(text(270, 24, "CO₂ – Molekyyliorbitaalikaavio", size=14, weight="bold"))
        svg_lines.append(text(85, 525, "C", size=13, weight="bold", color=text_sub))
        svg_lines.append(text(270, 525, "CO₂", size=14, weight="bold"))
        svg_lines.append(text(455, 525, "2 × O", size=13, weight="bold", color=text_sub))

        svg_lines.append(corr(115, 385, 240, 475))
        svg_lines.append(corr(425, 440, 300, 475))
        svg_lines.append(corr(115, 385, 240, 330))
        svg_lines.append(corr(425, 440, 300, 330))
        svg_lines.append(corr(395, 185, 300, 330))

        svg_lines.append(corr(145, 150, 225, 230))
        svg_lines.append(corr(395, 185, 315, 230))
        svg_lines.append(corr(465, 185, 315, 160))

        svg_lines.append(corr(145, 150, 225, 95))
        svg_lines.append(corr(395, 185, 315, 95))
        svg_lines.append(corr(115, 385, 240, 42))
        svg_lines.append(corr(425, 440, 300, 42))

        # C
        svg_lines.append(line(60, 385, 115, 385))
        svg_lines.append(draw_pair(87.5, 385))
        svg_lines.append(text(40, 390, "2s", anchor="end", size=13, italic=True))
        for idx, (x1, x2) in enumerate([(50, 75), (85, 110), (120, 145)]):
            svg_lines.append(line(x1, 150, x2, 150))
            if idx < 2:
                svg_lines.append(draw_single((x1 + x2) / 2.0, 150))
        svg_lines.append(text(35, 155, "2p", anchor="end", size=13, italic=True))

        # 2x O
        svg_lines.append(line(415, 440, 445, 440))
        svg_lines.append(line(455, 440, 485, 440))
        svg_lines.append(draw_pair(430, 440))
        svg_lines.append(draw_pair(470, 440))
        svg_lines.append(text(505, 445, "2s SALC", anchor="start", size=12))

        for idx, (x1, x2) in enumerate([(395, 420), (430, 455), (465, 490)]):
            svg_lines.append(line(x1, 185, x2, 185))
            svg_lines.append(draw_pair((x1 + x2) / 2.0, 185))
        svg_lines.append(text(505, 190, "2p SALC", anchor="start", size=12))

        # MOt
        svg_lines.append(line(240, 475, 300, 475))
        svg_lines.append(draw_pair(270, 475))
        svg_lines.append(text(225, 480, "2σ_g, 1σ_u", anchor="end", size=12))

        svg_lines.append(line(240, 330, 300, 330))
        svg_lines.append(draw_pair(270, 330))
        svg_lines.append(text(225, 335, "3σ_g (s-p)", anchor="end", size=12))

        is_h_m1 = (selected_orb_key == "HOMO-1")
        c_hm1 = lvl_hi_occ if is_h_m1 else lvl_norm
        svg_lines.append(line(225, 230, 255, 230, color=c_hm1, width=4.5 if is_h_m1 else 2.5))
        svg_lines.append(line(285, 230, 315, 230, color=c_hm1, width=4.5 if is_h_m1 else 2.5))
        svg_lines.append(draw_pair(240, 230))
        svg_lines.append(draw_pair(300, 230))
        svg_lines.append(text(270, 252, "1π_u (HOMO-1, sitova)", anchor="middle", size=12, color=c_hm1))
        if is_h_m1:
            svg_lines.append(text(330, 235, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_homo = (selected_orb_key == "HOMO")
        c_h = lvl_hi_occ if is_homo else lvl_norm
        svg_lines.append(line(225, 160, 255, 160, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(line(285, 160, 315, 160, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(240, 160))
        svg_lines.append(draw_pair(300, 160))
        svg_lines.append(text(270, 182, "1π_g (HOMO, vapaat parit)", anchor="middle", size=12, color=c_h))
        if is_homo:
            svg_lines.append(text(330, 165, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_lumo = (selected_orb_key == "LUMO")
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        svg_lines.append(line(225, 95, 255, 95, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(line(285, 95, 315, 95, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(270, 118, "2π_u* (LUMO, π*)", anchor="middle", size=12, color=c_l))
        if is_lumo:
            svg_lines.append(text(330, 100, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        is_lp1 = (selected_orb_key == "LUMO+1")
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm
        svg_lines.append(line(240, 42, 300, 42, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(225, 47, "4σ_g* (LUMO+1)", anchor="end", size=12, color=c_lp1))
        if is_lp1:
            svg_lines.append(text(315, 47, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

    # --------------------------------------------------------------------------
    # 2.4. C2H4 – Kaksoissidos
    # --------------------------------------------------------------------------
    elif mol_key == "C2H4":
        svg_lines.append(text(270, 24, "C₂H₄ – Molekyyliorbitaalikaavio", size=14, weight="bold"))
        svg_lines.append(text(85, 525, "CH₂ (1)", size=13, weight="bold", color=text_sub))
        svg_lines.append(text(270, 525, "C₂H₄", size=14, weight="bold"))
        svg_lines.append(text(455, 525, "CH₂ (2)", size=13, weight="bold", color=text_sub))

        svg_lines.append(corr(115, 300, 240, 460))
        svg_lines.append(corr(425, 300, 300, 460))
        svg_lines.append(corr(115, 300, 240, 230))
        svg_lines.append(corr(425, 300, 300, 230))

        svg_lines.append(corr(115, 160, 240, 160))
        svg_lines.append(corr(425, 160, 300, 160))
        svg_lines.append(corr(115, 160, 240, 80))
        svg_lines.append(corr(425, 160, 300, 80))

        svg_lines.append(corr(115, 300, 240, 35))
        svg_lines.append(corr(425, 300, 300, 35))

        # CH2 (1)
        svg_lines.append(line(60, 300, 115, 300))
        svg_lines.append(draw_pair(87.5, 300))
        svg_lines.append(text(40, 305, "σ(sp²)", anchor="end", size=12))

        svg_lines.append(line(60, 160, 115, 160))
        svg_lines.append(draw_single(87.5, 160))
        svg_lines.append(text(40, 165, "2p_x (π)", anchor="end", size=12))

        # CH2 (2)
        svg_lines.append(line(425, 300, 480, 300))
        svg_lines.append(draw_pair(452.5, 300))
        svg_lines.append(text(500, 305, "σ(sp²)", anchor="start", size=12))

        svg_lines.append(line(425, 160, 480, 160))
        svg_lines.append(draw_single(452.5, 160))
        svg_lines.append(text(500, 165, "2p_x (π)", anchor="start", size=12))

        # MOt
        svg_lines.append(line(240, 460, 300, 460))
        svg_lines.append(draw_pair(270, 460))
        svg_lines.append(text(225, 465, "σ (C-C sidos)", anchor="end", size=12))

        is_h_m1 = (selected_orb_key == "HOMO-1")
        c_hm1 = lvl_hi_occ if is_h_m1 else lvl_norm
        svg_lines.append(line(240, 230, 300, 230, color=c_hm1, width=4.5 if is_h_m1 else 2.5))
        svg_lines.append(draw_pair(270, 230))
        svg_lines.append(text(225, 235, "1b₁ᵤ (HOMO-1, σ)", anchor="end", size=12, color=c_hm1))
        if is_h_m1:
            svg_lines.append(text(315, 235, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_homo = (selected_orb_key == "HOMO")
        c_h = lvl_hi_occ if is_homo else lvl_norm
        svg_lines.append(line(240, 160, 300, 160, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(270, 160))
        svg_lines.append(text(225, 165, "1b₃ᵤ (HOMO, C=C π)", anchor="end", size=12, color=c_h))
        if is_homo:
            svg_lines.append(text(315, 165, "◄ 3D-malli (C=C π-sidos)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_lumo = (selected_orb_key == "LUMO")
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        svg_lines.append(line(240, 80, 300, 80, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(225, 85, "1b₂g* (LUMO, π*)", anchor="end", size=12, color=c_l))
        if is_lumo:
            svg_lines.append(text(315, 85, "◄ 3D-malli (π*-hajoittava)", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        is_lp1 = (selected_orb_key == "LUMO+1")
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm
        svg_lines.append(line(240, 35, 300, 35, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(225, 40, "3a_g* (LUMO+1, σ*)", anchor="end", size=12, color=c_lp1))
        if is_lp1:
            svg_lines.append(text(315, 40, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

    svg_lines.append("</svg>")
    return "".join(svg_lines)


# ==============================================================================
# 3. Rajaorbitaalien (HOMO & LUMO) pedagoginen selvitys
# ==============================================================================
def render_fmo_explanation_html(selected_orb: str = "HOMO") -> str:
    """
    Palauttaa pedagogisen ja selkeän selvityksen HOMO- ja LUMO-lyhenteistä,
    niiden fysikaalis-kemiallisesta taustasta sekä rajaorbitaaliteoriasta (FMO).
    """
    is_homo = (selected_orb == "HOMO")
    is_homo_m1 = (selected_orb == "HOMO-1")
    is_lumo = (selected_orb == "LUMO")
    is_lumo_p1 = (selected_orb == "LUMO+1")

    homo_border = "#0284c7" if is_homo else ("#38bdf8" if is_homo_m1 else "#e2e8f0")
    homo_bg = "#f0f9ff" if (is_homo or is_homo_m1) else "#ffffff"
    homo_badge = " <span style='background:#0284c7; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>VALITTUNA</span>" if is_homo else (" <span style='background:#0284c7; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>HOMO-1 VALITTUNA</span>" if is_homo_m1 else "")

    lumo_border = "#d97706" if is_lumo else ("#fbbf24" if is_lumo_p1 else "#e2e8f0")
    lumo_bg = "#fffbeb" if (is_lumo or is_lumo_p1) else "#ffffff"
    lumo_badge = " <span style='background:#d97706; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>VALITTUNA</span>" if is_lumo else (" <span style='background:#d97706; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>LUMO+1 VALITTUNA</span>" if is_lumo_p1 else "")

    return f"""
    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:8px; padding:16px 20px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:#1e293b; box-shadow:0 1px 3px rgba(0,0,0,0.04); margin-top:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
        <h4 style="margin:0; font-size:15px; color:#0f172a; font-weight:700; display:flex; align-items:center; gap:6px;">
          📚 HOMO ja orbitaalit (Rajaorbitaalit / Frontier Molecular Orbitals)
        </h4>
        <span style="font-size:11.5px; color:#64748b; background:#f1f5f9; padding:2px 8px; border-radius:12px;">Nobel-palkittu teoria (1981)</span>
      </div>

      <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
        <!-- HOMO Kortti -->
        <div style="background:{homo_bg}; border:2px solid {homo_border}; border-radius:8px; padding:12px 15px; transition:all 0.2s ease;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13.5px; font-weight:700; color:#0284c7;">🔵 <b>H</b>ighest <b>O</b>ccupied <b>M</b>olecular <b>O</b>rbital<br> HOMO</span>
            {homo_badge}
          </div>
          <div style="font-size:12px; font-weight:600; color:#334155; margin-bottom:6px;">
            <span style="color:#64748b; font-weight:400; font-style:italic;">(Korkein miehitetty molekyyliorbitaali)</span>
          </div>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • Molekyylin perustilan miehitetty orbitaali, jolla on korkein energia. Nämä elektronit ovat heikoimmin sitoutuneet molekyyliin.
          </p>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • Koska nämä elektronit ovat heikoimmin sitoutuneet, molekyyli tyypillisesti <b>luovuttaa elektroneja</b> juuri HOMOlta.
          </p>
          <div style="font-size:11.5px; background:rgba(2,132,199,0.08); padding:4px 8px; border-radius:4px; color:#0369a1;">
            <b>HOMO-1:</b> Toiseksi korkein miehitetty orbitaali.
          </div>
        </div>

        <!-- LUMO Kortti -->
        <div style="background:{lumo_bg}; border:2px solid {lumo_border}; border-radius:8px; padding:12px 15px; transition:all 0.2s ease;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13.5px; font-weight:700; color:#d97706;">🟠 <b>L</b>owest <b>U</b>noccupied <b>M</b>olecular <b>O</b>rbital<br> LUMO</span>
            {lumo_badge}
          </div>
          <div style="font-size:12px; font-weight:600; color:#334155; margin-bottom:6px;">
            <span style="color:#64748b; font-weight:400; font-style:italic;">(Matalin miehittämätön / tyhjä molekyyliorbitaali)</span>
          </div>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • Molekyylin miehittämätön, alimman energian orbitaali. Tällä orbitaalilla ei perustilassa ole elektroneja.
          </p>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • Koska LUMO on tyhjistä orbitaaleista energialtaan alin, molekyylin on edullisinta <b>vastaanottaa elektroneja</b> juuri LUMOlle.
          </p>
          <div style="font-size:11.5px; background:rgba(217,119,6,0.08); padding:4px 8px; border-radius:4px; color:#b45309;">
            <b>LUMO+1:</b> Toiseksi matalin tyhjä orbitaali (heti LUMOn yläpuolella).
          </div>
        </div>
      </div>

      <!-- FMO teoria -->
      <div style="background:#f8fafc; border-left:4px solid #6366f1; padding:9px 13px; border-radius:0 6px 6px 0; font-size:12px; line-height:1.5; color:#334155;">
        <b>💡 Rajaorbitaaliteoria (Frontier Molecular Orbital, FMO):</b><br>
        Molekyylin <b>HOMO</b> ja <b>LUMO</b> tunnetaan yhdessä <i>rajaorbitaaleina</i>. Japanilainen kemisti <b>Kenichi Fukui</b> osoitti, että useimmissa kemiallisissa reaktioissa uusi kovalenttinen sidos muodostuu, kun toisen molekyylin elektronirikas <b>HOMO</b> menee päällekkäin toisen molekyylin tyhjän <b>LUMO</b>:n kanssa (rakentava aaltointerferenssi). Tästä uraauurtavasta oivalluksesta Fukui ja Roald Hoffmann palkittiin <b>Nobelin kemianpalkinnolla vuonna 1981</b>. <i>(Katso havainnollinen esimerkki 5. välilehdeltä: syanidi-ionin HOMO luovuttaa elektroniparin formaldehydin LUMOlle!)</i>
      </div>
    </div>
    """


# ==============================================================================
# 4. Studion Päänäkymä
# ==============================================================================
def create_orbital_studio() -> widgets.VBox:
    """Rakentaa ja palauttaa molekyyliorbitaalien 3D-studion ja suuren MO-kaavion."""
    state = {
        "mol": "CO",
        "orb": "HOMO",
    }

    # Piilotettu ohjauskenttä Pythonista selaimen 3D-iframeen (ei riko kameraa!)
    cmd_to_3d = widgets.Text(value="", layout=widgets.Layout(display="none"))
    cmd_to_3d.add_class("orbital-cmd-to-iframe")

    viewer_3d_box = widgets.HTML(
        value=render_molecule_3d_iframe(state["mol"], state["orb"], width=580, height=540),
        layout=widgets.Layout(width="100%", height="545px")
    )

    mo_diagram_box = widgets.HTML(
        value=render_mo_diagram_svg(state["mol"], state["orb"]),
        layout=widgets.Layout(width="100%", height="545px")
    )

    fmo_explanation_box = widgets.HTML(
        value=render_fmo_explanation_html(state["orb"]),
        layout=widgets.Layout(width="100%")
    )

    # Painikkeet vain kemiallisilla kaavoilla
    mol_btns = {
        "CO": widgets.Button(
            description="CO",
            button_style="primary" if state["mol"] == "CO" else "",
            layout=widgets.Layout(width="95px", height="34px")
        ),
        "H2O": widgets.Button(
            description="H₂O",
            button_style="primary" if state["mol"] == "H2O" else "",
            layout=widgets.Layout(width="95px", height="34px")
        ),
        "CO2": widgets.Button(
            description="CO₂",
            button_style="primary" if state["mol"] == "CO2" else "",
            layout=widgets.Layout(width="95px", height="34px")
        ),
        "C2H4": widgets.Button(
            description="C₂H₄",
            button_style="primary" if state["mol"] == "C2H4" else "",
            layout=widgets.Layout(width="95px", height="34px")
        ),
    }

    orb_btns = {
        "HOMO-1": widgets.Button(
            description="HOMO-1",
            button_style="" if state["orb"] != "HOMO-1" else "info",
            layout=widgets.Layout(width="105px", height="34px")
        ),
        "HOMO": widgets.Button(
            description="HOMO",
            button_style="info" if state["orb"] == "HOMO" else "",
            layout=widgets.Layout(width="105px", height="34px")
        ),
        "LUMO": widgets.Button(
            description="LUMO",
            button_style="" if state["orb"] != "LUMO" else "warning",
            layout=widgets.Layout(width="105px", height="34px")
        ),
        "LUMO+1": widgets.Button(
            description="LUMO+1",
            button_style="" if state["orb"] != "LUMO+1" else "warning",
            layout=widgets.Layout(width="105px", height="34px")
        ),
    }

    def on_mol_button_clicked(m_key: str):
        if state["mol"] == m_key:
            return
        state["mol"] = m_key

        for k, btn in mol_btns.items():
            btn.button_style = "primary" if k == m_key else ""

        # Uuden molekyylin lataus iframessa (asettuu poikittain)
        viewer_3d_box.value = render_molecule_3d_iframe(state["mol"], state["orb"], width=580, height=540)
        mo_diagram_box.value = render_mo_diagram_svg(state["mol"], state["orb"])

    def on_orb_button_clicked(o_key: str):
        if state["orb"] == o_key:
            return
        state["orb"] = o_key

        for k, btn in orb_btns.items():
            if k in ["HOMO-1", "HOMO"]:
                btn.button_style = "info" if k == o_key else ""
            else:
                btn.button_style = "warning" if k == o_key else ""

        # Päivitetään MO-kaavio oikealla
        mo_diagram_box.value = render_mo_diagram_svg(state["mol"], state["orb"])
        # Päivitetään selityslaatikko korostaen valittua orbitaalia
        fmo_explanation_box.value = render_fmo_explanation_html(state["orb"])

        # Lähetetään orbitaalivaihtokomento iframeen ILMAN iframen uudelleenlatausta!
        # Tämä säilyttää käyttäjän kääntämän orientaation ja zoomauksen täysin!
        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "orb": o_key
        })

    mol_btns["CO"].on_click(lambda _: on_mol_button_clicked("CO"))
    mol_btns["H2O"].on_click(lambda _: on_mol_button_clicked("H2O"))
    mol_btns["CO2"].on_click(lambda _: on_mol_button_clicked("CO2"))
    mol_btns["C2H4"].on_click(lambda _: on_mol_button_clicked("C2H4"))

    orb_btns["HOMO-1"].on_click(lambda _: on_orb_button_clicked("HOMO-1"))
    orb_btns["HOMO"].on_click(lambda _: on_orb_button_clicked("HOMO"))
    orb_btns["LUMO"].on_click(lambda _: on_orb_button_clicked("LUMO"))
    orb_btns["LUMO+1"].on_click(lambda _: on_orb_button_clicked("LUMO+1"))

    # Asettelu
    header_html = widgets.HTML("""
    <div style="font-size:16px; font-weight:700; color:#0f172a; margin-bottom:10px; letter-spacing:-0.01em;">
      ⚛️ Molekyyliorbitaalit (3D-pinnat & MO-korrelaatiokaavio)
    </div>
    """)

    mol_row = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; margin-right:8px;'>Molekyyli:</b>"),
        mol_btns["CO"],
        mol_btns["H2O"],
        mol_btns["CO2"],
        mol_btns["C2H4"],
    ], layout=widgets.Layout(align_items="center", margin="0 0 8px 0", gap="8px"))

    orb_row = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; margin-right:8px;'>Orbitaali:</b>"),
        orb_btns["HOMO-1"],
        orb_btns["HOMO"],
        orb_btns["LUMO"],
        orb_btns["LUMO+1"],
    ], layout=widgets.Layout(align_items="center", gap="8px"))

    control_card = widgets.VBox([
        mol_row,
        orb_row
    ], layout=widgets.Layout(
        background="#ffffff",
        padding="12px 18px",
        border_radius="8px",
        border="1px solid #cbd5e1",
        margin="0 0 14px 0"
    ))

    viewer_hint_html = widgets.HTML("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:7px 14px; background:#f8fafc; border:1px solid #cbd5e1; border-top:none; border-radius:0 0 8px 8px; font-size:12px; color:#475569;">
      <div>
        <span style="color:#0ea5e9; font-weight:600;">🔵 + Vaihe (ψ > 0)</span> &nbsp;|&nbsp;
        <span style="color:#f43f5e; font-weight:600;">🔴 - Vaihe (ψ < 0)</span> &nbsp;|&nbsp;
        <span style="color:#64748b;">⚪ H</span> &nbsp;|&nbsp;
        <span style="color:#334155;">⚫ C</span> &nbsp;|&nbsp;
        <span style="color:#dc2626;">🔴 O</span>
      </div>
      <div>
        💡 <i>Kierrä hiirellä &bull; Zoomaa rullalla &bull; Orientaatio säilyy orbitaalia vaihdettaessa</i>
      </div>
    </div>
    """)

    left_column = widgets.VBox([
        viewer_3d_box,
        viewer_hint_html
    ], layout=widgets.Layout(
        width="48%",
        min_width="480px",
        background="#ffffff",
        border_radius="8px",
        border="1px solid #cbd5e1",
        overflow="hidden"
    ))

    right_column = widgets.VBox([
        mo_diagram_box
    ], layout=widgets.Layout(
        width="52%",
        min_width="520px",
        background="#ffffff",
        border_radius="8px",
        border="1px solid #cbd5e1",
        overflow="hidden"
    ))

    main_content_row = widgets.HBox([
        left_column,
        right_column
    ], layout=widgets.Layout(width="100%", gap="14px"))

    return widgets.VBox([
        cmd_to_3d,
        header_html,
        control_card,
        main_content_row,
        fmo_explanation_box
    ], layout=widgets.Layout(width="100%", padding="4px 0"))


if __name__ == "__main__":
    studio = create_orbital_studio()
    display(studio)

