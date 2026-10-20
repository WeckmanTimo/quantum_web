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
    Palauttaa 3Dmol.js iframe -näkymän, joka sisältää kaikki orbitaalit
    kyseiselle molekyylille tai atomille. Orbitaalin vaihto tapahtuu JavaScript-tasolla
    ilman iframen uudelleenlatausta, jolloin käyttäjän kääntämä orientaatio säilyy!
    """
    cubes_data = {}
    if mol_key == "Atom":
        possible_orbs = ["1s", "2s", "2pz", "2px", "2py", "sp3", "3dz2", "3dxz", "3dyz", "3dxy", "3dx2y2"]
    elif mol_key == "C2H4":
        possible_orbs = ["HOMO-1", "HOMO", "LUMO", "LUMO+1", "HOMO_AO", "LUMO_AO"]
    else:
        possible_orbs = ["HOMO-1", "HOMO", "LUMO", "LUMO+1"]

    for orb in possible_orbs:
        fn = os.path.join(CUBES_DIR, f"{mol_key}_{orb}.cube")
        if os.path.exists(fn):
            with open(fn, "r", encoding="utf-8") as f:
                cubes_data[orb] = f.read()

    cubes_json = json.dumps(cubes_data)
    initial_cube = cubes_data.get(initial_orb_key, "")
    if not initial_cube and cubes_data:
        initial_orb_key = list(cubes_data.keys())[0]
        initial_cube = cubes_data[initial_orb_key]

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
        const cRadius = ("{mol_key}" === "Atom") ? 0.22 : 0.62;
        viewer.addStyle({{ elem: "C" }}, {{ sphere: {{ radius: cRadius, color: "#334155" }} }});
        viewer.addStyle({{ elem: "O" }}, {{ sphere: {{ radius: 0.58, color: "#DC2626" }} }});

        // Lisätään orbitaalipinnat (1s-orbitaalilla pienempi isopinta 0.008 jotta pinta on suurempi)
        addOrbitalSurfaces(baseCube, currentOrb);

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

    function addOrbitalSurfaces(cubeData, orbName) {{
      clearOrbitalSurfaces();
      const voldata = new $3Dmol.VolumeData(cubeData, "cube");
      let isoval = {FIXED_ISOVAL};
      if (orbName === "1s") {{
        isoval = 0.008; // Pienempi isopinta 1s:lle, jotta näkyvä pallopinta on selkeästi suurempi!
      }}
      // Positiivinen vaihe (+isoval)
      const sPos = viewer.addIsosurface(voldata, {{
        isoval: isoval,
        color: "#0ea5e9",
        opacity: {FIXED_OPACITY},
        smoothness: 3
      }});
      if (sPos) orbitalShapes.push(sPos);

      // Negatiivinen vaihe (-isoval)
      const sNeg = viewer.addIsosurface(voldata, {{
        isoval: -isoval,
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
      addOrbitalSurfaces(cubeData, newOrb);

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


def render_mo_diagram_svg(mol_key: str, selected_orb_key: str, lcao_mode: str = "MO") -> str:
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

        # Jos LCAO AO -tila on valittu, korostetaan molemmat 2p_z lähtö-AO:t
        is_ao_mode = (lcao_mode == "AO")
        pz1_col = lvl_hi_occ if is_ao_mode else lvl_norm
        pz1_w = 4.5 if is_ao_mode else 2.5
        svg_lines.append(line(60, 160, 115, 160, color=pz1_col, width=pz1_w))
        svg_lines.append(draw_single(87.5, 160))
        svg_lines.append(text(40, 165, "2p_z (π)", anchor="end", size=12, weight="bold" if is_ao_mode else "normal", color=pz1_col))
        if is_ao_mode:
            svg_lines.append(text(87.5, 142, "AO (1) ◄", anchor="middle", size=11, weight="bold", color=lvl_hi_occ))

        # CH2 (2)
        svg_lines.append(line(425, 300, 480, 300))
        svg_lines.append(draw_pair(452.5, 300))
        svg_lines.append(text(500, 305, "σ(sp²)", anchor="start", size=12))

        pz2_col = lvl_hi_occ if is_ao_mode else lvl_norm
        pz2_w = 4.5 if is_ao_mode else 2.5
        svg_lines.append(line(425, 160, 480, 160, color=pz2_col, width=pz2_w))
        svg_lines.append(draw_single(452.5, 160))
        svg_lines.append(text(500, 165, "2p_z (π)", anchor="start", size=12, weight="bold" if is_ao_mode else "normal", color=pz2_col))
        if is_ao_mode:
            svg_lines.append(text(452.5, 142, "► AO (2)", anchor="middle", size=11, weight="bold", color=lvl_hi_occ))

        # MOt
        svg_lines.append(line(240, 460, 300, 460))
        svg_lines.append(draw_pair(270, 460))
        svg_lines.append(text(225, 465, "σ (C-C sidos)", anchor="end", size=12))

        is_h_m1 = (selected_orb_key == "HOMO-1" and not is_ao_mode)
        c_hm1 = lvl_hi_occ if is_h_m1 else lvl_norm
        svg_lines.append(line(240, 230, 300, 230, color=c_hm1, width=4.5 if is_h_m1 else 2.5))
        svg_lines.append(draw_pair(270, 230))
        svg_lines.append(text(225, 235, "1b₁ᵤ (HOMO-1, σ)", anchor="end", size=12, color=c_hm1))
        if is_h_m1:
            svg_lines.append(text(315, 235, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_homo = (selected_orb_key == "HOMO" and not is_ao_mode)
        c_h = lvl_hi_occ if is_homo else lvl_norm
        svg_lines.append(line(240, 160, 300, 160, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(270, 160))
        svg_lines.append(text(225, 165, "1b₃ᵤ (HOMO, C=C π)", anchor="end", size=12, color=c_h))
        if is_homo:
            svg_lines.append(text(315, 165, "◄ 3D-malli (C=C π-sidos)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        is_lumo = (selected_orb_key == "LUMO" and not is_ao_mode)
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        svg_lines.append(line(240, 80, 300, 80, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(225, 85, "1b₂g* (LUMO, π*)", anchor="end", size=12, color=c_l))
        if is_lumo:
            svg_lines.append(text(315, 85, "◄ 3D-malli (π*-hajoittava)", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        is_lp1 = (selected_orb_key == "LUMO+1" and not is_ao_mode)
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm
        svg_lines.append(line(240, 35, 300, 35, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(225, 40, "3a_g* (LUMO+1, σ*)", anchor="end", size=12, color=c_lp1))
        if is_lp1:
            svg_lines.append(text(315, 40, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        if is_ao_mode:
            svg_lines.append(text(270, 125, "◄ 3D-malli: Lähtö-AO:t (2 × C 2p_z)", anchor="middle", size=12, weight="bold", color=lvl_hi_occ))

    # --------------------------------------------------------------------------
    # 2.5. H2CO (Formaldehydi) – Karbonyyliryhmän MO-kaavio ja polarisaatio
    # --------------------------------------------------------------------------
    elif mol_key == "H2CO":
        svg_lines.append(text(270, 24, "H₂CO (Formaldehydi) – Karbonyylin MO-kaavio", size=14, weight="bold"))
        svg_lines.append(text(85, 525, "H₂C-lohko", size=13, weight="bold", color=text_sub))
        svg_lines.append(text(270, 525, "H₂C=O", size=14, weight="bold"))
        svg_lines.append(text(455, 525, "Happi (O)", size=13, weight="bold", color=text_sub))

        # Korrelaatioviivat
        svg_lines.append(corr(115, 360, 240, 460))
        svg_lines.append(corr(425, 430, 300, 460))

        svg_lines.append(corr(115, 360, 240, 370))
        svg_lines.append(corr(425, 430, 300, 370))

        svg_lines.append(corr(115, 200, 240, 290))
        svg_lines.append(corr(425, 210, 300, 290))

        svg_lines.append(corr(115, 150, 240, 220))
        svg_lines.append(corr(425, 210, 300, 220))

        svg_lines.append(corr(425, 210, 300, 160))

        svg_lines.append(corr(115, 150, 240, 95))
        svg_lines.append(corr(425, 210, 300, 95))

        svg_lines.append(corr(115, 200, 240, 40))
        svg_lines.append(corr(425, 210, 300, 40))

        # Vasen: H2C (sp2 + 2pz)
        svg_lines.append(line(60, 360, 115, 360))
        svg_lines.append(draw_pair(87.5, 360))
        svg_lines.append(text(40, 365, "2s / σ", anchor="end", size=12))

        svg_lines.append(line(60, 200, 115, 200))
        svg_lines.append(draw_single(87.5, 200))
        svg_lines.append(text(40, 205, "σ(sp²)", anchor="end", size=12))

        svg_lines.append(line(60, 150, 115, 150))
        svg_lines.append(draw_single(87.5, 150))
        svg_lines.append(text(40, 155, "2p_z (π)", anchor="end", size=12))

        # Oikea: O (2s, 2p)
        svg_lines.append(line(425, 430, 480, 430))
        svg_lines.append(draw_pair(452.5, 430))
        svg_lines.append(text(500, 435, "2s", anchor="start", size=12))

        for idx, (x1, x2) in enumerate([(395, 420), (430, 455), (465, 490)]):
            svg_lines.append(line(x1, 210, x2, 210))
            if idx == 0:
                svg_lines.append(draw_pair((x1 + x2) / 2.0, 210))
            else:
                svg_lines.append(draw_single((x1 + x2) / 2.0, 210))
        svg_lines.append(text(505, 215, "2p", anchor="start", size=12))

        # Keskellä: MO-tasot
        svg_lines.append(line(240, 460, 300, 460))
        svg_lines.append(draw_pair(270, 460))
        svg_lines.append(text(225, 465, "1a₁ (σ_O)", anchor="end", size=12))

        svg_lines.append(line(240, 370, 300, 370))
        svg_lines.append(draw_pair(270, 370))
        svg_lines.append(text(225, 375, "2a₁ (σ_CH)", anchor="end", size=12))

        svg_lines.append(line(240, 290, 300, 290))
        svg_lines.append(draw_pair(270, 290))
        svg_lines.append(text(225, 295, "1b₂ (σ_CO)", anchor="end", size=12))

        # HOMO-1: π(C=O) (1b1)
        is_hm1 = (selected_orb_key == "HOMO-1")
        c_hm1 = lvl_hi_occ if is_hm1 else lvl_norm
        svg_lines.append(line(240, 220, 300, 220, color=c_hm1, width=4.5 if is_hm1 else 2.5))
        svg_lines.append(draw_pair(270, 220))
        svg_lines.append(text(225, 225, "1b₁ (HOMO-1, π_CO)", anchor="end", size=12, color=c_hm1))
        if is_hm1:
            svg_lines.append(text(315, 225, "◄ 3D-malli (C=O π-sidos)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        # HOMO: n_O (2b2) hapen vapaa pari
        is_homo = (selected_orb_key == "HOMO")
        c_h = lvl_hi_occ if is_homo else lvl_norm
        svg_lines.append(line(240, 160, 300, 160, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(270, 160))
        svg_lines.append(text(225, 165, "2b₂ (HOMO, n_O)", anchor="end", size=12, color=c_h))
        if is_homo:
            svg_lines.append(text(315, 165, "◄ 3D-malli (hapen vapaa pari)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        # LUMO: π*(C=O) (2b1*)
        is_lumo = (selected_orb_key == "LUMO")
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        svg_lines.append(line(240, 95, 300, 95, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(225, 100, "2b₁* (LUMO, π*)", anchor="end", size=12, color=c_l))
        if is_lumo:
            svg_lines.append(text(315, 100, "◄ 3D-malli (reaktiokeskus)", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        # LUMO+1: σ* (3a1*)
        is_lp1 = (selected_orb_key == "LUMO+1")
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm
        svg_lines.append(line(240, 40, 300, 40, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(225, 45, "3a₁* (LUMO+1, σ*)", anchor="end", size=12, color=c_lp1))
        if is_lp1:
            svg_lines.append(text(315, 45, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

    # --------------------------------------------------------------------------
    # 2.6. Bentseeni (C6H6) – Aromaattinen 6π-systeemi (Hückel / Frost-ympyrä)
    # --------------------------------------------------------------------------
    elif mol_key == "Benzene":
        svg_lines.append(text(270, 24, "Bentseeni (C₆H₆) – Aromaattinen π-molekyyliorbitaalikaavio", size=14, weight="bold"))
        svg_lines.append(text(270, 42, "Hückel-sääntö: 4n + 2 = 6 π-elektronia (n = 1, suljettu kuori)", size=12, color=text_sub))

        # 6 × C 2p_z lähtötaso katkoviivalla y = 220
        svg_lines.append(line(70, 220, 130, 220, color="#64748b", width=1.5, dash="3,3"))
        svg_lines.append(line(410, 220, 470, 220, color="#64748b", width=1.5, dash="3,3"))
        svg_lines.append(text(100, 240, "3 × C(2p_z)", anchor="middle", size=11, color=text_sub))
        svg_lines.append(text(440, 240, "3 × C(2p_z)", anchor="middle", size=11, color=text_sub))

        # Korrelaatiot Frost-syklille
        svg_lines.append(corr(100, 220, 240, 390))
        svg_lines.append(corr(440, 220, 300, 390))
        svg_lines.append(corr(100, 220, 225, 235))
        svg_lines.append(corr(440, 220, 315, 235))
        svg_lines.append(corr(100, 220, 225, 115))
        svg_lines.append(corr(440, 220, 315, 115))
        svg_lines.append(corr(100, 220, 240, 55))
        svg_lines.append(corr(440, 220, 300, 55))

        # π1 (a2u): Alin sitova orbitaali (kaikki lohkot samassa vaiheessa renkaan yläpuolella)
        svg_lines.append(line(240, 390, 300, 390, color=lvl_norm, width=2.5))
        svg_lines.append(draw_pair(270, 390))
        svg_lines.append(text(225, 395, "π₁ (a₂ᵤ, sitova)", anchor="end", size=12))

        # π2 & π3 (e1g): Degeneroitunut HOMO-kaksikko (y = 235, -9.17 eV)
        is_hm1 = (selected_orb_key == "HOMO-1")
        is_homo = (selected_orb_key == "HOMO")
        c_hm1 = lvl_hi_occ if is_hm1 else lvl_norm
        c_h = lvl_hi_occ if is_homo else lvl_norm

        svg_lines.append(line(195, 235, 255, 235, color=c_hm1, width=4.5 if is_hm1 else 2.5))
        svg_lines.append(draw_pair(225, 235))
        svg_lines.append(text(225, 258, "π₂ (HOMO-1)", anchor="middle", size=11, weight="bold" if is_hm1 else "normal", color=c_hm1))

        svg_lines.append(line(285, 235, 345, 235, color=c_h, width=4.5 if is_homo else 2.5))
        svg_lines.append(draw_pair(315, 235))
        svg_lines.append(text(315, 258, "π₃ (HOMO)", anchor="middle", size=11, weight="bold" if is_homo else "normal", color=c_h))

        svg_lines.append(text(180, 239, "e₁_g (HOMO pari)", anchor="end", size=12, weight="bold" if (is_hm1 or is_homo) else "normal"))

        if is_hm1:
            svg_lines.append(text(360, 239, "◄ 3D: π₂ (solmutaso C-C)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))
        elif is_homo:
            svg_lines.append(text(360, 239, "◄ 3D: π₃ (solmutaso atomien läpi)", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        # π4* & π5* (e2u): Degeneroitunut LUMO-kaksikko (y = 115, +3.62 eV)
        is_lumo = (selected_orb_key == "LUMO")
        is_lp1 = (selected_orb_key == "LUMO+1")
        c_l = lvl_hi_virt if is_lumo else lvl_norm
        c_lp1 = lvl_hi_virt if is_lp1 else lvl_norm

        svg_lines.append(line(195, 115, 255, 115, color=c_l, width=4.5 if is_lumo else 2.5))
        svg_lines.append(text(225, 96, "π₄* (LUMO)", anchor="middle", size=11, weight="bold" if is_lumo else "normal", color=c_l))

        svg_lines.append(line(285, 115, 345, 115, color=c_lp1, width=4.5 if is_lp1 else 2.5))
        svg_lines.append(text(315, 96, "π₅* (LUMO+1)", anchor="middle", size=11, weight="bold" if is_lp1 else "normal", color=c_lp1))

        svg_lines.append(text(180, 119, "e₂_u (LUMO pari)", anchor="end", size=12, weight="bold" if (is_lumo or is_lp1) else "normal"))

        if is_lumo:
            svg_lines.append(text(360, 119, "◄ 3D: π₄* (4 solmulohkoa)", anchor="start", size=12, weight="bold", color=lvl_hi_virt))
        elif is_lp1:
            svg_lines.append(text(360, 119, "◄ 3D: π₅* (vastin-LUMO)", anchor="start", size=12, weight="bold", color=lvl_hi_virt))

        # π6* (b2g): Ylin hajottava orbitaali (kaikki vierekkäiset lohkot vastakkaisissa vaiheissa)
        svg_lines.append(line(240, 55, 300, 55, color=lvl_norm, width=2.5))
        svg_lines.append(text(225, 59, "π₆* (b₂g, hajottava)", anchor="end", size=12))

        # Alateksti
        svg_lines.append(text(270, 480, "Frost–Musulin ympyrämalli: 1 alin sitova + 2 degeneroitunutta sitovaa (6 e⁻)", size=11.5, color=text_sub))
        svg_lines.append(text(270, 500, "Täydellisesti suljettu kuori takaa bentseenin aromaattisen vakauden!", size=11.5, color=text_sub))

    # --------------------------------------------------------------------------
    # 2.7. Atomi (Hiili, C) – Aufbau ja Hundin sääntö
    # --------------------------------------------------------------------------
    elif mol_key == "Atom":
        svg_lines.append(text(270, 22, "Hiiliatomi (C) – Atomiorbitaalien energiatasot", size=14, weight="bold"))
        svg_lines.append(text(270, 40, "Elektronikonfiguraatio 1s² 2s² 2p² (Aufbau & Hund)", size=12, color=text_sub))

        # Pystyakseli (Energia)
        svg_lines.append(line(50, 485, 50, 65, color="#64748b", width=2.0))
        svg_lines.append('<polygon points="50,55 45,69 55,69" fill="#64748b"/>')
        svg_lines.append(text(40, 60, "E", anchor="end", size=13, weight="bold", color="#64748b"))

        # ----------------------------------------------------------------------
        # 3d degeneroidut tasot (y = 105): 3d_xy, 3d_yz, 3d_z², 3d_xz, 3d_x²-y²
        # ----------------------------------------------------------------------
        is_3dxy = (selected_orb_key == "3dxy")
        is_3dyz = (selected_orb_key == "3dyz")
        is_3dz2 = (selected_orb_key == "3dz2")
        is_3dxz = (selected_orb_key == "3dxz")
        is_3dx2y2 = (selected_orb_key == "3dx2y2")

        col_3dxy = lvl_hi_virt if is_3dxy else lvl_norm
        col_3dyz = lvl_hi_virt if is_3dyz else lvl_norm
        col_3dz2 = lvl_hi_virt if is_3dz2 else lvl_norm
        col_3dxz = lvl_hi_virt if is_3dxz else lvl_norm
        col_3dx2y2 = lvl_hi_virt if is_3dx2y2 else lvl_norm

        w_3dxy = 4.5 if is_3dxy else 2.5
        w_3dyz = 4.5 if is_3dyz else 2.5
        w_3dz2 = 4.5 if is_3dz2 else 2.5
        w_3dxz = 4.5 if is_3dxz else 2.5
        w_3dx2y2 = 4.5 if is_3dx2y2 else 2.5

        # 3d_xy (100 - 145)
        svg_lines.append(line(100, 105, 145, 105, color=col_3dxy, width=w_3dxy))
        svg_lines.append(text(122, 124, "3d_xy", anchor="middle", size=11, weight="bold", color=col_3dxy))

        # 3d_yz (155 - 200)
        svg_lines.append(line(155, 105, 200, 105, color=col_3dyz, width=w_3dyz))
        svg_lines.append(text(177, 124, "3d_yz", anchor="middle", size=11, weight="bold", color=col_3dyz))

        # 3d_z² (210 - 255)
        svg_lines.append(line(210, 105, 255, 105, color=col_3dz2, width=w_3dz2))
        svg_lines.append(text(232, 124, "3d_z²", anchor="middle", size=11, weight="bold", color=col_3dz2))

        # 3d_xz (265 - 310)
        svg_lines.append(line(265, 105, 310, 105, color=col_3dxz, width=w_3dxz))
        svg_lines.append(text(287, 124, "3d_xz", anchor="middle", size=11, weight="bold", color=col_3dxz))

        # 3d_x²-y² (320 - 365)
        svg_lines.append(line(320, 105, 365, 105, color=col_3dx2y2, width=w_3dx2y2))
        svg_lines.append(text(342, 124, "3d_x²-y²", anchor="middle", size=11, weight="bold", color=col_3dx2y2))

        svg_lines.append(text(85, 109, "3d", anchor="end", size=13, weight="bold", color=text_main))

        if is_3dxy:
            svg_lines.append(text(122, 88, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))
        elif is_3dyz:
            svg_lines.append(text(177, 88, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))
        elif is_3dz2:
            svg_lines.append(text(232, 88, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))
        elif is_3dxz:
            svg_lines.append(text(287, 88, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))
        elif is_3dx2y2:
            svg_lines.append(text(342, 88, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))

        # ----------------------------------------------------------------------
        # 2p degeneroidut tasot (y = 205): 2p_x, 2p_y, 2p_z
        # ----------------------------------------------------------------------
        is_2px = (selected_orb_key == "2px")
        is_2py = (selected_orb_key == "2py")
        is_2pz = (selected_orb_key == "2pz")

        col_2px = lvl_hi_occ if is_2px else lvl_norm
        col_2py = lvl_hi_occ if is_2py else lvl_norm
        col_2pz = lvl_hi_virt if is_2pz else lvl_norm

        w_2px = 4.5 if is_2px else 2.5
        w_2py = 4.5 if is_2py else 2.5
        w_2pz = 4.5 if is_2pz else 2.5

        # 2p_x (130 - 180)
        svg_lines.append(line(130, 205, 180, 205, color=col_2px, width=w_2px))
        svg_lines.append(draw_single(155, 205))
        svg_lines.append(text(155, 226, "2p_x", anchor="middle", size=12, weight="bold", color=col_2px))

        # 2p_y (215 - 265)
        svg_lines.append(line(215, 205, 265, 205, color=col_2py, width=w_2py))
        svg_lines.append(draw_single(240, 205))
        svg_lines.append(text(240, 226, "2p_y", anchor="middle", size=12, weight="bold", color=col_2py))

        # 2p_z (300 - 350)
        svg_lines.append(line(300, 205, 350, 205, color=col_2pz, width=w_2pz))
        svg_lines.append(text(325, 226, "2p_z", anchor="middle", size=12, weight="bold", color=col_2pz))

        svg_lines.append(text(110, 209, "2p", anchor="end", size=13, weight="bold", color=text_main))

        if is_2px:
            svg_lines.append(text(155, 185, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_occ))
        elif is_2py:
            svg_lines.append(text(240, 185, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_occ))
        elif is_2pz:
            svg_lines.append(text(325, 185, "◄ 3D", anchor="middle", size=11, weight="bold", color=lvl_hi_virt))

        # ----------------------------------------------------------------------
        # sp^3 hybriditaso (y = 265, katkoviiva s- ja p-tasojen välissä)
        # ----------------------------------------------------------------------
        is_sp3 = (selected_orb_key == "sp3")
        col_sp3 = "#8b5cf6" if is_sp3 else "#94a3b8"
        w_sp3 = 3.5 if is_sp3 else 1.8
        dash_sp3 = "" if is_sp3 else "4,3"
        svg_lines.append(line(160, 265, 320, 265, color=col_sp3, width=w_sp3, dash=dash_sp3))
        svg_lines.append(text(145, 269, "sp³", anchor="end", size=12, weight="bold", color=col_sp3))
        if is_sp3:
            svg_lines.append(text(335, 269, "◄ 3D (tetraedri)", anchor="start", size=12, weight="bold", color="#8b5cf6"))

        # ----------------------------------------------------------------------
        # 2s (y = 340)
        # ----------------------------------------------------------------------
        is_2s = (selected_orb_key == "2s")
        col_2s = lvl_hi_occ if is_2s else lvl_norm
        w_2s = 4.5 if is_2s else 2.5
        svg_lines.append(line(170, 340, 310, 340, color=col_2s, width=w_2s))
        svg_lines.append(draw_pair(240, 340))
        svg_lines.append(text(150, 344, "2s", anchor="end", size=13, weight="bold", color=col_2s))
        if is_2s:
            svg_lines.append(text(325, 344, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        # ----------------------------------------------------------------------
        # 1s (y = 445)
        # ----------------------------------------------------------------------
        is_1s = (selected_orb_key == "1s")
        col_1s = lvl_hi_occ if is_1s else lvl_norm
        w_1s = 4.5 if is_1s else 2.5
        svg_lines.append(line(170, 445, 310, 445, color=col_1s, width=w_1s))
        svg_lines.append(draw_pair(240, 445))
        svg_lines.append(text(150, 449, "1s", anchor="end", size=13, weight="bold", color=col_1s))
        if is_1s:
            svg_lines.append(text(325, 449, "◄ 3D-malli", anchor="start", size=12, weight="bold", color=lvl_hi_occ))

        # Alareunan säännöt
        svg_lines.append('<rect x="30" y="490" width="480" height="52" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
        svg_lines.append(text(270, 510, "Aufbau-periaate: 1s < 2s < 2p < 3d täyttyy energian mukaan", anchor="middle", size=11.5, weight="bold", color=text_main))
        svg_lines.append(text(270, 528, "Hundin sääntö: degeneroidut p- ja d-tilat miehittyvät yksittäin samansuuntaisin spinein", anchor="middle", size=11, color=text_sub))

    svg_lines.append("</svg>")
    return "".join(svg_lines)


# ==============================================================================
# 3. Rajaorbitaalien (HOMO & LUMO) pedagoginen selvitys
# ==============================================================================
def render_fmo_explanation_html(
    selected_orb: str = "HOMO",
    selected_mol: str = "CO",
    lcao_mode: str = "MO"
) -> str:
    """
    Palauttaa pedagogisen ja selkeän selvityksen HOMO- ja LUMO-lyhenteistä,
    atomiorbitaalien kvanttiluvuista ja solmuista tai LCAO-periaatteesta.
    """
    if selected_mol == "Atom":
        is_s = selected_orb in ["1s", "2s"]
        is_p = selected_orb in ["2px", "2py", "2pz"]
        is_sp3 = (selected_orb == "sp3")
        is_d = selected_orb in ["3dz2", "3dxz", "3dyz", "3dxy", "3dx2y2"]

        s_border = "#0284c7" if is_s else "#e2e8f0"
        s_bg = "#f0f9ff" if is_s else "#ffffff"
        s_badge = f" <span style='background:#0284c7; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>{selected_orb.upper()} VALITTUNA</span>" if is_s else ""

        p_border = "#d97706" if is_p else ("#8b5cf6" if is_sp3 else "#e2e8f0")
        p_bg = "#fffbeb" if is_p else ("#f5f3ff" if is_sp3 else "#ffffff")
        p_badge = f" <span style='background:#d97706; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>{selected_orb.upper()} VALITTUNA</span>" if is_p else (
            " <span style='background:#8b5cf6; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>SP³ VALITTUNA</span>" if is_sp3 else ""
        )

        d_border = "#059669" if is_d else "#e2e8f0"
        d_bg = "#ecfdf5" if is_d else "#ffffff"
        d_display_map = {
            "3dz2": "3d_z²",
            "3dxz": "3d_xz",
            "3dyz": "3d_yz",
            "3dxy": "3d_xy",
            "3dx2y2": "3d_x²-y²",
        }
        d_badge_text = d_display_map.get(selected_orb, "3d")
        d_badge = f" <span style='background:#059669; color:white; font-size:10px; font-weight:700; padding:2px 7px; border-radius:10px;'>{d_badge_text.upper()} VALITTUNA</span>" if is_d else ""

        return f"""
    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:8px; padding:16px 20px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:#1e293b; box-shadow:0 1px 3px rgba(0,0,0,0.04); margin-top:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
        <h4 style="margin:0; font-size:15px; color:#0f172a; font-weight:700; display:flex; align-items:center; gap:6px;">
          ⚛️ Atomiorbitaalit (AO), aaltofunktio ψ ja kvanttiluvut
        </h4>
        <span style="font-size:11.5px; color:#64748b; background:#f1f5f9; padding:2px 8px; border-radius:12px;">Schrödingerin yhtälön ratkaisut</span>
      </div>

      <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; margin-bottom:14px;">
        <!-- s-orbitaalit ja aaltofunktio -->
        <div style="background:{s_bg}; border:2px solid {s_border}; border-radius:8px; padding:12px 15px; transition:all 0.2s ease;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13.5px; font-weight:700; color:#0284c7;">🔵 s-orbitaalit (l = 0) & Aaltofunktio ψ</span>
            {s_badge}
          </div>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>Pallosymmetrinen aalto:</b> s-orbitaaleilla sivukvanttiluku on <i>l = 0</i>. Elektronitiheys riippuu vain etäisyydestä ytimeen, ei suunnasta.
          </p>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>Aallon vaihe (+ / -):</b> 3D-kuvassa sininen väri kuvaa aaltofunktion positiivista arvoa (ψ > 0) ja punainen negatiivista (ψ < 0). Elektronin löytymistodennäköisyys on |ψ|², joten molemmat vaiheet kuvaavat yhtä lailla elektronitiheyttä.
          </p>
          <div style="font-size:11.5px; background:rgba(2,132,199,0.08); padding:4px 8px; border-radius:4px; color:#0369a1;">
            <b>Radiaalinen solmupinta:</b> 1s-orbitaalilla ei ole solmuja; 2s-orbitaalilla on 1 pallomainen solmupinta ytimen lähellä, jossa ψ vaihtaa merkkiä.
          </div>
        </div>

        <!-- p-orbitaalit ja hybridisaatio -->
        <div style="background:{p_bg}; border:2px solid {p_border}; border-radius:8px; padding:12px 15px; transition:all 0.2s ease;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13.5px; font-weight:700; color:#d97706;">🟠 p-orbitaalit (l = 1) & sp³</span>
            {p_badge}
          </div>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>Suunnatut lohkot ja kulmasolmu:</b> p-orbitaaleilla (l = 1) on kaksi vastakkaismerkkistä lohkoa (+ ja -), joiden välissä ytimen läpi kulkee tasomainen solmupinta (ψ = 0).
          </p>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>Avaruudelliset suuntaukset:</b> 2p_x, 2p_y ja 2p_z ovat keskenään kohtisuorassa (magneettinen kvanttiluku m_l = -1, 0, +1).
          </p>
          <div style="font-size:11.5px; background:rgba(139,92,246,0.1); padding:4px 8px; border-radius:4px; color:#6d28d9;">
            <b>sp³-hybridisaatio:</b> Yhdistää yhden 2s- ja kolme 2p-orbitaalia neljäksi suunnatuksi hybridiorbitaaliksi (tetraedrikulma 109.5°), kuten hiilellä metaanissa (CH₄).
          </div>
        </div>

        <!-- d-orbitaalit ja siirtymämetallit -->
        <div style="background:{d_bg}; border:2px solid {d_border}; border-radius:8px; padding:12px 15px; transition:all 0.2s ease;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:13.5px; font-weight:700; color:#059669;">🟢 d-orbitaalit (l = 2) & Siirtymämetallit</span>
            {d_badge}
          </div>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>5 avaruudellista muotoa:</b> Neljä neliapilamaista orbitaalia (<i>3d_xy, 3d_yz, 3d_xz, 3d_x²-y²</i>) ja <i>3d_z²</i>, jolla on kaksi lohkoa z-akselilla sekä toroidimainen rengas (donitsi) xy-tasossa.
          </p>
          <p style="font-size:12px; line-height:1.45; color:#475569; margin:0 0 6px 0;">
            • <b>Kaksi solmupintaa:</b> Kaikilla d-orbitaaleilla on 2 kulmasolmutasoa tai -kartiopintaa, joissa aaltofunktio vaihtaa merkkiä (ψ = 0).
          </p>
          <div style="font-size:11.5px; background:rgba(5,150,105,0.08); padding:4px 8px; border-radius:4px; color:#047857;">
            <b>Ligandikenttäteoria:</b> d-orbitaalit ovat keskeisiä siirtymämetallien sidoksissa, katalyysissä ja kompleksiyhdisteiden oktaedrisessä (t₂_g / e_g) energiatasojen hajoamisessa.
          </div>
        </div>
      </div>

      <!-- LCAO periaate -->
      <div style="background:#f8fafc; border-left:4px solid #0ea5e9; padding:9px 13px; border-radius:0 6px 6px 0; font-size:12px; line-height:1.5; color:#334155;">
        <b>💡 Miten atomiorbitaaleista syntyy molekyyliorbitaaleja (LCAO)?</b><br>
        Molekyyliorbitaaliteoria perustuu atomiorbitaalien lineaariseen yhdistämiseen (<i>Linear Combination of Atomic Orbitals, LCAO</i>): <b>ψ_MO = c₁ φ_AO,1 ± c₂ φ_AO,2</b>. 
        Kun samanvaiheiset lohkot (+ ja +) kohtaavat, aallot vahvistavat toisiaan muodostaen <b>sitovan molekyyliorbitaalin</b>. Kun erivaiheiset lohkot (+ ja -) kohtaavat, aallot kumoavat toisensa muodostaen <b>hajottavan orbitaalin</b> solmupintoineen.<br>
        <i>Valitse ylhäältä <b>C₂H₄</b>, niin voit kokeilla suoraan LCAO-näkymää ja nähdä miten kahdesta hiilen 2p_z-atomiorbitaalista syntyy eteenin π-sidos!</i>
      </div>
    </div>
    """

    # Molekyylit (CO, H2O, CO2, C2H4, H2CO, Benzene)
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

    mol_banner = ""
    if selected_mol == "C2H4" and lcao_mode == "AO":
        mol_banner = """
      <div style="background:#eff6ff; border:1px solid #3b82f6; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:12.5px; color:#1e40af;">
        <b>🔬 LCAO-tila aktiivinen (Linear Combination of Atomic Orbitals):</b><br>
        Tarkastelet nyt kahden hiiliatomin erillisiä <b>2p_z-atomiorbitaaleja</b> ennen sidoksen muodostumista. 
        Klikkaa ylhäältä <b>"Muodostunut MO"</b> nähdäksesi, miten nämä atomiorbitaalit interferoivat keskenään muodostaen joko sitovan <b>π-kaksoissidoksen (HOMO)</b> tai hajottavan <b>π*-orbitaalin (LUMO)</b> ilman että 3D-kamera kääntyy!
      </div>
        """
    elif selected_mol == "H2CO":
        mol_banner = """
      <div style="background:#f0fdf4; border:1px solid #22c55e; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:12.5px; color:#15803d;">
        <b>🎯 Formaldehydin (H₂C=O) rajaorbitaalit ja karbonyylin reaktiivisuus:</b><br>
        • <b>HOMO (2b₂):</b> Hapen ei-sitova vapaa elektronipari (n_O). Toimii elektronitiheänä emäksisenä keskuksena (esim. protonoituu happamissa reaktioissa).<br>
        • <b>LUMO (2b₁*):</b> C=O kaksoissidoksen hajottava π*-orbitaali. Elektronegatiivisuuseron vuoksi <b>LUMO:n aaltofunktio on polarisoitunut voimakkaasti hiilelle</b> (hiilen lohko on paljon happea suurempi!). Siksi nukleofiilit (kuten CN⁻, hydridit tai Grignard-reagenssit) hyökkäävät nimenomaan karbonyylihiileen! <i>(Katso havainnollistus 4. välilehdellä!)</i>
      </div>
        """
    elif selected_mol == "Benzene":
        mol_banner = """
      <div style="background:#faf5ff; border:1px solid #a855f7; border-radius:6px; padding:10px 14px; margin-bottom:12px; font-size:12.5px; color:#6b21a8;">
        <b>⬡ Bentseenin aromaattinen sekstetti ja Hückelin sääntö (4n + 2):</b><br>
        • <b>Degeneroitunut HOMO-pari (π₂ ja π₃, -9.17 eV):</b> Bentseenillä on kaksi täsmälleen samanenergistä HOMO-orbitaalia. Valitse <b>HOMO</b> tai <b>HOMO-1</b> nähdäksesi niiden toisiaan vastaan kohtisuorassa olevat solmutasot (toisella solmutaso kulkee kahden vastakkaisen hiilen läpi, toisella C-C-sidosten välistä)!<br>
        • <b>Degeneroitunut LUMO-pari (π₄* ja π₅*, +3.62 eV):</b> Kaksi tyhjää hajottavaa π*-orbitaalia, joissa on neljä erivaiheista lohkoa.<br>
        • <b>Aromaattinen vakaus:</b> Koska kaikki kolme sitovaa π-orbitaalia ovat täysin miehitettyjä (suljettu kuori) ja HOMO-LUMO-energiaväli on erittäin suuri (~12.8 eV), bentseeni on poikkeuksellisen stabiili eikä reagoi herkästi additiolla, vaan suosii elektrofiilistä aromaattista substituutiota (EAS)!
      </div>
        """

    return f"""
    <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:8px; padding:16px 20px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; color:#1e293b; box-shadow:0 1px 3px rgba(0,0,0,0.04); margin-top:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
        <h4 style="margin:0; font-size:15px; color:#0f172a; font-weight:700; display:flex; align-items:center; gap:6px;">
          📚 HOMO ja orbitaalit (Rajaorbitaalit / Frontier Molecular Orbitals)
        </h4>
        <span style="font-size:11.5px; color:#64748b; background:#f1f5f9; padding:2px 8px; border-radius:12px;">Nobel-palkittu teoria (1981)</span>
      </div>

      {mol_banner}

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
        "atom_orb": "2pz",
        "lcao_mode": "MO",
    }

    # Piilotettu ohjauskenttä Pythonista selaimen 3D-iframeen (ei riko kameraa!)
    cmd_to_3d = widgets.Text(value="", layout=widgets.Layout(display="none"))
    cmd_to_3d.add_class("orbital-cmd-to-iframe")

    viewer_3d_box = widgets.HTML(
        value=render_molecule_3d_iframe(state["mol"], state["orb"], width=580, height=540),
        layout=widgets.Layout(width="100%", height="545px")
    )

    mo_diagram_box = widgets.HTML(
        value=render_mo_diagram_svg(state["mol"], state["orb"], state["lcao_mode"]),
        layout=widgets.Layout(width="100%", height="545px")
    )

    fmo_explanation_box = widgets.HTML(
        value=render_fmo_explanation_html(state["orb"], selected_mol=state["mol"], lcao_mode=state["lcao_mode"]),
        layout=widgets.Layout(width="100%")
    )

    # Painikkeet vain kemiallisilla kaavoilla
    mol_btns = {
        "CO": widgets.Button(
            description="CO",
            button_style="primary" if state["mol"] == "CO" else "",
            layout=widgets.Layout(width="65px", height="34px")
        ),
        "H2O": widgets.Button(
            description="H₂O",
            button_style="primary" if state["mol"] == "H2O" else "",
            layout=widgets.Layout(width="65px", height="34px")
        ),
        "CO2": widgets.Button(
            description="CO₂",
            button_style="primary" if state["mol"] == "CO2" else "",
            layout=widgets.Layout(width="65px", height="34px")
        ),
        "C2H4": widgets.Button(
            description="C₂H₄",
            button_style="primary" if state["mol"] == "C2H4" else "",
            layout=widgets.Layout(width="65px", height="34px")
        ),
        "H2CO": widgets.Button(
            description="H₂CO",
            button_style="primary" if state["mol"] == "H2CO" else "",
            layout=widgets.Layout(width="70px", height="34px")
        ),
        "Benzene": widgets.Button(
            description="Bentseeni",
            button_style="primary" if state["mol"] == "Benzene" else "",
            layout=widgets.Layout(width="95px", height="34px")
        ),
        "Atom": widgets.Button(
            description="Atomi (AO)",
            button_style="primary" if state["mol"] == "Atom" else "",
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

    atom_orb_btns = {
        "1s": widgets.Button(
            description="1s",
            button_style="info" if state["atom_orb"] == "1s" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "2s": widgets.Button(
            description="2s",
            button_style="info" if state["atom_orb"] == "2s" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "2pz": widgets.Button(
            description="2p_z",
            button_style="warning" if state["atom_orb"] == "2pz" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "2px": widgets.Button(
            description="2p_x",
            button_style="info" if state["atom_orb"] == "2px" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "2py": widgets.Button(
            description="2p_y",
            button_style="info" if state["atom_orb"] == "2py" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "sp3": widgets.Button(
            description="sp³",
            button_style="success" if state["atom_orb"] == "sp3" else "",
            layout=widgets.Layout(width="72px", height="34px")
        ),
        "3dz2": widgets.Button(
            description="3d_z²",
            button_style="warning" if state["atom_orb"] == "3dz2" else "",
            layout=widgets.Layout(width="78px", height="34px")
        ),
        "3dxz": widgets.Button(
            description="3d_xz",
            button_style="warning" if state["atom_orb"] == "3dxz" else "",
            layout=widgets.Layout(width="78px", height="34px")
        ),
        "3dyz": widgets.Button(
            description="3d_yz",
            button_style="warning" if state["atom_orb"] == "3dyz" else "",
            layout=widgets.Layout(width="78px", height="34px")
        ),
        "3dxy": widgets.Button(
            description="3d_xy",
            button_style="warning" if state["atom_orb"] == "3dxy" else "",
            layout=widgets.Layout(width="78px", height="34px")
        ),
        "3dx2y2": widgets.Button(
            description="3d_x²-y²",
            button_style="warning" if state["atom_orb"] == "3dx2y2" else "",
            layout=widgets.Layout(width="84px", height="34px")
        ),
    }

    btn_lcao_mo = widgets.Button(
        description="Muodostunut MO",
        button_style="primary" if state["lcao_mode"] == "MO" else "",
        layout=widgets.Layout(width="135px", height="32px")
    )
    btn_lcao_ao = widgets.Button(
        description="Lähtö-AO:t (2 × 2p_z)",
        button_style="primary" if state["lcao_mode"] == "AO" else "",
        layout=widgets.Layout(width="165px", height="32px")
    )

    def get_c2h4_cube_key() -> str:
        if state["lcao_mode"] == "AO":
            if state["orb"] == "HOMO":
                return "HOMO_AO"
            elif state["orb"] == "LUMO":
                return "LUMO_AO"
        return state["orb"]

    def on_mol_button_clicked(m_key: str):
        if state["mol"] == m_key:
            return
        state["mol"] = m_key

        for k, btn in mol_btns.items():
            btn.button_style = "primary" if k == m_key else ""

        if m_key == "Atom":
            orb_row.layout.display = "none"
            lcao_row.layout.display = "none"
            atom_orb_row.layout.display = "flex"
            cur_orb = state["atom_orb"]
            viewer_3d_box.value = render_molecule_3d_iframe("Atom", cur_orb, width=580, height=540)
            mo_diagram_box.value = render_mo_diagram_svg("Atom", cur_orb)
            fmo_explanation_box.value = render_fmo_explanation_html(cur_orb, selected_mol="Atom")
        elif m_key == "C2H4":
            orb_row.layout.display = "flex"
            lcao_row.layout.display = "flex"
            atom_orb_row.layout.display = "none"
            cur_cube = get_c2h4_cube_key()
            viewer_3d_box.value = render_molecule_3d_iframe("C2H4", cur_cube, width=580, height=540)
            mo_diagram_box.value = render_mo_diagram_svg("C2H4", state["orb"], state["lcao_mode"])
            fmo_explanation_box.value = render_fmo_explanation_html(state["orb"], selected_mol="C2H4", lcao_mode=state["lcao_mode"])
        else:
            orb_row.layout.display = "flex"
            lcao_row.layout.display = "none"
            atom_orb_row.layout.display = "none"
            viewer_3d_box.value = render_molecule_3d_iframe(state["mol"], state["orb"], width=580, height=540)
            mo_diagram_box.value = render_mo_diagram_svg(state["mol"], state["orb"])
            fmo_explanation_box.value = render_fmo_explanation_html(state["orb"], selected_mol=state["mol"])

    def on_orb_button_clicked(o_key: str):
        state["orb"] = o_key

        for k, btn in orb_btns.items():
            if k in ["HOMO-1", "HOMO"]:
                btn.button_style = "info" if k == o_key else ""
            else:
                btn.button_style = "warning" if k == o_key else ""

        target_cube = o_key
        if state["mol"] == "C2H4":
            target_cube = get_c2h4_cube_key()

        mo_diagram_box.value = render_mo_diagram_svg(state["mol"], state["orb"], state["lcao_mode"])
        fmo_explanation_box.value = render_fmo_explanation_html(state["orb"], selected_mol=state["mol"], lcao_mode=state["lcao_mode"])

        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "orb": target_cube
        })

    def on_lcao_mode_clicked(mode: str):
        if state["lcao_mode"] == mode:
            return
        state["lcao_mode"] = mode

        btn_lcao_mo.button_style = "primary" if mode == "MO" else ""
        btn_lcao_ao.button_style = "primary" if mode == "AO" else ""

        if mode == "AO" and state["orb"] not in ["HOMO", "LUMO"]:
            state["orb"] = "HOMO"
            for k, btn in orb_btns.items():
                btn.button_style = "info" if k == "HOMO" else ""

        target_cube = get_c2h4_cube_key()
        mo_diagram_box.value = render_mo_diagram_svg("C2H4", state["orb"], state["lcao_mode"])
        fmo_explanation_box.value = render_fmo_explanation_html(state["orb"], selected_mol="C2H4", lcao_mode=state["lcao_mode"])

        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "orb": target_cube
        })

    def on_atom_orb_clicked(a_key: str):
        state["atom_orb"] = a_key

        for k, btn in atom_orb_btns.items():
            if k == "sp3":
                btn.button_style = "success" if k == a_key else ""
            elif k in ["1s", "2s", "2px", "2py"]:
                btn.button_style = "info" if k == a_key else ""
            else:
                btn.button_style = "warning" if k == a_key else ""

        mo_diagram_box.value = render_mo_diagram_svg("Atom", state["atom_orb"])
        fmo_explanation_box.value = render_fmo_explanation_html(state["atom_orb"], selected_mol="Atom")

        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "orb": a_key
        })

    mol_btns["CO"].on_click(lambda _: on_mol_button_clicked("CO"))
    mol_btns["H2O"].on_click(lambda _: on_mol_button_clicked("H2O"))
    mol_btns["CO2"].on_click(lambda _: on_mol_button_clicked("CO2"))
    mol_btns["C2H4"].on_click(lambda _: on_mol_button_clicked("C2H4"))
    mol_btns["H2CO"].on_click(lambda _: on_mol_button_clicked("H2CO"))
    mol_btns["Benzene"].on_click(lambda _: on_mol_button_clicked("Benzene"))
    mol_btns["Atom"].on_click(lambda _: on_mol_button_clicked("Atom"))

    orb_btns["HOMO-1"].on_click(lambda _: on_orb_button_clicked("HOMO-1"))
    orb_btns["HOMO"].on_click(lambda _: on_orb_button_clicked("HOMO"))
    orb_btns["LUMO"].on_click(lambda _: on_orb_button_clicked("LUMO"))
    orb_btns["LUMO+1"].on_click(lambda _: on_orb_button_clicked("LUMO+1"))

    btn_lcao_mo.on_click(lambda _: on_lcao_mode_clicked("MO"))
    btn_lcao_ao.on_click(lambda _: on_lcao_mode_clicked("AO"))

    for k, btn in atom_orb_btns.items():
        btn.on_click(lambda _, key=k: on_atom_orb_clicked(key))

    # Asettelu
    header_html = widgets.HTML("""
    <div style="font-size:16px; font-weight:700; color:#0f172a; margin-bottom:10px; letter-spacing:-0.01em;">
      ⚛️ Molekyyli- ja atomiorbitaalit (3D-pinnat, MO-kaavio & LCAO)
    </div>
    """)

    mol_row = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; margin-right:8px;'>Kohde:</b>"),
        mol_btns["CO"],
        mol_btns["H2O"],
        mol_btns["CO2"],
        mol_btns["C2H4"],
        mol_btns["H2CO"],
        mol_btns["Benzene"],
        mol_btns["Atom"],
    ], layout=widgets.Layout(align_items="center", margin="0 0 8px 0", gap="6px"))

    orb_row = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; margin-right:8px;'>Molekyyliorbitaali:</b>"),
        orb_btns["HOMO-1"],
        orb_btns["HOMO"],
        orb_btns["LUMO"],
        orb_btns["LUMO+1"],
    ], layout=widgets.Layout(align_items="center", margin="0 0 6px 0", gap="8px"))

    lcao_row = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; margin-right:8px;'>LCAO-näkymä:</b>"),
        btn_lcao_mo,
        btn_lcao_ao,
        widgets.HTML("<span style='color:#64748b; font-size:11.5px; margin-left:6px;'><i>(Kierrä molekyyliä ja vertaa hiilen 2p_z lähtöorbitaaleja ja niistä muodostuvaa π-sidosta)</i></span>"),
    ], layout=widgets.Layout(align_items="center", display="none", margin="0 0 6px 0", gap="8px"))

    atom_orb_row1 = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; min-width:130px;'>s- ja p-orbitaalit:</b>"),
        atom_orb_btns["1s"],
        atom_orb_btns["2s"],
        atom_orb_btns["2px"],
        atom_orb_btns["2py"],
        atom_orb_btns["2pz"],
        atom_orb_btns["sp3"],
    ], layout=widgets.Layout(align_items="center", gap="8px"))

    atom_orb_row2 = widgets.HBox([
        widgets.HTML("<b style='color:#334155; font-size:13px; min-width:130px;'>3d-orbitaalit (l = 2):</b>"),
        atom_orb_btns["3dz2"],
        atom_orb_btns["3dxz"],
        atom_orb_btns["3dyz"],
        atom_orb_btns["3dxy"],
        atom_orb_btns["3dx2y2"],
    ], layout=widgets.Layout(align_items="center", gap="8px"))

    atom_orb_row = widgets.VBox([
        atom_orb_row1,
        atom_orb_row2
    ], layout=widgets.Layout(display="none", margin="0 0 6px 0", gap="6px"))

    control_card = widgets.VBox([
        mol_row,
        orb_row,
        lcao_row,
        atom_orb_row
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

