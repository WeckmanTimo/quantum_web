"""
Comp-Chem-Lab: Nukleofiilisen addition 3D-demonstraatiostudio (Formaldehydi + Syanidi CN-)
=======================================================================================
Havainnollistaa orgaanisen kemian perusreaktion: nukleofiilimolekyylin (syanidi-ioni, CN-)
hyökkäyksen karbonyylin (formaldehydi, H₂CO) LUMO-orbitaalille.

Reaktio: H₂C=O + CN⁻ ➔ NC–CH₂–O⁻ (Syanohydriinin muodostuminen, uusi C-C sidos)

Tilat:
0. Pelkkä atomirakenne (H₂CO + CN⁻ hyökkäysasennossa ilman orbitaalipintoja)
1. Karbonyyli (H₂CO) LUMO (π*)
2. Nukleofiili (CN⁻) HOMO (hiilen sp-vapaa elektronipari)
3. Hyökkäys: Erilliset orbitaalit (Bürgi-Dunitz -asento, nukleofiili eri sävyllä!)
4. Hyökkäys: Muodostunut σ-sidos (hybridisoitunut / rakentava interferenssi)
5. Hyökkäys: Hajoittava σ*-orbitaali
"""
from __future__ import annotations
import html
import json
import os
import time
from typing import Dict

import ipywidgets as widgets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
CUBES_DIR = os.path.join(SCRIPT_DIR, "cubes")

FIXED_ISOVAL = 0.03
FIXED_OPACITY = 1.0

CUBE_FILES = {
    "carb_only": "addition_carbonyl_lumo.cube",
    "nuc_only": "addition_nucleophile_homo.cube",
    "carb_comp": "addition_carb_in_comp.cube",
    "nuc_comp": "addition_nuc_in_comp.cube",
    "bonding": "addition_hybridized_bonding.cube",
    "antibonding": "addition_hybridized_antibonding.cube",
    "homo": "addition_hybridized_homo.cube",
}

EXPLANATIONS: Dict[str, Dict[str, str]] = {
    "structure": {
        "title": "Atomirakenne – Formaldehydi ja syanidi-ioni (H₂C=O + CN⁻)",
        "tag": "Pelkkä molekyyligeometria",
        "tag_color": "#475569",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Reaktion molekyylirakenne ja atomigeometria ilman orbitaalipintoja:
          elektrofiilinen <b>formaldehydi (H₂C=O)</b> ja lähestyvä nukleofiilinen <b>syanidi-ioni (C≡N⁻)</b>.
        </p>
        <div style="background:#f8fafc; border-left:3px solid #475569; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#334155; line-height:1.5;">
          <b>Bürgi–Dunitz -lähestymiskulma ja -geometria:</b><br>
          • Syanidi-ionin hiiliatomi lähestyy karbonyylihiiltä <b>~107° kulmassa</b> suhteessa C=O-sidokseen.<br>
          • Lähestymisetäisyys reaktion alkutilassa on <b>2.05 Å</b> (ennen uuden kovalenttisen C–C-sidoksen muodostumista).<br>
          • Formaldehydin hiiliatomi on vielä tasomainen (sp²-hybridisaatio), mutta siirtyy reaktion edetessä kohti tetraedristä sp³-rakennetta.
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • <b>Atomivärit (CPK)</b>: ⚪ Vety (H), ⚫ Hiili (C), 🔴 Happi (O), 🔵 Typpi (N).<br>
          • <i>Kierrä molekyyliparia hiirellä ja tarkastele lähestymiskulmaa ja atomien sijoittumista avaruudessa mistä tahansa kulmasta!</i>
        </p>
        """
    },
    "carbonyl": {
        "title": "1. Karbonyyli (H₂CO) – LUMO (π*)",
        "tag": "Elektrofiili",
        "tag_color": "#2563eb",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Formaldehydin alhaisin miehittämätön molekyyliorbitaali (<b>LUMO</b>) on karbonyyliryhmän
          hajottava <b>π*-orbitaali</b>.
        </p>
        <div style="background:#f8fafc; border-left:3px solid #2563eb; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#334155; line-height:1.5;">
          <b>Miksi lohko on suurempi hiilessä?</b><br>
          Happi on hiiltä elektronegatiivisempi, joten sitovan π-orbitaalin elektronit keskittyvät hapelle.
          Vastavuoroisesti hajottava π*-orbitaali työntyy voimakkaammin <b>hiiliatomin puolelle</b>.
          Tästä syystä saapuva nukleofiili kohdistaa hyökkäyksensä nimenomaan hiileen!
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • <b>Värit</b>: 🔵 Sininen (+ vaihe) ja 🔴 Punainen (- vaihe).<br>
          • Nooditaso kulkee C=O-tason läpi, ja vastaanottavat lohkot osoittavat tason ylä- ja alapuolelle.
        </p>
        """
    },
    "nucleophile": {
        "title": "2. Nukleofiili (CN⁻) – HOMO (sp-vapaa pari)",
        "tag": "Nukleofiilimolekyyli",
        "tag_color": "#3b82f6",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Syanidi-ionin (C≡N⁻) korkein miehitetty molekyyliorbitaali (<b>HOMO</b>) on hiiliatomin
          suuntautunut <b>sp-hybridiorbitaali</b> (vapaa elektronipari).
        </p>
        <div style="background:#f0f9ff; border-left:3px solid #3b82f6; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#0369a1; line-height:1.5;">
          <b>Hiilinukleofiili & uuden C–C-sidoksen muodostaja:</b><br>
          Syanidi on kaksiaatominen molekyylianioni. Sen hiilessä oleva vapaa elektronipari työntyy
          suoraan ulospäin C≡N-akselin suuntaisesti ja toimii erittäin voimakkaana nukleofiilisenä keskuksena.
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • <b>Värit</b>: 🔷 Hieman vaaleampi sininen (+ vaihe) ja 🌸 Punainen (- vaihe).<br>
          • Suuri suuntautunut sininen lohko hiilessä osoittaa suoraan hyökkäyssuuntaan.
        </p>
        """
    },
    "unmixed": {
        "title": "3. Hyökkäysasento – Erilliset orbitaalit",
        "tag": "Kohtaaminen & Liitos",
        "tag_color": "#2563eb",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Syanidi lähestyy karbonyylihiiltä klassisessa <b>Bürgi–Dunitz -kulmassa</b> (~107° suhteessa C=O-sidokseen, etäisyys 2.05 Å).
        </p>
        <div style="background:#f0f9ff; border-left:3px solid #2563eb; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#0c4a6e; line-height:1.5;">
          <b>Erilliset orbitaalit hyvin lähellä toisiaan olevilla sävyillä:</b><br>
          • <b>Karbonyyli (H₂CO)</b>: 🔵 Sininen ja 🔴 Punainen.<br>
          • <b>Nukleofiili (CN⁻)</b>: 🔷 Hieman vaaleampi sininen ja 🌸 Punainen.<br>
          Kiinteät, peittävät pinnat (opasiteetti 1.0) näyttävät tarkasti, miten syanidin hiilen sininen lohko
          kohtaa karbonyylihiilen samanvaiheisen sinisen π*-lohkon ennen sidoksen muodostumista.
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • <b>Miksi kulma on ~107°?</b> C=O-sidoksen happi hylkii nukleofiiliä, ja hiilen π*-lohko osoittaa hieman taaksepäin.<br>
          • Molekyylit ovat täydellisessä linjassa rakentavaa interferenssiä varten!
        </p>
        """
    },
    "bonding": {
        "title": "4. Hybridisoitunut – Muodostunut σ(C–C)-sidos",
        "tag": "Rakentava vuorovaikutus",
        "tag_color": "#2563eb",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Kvanttimekaaninen vuorovaikutus ja hybridisoituminen: uusi <b>σ(C–C)-sidos</b> muodostuu!
        </p>
        <div style="background:#eff6ff; border-left:3px solid #2563eb; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#1e3a8a; line-height:1.5;">
          <b>Rakentava interferenssi (Constructive Overlap):</b><br>
          Syanidin hiilen elektronipari ja karbonyylihiilen π*-lohko sulautuvat yhtenäiseksi kovalenttiseksi sidoselektronipilveksi.
          Kaksi erillistä orbitaalia yhdistyy yhdeksi vahvaksi C–C-sidokseksi!
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • C=O-sidoksen π-elektronit siirtyvät hapelle uudeksi vapaaksi pariksi.<br>
          • Tasomainen sp²-karbonyylihiili taipuu kohti tetraedristä sp³-muotoa (syntyy syanohydriinianioni, NC–CH₂–O⁻).
        </p>
        """
    },
    "antibonding": {
        "title": "5. Hajoittava σ*(C–C)-orbitaali",
        "tag": "Hajoittava vastinpari",
        "tag_color": "#e11d48",
        "html": """
        <p style="margin-top:0; color:#1e293b; line-height:1.55; font-size:13px;">
          Uuden C–C-sidoksen vastinpari: <b>hajoittava σ*(C–C)-orbitaali</b>.
        </p>
        <div style="background:#fef2f2; border-left:3px solid #e11d48; padding:8px 12px; margin:10px 0; font-size:12.5px; color:#991b1b; line-height:1.5;">
          <b>Noodipinta hiiliatomien välissä:</b><br>
          Karbonyylihiilen ja syanidihiilen väliin syntyy selkeä solmupinta (vaihe vaihtuu sinisestä punaiseen), jossa elektronitiheys on nolla.
        </div>
        <p style="color:#475569; font-size:12.5px; line-height:1.5;">
          • Tämä orbitaali on tyhjä perustilassa, mutta sen muoto vahvistaa MO-teorian ennusteen sidoksen ja vastinsidoksen parillisuudesta.
        </p>
        """
    }
}


def render_addition_3d_iframe(initial_mode: str = "unmixed", height: int = 540) -> str:
    """
    Palauttaa 3Dmol.js -iframen, joka sisältää kaikki formaldehydi + syanidi -addition tilat.
    Tilan vaihto tapahtuu lennossa ilman kameran nollausta.
    """
    cubes_data = {}
    for key, fname in CUBE_FILES.items():
        fpath = os.path.join(CUBES_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                cubes_data[key] = f.read()

    cubes_json = json.dumps(cubes_data)

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
    let currentMode = "{initial_mode}";
    let viewer = null;
    let orbitalShapes = [];

    function initViewer() {{
      const container = document.getElementById("viewer-container");
      viewer = $3Dmol.createViewer(container, {{
        backgroundColor: "#ffffff",
        antialias: true
      }});
      loadState(currentMode, true);
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

    function setAtomStyles() {{
      viewer.setStyle({{}}, {{
        stick: {{ radius: 0.14, color: "#64748b" }}
      }});
      viewer.addStyle({{ elem: "H" }}, {{ sphere: {{ radius: 0.42, color: "#FFFFFF" }} }});
      viewer.addStyle({{ elem: "C" }}, {{ sphere: {{ radius: 0.62, color: "#334155" }} }});
      viewer.addStyle({{ elem: "O" }}, {{ sphere: {{ radius: 0.58, color: "#DC2626" }} }});
      viewer.addStyle({{ elem: "N" }}, {{ sphere: {{ radius: 0.60, color: "#2563EB" }} }});
    }}

    function loadState(modeKey, initial = false) {{
      currentMode = modeKey;
      if (!viewer) return;

      clearOrbitalSurfaces();

      if (modeKey === "structure") {{
        // Pelkkä atomirakenne: molempien molekyylien atomit hyökkäysasennossa ilman orbitaaleja
        const cBase = CUBES["carb_comp"];
        if (cBase) {{
          viewer.removeAllModels();
          viewer.addModel(cBase, "cube");
          setAtomStyles();
        }}
      }}
      else if (modeKey === "carbonyl") {{
        // Vain formaldehydi yksin
        const cData = CUBES["carb_only"];
        if (cData) {{
          viewer.removeAllModels();
          viewer.addModel(cData, "cube");
          setAtomStyles();

          const voldata = new $3Dmol.VolumeData(cData, "cube");
          const sPos = viewer.addIsosurface(voldata, {{
            isoval: {FIXED_ISOVAL},
            color: "#2563eb", // Sininen
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sPos) orbitalShapes.push(sPos);

          const sNeg = viewer.addIsosurface(voldata, {{
            isoval: -{FIXED_ISOVAL},
            color: "#e11d48", // Punainen
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNeg) orbitalShapes.push(sNeg);
        }}
      }}
      else if (modeKey === "nucleophile") {{
        // Vain syanidi yksin
        const cData = CUBES["nuc_only"];
        if (cData) {{
          viewer.removeAllModels();
          viewer.addModel(cData, "cube");
          setAtomStyles();

          const voldata = new $3Dmol.VolumeData(cData, "cube");
          // Nukleofiili hyvin lähellä olevalla vaaleammalla sinisellä sävyllä
          const sPos = viewer.addIsosurface(voldata, {{
            isoval: {FIXED_ISOVAL},
            color: "#3b82f6", // Hieman vaaleampi sininen
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sPos) orbitalShapes.push(sPos);

          const sNeg = viewer.addIsosurface(voldata, {{
            isoval: -{FIXED_ISOVAL},
            color: "#f43f5e", // Punainen
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNeg) orbitalShapes.push(sNeg);
        }}
      }}
      else if (modeKey === "unmixed") {{
        // Hyökkäysasento: molemmat molekyylit, orbitaalit ERILLÄÄN hyvin lähellä toisiaan olevin sävyin!
        const cBase = CUBES["carb_comp"];
        const cNuc = CUBES["nuc_comp"];
        if (cBase && cNuc) {{
          viewer.removeAllModels();
          viewer.addModel(cBase, "cube"); // Sisältää kaikki 6 atomia
          setAtomStyles();

          // 1. Karbonyylin LUMO (Sininen & Punainen, kiinteä pinta)
          const volCarb = new $3Dmol.VolumeData(cBase, "cube");
          const sCarbPos = viewer.addIsosurface(volCarb, {{
            isoval: {FIXED_ISOVAL},
            color: "#2563eb", // Sininen (+ vaihe)
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sCarbPos) orbitalShapes.push(sCarbPos);

          const sCarbNeg = viewer.addIsosurface(volCarb, {{
            isoval: -{FIXED_ISOVAL},
            color: "#e11d48", // Punainen (- vaihe)
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sCarbNeg) orbitalShapes.push(sCarbNeg);

          // 2. Nukleofiilin HOMO lähes samalla, aavistuksen vaaleammalla sinisellä
          const volNuc = new $3Dmol.VolumeData(cNuc, "cube");
          const sNucPos = viewer.addIsosurface(volNuc, {{
            isoval: {FIXED_ISOVAL},
            color: "#3b82f6", // Hieman vaaleampi sininen (+ vaihe)
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNucPos) orbitalShapes.push(sNucPos);

          const sNucNeg = viewer.addIsosurface(volNuc, {{
            isoval: -{FIXED_ISOVAL},
            color: "#f43f5e", // Punainen (- vaihe)
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNucNeg) orbitalShapes.push(sNucNeg);
        }}
      }}
      else if (modeKey === "bonding") {{
        // Vuorovaikuttanut kompleksi: muodostunut sigma(C-C)-sidos
        const cData = CUBES["bonding"];
        if (cData) {{
          viewer.removeAllModels();
          viewer.addModel(cData, "cube");
          setAtomStyles();

          const voldata = new $3Dmol.VolumeData(cData, "cube");
          const sPos = viewer.addIsosurface(voldata, {{
            isoval: {FIXED_ISOVAL},
            color: "#2563eb", // Yhtenäinen sininen sidos
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sPos) orbitalShapes.push(sPos);

          const sNeg = viewer.addIsosurface(voldata, {{
            isoval: -{FIXED_ISOVAL},
            color: "#e11d48",
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNeg) orbitalShapes.push(sNeg);
        }}
      }}
      else if (modeKey === "antibonding") {{
        // Hajoittava sigma*(C-C)-orbitaali
        const cData = CUBES["antibonding"];
        if (cData) {{
          viewer.removeAllModels();
          viewer.addModel(cData, "cube");
          setAtomStyles();

          const voldata = new $3Dmol.VolumeData(cData, "cube");
          const sPos = viewer.addIsosurface(voldata, {{
            isoval: {FIXED_ISOVAL},
            color: "#2563eb",
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sPos) orbitalShapes.push(sPos);

          const sNeg = viewer.addIsosurface(voldata, {{
            isoval: -{FIXED_ISOVAL},
            color: "#e11d48",
            opacity: {FIXED_OPACITY},
            smoothness: 3
          }});
          if (sNeg) orbitalShapes.push(sNeg);
        }}
      }}

      if (initial) {{
        viewer.zoomTo();
        // Asetetaan viisto kulma, josta sekä C=O-taso että CN-lähestymiskulma näkyvät kauniisti
        viewer.rotate(24, "y");
        viewer.rotate(-16, "x");
      }}
      viewer.render();
    }}

    function setMode(newMode) {{
      if (newMode === currentMode && (orbitalShapes.length > 0 || currentMode === "structure")) return;
      loadState(newMode, false);
    }}

    // Kuunnellaan komentoja isännältä
    let lastCmd = "";
    function checkParentCommand() {{
      try {{
        const inp = window.parent.document.querySelector(".addition-cmd-to-iframe input");
        if (inp && inp.value && inp.value !== lastCmd) {{
          lastCmd = inp.value;
          const cmd = JSON.parse(inp.value);
          if (cmd && cmd.mode) {{
            setMode(cmd.mode);
          }}
        }}
      }} catch (err) {{}}
    }}
    setInterval(checkParentCommand, 40);

    window.addEventListener("message", (e) => {{
      if (e.data && e.data.mode) {{
        setMode(e.data.mode);
      }}
    }});

    window.addEventListener("DOMContentLoaded", initViewer);
  </script>
</body>
</html>
"""
    escaped_html = html.escape(inner_html, quote=True)
    return (
        f'<iframe class="addition-3d-iframe" srcdoc="{escaped_html}" '
        f'style="width:100%; height:{height}px; border:none; border-radius:8px; background:#ffffff;" '
        f'frameborder="0"></iframe>'
    )


def create_nucleophilic_addition_studio() -> widgets.VBox:
    """Luo ja palauttaa nukleofiilisen addition demonstraatiostudion."""
    state = {
        "mode": "unmixed"
    }

    cmd_to_3d = widgets.Text(value="", layout=widgets.Layout(display="none"))
    cmd_to_3d.add_class("addition-cmd-to-iframe")

    viewer_3d_box = widgets.HTML(
        value=render_addition_3d_iframe(state["mode"], height=520),
        layout=widgets.Layout(width="100%", height="525px")
    )

    def get_info_html(mode: str) -> str:
        info = EXPLANATIONS.get(mode, EXPLANATIONS["unmixed"])
        return f"""
        <div style="background:#ffffff; border-radius:8px; padding:16px 20px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; height:100%; box-sizing:border-box;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
            <div style="font-size:15px; font-weight:700; color:#0f172a;">{info['title']}</div>
            <span style="background:{info['tag_color']}; color:#ffffff; font-size:11px; font-weight:600; padding:2px 8px; border-radius:12px;">{info['tag']}</span>
          </div>
          {info['html']}
        </div>
        """

    info_box = widgets.HTML(
        value=get_info_html(state["mode"]),
        layout=widgets.Layout(width="100%", height="525px")
    )

    # Painikkeet
    buttons = {
        "structure": widgets.Button(
            description="Pelkkä atomirakenne",
            button_style="primary" if state["mode"] == "structure" else "",
            layout=widgets.Layout(width="165px", height="36px")
        ),
        "carbonyl": widgets.Button(
            description="Karbonyyli (H₂CO LUMO)",
            button_style="" if state["mode"] != "carbonyl" else "primary",
            layout=widgets.Layout(width="175px", height="36px")
        ),
        "nucleophile": widgets.Button(
            description="Nukleofiili (CN⁻ HOMO)",
            button_style="" if state["mode"] != "nucleophile" else "success",
            layout=widgets.Layout(width="165px", height="36px")
        ),
        "unmixed": widgets.Button(
            description="Hyökkäys: Erilliset orbitaalit",
            button_style="warning" if state["mode"] == "unmixed" else "",
            layout=widgets.Layout(width="195px", height="36px")
        ),
        "bonding": widgets.Button(
            description="Hyökkäys: Muodostunut σ-sidos",
            button_style="info" if state["mode"] == "bonding" else "",
            layout=widgets.Layout(width="200px", height="36px")
        ),
        "antibonding": widgets.Button(
            description="Hajoittava σ*",
            button_style="danger" if state["mode"] == "antibonding" else "",
            layout=widgets.Layout(width="120px", height="36px")
        ),
    }

    def on_button_click(m_key: str):
        if state["mode"] == m_key:
            return
        state["mode"] = m_key

        for k, btn in buttons.items():
            if k == m_key:
                if k == "structure":
                    btn.button_style = "primary"
                elif k == "carbonyl":
                    btn.button_style = "primary"
                elif k == "nucleophile":
                    btn.button_style = "success"
                elif k == "unmixed":
                    btn.button_style = "warning"
                elif k == "bonding":
                    btn.button_style = "info"
                elif k == "antibonding":
                    btn.button_style = "danger"
            else:
                btn.button_style = ""

        # Päivitetään selityslaatikko
        info_box.value = get_info_html(m_key)

        # Lähetetään komento iframeen (säilyttää käyttäjän kiertämän kameran!)
        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "mode": m_key
        })

    for k, btn in buttons.items():
        btn.on_click(lambda _, key=k: on_button_click(key))

    header_html = widgets.HTML("""
    <div style="font-size:16px; font-weight:700; color:#0f172a; margin-bottom:4px; letter-spacing:-0.01em;">
      🎯 Nukleofiilinen additio karbonyylin LUMOlle
    </div>
    <div style="font-size:13px; color:#64748b; margin-bottom:12px;">
      Reaktio: <b>H₂C=O + CN⁻ ➔ NC–CH₂–O⁻</b> &nbsp;|&nbsp;
      Syanidin hiilen sp-elektronipari hyökkää formaldehydin π*-LUMO-lohkoon Bürgi–Dunitz -kulmassa (~107°).
    </div>
    """)

    btn_row = widgets.HBox([
        buttons["structure"],
        buttons["carbonyl"],
        buttons["nucleophile"],
        buttons["unmixed"],
        buttons["bonding"],
        buttons["antibonding"],
    ], layout=widgets.Layout(align_items="center", gap="8px", margin="0 0 12px 0", flex_wrap="wrap"))

    control_card = widgets.VBox([
        btn_row
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
        <span style="color:#2563eb; font-weight:600;">🔵 H₂CO (+)</span> &nbsp;|&nbsp;
        <span style="color:#e11d48; font-weight:600;">🔴 H₂CO (-)</span> &nbsp;|&nbsp;
        <span style="color:#3b82f6; font-weight:600;">🔷 CN⁻ (+)</span> &nbsp;|&nbsp;
        <span style="color:#f43f5e; font-weight:600;">🌸 CN⁻ (-)</span> &nbsp;|&nbsp;
        <span style="color:#334155;">⚫ C</span> &nbsp;
        <span style="color:#dc2626;">🔴 O</span> &nbsp;
        <span style="color:#2563eb;">🔵 N</span> &nbsp;
        <span style="color:#64748b;">⚪ H</span>
      </div>
      <div>
        💡 <i>Kiinteät 3D-pinnat (peittävä opasiteetti 1.0) &bull; Orientaatio säilyy</i>
      </div>
    </div>
    """)

    left_column = widgets.VBox([
        viewer_3d_box,
        viewer_hint_html
    ], layout=widgets.Layout(
        width="58%",
        min_width="540px",
        background="#ffffff",
        border_radius="8px",
        border="1px solid #cbd5e1",
        overflow="hidden"
    ))

    right_column = widgets.VBox([
        info_box
    ], layout=widgets.Layout(
        width="42%",
        min_width="400px",
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
        main_content_row
    ], layout=widgets.Layout(width="100%", padding="4px 0"))


if __name__ == "__main__":
    studio = create_nucleophilic_addition_studio()
    display(studio)
