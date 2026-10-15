"""
Comp-Chem-Lab: Molekyylimekaniikan ja Voimakenttäoptimoinnin Studio (MM)
========================================================================
Interaktiivinen työkalu pienen joustavan molekyylin (butaani, C4H10)
vapaaseen 3D-hiiriohjaukseen, konformaatioanalyysiin ja energiaoptimointiin.
Kaikkien 14 atomin (C ja H) liikuttelu suoraan 3D-näkymässä hiirellä.
"""
from __future__ import annotations
import html
import json
import os
import sys
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import forcefield as ff
from interactive_3d_editor import render_interactive_3d_editor


def get_conformation_name(phi: float) -> Tuple[str, str, str]:
    """Palauttaa konformaation nimen, kuvaustekstin ja väritunnuksen dihedraalikulman perusteella."""
    p = abs(phi)
    while p > 180.0:
        p = abs(360.0 - p)

    if 155.0 <= p <= 180.0:
        return "Anti (Staggered)", "Globaali energiaminimi (metyylit mahdollisimman kaukana toisistaan)", "#276749"
    elif 50.0 <= p <= 75.0:
        return "Gauche (Staggered)", "Paikallinen energiaminimi (stabiili porrastettu tila)", "#2B6CB0"
    elif 105.0 <= p <= 135.0:
        return "Eklipsoitunut C-H / C-CH₃", "Energiakynnys (C-H ja C-C sidokset kohdakkain)", "#DD6B20"
    elif p < 25.0:
        return "Syn-eklipsoitunut CH₃ / CH₃", "Korkein energiakynnys (metyyliryhmien voimakas steerinen este)", "#C53030"
    else:
        return "Vääristynyt / Välitila", "Relaksoitumaton geometrinen konformaatio", "#718096"


def create_mm_studio() -> widgets.VBox:
    """Rakentaa ja palauttaa voimakenttästudion käyttöliittymän 3D-hiiriohjauksella."""
    # Alustetaan anti-butaani
    initial_coords = ff.build_butane(phi_deg=180.0)
    current_coords = [initial_coords.copy()]
    opt_trajectory: List[Dict] = []

    # Piilotettu synkronointikenttä koordinaateille selaimesta Pythoniin
    coord_receiver = widgets.Text(value="", layout=widgets.Layout(display="none"))
    coord_receiver.add_class("butane-coord-receiver")

    # Piilotettu ohjauskenttä komennoille Pythonista selaimen 3D-editoriin
    cmd_to_3d = widgets.Text(value="", layout=widgets.Layout(display="none"))
    cmd_to_3d.add_class("butane-cmd-to-iframe")

    viewer_box = widgets.HTML()
    energy_box = widgets.HTML()
    status_box = widgets.HTML()

    # Toimintopainikkeet
    btn_optimize = widgets.Button(
        description="🚀 Minimoi energia (L-BFGS)",
        button_style="primary",
        icon="play",
        layout=widgets.Layout(width="100%", height="36px", font_weight="700")
    )
    btn_optimize.add_class("action-btn")

    btn_reset_anti = widgets.Button(
        description="↺ Anti (180°)",
        button_style="success",
        layout=widgets.Layout(width="112px", height="30px")
    )
    btn_reset_anti.add_class("preset-btn")

    btn_set_gauche = widgets.Button(
        description="🔹 Gauche (60°)",
        button_style="info",
        layout=widgets.Layout(width="116px", height="30px")
    )
    btn_set_gauche.add_class("preset-btn")

    btn_set_syn = widgets.Button(
        description="⚠️ Syn (0°)",
        button_style="danger",
        layout=widgets.Layout(width="108px", height="30px")
    )
    btn_set_syn.add_class("preset-btn")

    btn_perturb = widgets.Button(
        description="🎲 Häiritse",
        button_style="warning",
        layout=widgets.Layout(width="100px", height="30px")
    )
    btn_perturb.add_class("preset-btn")

    # Trajektoriasteikko optimoinnin tarkasteluun
    traj_slider = widgets.IntSlider(
        value=0, min=0, max=0, step=1,
        description="Optimointiaskel:",
        layout=widgets.Layout(width="420px"),
        disabled=True
    )

    # Torsioprofiilin kuvaaja (butaanin teoreettinen torsiopotentiaali 0 - 360 astetta)
    phi_grid = np.linspace(0.0, 360.0, 181)
    phi_rad = np.radians(phi_grid)
    v1, v2, v3 = ff.V1, ff.V2, ff.V3
    e_tors_profile = (0.5 * v1 * (1.0 + np.cos(phi_rad)) +
                      0.5 * v2 * (1.0 - np.cos(2.0 * phi_rad)) +
                      0.5 * v3 * (1.0 + np.cos(3.0 * phi_rad)))

    fig_tors = go.FigureWidget(
        data=[
            go.Scatter(
                x=phi_grid, y=e_tors_profile, mode="lines",
                name="Torsiopotentiaali",
                line=dict(color="#4A5568", width=2)
            ),
            go.Scatter(
                x=[180.0], y=[0.0], mode="markers",
                name="Nykyinen konformaatio",
                marker=dict(size=14, color="#DD6B20", symbol="diamond", line=dict(color="black", width=1.5))
            )
        ]
    )
    fig_tors.update_layout(
        title=dict(text="Butaanin torsiopotentiaali φ(C-C-C-C)", font=dict(size=12.5)),
        xaxis=dict(title="Dihedraalikulma φ [°]", tickvals=[0, 60, 120, 180, 240, 300, 360]),
        yaxis=dict(title="Torsioenergia [kJ/mol]"),
        width=520, height=220,
        margin=dict(l=55, r=20, t=35, b=35),
        showlegend=False
    )

    # Optimoinnin suppenemiskuvaaja
    fig_conv = go.FigureWidget(
        data=[
            go.Scatter(
                x=[0], y=[0.0], mode="lines+markers",
                name="Energia",
                line=dict(color="#2B6CB0", width=2.5),
                marker=dict(size=7, color="#2B6CB0", symbol="circle")
            ),
            go.Scatter(
                x=[0], y=[0.0], mode="markers",
                name="Aktiivinen askel",
                marker=dict(size=12, color="#DD6B20", symbol="diamond", line=dict(color="black", width=1.5))
            )
        ]
    )
    fig_conv.update_layout(
        title=dict(text="Geometrian minimoinnin eteneminen (L-BFGS)", font=dict(size=12.5)),
        xaxis=dict(title="Iteraatioaskel", dtick=5),
        yaxis=dict(title="Voimakenttäenergia E_MM [kJ/mol]"),
        width=520, height=200,
        margin=dict(l=55, r=20, t=35, b=35),
        showlegend=False
    )

    def render_energy_cards(res: Dict[str, float]) -> str:
        conf_name, conf_desc, conf_color = get_conformation_name(res["phi"])
        e_tot = res["E_tot"]
        badge_bg = "#EBF8FF" if e_tot < 5.0 else ("#FEFCBF" if e_tot < 20.0 else "#FED7D7")
        badge_color = "#2B6CB0" if e_tot < 5.0 else ("#B7791F" if e_tot < 20.0 else "#C53030")

        return f"""
        <div style="background:#F7FAFC; border:1px solid #E2E8F0; border-radius:6px; padding:10px 12px; margin-top:8px; font-size:12px;">
          <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #CBD5E0; padding-bottom:6px; margin-bottom:8px;">
            <div>
              <span style="font-weight:700; color:{conf_color}; font-size:13.5px;">{conf_name}</span><br>
              <span style="color:#718096; font-size:11px;">{conf_desc}</span>
            </div>
            <div style="background:{badge_bg}; color:{badge_color}; padding:4px 10px; border-radius:12px; font-weight:700; font-size:13px;">
              E_MM = {e_tot:.1f} kJ/mol
            </div>
          </div>

          <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:6px; text-align:center;">
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">Sidosvenytys</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_bond']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">Kulmataivutus</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_angle']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">Torsiopotentiaali</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_torsion']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">van der Waals</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_vdw']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
          </div>

          <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11.5px; color:#4A5568;">
            <span><b>Torsio φ(C-C-C-C):</b> {res['phi']:.1f}°</span>
            <span><b>r(C2-C3):</b> {res['r_cc']:.2f} Å</span>
            <span><b>∠(C-C-C):</b> {res['ccc_deg']:.1f}°</span>
          </div>
        </div>
        """

    is_optimizing = [False]

    def set_coords_and_update(coords: np.ndarray, update_3d: bool = True) -> None:
        current_coords[0] = coords.copy()
        res = ff.calc_energy_and_components(coords)

        if not viewer_box.value:
            viewer_box.value = render_interactive_3d_editor(coords, width=540, height=360, editable=True)
        elif update_3d:
            cmd_to_3d.value = json.dumps({
                "id": time.time(),
                "coords": coords.tolist(),
                "editable": not is_optimizing[0]
            })

        energy_box.value = render_energy_cards(res)

        phi = res["phi"] % 360.0
        with fig_tors.batch_update():
            fig_tors.data[1].x = [phi]
            fig_tors.data[1].y = [res["E_torsion"]]

        with fig_conv.batch_update():
            fig_conv.data[0].x = [0]
            fig_conv.data[0].y = [res["E_tot"]]
            fig_conv.data[1].x = [0]
            fig_conv.data[1].y = [res["E_tot"]]

    def on_coords_from_browser(change) -> None:
        if is_optimizing[0]:
            return
        raw = change["new"]
        if not raw:
            return
        try:
            arr = np.array(json.loads(raw), dtype=float)
            if arr.shape == (14, 3):
                set_coords_and_update(arr, update_3d=False)
                traj_slider.disabled = True
                status_box.value = "<div style='color:#2B6CB0; font-size:12px; margin-top:4px;'>✋ Rakennetta muokattu käsin hiirellä. Paina <b>Minimoi energia</b> nähdäksesi relaksaation askel askeleelta!</div>"
        except Exception:
            pass

    coord_receiver.observe(on_coords_from_browser, names="value")

    def on_optimize_clicked(_) -> None:
        if is_optimizing[0]:
            return
        is_optimizing[0] = True

        # 1. Lukitaan käyttöliittymä ja 3D-editorin atomien muokkaus välittömästi
        btn_optimize.description = "⏳ Optimoidaan..."
        btn_optimize.disabled = True
        btn_reset_anti.disabled = True
        btn_set_gauche.disabled = True
        btn_set_syn.disabled = True
        btn_perturb.disabled = True
        traj_slider.disabled = True

        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "editable": False
        })

        status_box.value = "<div style='color:#C53030; font-weight:700; font-size:12px; margin-top:4px;'>⏳ Optimointi käynnissä... Molekyylin muokkaus on lukittu.</div>"

        def run_optimization_animation():
            try:
                # Ajetaan L-BFGS-optimointi (max_iter=16 antaa n. 10-16 askelta, optimaalinen 1s animaatioon)
                opt_coords, traj = ff.optimize_butane(current_coords[0], max_iter=16)
                opt_trajectory.clear()
                opt_trajectory.extend(traj)
                total_steps = len(traj)

                # Alustetaan suppenemiskuvaaja lähtöpisteellä (askel 0)
                init_step = traj[0]
                with fig_conv.batch_update():
                    fig_conv.data[0].x = [init_step["step"]]
                    fig_conv.data[0].y = [init_step["energy"]]
                    fig_conv.data[1].x = [init_step["step"]]
                    fig_conv.data[1].y = [init_step["energy"]]

                # Päivitetään jokainen askel 3D-malliin ja kuvaajaan ~1s viiveellä
                for idx, t in enumerate(traj):
                    current_coords[0] = t["coords"].copy()

                    # 1. Päivitetään 3D-malli (ei välkettä, kamera säilyy)
                    cmd_to_3d.value = json.dumps({
                        "id": time.time(),
                        "coords": t["coords"].tolist(),
                        "editable": False,
                        "step": idx,
                        "total": total_steps
                    })

                    # 2. Tuodaan uusi piste suppenemiskuvaajaan
                    steps_so_far = [traj[k]["step"] for k in range(idx + 1)]
                    energies_so_far = [traj[k]["energy"] for k in range(idx + 1)]
                    with fig_conv.batch_update():
                        fig_conv.data[0].x = steps_so_far
                        fig_conv.data[0].y = energies_so_far
                        fig_conv.data[1].x = [t["step"]]
                        fig_conv.data[1].y = [t["energy"]]

                    # 3. Päivitetään torsiokuvaajan merkitsin
                    with fig_tors.batch_update():
                        fig_tors.data[1].x = [t["phi"] % 360.0]
                        fig_tors.data[1].y = [t["e_torsion"]]

                    # 4. Päivitetään energiakortit
                    res = ff.calc_energy_and_components(t["coords"])
                    energy_box.value = render_energy_cards(res)

                    # 5. Tilateksti
                    status_box.value = f"""
                    <div style="background:#FFF5F5; border:1px solid #FEB2B2; border-radius:6px; padding:6px 12px; margin-top:6px; font-size:12px; color:#9B2C2C;">
                      ⏳ <b>Optimointiaskel {idx + 1} / {total_steps}:</b> E_MM = <b>{t['energy']:.1f} kJ/mol</b>, φ = <b>{t['phi']:.1f}°</b><br>
                      <i>Molekyylin muokkaus on lukittu optimoinnin ajaksi.</i>
                    </div>
                    """

                    # Viive noin 1s askeleiden välillä
                    if idx < total_steps - 1:
                        time.sleep(1.0)

                # Lopputuloksen yhteenveto
                final_t = traj[-1]
                init_t = traj[0]
                delta_e = final_t["energy"] - init_t["energy"]

                traj_slider.disabled = False
                traj_slider.max = total_steps - 1
                traj_slider.value = total_steps - 1

                status_box.value = f"""
                <div style="background:#F0FFF4; border:1px solid #9AE6B4; border-radius:6px; padding:8px 12px; margin-top:6px; font-size:12px; color:#22543D;">
                  <b>✅ Optimointi valmis ({total_steps} askelta):</b><br>
                  Alkuenergia: <b>{init_t['energy']:.1f} kJ/mol</b> &rarr; Minimi: <b>{final_t['energy']:.1f} kJ/mol</b> (ΔE = <b>{delta_e:+.1f} kJ/mol</b>)<br>
                  Lopullinen geometria: φ = <b>{final_t['phi']:.1f}°</b>, r(C-C) = <b>{final_t['r_cc']:.2f} Å</b>.<br>
                  <i>Molekyylin muokkaus on taas aktiivinen! Voit tutkia askeleita liukusäätimellä tai muokata atomien paikkoja hiirellä.</i>
                </div>
                """
            finally:
                # Palautetaan muokkaus ja aktivoidaan painikkeet
                cmd_to_3d.value = json.dumps({
                    "id": time.time(),
                    "editable": True
                })
                btn_optimize.description = "🚀 Minimoi energia (L-BFGS)"
                btn_optimize.disabled = False
                btn_reset_anti.disabled = False
                btn_set_gauche.disabled = False
                btn_set_syn.disabled = False
                btn_perturb.disabled = False
                is_optimizing[0] = False

        threading.Thread(target=run_optimization_animation, daemon=True).start()

    def on_reset_anti_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = ff.build_butane(phi_deg=180.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#276749; font-size:12px; margin-top:4px;'>↺ Palautettu anti-perusrakenteeseen (180°).</div>"

    def on_set_gauche_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = ff.build_butane(phi_deg=60.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#2B6CB0; font-size:12px; margin-top:4px;'>🔹 Asetettu gauche-konformaatioon (60°).</div>"

    def on_set_syn_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = ff.build_butane(phi_deg=0.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#C53030; font-size:12px; margin-top:4px;'>⚠️ Asetettu syn-eklipsoituneeseen tilaan (0°, suuri steerinen este).</div>"

    def on_perturb_clicked(_) -> None:
        if is_optimizing[0]:
            return
        new_coords = ff.perturb_coords(current_coords[0], scale=0.25)
        set_coords_and_update(new_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#DD6B20; font-size:12px; margin-top:4px;'>🎲 Rakennetta häiritty satunnaisesti! Paina 'Minimoi energia' palauttaaksesi minimiin.</div>"

    def on_traj_slider_changed(change) -> None:
        if is_optimizing[0]:
            return
        idx = change["new"]
        if 0 <= idx < len(opt_trajectory):
            t = opt_trajectory[idx]
            current_coords[0] = t["coords"].copy()
            res = ff.calc_energy_and_components(t["coords"])
            cmd_to_3d.value = json.dumps({
                "id": time.time(),
                "coords": t["coords"].tolist(),
                "editable": True
            })
            energy_box.value = render_energy_cards(res)
            with fig_conv.batch_update():
                fig_conv.data[1].x = [t["step"]]
                fig_conv.data[1].y = [t["energy"]]
            with fig_tors.batch_update():
                fig_tors.data[1].x = [t["phi"] % 360.0]
                fig_tors.data[1].y = [t["e_torsion"]]

    btn_optimize.on_click(on_optimize_clicked)
    btn_reset_anti.on_click(on_reset_anti_clicked)
    btn_set_gauche.on_click(on_set_gauche_clicked)
    btn_set_syn.on_click(on_set_syn_clicked)
    btn_perturb.on_click(on_perturb_clicked)
    traj_slider.observe(on_traj_slider_changed, names="value")

    # Alustetaan näkymä
    set_coords_and_update(initial_coords, update_3d=True)

    # Painikkeiden ja asettelun tyylittely (varmistetaan, etteivät ne näytä välilehdiltä)
    btn_style_injector = widgets.HTML("""
    <style>
      .preset-btn button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 11.5px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
      }
      .preset-btn button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.18) !important;
      }
      .preset-btn button:active {
        transform: translateY(1px) !important;
      }
      .action-btn button {
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
      }
    </style>
    """)

    # Vasen sarake: 3D-editori ja pikaohjaukset
    presets_row = widgets.HBox([
        widgets.HTML("<b style='font-size:11.5px; color:#4A5568; line-height:28px; margin-right:4px;'>Pika-asettelut:</b>"),
        btn_reset_anti, btn_set_gauche, btn_set_syn, btn_perturb
    ], layout=widgets.Layout(margin="6px 0", gap="4px"))

    action_row = widgets.HBox([
        btn_optimize
    ], layout=widgets.Layout(margin="4px 0"))

    left_col = widgets.VBox([
        btn_style_injector,
        widgets.HTML("<h3 style='margin:0 0 6px 0; font-size:15px; color:#2D3748;'>1. Interaktiivinen 3D-molekyylieditori (Hiiriohjaus)</h3>"),
        viewer_box,
        coord_receiver,
        cmd_to_3d,
        presets_row,
        action_row,
        energy_box
    ], layout=widgets.Layout(width="560px", padding="12px", border="1px solid #CBD5E0", border_radius="8px", background="#FFFFFF"))

    # Oikea sarake: Energiaprofiili ja optimointi
    right_col = widgets.VBox([
        widgets.HTML("<h3 style='margin:0 0 6px 0; font-size:15px; color:#2D3748;'>2. Konformaatioanalyysi ja minimointi</h3>"),
        fig_tors,
        fig_conv,
        widgets.VBox([
            traj_slider,
            status_box
        ], layout=widgets.Layout(margin="4px 0"))
    ], layout=widgets.Layout(width="560px", padding="12px", border="1px solid #CBD5E0", border_radius="8px", background="#FFFFFF"))

    mm_layout = widgets.HBox([left_col, right_col], layout=widgets.Layout(gap="14px", margin="10px 0"))

    # Yläbanneri
    header = widgets.HTML("""
    <div style="background-color:#1E293B; color:white; padding:12px 18px; border-radius:8px; margin-bottom:10px; box-shadow:0 2px 4px rgba(0,0,0,0.12);">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="margin:0; font-size:19px; font-weight:700;">⚙️ Butaanin (C₄H₁₀) molekyylimekaniikka ja 3D-hiiriohjaus</h2>
          <p style="margin:3px 0 0 0; font-size:12.5px; opacity:0.9;">
            Kaikkien atomien vapaa liikuttelu hiirellä &bull; Reaaliaikainen voimakenttäenergia &bull; L-BFGS geometriaoptimointi
          </p>
        </div>
        <div style="text-align:right; font-size:11px; opacity:0.8; line-height:1.4;">
          Interaktiivinen 3D-fysiikkamalli<br>
          Konformaatiot & steerinen este
        </div>
      </div>
    </div>
    """)

    return widgets.VBox([header, mm_layout])


if __name__ == "__main__":
    from IPython.display import display
    app = create_mm_studio()
    display(app)
