"""
Comp-Chem-Lab: Bentseenirenkaan ja Hiiliketjun Studio (Propyylibentseeni)
========================================================================
Interaktiivinen työkalu bentseenirenkaan ja siihen kiinnittyneen alkyyliketjun
(propyylibentseeni, C6H5-C3H7, 21 atomia) 3D-hiiriohjaukseen ja energiaoptimointiin (L-BFGS).
"""
from __future__ import annotations
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

import benzene_forcefield as bff
from interactive_benzene_editor import render_interactive_benzene_editor


def get_benzene_conformation_name(phi_ring: float, phi_chain: float) -> Tuple[str, str, str]:
    """Palauttaa konformaation nimen, kuvauksen ja värikoodin."""
    pr = abs(phi_ring)
    while pr > 180.0:
        pr = abs(360.0 - pr)

    if 70.0 <= pr <= 110.0:
        pc = abs(phi_chain)
        while pc > 180.0:
            pc = abs(360.0 - pc)
        if 150.0 <= pc <= 180.0:
            return "Kohtisuora Anti (90°)", "Globaali energiaminimi: ketju kohtisuorassa renkaaseen, suora porrastettu ketju", "#276749"
        elif 45.0 <= pc <= 75.0:
            return "Kohtisuora Gauche", "Paikallinen energiaminimi: ketju kohtisuorassa, ketjussa taitos", "#2B6CB0"
        else:
            return "Kohtisuora välitila", "Ketju kohtisuorassa renkaaseen, ketju kiertynyt", "#319795"
    elif pr <= 25.0 or pr >= 155.0:
        return "Tason suuntainen (Koplaanaari)", "Korkea energiakynnys: ketju törmää orto-vetyyn (voimakas steerinen este)", "#C53030"
    else:
        return "Vääristynyt tila", "Relaksoitumaton konformaatio (vinossa renkaan tasoon nähden)", "#718096"


def create_benzene_studio() -> widgets.VBox:
    """Rakentaa ja palauttaa propyylibentseenin studion käyttöliittymän."""
    initial_coords = bff.build_propylbenzene(phi_ring_deg=90.0, phi_chain_deg=180.0)
    current_coords = [initial_coords.copy()]
    opt_trajectory: List[Dict] = []
    is_optimizing = [False]

    # Piilotetut synkronointikentät
    coord_receiver = widgets.Text(value="", layout=widgets.Layout(display="none"))
    coord_receiver.add_class("benzene-coord-receiver")

    cmd_to_3d = widgets.Text(value="", layout=widgets.Layout(display="none"))
    cmd_to_3d.add_class("benzene-cmd-to-iframe")

    viewer_box = widgets.HTML()
    energy_box = widgets.HTML()
    status_box = widgets.HTML()

    # Painikkeiden ja asettelun tyylittely
    btn_style_injector = widgets.HTML("""
    <style>
      .benzene-preset-btn button {
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 11.5px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12) !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease !important;
      }
      .benzene-preset-btn button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.18) !important;
      }
      .benzene-action-btn button {
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15) !important;
      }
    </style>
    """)

    # Toimintopainikkeet selkeillä väriteemoilla
    btn_optimize = widgets.Button(
        description="🚀 Minimoi energia (L-BFGS)",
        button_style="primary",
        icon="play",
        layout=widgets.Layout(width="100%", height="36px", font_weight="700")
    )
    btn_optimize.add_class("benzene-action-btn")

    btn_perp_anti = widgets.Button(
        description="⬡ Kohtisuora (90°)",
        button_style="success",
        layout=widgets.Layout(width="122px", height="30px")
    )
    btn_perp_anti.add_class("benzene-preset-btn")

    btn_perp_gauche = widgets.Button(
        description="🔹 Gauche (60°)",
        button_style="info",
        layout=widgets.Layout(width="114px", height="30px")
    )
    btn_perp_gauche.add_class("benzene-preset-btn")

    btn_coplanar = widgets.Button(
        description="⚠️ Tasossa (0°)",
        button_style="danger",
        layout=widgets.Layout(width="106px", height="30px")
    )
    btn_coplanar.add_class("benzene-preset-btn")

    btn_perturb = widgets.Button(
        description="🎲 Häiritse",
        button_style="warning",
        layout=widgets.Layout(width="98px", height="30px")
    )
    btn_perturb.add_class("benzene-preset-btn")

    # Liukusäädin optimointiaskeleille
    traj_slider = widgets.IntSlider(
        value=0, min=0, max=0, step=1,
        description="Optimointiaskel:",
        layout=widgets.Layout(width="420px"),
        disabled=True
    )

    # Suppenemiskuvaaja (L-BFGS minimointi)
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
        title=dict(text="Geometrian minimoinnin eteneminen (L-BFGS)", font=dict(size=13.0)),
        xaxis=dict(title="Iteraatioaskel", dtick=5),
        yaxis=dict(title="Voimakenttäenergia E_MM [kJ/mol]"),
        width=520, height=350,
        margin=dict(l=55, r=20, t=35, b=35),
        showlegend=False
    )

    def render_energy_cards(res: Dict[str, float]) -> str:
        conf_name, conf_desc, conf_color = get_benzene_conformation_name(res["phi_ring"], res["phi_chain"])
        e_tot = res["E_tot"]
        badge_bg = "#EBF8FF" if e_tot < 20.0 else ("#FEFCBF" if e_tot < 35.0 else "#FED7D7")
        badge_color = "#2B6CB0" if e_tot < 20.0 else ("#B7791F" if e_tot < 35.0 else "#C53030")

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
              <span style="color:#718096; font-size:10px;">Kulma & taso</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_angle']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">Torsio (rengas+ketju)</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_torsion']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-radius:4px; padding:5px 4px;">
              <span style="color:#718096; font-size:10px;">van der Waals</span><br>
              <b style="color:#2D3748; font-size:12.5px;">{res['E_vdw']:.1f}</b> <span style="font-size:9.5px;">kJ/mol</span>
            </div>
          </div>

          <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11.5px; color:#4A5568;">
            <span><b>φ_rengas (C2-C1-C7-C8):</b> {res['phi_ring']:.1f}°</span>
            <span><b>φ_ketju (C1-C7-C8-C9):</b> {res['phi_chain']:.1f}°</span>
          </div>
        </div>
        """

    def set_coords_and_update(coords: np.ndarray, update_3d: bool = True) -> None:
        current_coords[0] = coords.copy()
        res = bff.calc_energy_and_components(coords)

        if not viewer_box.value:
            viewer_box.value = render_interactive_benzene_editor(coords, width=540, height=360, editable=True)
        elif update_3d:
            cmd_to_3d.value = json.dumps({
                "id": time.time(),
                "coords": coords.tolist(),
                "editable": not is_optimizing[0]
            })

        energy_box.value = render_energy_cards(res)

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
            if arr.shape == (21, 3):
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
        btn_perp_anti.disabled = True
        btn_perp_gauche.disabled = True
        btn_coplanar.disabled = True
        btn_perturb.disabled = True
        traj_slider.disabled = True

        cmd_to_3d.value = json.dumps({
            "id": time.time(),
            "editable": False
        })

        status_box.value = "<div style='color:#C53030; font-weight:700; font-size:12px; margin-top:4px;'>⏳ Optimointi käynnissä... Molekyylin muokkaus on lukittu.</div>"

        def run_optimization_animation():
            try:
                # Ajetaan L-BFGS-optimointi (max_iter=16)
                opt_coords, traj = bff.optimize_propylbenzene(current_coords[0], max_iter=16)
                opt_trajectory.clear()
                opt_trajectory.extend(traj)
                total_steps = len(traj)

                # Alustetaan suppenemiskuvaaja lähtöpisteellä
                init_step = traj[0]
                with fig_conv.batch_update():
                    fig_conv.data[0].x = [init_step["step"]]
                    fig_conv.data[0].y = [init_step["energy"]]
                    fig_conv.data[1].x = [init_step["step"]]
                    fig_conv.data[1].y = [init_step["energy"]]

                # Askel askeleelta animaatio ~1s viiveellä
                for idx, t in enumerate(traj):
                    current_coords[0] = t["coords"].copy()

                    cmd_to_3d.value = json.dumps({
                        "id": time.time(),
                        "coords": t["coords"].tolist(),
                        "editable": False,
                        "step": idx,
                        "total": total_steps
                    })

                    steps_so_far = [traj[k]["step"] for k in range(idx + 1)]
                    energies_so_far = [traj[k]["energy"] for k in range(idx + 1)]
                    with fig_conv.batch_update():
                        fig_conv.data[0].x = steps_so_far
                        fig_conv.data[0].y = energies_so_far
                        fig_conv.data[1].x = [t["step"]]
                        fig_conv.data[1].y = [t["energy"]]

                    res = bff.calc_energy_and_components(t["coords"])
                    energy_box.value = render_energy_cards(res)

                    status_box.value = f"""
                    <div style="background:#FFF5F5; border:1px solid #FEB2B2; border-radius:6px; padding:6px 12px; margin-top:6px; font-size:12px; color:#9B2C2C;">
                      ⏳ <b>Optimointiaskel {idx + 1} / {total_steps}:</b> E_MM = <b>{t['energy']:.1f} kJ/mol</b>, φ_rengas = <b>{t['phi_ring']:.1f}°</b><br>
                      <i>Molekyylin muokkaus on lukittu optimoinnin ajaksi.</i>
                    </div>
                    """

                    if idx < total_steps - 1:
                        time.sleep(1.0)

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
                  Lopullinen geometria: φ_rengas = <b>{final_t['phi_ring']:.1f}°</b>, φ_ketju = <b>{final_t['phi_chain']:.1f}°</b>.<br>
                  <i>Molekyylin muokkaus on taas aktiivinen! Voit tutkia askeleita liukusäätimellä tai muokata atomien paikkoja hiirellä.</i>
                </div>
                """
            finally:
                cmd_to_3d.value = json.dumps({
                    "id": time.time(),
                    "editable": True
                })
                btn_optimize.description = "🚀 Minimoi energia (L-BFGS)"
                btn_optimize.disabled = False
                btn_perp_anti.disabled = False
                btn_perp_gauche.disabled = False
                btn_coplanar.disabled = False
                btn_perturb.disabled = False
                is_optimizing[0] = False

        threading.Thread(target=run_optimization_animation, daemon=True).start()

    def on_perp_anti_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = bff.build_propylbenzene(phi_ring_deg=90.0, phi_chain_deg=180.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#276749; font-size:12px; margin-top:4px;'>↺ Asetettu kohtisuoraan anti-perusrakenteeseen (90°, 180°).</div>"

    def on_perp_gauche_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = bff.build_propylbenzene(phi_ring_deg=90.0, phi_chain_deg=60.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#2B6CB0; font-size:12px; margin-top:4px;'>🔹 Asetettu kohtisuoraan gauche-konformaatioon (90°, 60°).</div>"

    def on_coplanar_clicked(_) -> None:
        if is_optimizing[0]:
            return
        base_coords = bff.build_propylbenzene(phi_ring_deg=5.0, phi_chain_deg=180.0)
        set_coords_and_update(base_coords, update_3d=True)
        traj_slider.disabled = True
        status_box.value = "<div style='color:#C53030; font-size:12px; margin-top:4px;'>⚠️ Asetettu tason suuntaiseen tilaan (~0°, voimakas steerinen este orto-vetyyn).</div>"

    def on_perturb_clicked(_) -> None:
        if is_optimizing[0]:
            return
        new_coords = bff.perturb_coords(current_coords[0], scale=0.20)
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
            res = bff.calc_energy_and_components(t["coords"])
            cmd_to_3d.value = json.dumps({
                "id": time.time(),
                "coords": t["coords"].tolist(),
                "editable": True
            })
            energy_box.value = render_energy_cards(res)
            with fig_conv.batch_update():
                fig_conv.data[1].x = [t["step"]]
                fig_conv.data[1].y = [t["energy"]]

    btn_optimize.on_click(on_optimize_clicked)
    btn_perp_anti.on_click(on_perp_anti_clicked)
    btn_perp_gauche.on_click(on_perp_gauche_clicked)
    btn_coplanar.on_click(on_coplanar_clicked)
    btn_perturb.on_click(on_perturb_clicked)
    traj_slider.observe(on_traj_slider_changed, names="value")

    # Alustetaan näkymä
    set_coords_and_update(initial_coords, update_3d=True)

    # Vasen sarake: 3D-editori ja pikaohjaukset
    presets_row = widgets.HBox([
        widgets.HTML("<b style='font-size:11.5px; color:#4A5568; line-height:28px; margin-right:4px;'>Pika-asettelut:</b>"),
        btn_perp_anti, btn_perp_gauche, btn_coplanar, btn_perturb
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

    # Oikea sarake: Geometrian minimointi
    right_col = widgets.VBox([
        widgets.HTML("<h3 style='margin:0 0 6px 0; font-size:15px; color:#2D3748;'>2. Geometrian minimointi (L-BFGS)</h3>"),
        fig_conv,
        widgets.VBox([
            traj_slider,
            status_box
        ], layout=widgets.Layout(margin="4px 0"))
    ], layout=widgets.Layout(width="560px", padding="12px", border="1px solid #CBD5E0", border_radius="8px", background="#FFFFFF"))

    benzene_layout = widgets.HBox([left_col, right_col], layout=widgets.Layout(gap="14px", margin="10px 0"))

    # Yläbanneri
    header = widgets.HTML("""
    <div style="background-color:#1E293B; color:white; padding:12px 18px; border-radius:8px; margin-bottom:10px; box-shadow:0 2px 4px rgba(0,0,0,0.12);">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <h2 style="margin:0; font-size:19px; font-weight:700;">⬡ Bentseenirengas & alkyyliketju (Propyylibentseeni, C₉H₁₂)</h2>
          <p style="margin:3px 0 0 0; font-size:12.5px; opacity:0.9;">
            Kaikkien 21 atomin vapaa liikuttelu hiirellä &bull; Aromaattinen tasorakenne &bull; Steerinen este orto-vetyyn &bull; L-BFGS minimointi
          </p>
        </div>
        <div style="text-align:right; font-size:11px; opacity:0.8; line-height:1.4;">
          Interaktiivinen 3D-fysiikkamalli<br>
          L-BFGS energiaminimointi
        </div>
      </div>
    </div>
    """)

    return widgets.VBox([header, benzene_layout])


if __name__ == "__main__":
    from IPython.display import display
    app = create_benzene_studio()
    display(app)
