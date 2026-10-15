"""
Interactive 3D Molecular Editor for Propylbenzene with Mouse Dragging (Three.js)
================================================================================
Mahdollistaa kaikkien propyylibentseenin 21 atomin (9 C, 12 H) vapaan
liikuttamisen suoraan 3D-näkymässä hiirellä raahaamalla.
Aromaattinen bentseenirengas + joustava alkyyliketju.
Reaaliaikainen voimakenttäenergian laskenta (60 FPS) ja kaksisuuntainen
synkronointi Python-taustaan.
"""
from __future__ import annotations
import html
import json
from typing import List
import numpy as np

import benzene_forcefield as bff

ELEMENTS = bff.ELEMENTS
ATOM_NAMES = bff.ATOM_NAMES
BONDS = bff.BONDS
ANGLES = bff.ANGLES
IMPROPERS = bff.IMPROPERS
LJ_PAIRS = bff.LJ_PAIRS
V2_RING = bff.V2_RING
V1_CHAIN = bff.V1_CHAIN
V2_CHAIN = bff.V2_CHAIN
V3_CHAIN = bff.V3_CHAIN


def render_interactive_benzene_editor(
    coords: np.ndarray,
    width: int = 560,
    height: int = 400,
    container_id: str = "benzene_editor",
    editable: bool = True
) -> str:
    """
    Palauttaa upotettavan iframe-HTML-koodin propyylibentseenin 3D-editorille.
    """
    initial_coords_json = json.dumps(coords.tolist())
    bonds_json = json.dumps([{"i": b[0], "j": b[1], "r0": b[2], "kb": b[3]} for b in BONDS])
    angles_json = json.dumps([{"i": a[0], "j": a[1], "k": a[2], "th0": a[3], "kth": a[4]} for a in ANGLES])
    impropers_json = json.dumps([{"i": m[0], "j": m[1], "k": m[2], "l": m[3], "koop": m[4]} for m in IMPROPERS])
    lj_pairs_json = json.dumps([{"i": p[0], "j": p[1], "sigma": p[2], "epsilon": p[3]} for p in LJ_PAIRS])
    atom_names_json = json.dumps(ATOM_NAMES)
    elements_json = json.dumps(ELEMENTS)
    editable_js = "true" if editable else "false"

    inner_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      user-select: none;
      background: radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 100%);
    }}
    #canvas-container {{
      width: 100vw;
      height: 100vh;
      position: absolute;
      top: 0;
      left: 0;
    }}
    .hud-compact {{
      position: absolute;
      top: 8px;
      left: 8px;
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 6px;
      padding: 3px 8px;
      color: #f8fafc;
      font-size: 11px;
      backdrop-filter: blur(6px);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
      pointer-events: none;
      display: flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
      z-index: 5;
    }}
    .badge {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 10.5px;
    }}
    .badge-perp {{ background: #166534; color: #bbf7d0; }}
    .badge-gauche {{ background: #1e40af; color: #bfdbfe; }}
    .badge-coplanar {{ background: #991b1b; color: #fecaca; }}
    .badge-other {{ background: #475569; color: #cbd5e1; }}
    
    .instructions {{
      position: absolute;
      bottom: 8px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.80);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 16px;
      padding: 4px 12px;
      color: #e2e8f0;
      font-size: 11px;
      pointer-events: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
      white-space: nowrap;
    }}
    .lock-indicator {{
      position: absolute;
      bottom: 8px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(185, 28, 28, 0.92);
      border: 1px solid #fca5a5;
      border-radius: 16px;
      padding: 4px 14px;
      color: #ffffff;
      font-size: 11px;
      font-weight: 700;
      pointer-events: none;
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.4);
      white-space: nowrap;
      display: none;
      letter-spacing: 0.2px;
    }}
    #atom-tooltip {{
      position: absolute;
      background: #0284c7;
      color: white;
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      pointer-events: none;
      display: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
      transform: translate(-50%, -125%);
      z-index: 10;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container"></div>

  <div class="hud-compact">
    <span id="conf-badge" class="badge badge-perp">Kohtisuora (90°)</span>
    <span style="color:#cbd5e1;">φ_rengas: <b id="phi-ring-val" style="color:#38bdf8;">90.0°</b></span>
    <span style="color:#64748b;">|</span>
    <span style="color:#cbd5e1;">E_MM: <b id="energy-val" style="color:#4ade80;">16.1 kJ/mol</b></span>
  </div>

  <div id="instructions-bar" class="instructions">
    🖱️ <b>Tartu ja raahaa atomia</b> hiirellä &bull; Pyöritä taustasta &bull; Zoomaa rullalla
  </div>

  <div id="lock-indicator" class="lock-indicator">
    ⏳ Optimointi käynnissä... Molekyylin muokkaus on lukittu
  </div>

  <div id="atom-tooltip"></div>

  <script>
    const ELEMENTS = {elements_json};
    const ATOM_NAMES = {atom_names_json};
    const BONDS = {bonds_json};
    const ANGLES = {angles_json};
    const IMPROPERS = {impropers_json};
    const LJ_PAIRS = {lj_pairs_json};
    const V2_RING = {V2_RING};
    const V1_CHAIN = {V1_CHAIN}, V2_CHAIN = {V2_CHAIN}, V3_CHAIN = {V3_CHAIN};

    let coords = {initial_coords_json};

    // Scene setup
    const container = document.getElementById("canvas-container");
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(4.8, 3.2, 6.8);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 1.0;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(8, 14, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x93c5fd, 0.4);
    dirLight2.position.set(-8, -6, -6);
    scene.add(dirLight2);

    // Meshes
    const atomMeshes = [];
    const bondMeshes = [];

    const geomC = new THREE.SphereGeometry(0.38, 32, 32);
    const geomH = new THREE.SphereGeometry(0.24, 24, 24);

    const matC = new THREE.MeshStandardMaterial({{
      color: 0x334155,
      roughness: 0.25,
      metalness: 0.15,
      emissive: 0x0f172a,
      emissiveIntensity: 0.2
    }});

    const matH = new THREE.MeshStandardMaterial({{
      color: 0xf8fafc,
      roughness: 0.2,
      metalness: 0.05
    }});

    const bondGeom = new THREE.CylinderGeometry(0.07, 0.07, 1.0, 16);
    const matBond = new THREE.MeshStandardMaterial({{
      color: 0x64748b,
      roughness: 0.35,
      metalness: 0.1
    }});

    // Build atoms
    for (let i = 0; i < coords.length; i++) {{
      const isC = ELEMENTS[i] === "C";
      const mesh = new THREE.Mesh(isC ? geomC : geomH, (isC ? matC : matH).clone());
      mesh.position.set(coords[i][0], coords[i][1], coords[i][2]);
      mesh.userData = {{ index: i, element: ELEMENTS[i], name: ATOM_NAMES[i] }};
      scene.add(mesh);
      atomMeshes.push(mesh);
    }}

    // Build bonds
    for (let b of BONDS) {{
      const mesh = new THREE.Mesh(bondGeom, matBond.clone());
      mesh.userData = {{ i: b.i, j: b.j }};
      scene.add(mesh);
      bondMeshes.push(mesh);
    }}

    function updateBondMesh(mesh) {{
      const p1 = atomMeshes[mesh.userData.i].position;
      const p2 = atomMeshes[mesh.userData.j].position;
      const dist = p1.distanceTo(p2);
      
      mesh.position.copy(p1).add(p2).multiplyScalar(0.5);
      mesh.scale.set(1, dist, 1);
      
      const v = new THREE.Vector3().subVectors(p2, p1).normalize();
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), v);
    }}

    for (let bMesh of bondMeshes) {{
      updateBondMesh(bMesh);
    }}

    // -------------------------------------------------------------
    // Energy and Dihedral Calculation in JS
    // -------------------------------------------------------------
    function calcDihedral(p0, p1, p2, p3) {{
      const b0 = new THREE.Vector3().subVectors(p0, p1);
      const b1 = new THREE.Vector3().subVectors(p2, p1);
      const b2 = new THREE.Vector3().subVectors(p3, p2);
      
      const b1_norm = b1.length();
      if (b1_norm < 1e-6) return 0.0;
      const b1_u = b1.clone().multiplyScalar(1.0 / b1_norm);
      
      const v = b0.clone().sub(b1_u.clone().multiplyScalar(b0.dot(b1_u)));
      const w = b2.clone().sub(b1_u.clone().multiplyScalar(b2.dot(b1_u)));
      
      const x = v.dot(w);
      const y = b1_u.cross(v).dot(w);
      return Math.atan2(y, x) * (180.0 / Math.PI);
    }}

    function calcEnergy() {{
      const pos = atomMeshes.map(m => m.position);
      
      // Bond energy
      let e_bond = 0.0;
      for (let b of BONDS) {{
        const r = pos[b.i].distanceTo(pos[b.j]);
        e_bond += 0.5 * b.kb * Math.pow(r - b.r0, 2);
      }}

      // Angle energy
      let e_angle = 0.0;
      for (let a of ANGLES) {{
        const v1 = new THREE.Vector3().subVectors(pos[a.i], pos[a.j]);
        const v2 = new THREE.Vector3().subVectors(pos[a.k], pos[a.j]);
        const denom = v1.length() * v2.length();
        if (denom > 1e-6) {{
          const cos_th = Math.max(-1.0, Math.min(1.0, v1.dot(v2) / denom));
          const th = Math.acos(cos_th);
          e_angle += 0.5 * a.kth * Math.pow(th - a.th0, 2);
        }}
      }}

      // Impropers (aromatic planarity)
      let e_oop = 0.0;
      for (let m of IMPROPERS) {{
        const phi = calcDihedral(pos[m.i], pos[m.j], pos[m.k], pos[m.l]);
        const phi_rad = phi * (Math.PI / 180.0);
        e_oop += 0.5 * m.koop * (1.0 - Math.cos(2.0 * phi_rad));
      }}

      // Torsions:
      // phi_ring: C2-C1-C7-C8 (atoms 1, 0, 6, 7)
      const phi_ring = calcDihedral(pos[1], pos[0], pos[6], pos[7]);
      const phi_r_rad = phi_ring * (Math.PI / 180.0);
      const e_tors_ring = 0.5 * V2_RING * (1.0 + Math.cos(2.0 * phi_r_rad));

      // phi_chain: C1-C7-C8-C9 (atoms 0, 6, 7, 8)
      const phi_chain = calcDihedral(pos[0], pos[6], pos[7], pos[8]);
      const phi_c_rad = phi_chain * (Math.PI / 180.0);
      const e_tors_chain = 0.5 * V1_CHAIN * (1.0 + Math.cos(phi_c_rad)) +
                           0.5 * V2_CHAIN * (1.0 - Math.cos(2.0 * phi_c_rad)) +
                           0.5 * V3_CHAIN * (1.0 + Math.cos(3.0 * phi_c_rad));

      const e_torsion = e_tors_ring + e_tors_chain;

      // vdW
      let e_vdw = 0.0;
      for (let p of LJ_PAIRS) {{
        let r = pos[p.i].distanceTo(pos[p.j]);
        if (r < 0.6) r = 0.6;
        const sr6 = Math.pow(p.sigma / r, 6);
        const sr12 = sr6 * sr6;
        e_vdw += 0.5 * 4.0 * p.epsilon * (sr12 - sr6);
      }}

      const e_tot = e_bond + e_angle + e_oop + e_torsion + e_vdw;
      return {{ e_tot, e_bond, e_angle: e_angle + e_oop, e_torsion, e_vdw, phi_ring, phi_chain }};
    }}

    function getConfBadge(phi_ring, phi_chain) {{
      let pr = Math.abs(phi_ring);
      while (pr > 180.0) pr = Math.abs(360.0 - pr);

      if (pr >= 70.0 && pr <= 110.0) {{
        let pc = Math.abs(phi_chain);
        while (pc > 180.0) pc = Math.abs(360.0 - pc);
        if (pc >= 150.0 && pc <= 180.0) {{
          return {{ text: "Kohtisuora Anti (90°)", cls: "badge-perp" }};
        }} else if (pc >= 45.0 && pc <= 75.0) {{
          return {{ text: "Kohtisuora Gauche", cls: "badge-gauche" }};
        }} else {{
          return {{ text: "Kohtisuora (90°)", cls: "badge-perp" }};
        }}
      }} else if (pr <= 25.0 || pr >= 155.0) {{
        return {{ text: "Tason suuntainen (Steerinen este)", cls: "badge-coplanar" }};
      }} else {{
        return {{ text: "Vääristynyt tila", cls: "badge-other" }};
      }}
    }}

    function updateHUD() {{
      const res = calcEnergy();
      const dihEl = document.getElementById("phi-ring-val");
      if (dihEl) dihEl.innerText = res.phi_ring.toFixed(1) + "°";

      const enEl = document.getElementById("energy-val");
      if (enEl) enEl.innerText = (res.e_tot >= 0 ? "+" : "") + res.e_tot.toFixed(1) + " kJ/mol";

      const conf = getConfBadge(res.phi_ring, res.phi_chain);
      const badge = document.getElementById("conf-badge");
      if (badge) {{
        badge.innerText = conf.text;
        badge.className = "badge " + conf.cls;
      }}
    }}

    updateHUD();

    // -------------------------------------------------------------
    // Mouse Interaction (Raycast Atom Dragging) & State Management
    // -------------------------------------------------------------
    let isEditable = {editable_js};
    let lastCmdValue = "";

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    const dragPlane = new THREE.Plane();
    const planeIntersect = new THREE.Vector3();

    let draggedAtom = null;
    let hoveredAtom = null;
    const tooltip = document.getElementById("atom-tooltip");

    function setEditableState(editable) {{
      isEditable = !!editable;
      const lockBanner = document.getElementById("lock-indicator");
      const instBar = document.getElementById("instructions-bar");
      if (lockBanner) lockBanner.style.display = isEditable ? "none" : "block";
      if (instBar) instBar.style.display = isEditable ? "block" : "none";
      if (!isEditable) {{
        document.body.style.cursor = "default";
        if (tooltip) tooltip.style.display = "none";
        if (draggedAtom) {{
          draggedAtom.material.emissive.setHex(draggedAtom.userData.element === "C" ? 0x0f172a : 0x000000);
          draggedAtom.material.emissiveIntensity = 0.2;
          draggedAtom = null;
          controls.enabled = true;
        }}
        if (hoveredAtom) {{
          hoveredAtom.material.emissive.setHex(hoveredAtom.userData.element === "C" ? 0x0f172a : 0x000000);
          hoveredAtom.material.emissiveIntensity = 0.2;
          hoveredAtom = null;
        }}
      }}
    }}
    setEditableState(isEditable);

    function applyCoords(newCoords) {{
      if (!newCoords || !Array.isArray(newCoords)) return;
      for (let i = 0; i < newCoords.length && i < atomMeshes.length; i++) {{
        atomMeshes[i].position.set(newCoords[i][0], newCoords[i][1], newCoords[i][2]);
      }}
      for (let bMesh of bondMeshes) {{
        updateBondMesh(bMesh);
      }}
      updateHUD();
    }}

    function applyCommand(cmd) {{
      if (!cmd) return;
      if (cmd.coords) {{
        applyCoords(cmd.coords);
      }}
      if (typeof cmd.editable === "boolean") {{
        setEditableState(cmd.editable);
      }}
    }}

    function checkForParentCommands() {{
      try {{
        const cmdInp = window.parent.document.querySelector(".benzene-cmd-to-iframe input");
        if (cmdInp && cmdInp.value && cmdInp.value !== lastCmdValue) {{
          lastCmdValue = cmdInp.value;
          const cmd = JSON.parse(cmdInp.value);
          applyCommand(cmd);
        }}
      }} catch (err) {{}}
    }}

    window.addEventListener("message", (e) => {{
      if (e.data && typeof e.data === "object") {{
        applyCommand(e.data);
      }}
    }});

    function getMousePos(e) {{
      const rect = renderer.domElement.getBoundingClientRect();
      return {{
        x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
        screenX: e.clientX,
        screenY: e.clientY
      }};
    }}

    window.addEventListener("pointermove", (e) => {{
      if (!isEditable) {{
        if (hoveredAtom) {{
          hoveredAtom.material.emissive.setHex(hoveredAtom.userData.element === "C" ? 0x0f172a : 0x000000);
          hoveredAtom.material.emissiveIntensity = 0.2;
          hoveredAtom = null;
        }}
        if (tooltip) tooltip.style.display = "none";
        document.body.style.cursor = "default";
        return;
      }}

      const m = getMousePos(e);
      mouse.x = m.x;
      mouse.y = m.y;

      if (draggedAtom) {{
        raycaster.setFromCamera(mouse, camera);
        if (raycaster.ray.intersectPlane(dragPlane, planeIntersect)) {{
          draggedAtom.position.copy(planeIntersect);

          // Update connected bonds
          for (let bMesh of bondMeshes) {{
            if (bMesh.userData.i === draggedAtom.userData.index || bMesh.userData.j === draggedAtom.userData.index) {{
              updateBondMesh(bMesh);
            }}
          }}

          updateHUD();
          tooltip.style.left = m.screenX + "px";
          tooltip.style.top = m.screenY + "px";
        }}
      }} else {{
        // Hover detection
        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(atomMeshes);
        if (intersects.length > 0) {{
          const hit = intersects[0].object;
          if (hoveredAtom !== hit) {{
            if (hoveredAtom) hoveredAtom.material.emissive.setHex(hoveredAtom.userData.element === "C" ? 0x0f172a : 0x000000);
            hoveredAtom = hit;
            hoveredAtom.material.emissive.setHex(0x38bdf8);
            hoveredAtom.material.emissiveIntensity = 0.7;
            document.body.style.cursor = "grab";
          }}
          tooltip.innerText = hit.userData.name;
          tooltip.style.left = m.screenX + "px";
          tooltip.style.top = m.screenY + "px";
          tooltip.style.display = "block";
        }} else {{
          if (hoveredAtom) {{
            hoveredAtom.material.emissive.setHex(hoveredAtom.userData.element === "C" ? 0x0f172a : 0x000000);
            hoveredAtom.material.emissiveIntensity = 0.2;
            hoveredAtom = null;
            document.body.style.cursor = "default";
            tooltip.style.display = "none";
          }}
        }}
      }}
    }});

    window.addEventListener("pointerdown", (e) => {{
      if (!isEditable) return;
      if (e.button !== 0) return;
      const m = getMousePos(e);
      mouse.x = m.x;
      mouse.y = m.y;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(atomMeshes);
      if (intersects.length > 0) {{
        draggedAtom = intersects[0].object;
        controls.enabled = false;
        document.body.style.cursor = "grabbing";

        const camDir = new THREE.Vector3();
        camera.getWorldDirection(camDir);
        dragPlane.setFromNormalAndCoplanarPoint(camDir.negate(), draggedAtom.position);

        draggedAtom.material.emissive.setHex(0xf59e0b);
        draggedAtom.material.emissiveIntensity = 0.9;
      }}
    }});

    function syncToPython() {{
      const currentCoords = atomMeshes.map(m => [
        parseFloat(m.position.x.toFixed(4)),
        parseFloat(m.position.y.toFixed(4)),
        parseFloat(m.position.z.toFixed(4))
      ]);
      try {{
        const inp = window.parent.document.querySelector(".benzene-coord-receiver input");
        if (inp) {{
          inp.value = JSON.stringify(currentCoords);
          inp.dispatchEvent(new Event("input", {{ bubbles: true }}));
        }}
      }} catch(err) {{
        console.log("Sync error:", err);
      }}
    }}

    window.addEventListener("pointerup", () => {{
      if (draggedAtom) {{
        draggedAtom.material.emissive.setHex(draggedAtom.userData.element === "C" ? 0x0f172a : 0x000000);
        draggedAtom.material.emissiveIntensity = 0.2;
        draggedAtom = null;
        controls.enabled = true;
        document.body.style.cursor = (isEditable && hoveredAtom) ? "grab" : "default";

        syncToPython();
      }}
    }});

    // Center view on benzene ring
    function centerMolecule() {{
      const center = new THREE.Vector3();
      for (let i = 0; i < 6; i++) {{
        center.add(atomMeshes[i].position);
      }}
      center.multiplyScalar(1.0 / 6.0);
      controls.target.copy(center);
    }}
    centerMolecule();

    // Animation loop
    function animate() {{
      requestAnimationFrame(animate);
      checkForParentCommands();
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();

    // Window resize
    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""

    escaped_html = html.escape(inner_html, quote=True)
    return (
        f'<iframe id="{container_id}" srcdoc="{escaped_html}" '
        f'style="width:{width}px; height:{height}px; border:1px solid #CBD5E0; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.08); background:#0f172a;" '
        f'frameborder="0"></iframe>'
    )
