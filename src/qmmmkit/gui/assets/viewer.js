/* qmmmkit NGL viewer + QtWebChannel bridge.
 *
 * Exposed JS API (called from Python via runJavaScript):
 *   loadFile(content, fmt)            - load structure from text
 *   setQmAtoms(indices)               - replace QM region selection
 *   addQmAtoms(indices)               - extend QM region selection
 *   removeQmAtoms(indices)            - remove from QM region
 *   clearQmAtoms()
 *   selectByResidue(atomIndex)        - returns indices via bridge.atomsForResidue
 *   selectWithin(radius)              - expand current QM by radius (Angstrom)
 *   setRepresentation(name)           - 'cartoon' | 'sticks' | 'spacefill'
 *   recentre()
 *
 * Signals out (Python listens via QWebChannel slot):
 *   bridge.atomPicked(JSON.stringify({index, name, resname, resno, chain, element}))
 *   bridge.viewerReady(natoms)
 *   bridge.selectionChanged(JSON.stringify(indices))
 */

(function () {
  "use strict";

  const ACCENT = "#4F8CFF";
  const QM_HIGHLIGHT_RADIUS = 0.55;

  let stage, structure, component;
  let baseRepr = null;
  let qmRepr = null;
  let qmAtoms = new Set();
  let bridge = null;
  let currentRepName = "sticks";

  const status = (msg, isErr) => {
    const el = document.getElementById("status");
    el.textContent = msg;
    el.style.color = isErr ? "#FF7A8F" : "#8A93A1";
  };

  const hoverTip = document.getElementById("hover-tip");

  function init() {
    stage = new NGL.Stage("viewport", {
      backgroundColor: "#0B0D10",
      sampleLevel: 2,
      cameraType: "perspective",
      ambientIntensity: 0.5,
      lightIntensity: 0.6,
      mousePreset: "default",
    });
    stage.setParameters({ tooltip: false });
    window.addEventListener("resize", () => stage.handleResize());

    stage.signals.clicked.add(onClicked);
    stage.signals.hovered.add(onHovered);

    setupBridge();
    setupToolbar();
    status("Ready. Drop a PDB onto the canvas or load one from the sidebar.");
  }

  function setupBridge() {
    if (typeof QWebChannel === "undefined") {
      status("QWebChannel not available (running outside Qt?)", true);
      return;
    }
    new QWebChannel(qt.webChannelTransport, (channel) => {
      bridge = channel.objects.bridge;
      bridge.viewerReady(0);
      status("Bridge ready.");
    });
  }

  function setupToolbar() {
    const map = {
      "btn-cartoon": "cartoon",
      "btn-sticks": "sticks",
      "btn-spacefill": "spacefill",
    };
    Object.entries(map).forEach(([id, name]) => {
      document.getElementById(id).addEventListener("click", () => {
        setRepresentation(name);
      });
    });
    document.getElementById("btn-center").addEventListener("click", () => recentre());
  }

  function setActiveButton(name) {
    ["cartoon", "sticks", "spacefill"].forEach((n) => {
      const btn = document.getElementById("btn-" + n);
      if (btn) btn.classList.toggle("active", n === name);
    });
  }

  // ----------- Loading ------------
  window.loadFile = function (content, fmt) {
    if (component) {
      stage.removeAllComponents();
      structure = null; component = null; baseRepr = null; qmRepr = null;
      qmAtoms = new Set();
    }
    const blob = new Blob([content], { type: "text/plain" });
    return stage.loadFile(blob, { ext: fmt || "pdb", defaultRepresentation: false }).then((c) => {
      component = c;
      structure = c.structure;
      buildBaseRepresentation();
      stage.autoView();
      const n = structure.atomCount;
      status("Loaded " + n + " atoms.");
      if (bridge) bridge.viewerReady(n);
    }).catch((err) => {
      status("Load failed: " + err.message, true);
      throw err;
    });
  };

  function buildBaseRepresentation() {
    if (baseRepr) component.removeRepresentation(baseRepr);
    const params = { quality: "high", colorScheme: "element", aspectRatio: 1.6 };
    if (currentRepName === "cartoon") {
      baseRepr = component.addRepresentation("cartoon", { quality: "high" });
      component.addRepresentation("ball+stick", { sele: "hetero and not water", aspectRatio: 1.5 });
    } else if (currentRepName === "spacefill") {
      baseRepr = component.addRepresentation("spacefill", params);
    } else {
      baseRepr = component.addRepresentation("ball+stick", params);
    }
    rebuildQmRepresentation();
  }

  // ----------- QM selection rendering ------------
  function rebuildQmRepresentation() {
    if (qmRepr) { component.removeRepresentation(qmRepr); qmRepr = null; }
    if (!qmAtoms.size) return;
    const sele = "@" + Array.from(qmAtoms).join(",");
    qmRepr = component.addRepresentation("ball+stick", {
      sele: sele,
      color: ACCENT,
      aspectRatio: 2.4,
      radius: QM_HIGHLIGHT_RADIUS,
      opacity: 1.0,
    });
  }

  function emitSelectionChanged() {
    if (!bridge) return;
    bridge.selectionChanged(JSON.stringify(Array.from(qmAtoms).sort((a, b) => a - b)));
  }

  window.setQmAtoms = function (indices) {
    qmAtoms = new Set(indices.map(Number));
    rebuildQmRepresentation();
    emitSelectionChanged();
  };
  window.addQmAtoms = function (indices) {
    indices.forEach((i) => qmAtoms.add(Number(i)));
    rebuildQmRepresentation();
    emitSelectionChanged();
  };
  window.removeQmAtoms = function (indices) {
    indices.forEach((i) => qmAtoms.delete(Number(i)));
    rebuildQmRepresentation();
    emitSelectionChanged();
  };
  window.clearQmAtoms = function () {
    qmAtoms.clear();
    rebuildQmRepresentation();
    emitSelectionChanged();
  };

  window.selectByResidue = function (atomIndex) {
    if (!structure) return;
    const ap = structure.getAtomProxy(atomIndex);
    const resno = ap.resno, chain = ap.chainname;
    const out = [];
    structure.eachAtom((a) => {
      if (a.resno === resno && a.chainname === chain) out.push(a.index);
    });
    out.forEach((i) => qmAtoms.add(i));
    rebuildQmRepresentation();
    emitSelectionChanged();
  };

  window.selectWithin = function (radius) {
    if (!structure || !qmAtoms.size) return;
    const seedIndices = Array.from(qmAtoms);
    const sele = "@" + seedIndices.join(",");
    const sel = new NGL.Selection(sele);
    const within = new NGL.Selection(
      "(" + sele + ") or (within " + Number(radius) + " of (" + sele + "))"
    );
    const idxs = [];
    structure.eachAtom((a) => { idxs.push(a.index); }, within);
    idxs.forEach((i) => qmAtoms.add(i));
    rebuildQmRepresentation();
    emitSelectionChanged();
  };

  window.setRepresentation = function (name) {
    currentRepName = name;
    setActiveButton(name);
    if (component) buildBaseRepresentation();
  };

  window.recentre = function () {
    if (component) stage.autoView(500);
  };

  // ----------- Picking & hover ------------
  function onClicked(picking) {
    if (!picking || !picking.atom) return;
    const a = picking.atom;
    const info = {
      index: a.index,
      name: a.atomname,
      resname: a.resname,
      resno: a.resno,
      chain: a.chainname,
      element: a.element,
    };
    if (bridge) bridge.atomPicked(JSON.stringify(info));
    // Default behaviour: toggle in QM region
    if (qmAtoms.has(a.index)) qmAtoms.delete(a.index);
    else qmAtoms.add(a.index);
    rebuildQmRepresentation();
    emitSelectionChanged();
  }

  function onHovered(picking) {
    if (!picking || !picking.atom) {
      hoverTip.style.visibility = "hidden";
      return;
    }
    const a = picking.atom;
    hoverTip.textContent = `${a.element}  ${a.atomname}  ${a.resname} ${a.resno} ${a.chainname}  #${a.index}`;
    hoverTip.style.left = (picking.canvasPosition.x) + "px";
    hoverTip.style.top = (window.innerHeight - picking.canvasPosition.y) + "px";
    hoverTip.style.visibility = "visible";
  }

  // ----------- File drop ------------
  document.getElementById("viewport").addEventListener("dragover", (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  });
  document.getElementById("viewport").addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const reader = new FileReader();
    const ext = f.name.split(".").pop().toLowerCase();
    reader.onload = () => loadFile(reader.result, ext);
    reader.readAsText(f);
  });

  if (typeof NGL === "undefined") {
    status("NGL failed to load (offline?). Check internet or bundle locally.", true);
  } else {
    init();
  }
})();
