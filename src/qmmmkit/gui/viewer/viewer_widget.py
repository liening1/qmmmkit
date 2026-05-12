"""QWebEngineView-based 3D viewer widget.

Hosts the NGL viewer HTML/JS and wires QWebChannel so JS can reach
``ViewerBridge`` (atom picks, selection changes) and Python can call
JS-side helpers (``loadFile``, ``setQmAtoms``, ``setRepresentation``, ...).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QObject, Signal, Slot, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .bridge import ViewerBridge

_ASSETS = Path(__file__).resolve().parents[1] / "assets"


class ViewerWidget(QWidget):
    """Compose the QWebEngineView, page settings, and JS bridge."""

    atomPicked = Signal(dict)            # full atom info dict from JS
    viewerReady = Signal(int)            # natoms after a structure load
    selectionChanged = Signal(list)      # current QM atom indices, sorted

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._view = QWebEngineView(self)
        self._view.setContextMenuPolicy(self._view.contextMenuPolicy())  # default
        page = self._view.page()
        s = page.settings()
        s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        s.setAttribute(QWebEngineSettings.WebGLEnabled, True)
        s.setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        s.setAttribute(QWebEngineSettings.LocalStorageEnabled, True)
        s.setAttribute(QWebEngineSettings.Accelerated2dCanvasEnabled, True)

        self._bridge = ViewerBridge(self)
        self._channel = QWebChannel(self)
        self._channel.registerObject("bridge", self._bridge)
        page.setWebChannel(self._channel)

        # Forward JS -> Qt signals
        self._bridge.atomPickedSignal.connect(self.atomPicked)
        self._bridge.viewerReadySignal.connect(self.viewerReady)
        self._bridge.selectionChangedSignal.connect(self.selectionChanged)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        index = _ASSETS / "viewer.html"
        if not index.exists():
            raise FileNotFoundError(f"viewer.html not found at {index}")
        self._view.load(QUrl.fromLocalFile(str(index)))

    # ----- Python -> JS -----
    def load_structure(self, path: str | Path) -> None:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        ext = path.suffix.lstrip(".").lower() or "pdb"
        self._call_js("loadFile", text, ext)

    def set_qm_atoms(self, indices: Iterable[int]) -> None:
        self._call_js("setQmAtoms", list(map(int, indices)))

    def add_qm_atoms(self, indices: Iterable[int]) -> None:
        self._call_js("addQmAtoms", list(map(int, indices)))

    def remove_qm_atoms(self, indices: Iterable[int]) -> None:
        self._call_js("removeQmAtoms", list(map(int, indices)))

    def clear_qm_atoms(self) -> None:
        self._call_js("clearQmAtoms")

    def select_by_residue(self, atom_index: int) -> None:
        self._call_js("selectByResidue", int(atom_index))

    def select_within(self, radius_A: float) -> None:
        self._call_js("selectWithin", float(radius_A))

    def set_representation(self, name: str) -> None:
        self._call_js("setRepresentation", name)

    def recentre(self) -> None:
        self._call_js("recentre")

    # ----- internals -----
    def _call_js(self, fname: str, *args) -> None:
        argstr = ", ".join(json.dumps(a) for a in args)
        self._view.page().runJavaScript(f"window.{fname}({argstr});")
