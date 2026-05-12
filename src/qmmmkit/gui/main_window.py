"""qmmmkit main window.

Layout:
    +-----------------------------------------------------------+
    |  Menubar                                                  |
    +---------------------------------+-------------------------+
    |                                 |  Selection panel        |
    |          NGL.js viewport        |  QM config              |
    |                                 |  MM/embedding config    |
    |                                 |  Run controls           |
    |                                 +-------------------------+
    |                                 |  Log / output           |
    +---------------------------------+-------------------------+
    |  Status bar                                               |
    +-----------------------------------------------------------+
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .panels import (
    JobStatusPanel,
    MMConfigPanel,
    QMConfigPanel,
    RunPanel,
    SelectionPanel,
)
from .panels._common import card, title_label
from .viewer import ViewerWidget
from .workers.calc_worker import DispatchRequest, start_worker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("qmmmkit")
        self.setMinimumSize(1200, 800)

        self._pdb_path: Path | None = None
        self._qm_atoms: list[int] = []
        self._last_picked: int | None = None
        self._thread: QThread | None = None
        self._worker = None

        self._build_central()
        self._build_menubar()
        self._build_statusbar()
        self._connect_signals()

    # ------------------------------------------------------------
    def _build_central(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        outer.addWidget(splitter, 1)

        # ---- left: viewer + log ----
        left = QSplitter(Qt.Vertical, splitter)
        left.setHandleWidth(6)
        left.setChildrenCollapsible(False)

        viewer_card, viewer_layout = card("3D viewer", left)
        viewer_layout.setContentsMargins(0, 12, 0, 0)
        self.viewer = ViewerWidget(viewer_card)
        self.viewer.setMinimumSize(640, 480)
        viewer_layout.addWidget(self.viewer, 1)
        left.addWidget(viewer_card)

        log_card, log_layout = card("Log", left)
        self.log = QPlainTextEdit(log_card)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        self.log.setStyleSheet("font-family: Consolas, 'JetBrains Mono', 'SF Mono', monospace; font-size: 10pt;")
        log_layout.addWidget(self.log)
        left.addWidget(log_card)
        left.setStretchFactor(0, 4)
        left.setStretchFactor(1, 1)

        splitter.addWidget(left)

        # ---- right: sidebar ----
        sidebar = QWidget(splitter)
        sidebar.setMinimumWidth(380)
        sidebar.setMaximumWidth(520)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        side_layout.addWidget(title_label("QM/MM workflow"))

        self.selection_panel = SelectionPanel(sidebar)
        self.qm_panel = QMConfigPanel(sidebar)
        self.mm_panel = MMConfigPanel(sidebar)
        self.run_panel = RunPanel(sidebar)
        self.job_panel = JobStatusPanel(sidebar)

        side_layout.addWidget(self.selection_panel)
        side_layout.addWidget(self.qm_panel)
        side_layout.addWidget(self.mm_panel)
        side_layout.addWidget(self.run_panel)
        side_layout.addWidget(self.job_panel)
        side_layout.addStretch(1)

        self._refresh_clusters()

        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1100, 400])

    # ------------------------------------------------------------
    def _build_menubar(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")
        act_open = QAction("Open structure...", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._open_structure)
        file_menu.addAction(act_open)

        act_save = QAction("Save QM region as JSON...", self)
        act_save.triggered.connect(self._save_qm_region)
        file_menu.addAction(act_save)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = mb.addMenu("&View")
        for name in ("cartoon", "sticks", "spacefill"):
            a = QAction(name.capitalize(), self)
            a.triggered.connect(lambda _=False, n=name: self.viewer.set_representation(n))
            view_menu.addAction(a)
        view_menu.addSeparator()
        act_centre = QAction("Recentre", self)
        act_centre.setShortcut("R")
        act_centre.triggered.connect(self.viewer.recentre)
        view_menu.addAction(act_centre)

        sel_menu = mb.addMenu("&Selection")
        act_clear = QAction("Clear QM region", self)
        act_clear.setShortcut("Esc")
        act_clear.triggered.connect(self.viewer.clear_qm_atoms)
        sel_menu.addAction(act_clear)

    # ------------------------------------------------------------
    def _build_statusbar(self) -> None:
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        sb.showMessage("Ready.")

    # ------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.viewer.atomPicked.connect(self._on_atom_picked)
        self.viewer.viewerReady.connect(self._on_viewer_ready)
        self.viewer.selectionChanged.connect(self._on_selection_changed)

        self.selection_panel.expandWithinRequested.connect(self.viewer.select_within)
        self.selection_panel.expandResidueRequested.connect(self._expand_residue)
        self.selection_panel.clearRequested.connect(self.viewer.clear_qm_atoms)

        self.run_panel.runRequested.connect(self._on_run)
        self.run_panel.abortRequested.connect(self._on_abort)
        self.run_panel.refreshClustersRequested.connect(self._refresh_clusters)

        self.job_panel.fetchRequested.connect(self._on_fetch_results)
        self.job_panel.cancelRequested.connect(self._on_abort)
        self.job_panel.openWorkdirRequested.connect(self._on_open_workdir)

    # ------------------------------------------------------------
    def _open_structure(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open structure",
            filter="Molecules (*.pdb *.cif *.mmcif *.xyz *.mol2 *.sdf);;All files (*)",
        )
        if not path:
            return
        self._pdb_path = Path(path)
        self.viewer.load_structure(path)
        self._log(f"Loaded {path}")
        self.statusBar().showMessage(f"Loaded {path}")

    def _save_qm_region(self) -> None:
        import json

        if not self._qm_atoms:
            QMessageBox.information(self, "qmmmkit", "QM region is empty.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save QM region", filter="JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps({"qm_atoms": self._qm_atoms}, indent=2))
        self._log(f"Wrote {path}")

    # ------------------------------------------------------------
    def _on_atom_picked(self, info: dict) -> None:
        self._last_picked = int(info.get("index", -1))
        self.statusBar().showMessage(
            f"Picked atom #{info.get('index')}  "
            f"{info.get('element','')} {info.get('atomname','')} "
            f"in {info.get('resname','')} {info.get('resno','')} "
            f"chain {info.get('chain','')}"
        )

    def _on_viewer_ready(self, natoms: int) -> None:
        if natoms:
            self._log(f"Structure ready ({natoms} atoms).")
            self.selection_panel.update_atom_count(natoms)

    def _on_selection_changed(self, indices: list) -> None:
        self._qm_atoms = list(indices)
        self.selection_panel.update_selection(self._qm_atoms)

    def _expand_residue(self) -> None:
        if self._last_picked is None:
            self.statusBar().showMessage("Click an atom first to choose a residue.")
            return
        self.viewer.select_by_residue(self._last_picked)

    # ------------------------------------------------------------
    def _refresh_clusters(self) -> None:
        try:
            from ..scheduler import list_profiles
            names = list_profiles()
        except Exception:  # noqa: BLE001 - no config file is fine
            names = []
        self.run_panel.populate_clusters(names)
        self._log(f"Cluster profiles: {['Local', *names]}")

    def _on_run(self, task_label: str, cluster: str) -> None:
        if not self._pdb_path:
            self._fail("Open a structure first.")
            return
        if not self._qm_atoms:
            self._fail("Select at least one atom for the QM region.")
            return

        task_map = {
            "Single point": "single_point",
            "Optimise (minimum)": "optimize",
            "Transition state": "transition_state",
            "PES scan": "pes_scan",
            "NEB": "neb",
            "EDA": "eda",
        }
        task_name = task_map.get(task_label, "single_point")

        from ..schemas import JobManifest, SystemSpec, TaskSpec

        qm_cfg, charge, mult = self.qm_panel.to_config()
        mm_cfg, emb_cfg, forcefield = self.mm_panel.to_config()

        system = SystemSpec(
            kind="pdb",
            structure=str(self._pdb_path),
            qm_atoms=list(self._qm_atoms),
            forcefield=forcefield,
            charge=int(charge),
            mult=int(mult),
            active_shell=6.0,
        )
        manifest = JobManifest(
            name=Path(self._pdb_path).stem,
            system=system,
            qm=qm_cfg,
            mm=mm_cfg,
            embedding=emb_cfg,
            task=TaskSpec(name=task_name, options={}),
            analyses=[],
        )

        from ..scheduler import LocalScheduler
        local_sched = LocalScheduler()  # used only for the workdir base, not the run itself
        workdir = local_sched.base_dir / f"gui-{task_name}-{Path(self._pdb_path).stem}"
        workdir.mkdir(parents=True, exist_ok=True)
        manifest_path = workdir / "manifest.yaml"
        manifest.save(manifest_path)
        self._log(f"Wrote manifest -> {manifest_path}")
        self._log(f"Submitting {task_name} ({cluster}) on {len(self._qm_atoms)} QM atoms...")

        self.job_panel.reset()
        request = DispatchRequest(
            manifest_path=str(manifest_path),
            workdir=str(workdir),
            cluster=None if cluster.lower() == "local" else cluster,
        )
        thread, worker = start_worker(request)
        worker.progress.connect(self._log)
        worker.log_line.connect(self._log)
        worker.state_changed.connect(self.job_panel.update_handle)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        thread.finished.connect(self._cleanup_worker)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _on_abort(self) -> None:
        if self._worker is not None:
            self._worker.abort()
            self._log("Abort requested.")

    def _on_fetch_results(self) -> None:
        if self._worker is None or getattr(self._worker, "_handle", None) is None:
            return
        handle = self._worker._handle
        sched = self._worker._scheduler
        if sched is None:
            return
        dest = QFileDialog.getExistingDirectory(self, "Fetch into directory")
        if not dest:
            return
        try:
            out = sched.fetch_results(handle, Path(dest))
            self._log(f"Fetched results -> {out}")
        except Exception as exc:  # noqa: BLE001
            self._fail(f"Fetch failed: {exc}")

    def _on_open_workdir(self) -> None:
        if self._worker is None:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        wd = self._worker.request.workdir
        QDesktopServices.openUrl(QUrl.fromLocalFile(wd))

    def _on_finished(self, payload: dict) -> None:
        self.run_panel.set_running(False)
        e = payload.get("energy")
        if e is not None:
            self.run_panel.report(f"E = {e:.8f} Ha")
        else:
            self.run_panel.report(f"State: {payload.get('state', 'done')}")
        self._log(f"Finished: {payload}")
        self.statusBar().showMessage("Done.")

    def _on_failed(self, msg: str) -> None:
        self.run_panel.set_running(False)
        self._log(f"ERROR: {msg}")
        self.run_panel.report("Failed")
        self.statusBar().showMessage("Failed: " + msg)

    def _cleanup_worker(self) -> None:
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    # ------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def _fail(self, msg: str) -> None:
        QMessageBox.warning(self, "qmmmkit", msg)
