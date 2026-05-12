"""Run controls: pick task, pick cluster (Local or any profile), run, abort."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._common import card, hint_label

LOCAL_LABEL = "Local"


class RunPanel(QWidget):
    runRequested = Signal(str, str)   # (task name, cluster name; "Local" or profile)
    abortRequested = Signal()
    refreshClustersRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        frame, layout = card("Run", self)
        outer.addWidget(frame)
        layout.addWidget(hint_label(
            "Pick a task and where to run it. Engines run on a worker thread; "
            "the UI stays responsive. Cluster jobs are dispatched over SSH+SLURM.",
            frame,
        ))

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._task = QComboBox(frame)
        self._task.addItems([
            "Single point", "Optimise (minimum)", "Transition state",
            "PES scan", "NEB", "EDA",
        ])
        form.addRow("Task", self._task)

        self._cluster = QComboBox(frame)
        self._cluster.addItem(LOCAL_LABEL)
        form.addRow("Run on", self._cluster)

        self._refresh_btn = QPushButton("Refresh clusters", frame)
        self._refresh_btn.setProperty("role", "ghost")
        self._refresh_btn.clicked.connect(self.refreshClustersRequested.emit)
        form.addRow("", self._refresh_btn)

        layout.addLayout(form)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._run_btn = QPushButton("Run", frame)
        self._run_btn.setProperty("role", "primary")
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)

        self._abort_btn = QPushButton("Abort", frame)
        self._abort_btn.setProperty("role", "danger")
        self._abort_btn.setEnabled(False)
        self._abort_btn.clicked.connect(self.abortRequested.emit)
        row.addWidget(self._abort_btn)
        layout.addLayout(row)

        self._progress = QProgressBar(frame)
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("Idle.", frame)
        self._status.setProperty("role", "hint")
        layout.addWidget(self._status)

        outer.addStretch(1)

    # ----------------------------------------------------------------
    def populate_clusters(self, names: list[str]) -> None:
        """Replace the cluster list. ``Local`` is always kept as the first entry."""
        current = self._cluster.currentText()
        self._cluster.blockSignals(True)
        self._cluster.clear()
        self._cluster.addItem(LOCAL_LABEL)
        for n in names:
            self._cluster.addItem(n)
        if current and current in [LOCAL_LABEL, *names]:
            self._cluster.setCurrentText(current)
        self._cluster.blockSignals(False)

    def _on_run(self) -> None:
        self.set_running(True)
        self.runRequested.emit(self._task.currentText(), self._cluster.currentText())

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._abort_btn.setEnabled(running)
        self._progress.setVisible(running)
        self._status.setText("Running..." if running else "Idle.")

    def report(self, message: str) -> None:
        self._status.setText(message)
