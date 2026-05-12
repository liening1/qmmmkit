"""Live job status panel.

Shows the current ``JobHandle`` (id / cluster / state / elapsed time) and a
condensed log feed for either local subprocess or remote SLURM jobs.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._common import card, hint_label


_STATE_COLOR = {
    "submitted": "#FFD580",
    "pending":   "#FFD580",
    "running":   "#80CFFF",
    "completed": "#7CE2A8",
    "failed":    "#FF7A8F",
    "cancelled": "#C39FFF",
    "unknown":   "#8A93A1",
}


class JobStatusPanel(QWidget):
    fetchRequested = Signal()
    cancelRequested = Signal()
    openWorkdirRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._submitted_at: float | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        frame, layout = card("Active job", self)
        outer.addWidget(frame)
        layout.addWidget(hint_label(
            "Most recent job dispatched from this window. "
            "For SLURM jobs, status is polled from the remote scheduler.",
            frame,
        ))

        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        self._id = QLabel("-", frame)
        self._cluster = QLabel("-", frame)
        self._state = QLabel("Idle", frame)
        self._state.setStyleSheet("font-weight: 600; color: #8A93A1;")
        self._elapsed = QLabel("0 s", frame)
        self._workdir = QLabel("-", frame)
        self._workdir.setWordWrap(True)
        self._workdir.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        form.addRow("State", self._state)
        form.addRow("Job id", self._id)
        form.addRow("Cluster", self._cluster)
        form.addRow("Elapsed", self._elapsed)
        form.addRow("Workdir", self._workdir)
        layout.addLayout(form)

        row = QHBoxLayout()
        self._fetch_btn = QPushButton("Fetch results", frame)
        self._fetch_btn.setEnabled(False)
        self._fetch_btn.clicked.connect(self.fetchRequested.emit)
        row.addWidget(self._fetch_btn)

        self._cancel_btn = QPushButton("Cancel", frame)
        self._cancel_btn.setProperty("role", "danger")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancelRequested.emit)
        row.addWidget(self._cancel_btn)

        self._open_btn = QPushButton("Open workdir", frame)
        self._open_btn.setProperty("role", "ghost")
        self._open_btn.clicked.connect(self.openWorkdirRequested.emit)
        row.addWidget(self._open_btn)
        layout.addLayout(row)

    # ----------------------------------------------------------------
    def reset(self) -> None:
        self._submitted_at = None
        self._id.setText("-")
        self._cluster.setText("-")
        self._state.setText("Idle")
        self._state.setStyleSheet("font-weight: 600; color: #8A93A1;")
        self._elapsed.setText("0 s")
        self._workdir.setText("-")
        self._cancel_btn.setEnabled(False)
        self._fetch_btn.setEnabled(False)

    def update_handle(self, handle: dict[str, Any]) -> None:
        state = handle.get("state", "unknown")
        color = _STATE_COLOR.get(state, "#E6E8EB")
        self._state.setText(state.upper())
        self._state.setStyleSheet(f"font-weight: 700; color: {color};")
        self._id.setText(str(handle.get("id", "-")))
        self._cluster.setText(str(handle.get("cluster") or "Local"))
        self._workdir.setText(str(handle.get("workdir", "-")))

        if self._submitted_at is None:
            self._submitted_at = handle.get("submitted_at") or time.time()
        elapsed = max(0.0, time.time() - float(self._submitted_at))
        self._elapsed.setText(f"{elapsed:.0f} s")

        terminal = state in ("completed", "failed", "cancelled")
        self._cancel_btn.setEnabled(not terminal)
        self._fetch_btn.setEnabled(state == "completed" and bool(handle.get("cluster")))
