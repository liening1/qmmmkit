"""QM region selection panel.

Shows the live atom indices in the QM region, lets the user expand by
residue / by radius / clear, and emits high-level intents the main
window dispatches to the viewer widget.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ._common import card, hint_label


class SelectionPanel(QWidget):
    expandWithinRequested = Signal(float)
    expandResidueRequested = Signal()  # uses last picked atom
    clearRequested = Signal()
    setQmRequested = Signal(list)  # explicit list of indices

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._atom_count = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        frame, layout = card("QM region", self)
        outer.addWidget(frame)

        self._counter = QLabel("0 atoms selected", frame)
        self._counter.setStyleSheet("font-size: 11pt; font-weight: 600; color: #F2F4F7;")
        layout.addWidget(self._counter)

        layout.addWidget(hint_label("Click atoms in the 3D view to toggle them in/out of the QM region.", frame))

        self._list = QListWidget(frame)
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.setUniformItemSizes(True)
        self._list.setMaximumHeight(180)
        layout.addWidget(self._list)

        # row of action buttons
        row = QHBoxLayout()
        row.setSpacing(6)
        self._btn_residue = QPushButton("+ Residue", frame)
        self._btn_residue.setToolTip("Promote the last clicked atom's residue into QM region")
        self._btn_residue.clicked.connect(self.expandResidueRequested.emit)
        row.addWidget(self._btn_residue)

        self._btn_within = QPushButton("+ Within", frame)
        self._btn_within.setToolTip("Add all atoms within radius (A) of the current QM region")
        row.addWidget(self._btn_within)

        self._radius = QDoubleSpinBox(frame)
        self._radius.setRange(0.5, 20.0)
        self._radius.setSingleStep(0.5)
        self._radius.setValue(3.0)
        self._radius.setSuffix(" Å")
        row.addWidget(self._radius)
        self._btn_within.clicked.connect(lambda: self.expandWithinRequested.emit(self._radius.value()))

        layout.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_clear = QPushButton("Clear", frame)
        self._btn_clear.setProperty("role", "danger")
        self._btn_clear.clicked.connect(self.clearRequested.emit)
        row2.addWidget(self._btn_clear)
        row2.addStretch(1)
        layout.addLayout(row2)

        outer.addStretch(1)

    # ---- slots from main window ----
    def update_selection(self, indices: Iterable[int]) -> None:
        indices = list(indices)
        self._list.clear()
        for i in indices:
            QListWidgetItem(f"#{i}", self._list)
        n = len(indices)
        self._counter.setText(f"{n} atom{'s' if n != 1 else ''} selected")

    def update_atom_count(self, natoms: int) -> None:
        self._atom_count = int(natoms)
