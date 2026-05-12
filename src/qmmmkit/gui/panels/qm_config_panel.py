"""QM region configuration panel: method, basis, charge, multiplicity."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...schemas import QMSpec
from ._common import card, hint_label


_DFT_FUNCTIONALS = {
    "B3LYP", "B3LYP-D3BJ", "PBE0", "wB97X-D", "M06-2X", "TPSS", "PBE", "BLYP",
    "CAM-B3LYP", "M06", "M06-L", "TPSSh",
}
_NON_DFT = {"HF", "MP2", "CCSD", "CCSD(T)", "CASSCF", "CASCI"}
_METHODS = sorted(_DFT_FUNCTIONALS) + sorted(_NON_DFT)
_BASES = [
    "STO-3G", "3-21G", "6-31G(d)", "6-31G(d,p)", "6-311G(d,p)",
    "def2-SVP", "def2-TZVP", "def2-QZVP", "cc-pVDZ", "cc-pVTZ",
]


class QMConfigPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        frame, layout = card("QM region", self)
        outer.addWidget(frame)

        layout.addWidget(hint_label("Settings for the PySCF QM calculation on the selected atoms.", frame))

        form = QFormLayout()
        form.setLabelAlignment(form.labelAlignment())
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._scf_type = QComboBox(frame)
        self._scf_type.addItems(["RKS", "UKS", "RHF", "UHF", "ROHF"])
        form.addRow("SCF type", self._scf_type)

        self._method = QComboBox(frame)
        self._method.addItems(_METHODS)
        self._method.setEditable(True)
        form.addRow("Method", self._method)

        self._basis = QComboBox(frame)
        self._basis.addItems(_BASES)
        self._basis.setEditable(True)
        self._basis.setCurrentText("def2-SVP")
        form.addRow("Basis", self._basis)

        self._dispersion = QComboBox(frame)
        self._dispersion.addItems(["", "d3", "d3bj", "d4"])
        self._dispersion.setCurrentText("d3bj")
        form.addRow("Dispersion", self._dispersion)

        self._aux_basis = QLineEdit(frame)
        self._aux_basis.setPlaceholderText("e.g. def2-universal-jfit (RI-J)")
        form.addRow("Aux basis", self._aux_basis)

        chargemult = QHBoxLayout()
        chargemult.setSpacing(6)
        self._charge = QSpinBox(frame)
        self._charge.setRange(-10, 10)
        self._charge.setPrefix("q ")
        self._mult = QSpinBox(frame)
        self._mult.setRange(1, 10)
        self._mult.setPrefix("2S+1 ")
        self._mult.setValue(1)
        chargemult.addWidget(self._charge)
        chargemult.addWidget(self._mult)
        form.addRow("Charge / mult", _wrap(chargemult, frame))

        self._nprocs = QSpinBox(frame)
        self._nprocs.setRange(1, 256)
        self._nprocs.setValue(1)
        form.addRow("Cores", self._nprocs)

        self._memory = QSpinBox(frame)
        self._memory.setRange(256, 1024 * 1024)
        self._memory.setSingleStep(256)
        self._memory.setValue(4000)
        self._memory.setSuffix(" MB")
        form.addRow("Memory", self._memory)

        layout.addLayout(form)

    # --------------------------------------------------------------
    def to_config(self) -> tuple[QMSpec, int, int]:
        method_text = self._method.currentText()
        if method_text in _NON_DFT:
            method, functional = method_text, None
        else:
            method, functional = "DFT", method_text.split("-")[0] if "-D" in method_text else method_text
        cfg = QMSpec(
            method=method,
            functional=functional,
            basis=self._basis.currentText(),
            aux_basis=self._aux_basis.text() or None,
            dispersion=self._dispersion.currentText() or None,
            scf_type=self._scf_type.currentText(),
            nprocs=self._nprocs.value(),
            memory=self._memory.value(),
        )
        return cfg, self._charge.value(), self._mult.value()


def _wrap(layout, parent):
    w = QWidget(parent)
    w.setLayout(layout)
    return w
