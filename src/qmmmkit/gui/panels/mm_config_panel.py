"""MM region configuration panel: forcefield + OpenMM platform."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...schemas import EmbeddingSpec, MMSpec
from ._common import card, hint_label


_FORCEFIELDS = [
    "amber14-all.xml + amber14/tip3p.xml",
    "amber14-all.xml + amber14/tip4pew.xml",
    "amber99sbildn.xml + tip3p.xml",
    "charmm36.xml + charmm36/water.xml",
]


class MMConfigPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        frame, layout = card("MM region & embedding", self)
        outer.addWidget(frame)
        layout.addWidget(hint_label("OpenMM forcefield and how QM polarises against MM.", frame))

        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._ff = QComboBox(frame)
        self._ff.addItems(_FORCEFIELDS)
        self._ff.setEditable(True)
        form.addRow("Forcefield", self._ff)

        self._platform = QComboBox(frame)
        self._platform.addItems(["CPU", "CUDA", "OpenCL", "Reference"])
        form.addRow("OpenMM platform", self._platform)

        self._embedding = QComboBox(frame)
        self._embedding.addItems(["electrostatic", "mechanical"])
        form.addRow("Embedding", self._embedding)

        self._link_atoms = QCheckBox("Use link atoms at QM/MM boundary", frame)
        self._link_atoms.setChecked(True)
        form.addRow("", self._link_atoms)

        self._cutoff = QLineEdit(frame)
        self._cutoff.setPlaceholderText("e.g. 1.0")
        form.addRow("Nonbonded cutoff (nm)", self._cutoff)

        layout.addLayout(form)

    def to_config(self) -> tuple[MMSpec, EmbeddingSpec, list[str]]:
        ff_text = self._ff.currentText()
        ff = [s.strip() for s in ff_text.replace("+", ",").split(",") if s.strip()]
        try:
            cutoff_nm = float(self._cutoff.text()) if self._cutoff.text() else 1.0
        except ValueError:
            cutoff_nm = 1.0
        # ASH expects the cutoff in Angstrom, not nm. The label still says "nm"
        # for user familiarity but we convert here at the boundary.
        mm = MMSpec(
            platform=self._platform.currentText(),
            periodic_nonbonded_cutoff=cutoff_nm * 10.0,
        )
        emb = EmbeddingSpec(
            scheme=self._embedding.currentText(),
            use_link_atoms=self._link_atoms.isChecked(),
        )
        return mm, emb, ff
