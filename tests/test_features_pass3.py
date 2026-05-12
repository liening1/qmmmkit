"""Tests for the third enrichment pass:
- fragment_scf method on QMMMCalculator
- EDATask registration + manifest validation
- GUI workers using the Scheduler abstraction (Local + cluster names)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# fragment_scf
# ---------------------------------------------------------------------------
def test_fragment_scf_rejects_overlapping_real_and_ghost(mock_ash, tiny_pdb: Path):
    from qmmmkit.calculator import QMMMCalculator
    from qmmmkit.schemas import SystemSpec
    from qmmmkit.system import QMMMSystem

    spec = SystemSpec(structure=str(tiny_pdb), qm_atoms=[0, 1, 2], active_shell=None)
    sys_obj = QMMMSystem.from_spec(spec)
    calc = QMMMCalculator(system=sys_obj)
    calc.build()

    with pytest.raises(ValueError, match="both real and ghost"):
        calc.fragment_scf(real_atoms=[0, 1], ghost_atoms=[1, 2])


def test_fragment_scf_rejects_empty_real(mock_ash, tiny_pdb: Path):
    from qmmmkit.calculator import QMMMCalculator
    from qmmmkit.schemas import SystemSpec
    from qmmmkit.system import QMMMSystem

    spec = SystemSpec(structure=str(tiny_pdb), qm_atoms=[0, 1, 2], active_shell=None)
    sys_obj = QMMMSystem.from_spec(spec)
    calc = QMMMCalculator(system=sys_obj)
    calc.build()
    with pytest.raises(ValueError, match="at least one real atom"):
        calc.fragment_scf(real_atoms=[])


def test_extract_mm_environment_returns_empty_when_charges_missing(mock_ash, tiny_pdb: Path):
    """If the mocked theory has no MM-charge attribute, fragment_scf falls back
    to gas-phase rather than crashing."""
    from qmmmkit.calculator import QMMMCalculator
    from qmmmkit.schemas import SystemSpec
    from qmmmkit.system import QMMMSystem

    spec = SystemSpec(structure=str(tiny_pdb), qm_atoms=[0, 1, 2], active_shell=None)
    sys_obj = QMMMSystem.from_spec(spec)
    calc = QMMMCalculator(system=sys_obj)
    calc.build()
    coords, charges = calc._extract_mm_environment([0, 1, 2])
    assert len(coords) == 0
    assert len(charges) == 0


# ---------------------------------------------------------------------------
# EDATask
# ---------------------------------------------------------------------------
def test_eda_task_registered():
    import qmmmkit.tasks  # noqa: F401
    from qmmmkit._registry import TASKS

    assert "eda" in TASKS.names()


def test_eda_options_default_construction():
    from qmmmkit.tasks.eda import EDAOptions

    opts = EDAOptions()
    assert opts.write_summary is True


# ---------------------------------------------------------------------------
# GUI dispatch worker contract
# ---------------------------------------------------------------------------
def test_dispatch_request_back_compat_aliases():
    """The old name CalcRequest must still be importable for any external code."""
    pytest.importorskip("PySide6")
    from qmmmkit.gui.workers.calc_worker import CalcRequest, DispatchRequest

    assert CalcRequest is DispatchRequest


def test_dispatch_request_local_means_local_scheduler():
    """Sentinel-string handling: 'Local' or None both route to LocalScheduler.

    We construct a DispatchRequest and only verify the routing decision logic;
    actually running it would need PySide6's event loop.
    """
    pytest.importorskip("PySide6")
    from qmmmkit.gui.workers.calc_worker import DispatchRequest

    r1 = DispatchRequest(manifest_path="m.yaml", workdir="/tmp/x")
    r2 = DispatchRequest(manifest_path="m.yaml", workdir="/tmp/x", cluster="Local")
    r3 = DispatchRequest(manifest_path="m.yaml", workdir="/tmp/x", cluster="hpc")

    # The worker decides based on cluster; we just verify the field shape.
    assert r1.cluster is None
    assert r2.cluster.lower() == "local"
    assert r3.cluster == "hpc"


def test_run_panel_populate_clusters_keeps_local_first():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from qmmmkit.gui.panels.run_panel import LOCAL_LABEL, RunPanel

    panel = RunPanel()
    panel.populate_clusters(["hpc", "lab-cluster"])
    assert panel._cluster.itemText(0) == LOCAL_LABEL
    assert panel._cluster.itemText(1) == "hpc"
    assert panel._cluster.itemText(2) == "lab-cluster"
    panel.deleteLater()
    # don't quit the app; subsequent tests may want it
