"""Shared fixtures for the qmmmkit test suite."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tiny PDB fixture so SystemSpec validation has a real file to point at.
# ---------------------------------------------------------------------------
_TINY_PDB = """\
HEADER    qmmmkit unit test
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.239   0.927   0.000  1.00  0.00           H
END
"""


@pytest.fixture
def tiny_pdb(tmp_path: Path) -> Path:
    p = tmp_path / "water.pdb"
    p.write_text(_TINY_PDB)
    return p


# ---------------------------------------------------------------------------
# Mock ash + pyscf + openmm so task / calculator imports work without the
# real packages installed. Scoped per-test so a real ASH install (if present)
# isn't permanently shadowed.
# ---------------------------------------------------------------------------
class _MockResult:
    def __init__(self, energy=-1.234567, gradient=None, qm_energy=-1.0, mm_energy=-0.234567):
        self.energy = energy
        self.gradient = gradient if gradient is not None else np.zeros((3, 3))
        self.qm_energy = qm_energy
        self.mm_energy = mm_energy
        self.converged = True
        self.nsteps = 1


class _MockTheory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mf = None
        self.QMenergy = -1.0
        self.MMenergy = -0.234567


class _MockFragment:
    def __init__(self, pdbfile=None, xyzfile=None, charge=0, mult=1, **kwargs):
        self.pdbfile = pdbfile
        self.charge = charge
        self.mult = mult
        # Fixed 3-atom water for the tiny_pdb
        self.coords = np.array([[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.239, 0.927, 0.0]])
        self.elems = ["O", "H", "H"]
        self.numatoms = 3
        self.atom_residues = [0, 0, 0]


def _make_mock_ash() -> types.ModuleType:
    mod = types.ModuleType("ash")
    mod.Fragment = _MockFragment
    mod.PySCFTheory = _MockTheory
    mod.OpenMMTheory = _MockTheory
    mod.QMMMTheory = _MockTheory
    mod.Singlepoint = lambda **kwargs: _MockResult()
    mod.Optimizer = lambda **kwargs: _MockResult()
    mod.NumFreq = lambda **kwargs: types.SimpleNamespace(
        normal_modes=np.eye(9), frequencies=np.array([-100.0, *([100.0] * 8)]),
    )
    return mod


@pytest.fixture
def mock_ash(monkeypatch):
    """Install a minimal ash mock in sys.modules for the duration of one test."""
    mod = _make_mock_ash()
    monkeypatch.setitem(sys.modules, "ash", mod)
    yield mod
