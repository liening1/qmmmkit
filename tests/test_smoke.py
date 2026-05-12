"""Smoke tests that don't need ASH / PySCF / OpenMM."""

from __future__ import annotations

import numpy as np
import pytest


def test_top_level_imports():
    import qmmmkit  # noqa: F401
    from qmmmkit import QMMMCalculator, QMMMSystem  # noqa: F401
    from qmmmkit.schemas import JobManifest  # noqa: F401


def test_scan_coordinate_validation():
    from qmmmkit.tasks.pes_scan import ScanCoordinate

    sc = ScanCoordinate(kind="bond", atoms=[0, 1], values=[1.0, 1.1, 1.2])
    assert sc.label == "bond_0_1"

    with pytest.raises(ValueError):
        ScanCoordinate(kind="bond", atoms=[0, 1, 2], values=[1.0])
    with pytest.raises(ValueError):
        ScanCoordinate(kind="quintuple", atoms=[0, 1], values=[1.0])


def test_atoms_within_shell():
    from qmmmkit.system import _atoms_within

    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 5, 0], [0, 0, 8]], dtype=float)
    assert _atoms_within(coords, seeds=[0], radius=2.0) == [0, 1]
    assert _atoms_within(coords, seeds=[0], radius=6.0) == [0, 1, 2]


def test_orthonormal_frame():
    from qmmmkit.analysis.charge_displacement import _orthonormal_frame

    e_z = np.array([0.0, 0.0, 1.0])
    e_x, e_y = _orthonormal_frame(e_z)
    assert abs(np.dot(e_x, e_z)) < 1e-10
    assert abs(np.dot(e_y, e_z)) < 1e-10
    assert abs(np.dot(e_x, e_y)) < 1e-10
    assert abs(np.linalg.norm(e_x) - 1.0) < 1e-10


def test_diagonalise_in_ao_recovers_occupations():
    from qmmmkit.analysis.natural_orbitals import _diagonalise_in_ao

    rng = np.random.default_rng(0)
    n = 6
    s = rng.standard_normal((n, n))
    s = s @ s.T + n * np.eye(n)
    occ_true = np.array([2.0, 1.95, 1.7, 0.3, 0.05, 0.0])
    a = rng.standard_normal((n, n))
    s_evals, s_evecs = np.linalg.eigh(s)
    s_inv_half = s_evecs @ np.diag(1.0 / np.sqrt(s_evals)) @ s_evecs.T
    c = s_inv_half @ a
    rdm = (c * occ_true) @ c.T
    occ, _ = _diagonalise_in_ao(rdm, s)
    assert np.allclose(np.sort(occ)[::-1], np.sort(occ_true)[::-1], atol=1e-8)
