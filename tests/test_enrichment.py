"""Tests for the new tasks (NEB), analyses (cube helpers, NCI), and the
checkpoint/restart contract on BaseTask.

We don't exercise the full NEB / NCI math here — that needs PySCF and
pysisyphus and a real molecule. The goal is to validate the framework
contracts: registry membership, schema validation, checkpoint round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_new_tasks_registered():
    import qmmmkit.tasks  # noqa: F401
    from qmmmkit._registry import TASKS

    assert "neb" in TASKS.names()


def test_new_analyses_registered():
    import qmmmkit.analysis  # noqa: F401
    from qmmmkit._registry import ANALYSES

    for kind in ("density_difference", "nci", "orbital_cube"):
        assert kind in ANALYSES.names()


def test_neb_schema_rejects_missing_image_files(tmp_path: Path):
    from qmmmkit.tasks.neb import NEBOptions
    from pydantic import ValidationError

    real = tmp_path / "real.xyz"
    real.write_text("3\n\nO 0 0 0\nH 1 0 0\nH 0 1 0\n")
    with pytest.raises(ValidationError):
        NEBOptions(images=[str(real), "/no/such/file.xyz"])


def test_nci_schema_defaults():
    from qmmmkit.analysis.nci import NCIOptions

    opts = NCIOptions()
    assert opts.spacing > 0
    assert opts.rho_cutoff > 0


def test_density_difference_options_validation():
    from qmmmkit.analysis.density_difference import DensityDifferenceOptions
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DensityDifferenceOptions(fragment_a_atoms=[], fragment_b_atoms=[5])


def test_orbital_cube_index_selection_falls_back_to_frontier():
    """When no explicit indices are given, the analysis should derive a
    frontier window from the supplied mo_occ vector."""
    from qmmmkit.analysis.orbital_cube import OrbitalCubeAnalysis, OrbitalCubeOptions

    occ = np.array([2.0, 2.0, 2.0, 0.0, 0.0])  # HOMO at idx 2
    idxs = OrbitalCubeAnalysis._select_indices(occ, OrbitalCubeOptions(frontier=2))
    assert idxs == [1, 2, 3, 4]


def test_cube_grid_shape_and_coverage():
    from qmmmkit.analysis._cube import make_grid

    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # Angstrom
    grid = make_grid(coords, spacing=0.5, margin=2.0)
    assert all(n >= 4 for n in grid.shape)
    pts = grid.points()
    # Grid should cover the molecule + margin
    pts_ang = pts / 1.8897259886  # Bohr -> Angstrom (approx)
    lo = pts_ang.min(axis=0)
    hi = pts_ang.max(axis=0)
    assert lo[0] <= -1.5  # margin
    assert hi[0] >= 4.5   # 3 + margin


def test_cube_write_roundtrip(tmp_path: Path):
    """Cube file format check: header + atom records + numeric block."""
    from qmmmkit.analysis._cube import CubeGrid, write_cube

    grid = CubeGrid(
        origin=np.zeros(3), axes=np.eye(3) * 0.5, shape=(2, 2, 2),
    )
    field = np.arange(8, dtype=float).reshape(2, 2, 2)
    out = tmp_path / "test.cube"
    write_cube(
        out, field, grid,
        atom_numbers=[8],
        atom_coords_bohr=np.array([[0.0, 0.0, 0.0]]),
        title="test",
    )
    text = out.read_text()
    lines = text.splitlines()
    assert lines[0] == "test"
    # Header line: natoms + origin
    parts = lines[2].split()
    assert int(parts[0]) == 1
    # Atom record
    atom_line = lines[6].split()
    assert int(atom_line[0]) == 8
    # Numeric block contains all 8 values
    nums = " ".join(lines[7:]).split()
    assert len(nums) == 8


# ---------------------------------------------------------------------------
# Checkpoint/restart contract
# ---------------------------------------------------------------------------
def test_basetask_checkpoint_round_trip(tmp_path: Path):
    from qmmmkit.tasks._base import BaseTask
    from qmmmkit.schemas.result import TaskResult

    class FakeTask(BaseTask):
        name = "fake"

        def run(self, calc, *, workdir, log) -> TaskResult:
            return TaskResult(name=self.name)

    t = FakeTask()
    assert t.load_checkpoint(tmp_path) is None
    t.save_checkpoint(tmp_path, {"step": 7, "energy": -1.234})
    state = t.load_checkpoint(tmp_path)
    assert state == {"step": 7, "energy": -1.234}
    t.clear_checkpoint(tmp_path)
    assert t.load_checkpoint(tmp_path) is None


def test_pes_scan_checkpoint_resumes_completed_points(tmp_path: Path, tiny_pdb, mock_ash):
    """A pre-existing checkpoint with some converged points should mean those
    points are skipped on the next run."""
    from qmmmkit.runner import execute_manifest
    from qmmmkit.schemas import (
        EmbeddingSpec, JobManifest, MMSpec, QMSpec, SystemSpec, TaskSpec,
    )

    grid_shape = (3,)
    completed_first = 2
    energies = [-1.0, -1.1, np.nan]
    converged = [1, 1, 0]

    workdir = tmp_path / "scan"
    workdir.mkdir()
    (workdir / "pes_scan_checkpoint.json").write_text(
        json.dumps({
            "grid_shape": list(grid_shape),
            "energies": energies,
            "converged": converged,
            "label": "pes",
            "relaxed": False,
        })
    )

    manifest = JobManifest(
        name="pes_resume",
        system=SystemSpec(structure=str(tiny_pdb), qm_atoms=[0, 1, 2], active_shell=None),
        qm=QMSpec(method="DFT", functional="B3LYP", basis="def2-SVP", scf_type="RKS"),
        mm=MMSpec(),
        embedding=EmbeddingSpec(),
        task=TaskSpec(name="pes_scan", options={
            "coordinates": [{"kind": "bond", "atoms": [0, 1], "values": [0.95, 1.00, 1.05]}],
            "relaxed": False,
            "label": "pes",
        }),
    )

    execute_manifest(manifest, workdir=workdir)

    # After run, energies array has all three points populated; the previously
    # converged ones kept their stored values and only the third was computed.
    energies_arr = np.load(workdir / "pes_energies.npy")
    assert energies_arr.shape == (3,)
    assert energies_arr[0] == -1.0     # preserved
    assert energies_arr[1] == -1.1     # preserved
    assert np.isfinite(energies_arr[2])  # newly computed by mocked engine
