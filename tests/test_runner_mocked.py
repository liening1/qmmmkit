"""End-to-end runner test with mocked QM/MM engines.

Verifies that the full pipeline (manifest -> system -> calculator -> task ->
provenance -> result.json) is wired correctly without requiring real PySCF /
OpenMM / ASH installs.
"""

from __future__ import annotations

import json
from pathlib import Path

from qmmmkit.runner import execute_manifest
from qmmmkit.schemas import (
    EmbeddingSpec,
    JobManifest,
    MMSpec,
    QMSpec,
    SystemSpec,
    TaskSpec,
)


def test_runner_single_point_with_mocked_engines(tmp_path: Path, tiny_pdb: Path, mock_ash):
    manifest = JobManifest(
        name="mocked_sp",
        system=SystemSpec(
            structure=str(tiny_pdb), qm_atoms=[0, 1, 2], active_shell=None,
        ),
        qm=QMSpec(method="DFT", functional="B3LYP", basis="def2-SVP", scf_type="RKS"),
        mm=MMSpec(),
        embedding=EmbeddingSpec(),
        task=TaskSpec(name="single_point", options={"gradient": False}),
    )
    result = execute_manifest(manifest, workdir=tmp_path)

    # Files produced
    assert (tmp_path / "log.txt").exists()
    assert (tmp_path / "events.log").exists()
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "state.json").exists()

    # State is terminal-completed
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "completed"

    # Result is a valid JobResult JSON
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["task"]["name"] == "single_point"
    assert payload["task"]["energy"] is not None
    assert payload["provenance"]["host"]
    assert "duration_seconds" in payload["provenance"]
    # The manifest is round-tripped inside the result
    assert payload["manifest"]["name"] == "mocked_sp"


def test_runner_propagates_failure(tmp_path: Path, tiny_pdb: Path, mock_ash):
    """If the task raises, state.json says failed and the exception propagates."""
    manifest = JobManifest(
        name="will_fail",
        system=SystemSpec(structure=str(tiny_pdb), qm_atoms=[0], active_shell=None),
        task=TaskSpec(name="not_a_real_task"),
    )
    import pytest

    with pytest.raises(KeyError):
        execute_manifest(manifest, workdir=tmp_path)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "failed"
    assert "error" in state


def test_runner_hashes_inputs_with_relative_manifest_paths(
    tmp_path: Path, tiny_pdb: Path, mock_ash, monkeypatch,
):
    """Regression: a manifest with a relative `structure` path must still produce
    input hashes (the provenance ``inputs`` block must contain the resolved file).

    Reproduces the bug where ``hash_inputs()`` ran before ``resolve_paths()`` and
    silently returned an empty dict whenever cwd differed from the manifest's
    directory.
    """
    # Place the manifest next to the PDB so the relative path resolves correctly,
    # but invoke the runner from a *different* cwd to expose any cwd-dependence.
    manifest_dir = tmp_path / "case"
    manifest_dir.mkdir()
    pdb_in_case = manifest_dir / "water.pdb"
    pdb_in_case.write_text(tiny_pdb.read_text())

    manifest_path = manifest_dir / "manifest.yaml"
    manifest_path.write_text(
        "qmmmkit_version: '0.2'\n"
        "name: rel_paths\n"
        "system:\n"
        "  kind: pdb\n"
        "  structure: ./water.pdb\n"   # <- deliberately relative
        "  qm_atoms: [0]\n"
        "  active_shell: null\n"
        "task:\n"
        "  name: single_point\n"
    )
    manifest = JobManifest.load(manifest_path)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    workdir = tmp_path / "workdir"
    execute_manifest(manifest, workdir=workdir)

    payload = json.loads((workdir / "result.json").read_text())
    inputs = payload["provenance"]["inputs"]
    assert inputs, "provenance.inputs must not be empty for a manifest with a relative structure path"
    assert any(k.endswith("water.pdb") for k in inputs)
