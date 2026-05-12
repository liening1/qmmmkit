"""Manifest + spec validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qmmmkit.schemas import (
    EmbeddingSpec,
    JobManifest,
    ManifestValidationError,
    MMSpec,
    QMSpec,
    SystemSpec,
    TaskSpec,
)


def test_qm_dft_requires_functional():
    with pytest.raises(ValidationError):
        QMSpec(method="DFT", functional=None)


def test_qm_post_hf_rejects_ks_scf():
    with pytest.raises(ValidationError):
        QMSpec(method="HF", scf_type="RKS")


def test_qm_casscf_requires_active_space():
    with pytest.raises(ValidationError):
        QMSpec(method="CASSCF", scf_type="RHF", cas_active_space=None)


def test_qm_casscf_with_active_space_ok():
    spec = QMSpec(method="CASSCF", scf_type="RHF", cas_active_space=(4, 4))
    assert spec.cas_active_space == (4, 4)


def test_system_qm_atoms_unique_and_sorted():
    spec = SystemSpec(structure="dummy.pdb", qm_atoms=[3, 1, 2])
    assert spec.qm_atoms == [1, 2, 3]
    with pytest.raises(ValidationError):
        SystemSpec(structure="dummy.pdb", qm_atoms=[1, 1, 2])


def test_system_rejects_negative_indices():
    with pytest.raises(ValidationError):
        SystemSpec(structure="dummy.pdb", qm_atoms=[-1, 0])


def test_embedding_to_ash_kwargs_translates_scheme():
    spec = EmbeddingSpec(scheme="electrostatic", use_link_atoms=True)
    kw = spec.to_ash_kwargs()
    assert kw["embedding"] == "elstat"
    assert kw["linkatom_method"] == "simple"
    assert kw["linkatom_type"] == "H"


def test_embedding_truncated_pc():
    spec = EmbeddingSpec(scheme="electrostatic", truncate_pc_radius=15.0)
    kw = spec.to_ash_kwargs()
    assert kw["TruncatedPC"] is True
    assert kw["TruncPCRadius"] == 15.0


def test_manifest_roundtrip(tmp_path: Path, tiny_pdb: Path):
    manifest = JobManifest(
        name="rt",
        system=SystemSpec(structure=str(tiny_pdb), qm_atoms=[0, 1, 2]),
        task=TaskSpec(name="single_point"),
    )
    out = manifest.save(tmp_path / "manifest.yaml")
    again = JobManifest.load(out)
    assert again.name == "rt"
    assert again.system.qm_atoms == [0, 1, 2]
    assert again.task.name == "single_point"


def test_manifest_validation_error_message(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "system:\n  structure: nonexistent.pdb\n  qm_atoms: []\n"
        "task:\n  name: single_point\n"
    )
    with pytest.raises(ManifestValidationError) as exc:
        JobManifest.load(bad)
    msg = str(exc.value)
    assert "system.qm_atoms" in msg or "qm_atoms" in msg


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        QMSpec(method="DFT", functional="B3LYP", basis="def2-SVP", non_existent=42)
