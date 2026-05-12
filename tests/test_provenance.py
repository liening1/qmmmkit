"""Provenance capture tests."""

from __future__ import annotations

import os
from pathlib import Path

from qmmmkit._provenance import (
    Provenance,
    collect_static,
    hash_inputs,
    start_timer,
    stop_timer,
)


def test_static_provenance_has_minimum_fields():
    p = collect_static()
    assert p.python_version
    assert p.python_executable
    assert p.host
    assert p.platform
    assert p.cpu_count >= 1
    # qmmmkit version is set by collect_static; engines may be empty if nothing installed.
    assert p.qmmmkit_version


def test_hash_inputs_is_deterministic_and_truncated(tmp_path: Path):
    f1 = tmp_path / "a.txt"
    f1.write_bytes(b"hello qmmmkit")
    h1 = hash_inputs([f1])
    h2 = hash_inputs([f1])
    assert h1 == h2
    assert all(len(v) == 12 for v in h1.values() if not v.startswith("skipped"))


def test_hash_inputs_skips_oversize(tmp_path: Path):
    f1 = tmp_path / "big.bin"
    f1.write_bytes(b"x" * 1024)
    h = hash_inputs([f1], max_bytes=512)
    assert h[str(f1)] == "skipped:too-large"


def test_hash_inputs_ignores_missing_paths(tmp_path: Path):
    h = hash_inputs([tmp_path / "does-not-exist.txt"])
    assert h == {}


def test_timer_round_trip():
    p = Provenance()
    start_timer(p)
    stop_timer(p, cpu_time=0.01)
    d = p.duration_seconds()
    assert d is not None and d >= 0
    assert p.cpu_time_seconds == 0.01


def test_slurm_env_capture(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "gpu")
    p = collect_static()
    assert p.slurm is not None
    assert p.slurm["SLURM_JOB_ID"] == "12345"
    assert p.slurm["SLURM_JOB_PARTITION"] == "gpu"
