"""Provenance: capture exactly what ran, where, with which versions.

Every Result envelope embeds a ``Provenance`` block so anyone (the user,
a referee, future-you) can answer "how was this number produced?"
without rerunning the calculation.

Captured fields:
- qmmmkit version, git SHA when running from a checkout
- Python version + interpreter path
- Host + OS + CPU + memory snapshot
- Engine versions: ASH, PySCF, OpenMM, geomeTRIC, NumPy
- Input file hashes (SHA-256, truncated)
- Wall-clock timing (start/end), CPU time
- SLURM context when present (job id, partition, node list)
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Provenance:
    qmmmkit_version: str = ""
    qmmmkit_git_sha: str | None = None
    python_version: str = ""
    python_executable: str = ""
    host: str = ""
    platform: str = ""
    cpu_count: int = 0
    memory_total_mb: int | None = None
    engines: dict[str, str] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)  # path -> sha256 (truncated)
    started_at: float | None = None
    finished_at: float | None = None
    cpu_time_seconds: float | None = None
    slurm: dict[str, str] | None = None

    def duration_seconds(self) -> float | None:
        if self.started_at is not None and self.finished_at is not None:
            return self.finished_at - self.started_at
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "qmmmkit_version": self.qmmmkit_version,
            "qmmmkit_git_sha": self.qmmmkit_git_sha,
            "python_version": self.python_version,
            "python_executable": self.python_executable,
            "host": self.host,
            "platform": self.platform,
            "cpu_count": self.cpu_count,
            "memory_total_mb": self.memory_total_mb,
            "engines": dict(self.engines),
            "inputs": dict(self.inputs),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds(),
            "cpu_time_seconds": self.cpu_time_seconds,
            "slurm": self.slurm,
        }
        return {k: v for k, v in d.items() if v is not None}


def collect_static() -> Provenance:
    """Snapshot the process-level provenance fields that don't change per run."""
    from . import __version__

    p = Provenance(
        qmmmkit_version=__version__,
        qmmmkit_git_sha=_git_sha(),
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        host=platform.node(),
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        cpu_count=os.cpu_count() or 0,
        memory_total_mb=_memory_total_mb(),
        slurm=_slurm_env(),
    )
    p.engines = collect_engine_versions()
    return p


def collect_engine_versions() -> dict[str, str]:
    """Versions of the QM/MM engines we care about. Missing engines are skipped."""
    out: dict[str, str] = {}
    for name, modname in (
        ("ash", "ash"),
        ("pyscf", "pyscf"),
        ("openmm", "openmm"),
        ("geometric", "geometric"),
        ("pysisyphus", "pysisyphus"),
        ("ase", "ase"),
        ("mdtraj", "mdtraj"),
        ("parmed", "parmed"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
    ):
        v = _module_version(modname)
        if v is not None:
            out[name] = v
    return out


def hash_inputs(paths: list[Path | str], *, max_bytes: int = 64_000_000) -> dict[str, str]:
    """Return ``{path: sha256-12}`` for each existing file. Skip files larger than max_bytes."""
    out: dict[str, str] = {}
    for raw in paths:
        p = Path(raw)
        if not p.exists() or not p.is_file():
            continue
        if p.stat().st_size > max_bytes:
            out[str(p)] = "skipped:too-large"
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(p)] = h.hexdigest()[:12]
    return out


def start_timer(p: Provenance) -> None:
    p.started_at = time.time()


def stop_timer(p: Provenance, *, cpu_time: float | None = None) -> None:
    p.finished_at = time.time()
    if cpu_time is not None:
        p.cpu_time_seconds = cpu_time


# ---------------------------------------------------------------------------
def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None) or _via_metadata(name)
    except ImportError:
        return None


def _via_metadata(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _git_sha() -> str | None:
    """Return short git SHA when ``qmmmkit`` is installed editable from a checkout."""
    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        git_dir = ancestor.parent / ".git"
        if git_dir.exists():
            try:
                head_path = git_dir / "HEAD"
                if not head_path.exists():
                    return None
                head = head_path.read_text().strip()
                if head.startswith("ref:"):
                    ref = head.split(" ", 1)[1].strip()
                    sha_path = git_dir / ref
                    if sha_path.exists():
                        return sha_path.read_text().strip()[:12]
                return head[:12]
            except OSError:
                return None
    return None


def _memory_total_mb() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        pass
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return int(pages * page_size / (1024 * 1024))
        except (ValueError, OSError):
            pass
    return None


def _slurm_env() -> dict[str, str] | None:
    keys = (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_NNODES", "SLURM_JOB_ACCOUNT",
        "SLURM_SUBMIT_DIR", "SLURM_CLUSTER_NAME", "SLURM_GPUS_ON_NODE",
    )
    out = {k: v for k, v in os.environ.items() if k in keys}
    return out or None


__all__ = [
    "Provenance",
    "collect_static",
    "collect_engine_versions",
    "hash_inputs",
    "start_timer",
    "stop_timer",
]
