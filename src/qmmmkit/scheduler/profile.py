"""Cluster profiles: per-HPC connection + sbatch defaults.

Stored in ``~/.config/qmmmkit/clusters.yaml``::

    clusters:
      hpc:
        host: hpc.example.edu
        user: jdoe
        port: 22
        identity_file: ~/.ssh/id_ed25519
        workdir: /scratch/jdoe/qmmmkit
        sbatch:
          account: my-allocation
          partition: gpu
          qos: normal
          time: "12:00:00"
          mem: "64G"
          cpus_per_task: 8
          ntasks: 1
          gres: "gpu:1"
        modules:
          - cuda/12.4
          - openmpi/4.1
        conda_env: qmmmkit       # activated before running runner
        apptainer_image: null    # if set, runner is invoked inside the image
        env: {}                  # extra env vars set in the sbatch script
        scratch: /tmp/$SLURM_JOB_ID   # optional fast scratch dir
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path.home() / ".config" / "qmmmkit" / "clusters.yaml"


@dataclass
class SbatchDefaults:
    account: str | None = None
    partition: str | None = None
    qos: str | None = None
    time: str = "04:00:00"
    mem: str | None = "16G"
    cpus_per_task: int = 4
    ntasks: int = 1
    nodes: int = 1
    gres: str | None = None
    constraint: str | None = None
    extra: list[str] = field(default_factory=list)


@dataclass
class ClusterProfile:
    name: str
    host: str
    user: str
    workdir: str
    port: int = 22
    identity_file: str | None = None
    sbatch: SbatchDefaults = field(default_factory=SbatchDefaults)
    modules: list[str] = field(default_factory=list)
    conda_env: str | None = None
    apptainer_image: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    scratch: str | None = None
    runner_python: str = "python"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sbatch"] = {k: v for k, v in d["sbatch"].items() if v not in (None, [], {})}
        return d

    @classmethod
    def from_dict(cls, name: str, raw: dict) -> "ClusterProfile":
        sbatch_raw = raw.get("sbatch", {}) or {}
        return cls(
            name=name,
            host=raw["host"],
            user=raw["user"],
            workdir=raw["workdir"],
            port=int(raw.get("port", 22)),
            identity_file=raw.get("identity_file"),
            sbatch=SbatchDefaults(**sbatch_raw),
            modules=list(raw.get("modules", [])),
            conda_env=raw.get("conda_env"),
            apptainer_image=raw.get("apptainer_image"),
            env=dict(raw.get("env", {})),
            scratch=raw.get("scratch"),
            runner_python=raw.get("runner_python", "python"),
        )


def _config_path(path: str | Path | None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_CONFIG


def list_profiles(path: str | Path | None = None) -> list[str]:
    p = _config_path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return sorted((raw.get("clusters") or {}).keys())


def load_profile(name: str, path: str | Path | None = None) -> ClusterProfile:
    p = _config_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No cluster config at {p}. Run `qmmmkit clusters add {name}`.")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    profiles = (raw.get("clusters") or {})
    if name not in profiles:
        raise KeyError(f"Cluster profile '{name}' not found in {p}. Have: {sorted(profiles)}")
    return ClusterProfile.from_dict(name, profiles[name])


def save_profile(profile: ClusterProfile, *, path: str | Path | None = None) -> Path:
    p = _config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = {"clusters": {}}
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {"clusters": {}}
    raw.setdefault("clusters", {})[profile.name] = profile.to_dict()
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    return p
