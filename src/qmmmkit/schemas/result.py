"""Structured result + provenance schema.

Every job emits one ``JobResult`` containing:
- the verbatim manifest that was run (so you can replay it),
- a ``ProvenanceModel`` block with engine versions / host / hashes / timing,
- the ``TaskResult`` for the main task,
- a list of ``AnalysisResult`` for each analysis.

This is what gets written to ``result.json`` in the workdir. The GUI
reads this back to render the result view.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .manifest import JobManifest


class ProvenanceModel(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    qmmmkit_version: str = ""
    qmmmkit_git_sha: str | None = None
    python_version: str = ""
    python_executable: str = ""
    host: str = ""
    platform: str = ""
    cpu_count: int = 0
    memory_total_mb: int | None = None
    engines: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float | None = None
    cpu_time_seconds: float | None = None
    slurm: dict[str, str] | None = None


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: str
    converged: bool = True
    energy: float | None = None
    qm_energy: float | None = None
    mm_energy: float | None = None
    gradient_path: str | None = Field(default=None, description="Path to a .npy gradient if saved.")
    nsteps: int | None = None
    final_geometry: str | None = Field(default=None, description="Path to final-geometry XYZ.")
    trajectory: str | None = Field(default=None, description="Path to optimisation trajectory XYZ.")
    extra: dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    kind: str
    method: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict, description="kind -> path on disk")


class JobResult(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    manifest: JobManifest
    provenance: ProvenanceModel
    task: TaskResult
    analyses: list[AnalysisResult] = Field(default_factory=list)
    workdir: str
    log_path: str | None = None
    events_path: str | None = None
