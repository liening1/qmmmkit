"""Scheduler abstract base class + job handle types."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..manifest import JobManifest


class JobState(enum.Enum):
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    def is_terminal(self) -> bool:
        return self in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}


@dataclass
class JobHandle:
    """Opaque-ish reference returned by ``Scheduler.submit``.

    All schedulers fill at least ``id``, ``state``, ``workdir``. Backend
    specifics live in ``extra`` so callers don't depend on them.
    """

    id: str
    state: JobState
    workdir: Path
    cluster: str | None = None
    submitted_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    log_tail: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "workdir": str(self.workdir),
            "cluster": self.cluster,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobHandle":
        return cls(
            id=d["id"],
            state=JobState(d.get("state", "unknown")),
            workdir=Path(d["workdir"]),
            cluster=d.get("cluster"),
            submitted_at=d.get("submitted_at"),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            exit_code=d.get("exit_code"),
            extra=d.get("extra", {}),
        )


class SchedulerError(RuntimeError):
    """Raised for any scheduler-level failure (SSH, sbatch parse, etc.)."""


class Scheduler(ABC):
    """Run a JobManifest somewhere. Implementations: Local, SLURM, ..."""

    name: str = "abstract"

    @abstractmethod
    def submit(self, manifest: JobManifest) -> JobHandle: ...

    @abstractmethod
    def poll(self, handle: JobHandle) -> JobHandle:
        """Return an updated handle (state / exit_code / timestamps)."""

    @abstractmethod
    def cancel(self, handle: JobHandle) -> JobHandle: ...

    @abstractmethod
    def stream_log(self, handle: JobHandle, *, follow: bool = True) -> Iterator[str]:
        """Yield log lines as they appear, optionally tailing while RUNNING."""

    @abstractmethod
    def fetch_results(self, handle: JobHandle, dest: Path) -> Path:
        """Download/copy the workdir contents into ``dest`` and return the destination."""
