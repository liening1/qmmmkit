"""Base class for tasks.

A task is a one-shot computation: single point, opt, TS, IRC, scan, NEB, ...
Subclasses implement ``run`` and register themselves with
``@register_task("name")``.

Contract:
- ``options_schema`` is a pydantic model class describing the task's
  YAML options. Validated automatically at submit time.
- ``run`` returns a ``TaskResult``. The runner is responsible for
  surrounding it with provenance + saving result.json.
- ``save_checkpoint(workdir, state)`` writes a JSON-serialisable resumable
  state into the task's well-known checkpoint file (``<task>_checkpoint.json``).
- ``load_checkpoint(workdir)`` reads it back (returns None if absent).

Checkpointing is opt-in: a task that doesn't call save/load behaves as
before. Tasks that do call them gain free crash-recovery: if a previous
run wrote a checkpoint and crashed, the next invocation in the same
workdir picks up where it left off.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult


class BaseTask(ABC):
    """Concrete tasks subclass this and decorate with ``@register_task(name)``."""

    name: str = "abstract"
    options_schema: Type[BaseModel] | None = None  # set by subclasses

    def __init__(self, options: dict | BaseModel | None = None) -> None:
        if isinstance(options, BaseModel):
            self.options = options
        elif options is None:
            self.options = self.options_schema() if self.options_schema else _EmptyOptions()
        else:
            schema = self.options_schema or _EmptyOptions
            self.options = schema(**options)

    # ------------------------------------------------------------
    @abstractmethod
    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult: ...

    # ------------------------------------------------------------
    def checkpoint_path(self, workdir: Path) -> Path:
        """Where this task stores its checkpoint within a workdir."""
        return Path(workdir) / f"{self.name}_checkpoint.json"

    def save_checkpoint(self, workdir: Path, state: dict[str, Any]) -> Path:
        """Atomically write a JSON checkpoint. Returns the path."""
        target = self.checkpoint_path(workdir)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.replace(target)
        return target

    def load_checkpoint(self, workdir: Path) -> dict[str, Any] | None:
        """Read the previous checkpoint, or return None if absent / corrupt."""
        target = self.checkpoint_path(workdir)
        if not target.exists():
            return None
        try:
            return json.loads(target.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear_checkpoint(self, workdir: Path) -> None:
        target = self.checkpoint_path(workdir)
        if target.exists():
            target.unlink()


class _EmptyOptions(BaseModel):
    """Marker pydantic model for tasks that take no options."""


__all__ = ["BaseTask"]
