"""JobManifest: the top-level YAML object users author.

Holds a single calculation. The runner consumes it; the GUI builds it;
the SLURM worker re-validates it on the remote side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .analysis import AnalysisSpec
from .embedding import EmbeddingSpec
from .mm import MMSpec
from .qm import QMSpec
from .system import SystemSpec
from .task import TaskSpec

CURRENT_SCHEMA = "0.2"


class ManifestValidationError(ValueError):
    """Wraps pydantic ValidationError with a friendlier message."""


class JobManifest(BaseModel):
    """Reproducible description of a single QM/MM calculation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    qmmmkit_schema: str = Field(default=CURRENT_SCHEMA, alias="qmmmkit_version")
    name: str = Field(default="qmmmkit_job")
    notes: str | None = None

    system: SystemSpec
    qm: QMSpec = Field(default_factory=QMSpec)
    mm: MMSpec = Field(default_factory=MMSpec)
    embedding: EmbeddingSpec = Field(default_factory=EmbeddingSpec)
    task: TaskSpec
    analyses: list[AnalysisSpec] = Field(default_factory=list)

    # Not part of the YAML; populated after load() so relative paths can be resolved.
    source_path: Path | None = Field(default=None, exclude=True)

    # ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "JobManifest":
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        try:
            obj = cls.model_validate(raw)
        except ValidationError as e:
            raise ManifestValidationError(_pretty_errors(e, path)) from e
        obj.source_path = path
        return obj

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "JobManifest":
        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ManifestValidationError(_pretty_errors(e)) from e

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=exclude_none)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        return path

    # ------------------------------------------------------------
    def resolve_paths(self) -> None:
        """Resolve relative paths against the manifest's directory (or cwd)."""
        base = self.source_path.parent if self.source_path else Path.cwd()
        self.system.resolve_paths(base)

    def input_files(self) -> list[Path]:
        return self.system.input_files()


def _pretty_errors(err: ValidationError, path: Path | None = None) -> str:
    lines = [f"Manifest validation failed{f' for {path}' if path else ''}:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"])
        lines.append(f"  - {loc}: {e['msg']}")
    return "\n".join(lines)
