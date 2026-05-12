"""Pydantic v2 schemas for configs, manifests, and results.

Schemas are the contract between every layer:
- the GUI builds forms from the JSON Schema
- the CLI accepts YAML and validates it here
- the runner consumes manifests after validation
- the SLURM worker re-validates on the remote side (defence in depth)

Anything that ends up in ``manifest.yaml`` or ``result.json`` is
defined here.
"""

from .system import SystemSpec
from .qm import QMSpec
from .mm import MMSpec
from .embedding import EmbeddingSpec
from .task import TaskSpec
from .analysis import AnalysisSpec
from .manifest import JobManifest, ManifestValidationError
from .result import (
    AnalysisResult,
    JobResult,
    ProvenanceModel,
    TaskResult,
)

__all__ = [
    "SystemSpec",
    "QMSpec",
    "MMSpec",
    "EmbeddingSpec",
    "TaskSpec",
    "AnalysisSpec",
    "JobManifest",
    "ManifestValidationError",
    "AnalysisResult",
    "JobResult",
    "ProvenanceModel",
    "TaskResult",
]
