"""Backwards-compatible facade over the new pydantic schemas.

The schemas live in ``qmmmkit.schemas``. We re-export the canonical names
here so old code (and external users who import from this module) keeps
working.
"""

from __future__ import annotations

from .schemas import (
    AnalysisSpec,
    EmbeddingSpec,
    JobManifest,
    ManifestValidationError,
    MMSpec,
    QMSpec,
    SystemSpec,
    TaskSpec,
)
from .schemas.manifest import CURRENT_SCHEMA

__all__ = [
    "AnalysisSpec",
    "CURRENT_SCHEMA",
    "EmbeddingSpec",
    "JobManifest",
    "ManifestValidationError",
    "MMSpec",
    "QMSpec",
    "SystemSpec",
    "TaskSpec",
]
