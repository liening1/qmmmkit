"""Base class for post-task analyses (population, NOs, CDA, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from ..calculator import QMMMCalculator
from ..schemas.result import AnalysisResult


class BaseAnalysis(ABC):
    """Concrete analyses subclass and decorate with ``@register_analysis(name)``."""

    name: str = "abstract"
    options_schema: Type[BaseModel] | None = None

    def __init__(self, options: dict | BaseModel | None = None) -> None:
        if isinstance(options, BaseModel):
            self.options = options
        elif options is None:
            self.options = self.options_schema() if self.options_schema else _Empty()
        else:
            schema = self.options_schema or _Empty
            self.options = schema(**options)

    @abstractmethod
    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> AnalysisResult: ...


class _Empty(BaseModel):
    pass


__all__ = ["BaseAnalysis"]
