"""Analysis spec: name + options. Validation is delegated to BaseAnalysis subclasses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnalysisSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: str = Field(..., description="Registered analysis name (e.g. 'population', 'natural_orbitals').")
    options: dict = Field(default_factory=dict)
