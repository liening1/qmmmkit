"""Task spec: which task to run + free-form options.

Validation of the option payload happens inside the task class itself
(via ``BaseTask.options_schema``), keeping this top-level model simple.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Registered task name (e.g. 'single_point', 'optimize').")
    options: dict = Field(default_factory=dict, description="Task-specific options dict.")
    checkpoint: str | None = Field(
        default=None, description="Path to a checkpoint directory to resume from. None = fresh run."
    )
