"""Single-point energy + (optional) gradient on the QM/MM PES."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_task
from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult
from ._base import BaseTask


class SinglePointOptions(BaseModel):
    gradient: bool = Field(default=False, description="Also compute the QM/MM gradient.")


@register_task("single_point")
class SinglePointTask(BaseTask):
    name = "single_point"
    options_schema = SinglePointOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        import ash

        opts: SinglePointOptions = self.options  # type: ignore[assignment]

        log.info("singlepoint.start", grad=opts.gradient)
        result = ash.Singlepoint(
            theory=calc.theory,
            fragment=calc.system.fragment,
            Grad=opts.gradient,
            charge=calc.system.spec.charge,
            mult=calc.system.spec.mult,
            result_write_to_disk=False,
            printlevel=1,
        )

        energy = float(getattr(result, "energy"))
        qm_e = _maybe_float(getattr(result, "qm_energy", None) or getattr(calc.theory, "QMenergy", None))
        mm_e = _maybe_float(getattr(result, "mm_energy", None) or getattr(calc.theory, "MMenergy", None))
        gradient_path: str | None = None
        if opts.gradient:
            grad = np.asarray(getattr(result, "gradient"))
            gpath = workdir / "gradient.npy"
            np.save(gpath, grad)
            gradient_path = str(gpath)

        log.info("singlepoint.done", energy=energy, qm_energy=qm_e, mm_energy=mm_e)
        return TaskResult(
            name=self.name,
            converged=True,
            energy=energy,
            qm_energy=qm_e,
            mm_energy=mm_e,
            gradient_path=gradient_path,
        )


def _maybe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
