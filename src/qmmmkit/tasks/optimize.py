"""Geometry optimisation: minima and transition states.

Both backed by ``ash.Optimizer`` (alias for ``geomeTRICOptimizer``). For
QM/MM we always opt the *active* region, which is required for tractable
biomolecular jobs and is the same convention every modern QM/MM tool
uses (ChemShell, Q-Chem, etc.).

Real ASH kwargs (verified against ash.modules.module_Optimizer):
    Optimizer(theory, fragment, coordsystem='tric', frozenatoms=None,
              constraints=None, maxiter=50, ActiveRegion=False, actatoms=None,
              convergence_setting=None, conv_criteria=None,
              TSOpt=False, hessian=None,
              charge=None, mult=None)

Hessian options for TS search: 'first', 'each', 'never', 'file:<path>'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_task
from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult
from ._base import BaseTask


class OptOptions(BaseModel):
    coordsystem: Literal["tric", "hdlc", "dlc", "cartesian"] = "tric"
    maxiter: int = Field(default=250, ge=1)
    convergence: Literal["GAU", "GAU_TIGHT", "GAU_LOOSE", "ORCA", "ORCA_TIGHT", None] = "GAU"
    constraints: dict | None = None
    frozen_atoms: list[int] | None = None
    write_trajectory: bool = True


class TSOptions(OptOptions):
    convergence: Literal["GAU", "GAU_TIGHT", "GAU_LOOSE", "ORCA", "ORCA_TIGHT", None] = "GAU_TIGHT"
    hessian: Literal["first", "each", "never"] | str = "first"
    coordsystem: Literal["tric", "hdlc", "dlc", "cartesian"] = "tric"


def _run_optimizer(
    calc: QMMMCalculator, *, ts: bool, opts: OptOptions, workdir: Path, log, task,
) -> TaskResult:
    import ash

    actatoms = calc.system.active_atoms
    extra_kwargs: dict = {}
    if ts:
        extra_kwargs["TSOpt"] = True
        extra_kwargs["hessian"] = getattr(opts, "hessian", "first")

    # Resume from a checkpoint if one is present in the workdir. The checkpoint
    # carries the last geometry; ASH's Optimizer is one-shot so we can only
    # restart from the most recent checkpoint, not mid-iteration.
    state = task.load_checkpoint(workdir) if task is not None else None
    if state and "coords" in state:
        coords = np.asarray(state["coords"])
        if coords.shape == calc.system.coords.shape:
            calc.system.coords = coords
            log.info("optimize.restart", from_step=state.get("step"),
                     prev_energy=state.get("energy"))
        else:
            log.warning("optimize.restart_shape_mismatch",
                        expected=list(calc.system.coords.shape), got=list(coords.shape))

    log.info("optimize.start", ts=ts, coordsystem=opts.coordsystem, maxiter=opts.maxiter,
             active_atoms=len(actatoms or []),
             resumed=state is not None and "coords" in state)
    result = ash.Optimizer(
        theory=calc.theory,
        fragment=calc.system.fragment,
        coordsystem=opts.coordsystem,
        maxiter=opts.maxiter,
        ActiveRegion=actatoms is not None,
        actatoms=actatoms,
        convergence_setting=opts.convergence,
        constraints=opts.constraints,
        frozenatoms=opts.frozen_atoms,
        charge=calc.system.spec.charge,
        mult=calc.system.spec.mult,
        **extra_kwargs,
    )

    energy = float(getattr(result, "energy", float("nan")))
    converged = bool(getattr(result, "converged", True))
    nsteps = int(getattr(result, "nsteps", getattr(result, "iterations", 0) or 0))
    log.info("optimize.done", energy=energy, converged=converged, nsteps=nsteps)

    final_xyz = None
    if opts.write_trajectory:
        final_xyz = _write_xyz(calc.system, workdir=workdir, log=log)

    if task is not None:
        if converged:
            # Clear the checkpoint on successful convergence so a re-run with
            # the same workdir starts fresh from the optimised geometry only
            # if the user explicitly wants to (by editing the manifest).
            task.clear_checkpoint(workdir)
        else:
            # Save the latest geometry so a follow-up run resumes from here.
            task.save_checkpoint(workdir, {
                "coords": calc.system.coords.tolist(),
                "step": nsteps,
                "energy": energy,
                "ts": ts,
            })
            log.info("optimize.checkpoint_saved", path=str(task.checkpoint_path(workdir)))

    return TaskResult(
        name="transition_state" if ts else "optimize",
        converged=converged,
        energy=energy,
        nsteps=nsteps,
        final_geometry=final_xyz,
    )


@register_task("optimize")
class OptimizeTask(BaseTask):
    name = "optimize"
    options_schema = OptOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        return _run_optimizer(  # type: ignore[arg-type]
            calc, ts=False, opts=self.options, workdir=workdir, log=log, task=self,
        )


@register_task("transition_state")
class TransitionStateTask(BaseTask):
    name = "transition_state"
    options_schema = TSOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        return _run_optimizer(  # type: ignore[arg-type]
            calc, ts=True, opts=self.options, workdir=workdir, log=log, task=self,
        )


def _write_xyz(system, *, workdir: Path, log) -> str | None:
    """Dump the current geometry of ``system`` to ``<workdir>/final.xyz``."""
    try:
        coords = system.coords
        elems = list(getattr(system.fragment, "elems", []))
        if not elems:
            log.warning("optimize.no_elements_for_xyz")
            return None
        out = Path(workdir) / "final.xyz"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"{len(elems)}\n")
            f.write("qmmmkit final geometry\n")
            for el, (x, y, z) in zip(elems, coords):
                f.write(f"{el:<2}  {x: .8f}  {y: .8f}  {z: .8f}\n")
        return str(out)
    except Exception as e:  # noqa: BLE001
        log.warning("optimize.xyz_write_failed", error=str(e))
        return None
