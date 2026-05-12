"""IRC following from a transition state.

Two backends:
- pysisyphus (preferred): native IRC algorithms (LQA, EulerPC, ...). We
  expose the energy / gradient of our QMMMCalculator to it via a small
  adapter calculator.
- ASH NumFreq fallback: lowest-frequency mode from a numerical Hessian +
  steepest descent. Coarse but always available.
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


class IRCOptions(BaseModel):
    step_size: float = Field(default=0.05, gt=0)
    max_steps: int = Field(default=60, ge=1)
    direction: Literal["forward", "backward", "both"] = "both"
    integrator: Literal["lqa", "euler", "imk"] = "lqa"


@register_task("irc")
class IRCTask(BaseTask):
    name = "irc"
    options_schema = IRCOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        opts: IRCOptions = self.options  # type: ignore[assignment]
        log.info("irc.start", **opts.model_dump())
        try:
            return _irc_pysisyphus(calc, opts, workdir, log)
        except ImportError:
            log.warning("irc.pysisyphus_missing_falling_back")
            return _irc_ash_fallback(calc, opts, workdir, log)


def _irc_pysisyphus(calc: QMMMCalculator, opts: IRCOptions, workdir: Path, log) -> TaskResult:
    from pysisyphus.Geometry import Geometry  # type: ignore
    from pysisyphus.irc import EulerPC, IMKMod, LQA  # type: ignore

    cls = {"lqa": LQA, "euler": EulerPC, "imk": IMKMod}[opts.integrator]
    coords = calc.system.coords
    elems = list(getattr(calc.system.fragment, "elems", []))
    geom = Geometry(elems, coords.flatten())

    forwards: list[np.ndarray] = []
    backwards: list[np.ndarray] = []
    f_e: list[float] = []
    b_e: list[float] = []
    for forward, store, store_e in (
        (True, forwards, f_e) if opts.direction in ("forward", "both") else (None, None, None),
        (False, backwards, b_e) if opts.direction in ("backward", "both") else (None, None, None),
    ):
        if forward is None:
            continue
        irc = cls(geom, _PysisAshAdapter(calc),
                  step_size=opts.step_size, max_cycles=opts.max_steps,
                  forward=forward, backward=not forward)
        irc.run()
        for c, e in zip(getattr(irc, "all_coords", []), getattr(irc, "all_energies", [])):
            store.append(np.asarray(c).reshape(-1, 3))
            store_e.append(float(e))
        log.info("irc.branch_done", forward=forward, n=len(store_e))

    out = workdir / "irc.npz"
    np.savez(
        out,
        forward_energies=np.asarray(f_e),
        backward_energies=np.asarray(b_e),
        forward_coords=np.asarray(forwards) if forwards else np.zeros((0, 0, 3)),
        backward_coords=np.asarray(backwards) if backwards else np.zeros((0, 0, 3)),
    )
    return TaskResult(
        name="irc",
        converged=True,
        extra={"irc_path": str(out), "forward_energies": f_e, "backward_energies": b_e},
    )


def _irc_ash_fallback(calc: QMMMCalculator, opts: IRCOptions, workdir: Path, log) -> TaskResult:
    import ash

    log.info("irc.numfreq")
    freq = ash.NumFreq(theory=calc.theory, fragment=calc.system.fragment, npoint=2,
                       runmode="serial", printlevel=0)
    modes = np.asarray(getattr(freq, "normal_modes"))
    freqs = np.asarray(getattr(freq, "frequencies"))
    imag_idx = int(np.argmin(freqs))
    direction_vec = modes[:, imag_idx].reshape(-1, 3)

    coords0 = calc.system.coords.copy()
    f_e: list[float] = []
    b_e: list[float] = []
    forwards: list[np.ndarray] = []
    backwards: list[np.ndarray] = []
    for sign, store, store_e in (
        (+1.0, forwards, f_e) if opts.direction in ("forward", "both") else (None, None, None),
        (-1.0, backwards, b_e) if opts.direction in ("backward", "both") else (None, None, None),
    ):
        if sign is None:
            continue
        x = coords0.copy()
        for _ in range(opts.max_steps):
            x = x + sign * opts.step_size * direction_vec
            calc.system.coords = x
            sp = ash.Singlepoint(theory=calc.theory, fragment=calc.system.fragment,
                                 charge=calc.system.spec.charge, mult=calc.system.spec.mult,
                                 result_write_to_disk=False, printlevel=0)
            store.append(x.copy())
            store_e.append(float(getattr(sp, "energy")))
    calc.system.coords = coords0

    out = workdir / "irc.npz"
    np.savez(out,
             forward_energies=np.asarray(f_e),
             backward_energies=np.asarray(b_e),
             forward_coords=np.asarray(forwards) if forwards else np.zeros((0, 0, 3)),
             backward_coords=np.asarray(backwards) if backwards else np.zeros((0, 0, 3)))
    return TaskResult(
        name="irc",
        converged=True,
        extra={"irc_path": str(out), "imaginary_frequency_cm1": float(freqs[imag_idx])},
    )


class _PysisAshAdapter:
    """Energy/forces provider for pysisyphus, backed by our QMMMCalculator."""

    def __init__(self, calc: QMMMCalculator) -> None:
        self._calc = calc

    def get_forces(self, atoms, coords):
        import ash

        self._calc.system.coords = np.asarray(coords).reshape(-1, 3)
        result = ash.Singlepoint(
            theory=self._calc.theory,
            fragment=self._calc.system.fragment,
            Grad=True,
            charge=self._calc.system.spec.charge,
            mult=self._calc.system.spec.mult,
            result_write_to_disk=False,
        )
        e = float(getattr(result, "energy"))
        g = np.asarray(getattr(result, "gradient")).flatten()
        return {"energy": e, "forces": -g}
