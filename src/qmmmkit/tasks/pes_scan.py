"""Relaxed/rigid PES scans over an arbitrary set of internal coordinates."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_task
from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult
from ._base import BaseTask


class ScanCoordinate(BaseModel):
    """One scan dimension: a bond, angle, or dihedral, plus the values to visit."""

    kind: Literal["bond", "angle", "dihedral"]
    atoms: list[int]
    values: list[float]
    label: str | None = None

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        natoms = {"bond": 2, "angle": 3, "dihedral": 4}[self.kind]
        if len(self.atoms) != natoms:
            raise ValueError(f"{self.kind} requires {natoms} atom indices.")
        if not self.label:
            self.label = f"{self.kind}_{'_'.join(map(str, self.atoms))}"


class PESScanOptions(BaseModel):
    coordinates: list[ScanCoordinate]
    relaxed: bool = True
    label: str = "pes"


@register_task("pes_scan")
class PESScanTask(BaseTask):
    name = "pes_scan"
    options_schema = PESScanOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        import ash

        opts: PESScanOptions = self.options  # type: ignore[assignment]
        coords = opts.coordinates
        grid_shape = tuple(len(c.values) for c in coords)
        energies, converged, completed_count = self._initialise_grid(grid_shape, workdir, log)

        log.info("pes_scan.start", grid=list(grid_shape), relaxed=opts.relaxed,
                 dimensions=[c.label for c in coords],
                 already_completed=int(completed_count),
                 total=int(np.prod(grid_shape)))

        for idx in product(*[range(n) for n in grid_shape]):
            if converged[idx]:
                continue  # restart skips already-converged points
            constraints = _build_constraints(coords, idx)
            _seed_geometry(calc.system, coords, idx)

            if opts.relaxed:
                res = ash.Optimizer(
                    theory=calc.theory,
                    fragment=calc.system.fragment,
                    coordsystem="hdlc",
                    ActiveRegion=calc.system.active_atoms is not None,
                    actatoms=calc.system.active_atoms,
                    constraints=constraints,
                    constrainvalue=True,
                    charge=calc.system.spec.charge,
                    mult=calc.system.spec.mult,
                )
                e = float(getattr(res, "energy", float("nan")))
                converged[idx] = bool(getattr(res, "converged", True))
            else:
                res = ash.Singlepoint(
                    theory=calc.theory, fragment=calc.system.fragment,
                    charge=calc.system.spec.charge, mult=calc.system.spec.mult,
                    result_write_to_disk=False, printlevel=0,
                )
                e = float(getattr(res, "energy", float("nan")))
                converged[idx] = True
            energies[idx] = e
            log.info("pes_scan.point", idx=list(idx), energy=e)
            self._checkpoint(workdir, energies, converged, grid_shape, opts)

        out_path = workdir / f"{opts.label}_energies.npy"
        np.save(out_path, energies)
        if bool(converged.all()):
            self.clear_checkpoint(workdir)
        log.info("pes_scan.done", saved=str(out_path))

        return TaskResult(
            name=self.name,
            converged=bool(converged.all()),
            energy=float(energies.min()) if np.isfinite(energies).any() else None,
            extra={
                "grid_shape": list(grid_shape),
                "energies_path": str(out_path),
                "coordinates": [c.model_dump() for c in coords],
            },
        )

    # ------------------------------------------------------------
    def _initialise_grid(self, grid_shape: tuple[int, ...], workdir: Path, log):
        """Either start fresh or resume from a saved checkpoint."""
        state = self.load_checkpoint(workdir)
        if state and tuple(state.get("grid_shape", ())) == grid_shape:
            energies = np.asarray(state["energies"], dtype=float).reshape(grid_shape)
            converged = np.asarray(state["converged"], dtype=bool).reshape(grid_shape)
            completed = int(converged.sum())
            log.info("pes_scan.restart", completed_points=completed,
                     total=int(np.prod(grid_shape)))
            return energies, converged, completed
        return (
            np.full(grid_shape, np.nan),
            np.zeros(grid_shape, dtype=bool),
            0,
        )

    def _checkpoint(self, workdir: Path, energies: np.ndarray, converged: np.ndarray,
                    grid_shape: tuple[int, ...], opts: PESScanOptions) -> None:
        self.save_checkpoint(workdir, {
            "grid_shape": list(grid_shape),
            "energies": energies.tolist(),
            "converged": converged.astype(int).tolist(),
            "label": opts.label,
            "relaxed": opts.relaxed,
        })


# ---------------------------------------------------------------------------
def _build_constraints(coords: Sequence[ScanCoordinate], idx: tuple[int, ...]) -> dict:
    bonds, angles, dihedrals = [], [], []
    for c, i in zip(coords, idx):
        v = float(c.values[i])
        if c.kind == "bond":
            bonds.append([list(c.atoms), v])
        elif c.kind == "angle":
            angles.append([list(c.atoms), v])
        else:
            dihedrals.append([list(c.atoms), v])
    out: dict = {}
    if bonds:
        out["bond"] = bonds
    if angles:
        out["angle"] = angles
    if dihedrals:
        out["dihedral"] = dihedrals
    return out


def _seed_geometry(system, coords: Sequence[ScanCoordinate], idx: tuple[int, ...]) -> None:
    """Initial guess for bond stretches; angles/dihedrals are reached by the constraint."""
    xyz = system.coords.copy()
    for c, i in zip(coords, idx):
        if c.kind != "bond":
            continue
        a, b = c.atoms
        target = float(c.values[i])
        v = xyz[b] - xyz[a]
        d = float(np.linalg.norm(v))
        if d < 1e-8:
            continue
        xyz[b] = xyz[a] + v * (target / d)
    system.coords = xyz
