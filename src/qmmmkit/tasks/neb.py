"""Nudged Elastic Band reaction-path search.

State-of-the-art use case: you have reactant and product geometries (and
optionally a few intermediate guesses) and want to discover the connecting
transition state without already knowing it.

Backend: pysisyphus's NEB implementation, with the climbing-image variant
on by default. Each image's energy / gradient is evaluated through our
QMMMCalculator via a small adapter so the NEB sees the QM/MM PES.

Outputs:
- ``neb_energies.npy``      energies along the path, shape (n_images,)
- ``neb_path.npy``          coordinates along the path, shape (n_images, n_atoms, 3)
- ``neb_climbing.xyz``      the highest-energy image, the TS guess
- ``neb_path.xyz``          full optimized path as multi-frame XYZ
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .._registry import register_task
from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult
from ._base import BaseTask


class NEBOptions(BaseModel):
    images: list[str] = Field(
        ..., min_length=2,
        description="XYZ files defining the reaction path. The first and last are "
                    "fixed endpoints; intermediates are optional starting guesses. "
                    "If only two are given, the band is interpolated linearly.",
    )
    n_images: int = Field(
        default=11, ge=3,
        description="Total band size including endpoints. The interior is linearly "
                    "interpolated between the supplied images.",
    )
    climbing_image: bool = Field(
        default=True,
        description="Switch to climbing-image NEB after a few cycles to converge "
                    "the highest image onto the saddle point.",
    )
    spring_constant: float = Field(default=0.01, gt=0)
    max_cycles: int = Field(default=200, ge=1)
    rms_force_threshold: float = Field(default=5e-3, gt=0, description="Hartree/Bohr.")
    max_force_threshold: float = Field(default=1e-2, gt=0, description="Hartree/Bohr.")
    align_method: Literal["procrustes", "none"] = "procrustes"
    parallel_eval: bool = Field(
        default=False,
        description="Evaluate image gradients in parallel (off by default to avoid "
                    "thrashing a single-node QM workflow).",
    )

    @field_validator("images")
    @classmethod
    def _images_exist(cls, v: list[str]) -> list[str]:
        for p in v:
            if not Path(p).exists():
                raise ValueError(f"Image file does not exist: {p}")
        return v


@register_task("neb")
class NEBTask(BaseTask):
    name = "neb"
    options_schema = NEBOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        try:
            from pysisyphus.Geometry import Geometry  # type: ignore
            from pysisyphus.cos.NEB import NEB  # type: ignore
            from pysisyphus.optimizers.LBFGS import LBFGS  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "NEB requires pysisyphus. Install with `pip install qmmmkit[optimizers]`."
            ) from e

        opts: NEBOptions = self.options  # type: ignore[assignment]
        log.info("neb.start", images=len(opts.images), n_images=opts.n_images,
                 climbing=opts.climbing_image)

        geometries = self._load_endpoints(opts.images)
        band = self._interpolate_band(geometries, opts.n_images)
        log.info("neb.band_built", n=len(band))

        adapter = _PysisAshAdapter(calc)
        for g in band:
            g.set_calculator(adapter)

        cos = NEB(
            band,
            k=opts.spring_constant,
            climb=opts.climbing_image,
            align=opts.align_method != "none",
            parallel=opts.parallel_eval,
        )
        opt = LBFGS(
            cos,
            max_cycles=opts.max_cycles,
            rms_force=opts.rms_force_threshold,
            max_force=opts.max_force_threshold,
            dump=False,
        )
        opt.run()
        log.info("neb.optimized", cycles=opt.cur_cycle, converged=opt.is_converged)

        # Pull the optimized band coordinates + energies
        path_coords = np.array([g.coords.reshape(-1, 3) for g in cos.images])
        energies = np.array([g.energy for g in cos.images])
        ts_idx = int(np.argmax(energies))

        np.save(workdir / "neb_energies.npy", energies)
        np.save(workdir / "neb_path.npy", path_coords)
        elems = list(getattr(calc.system.fragment, "elems", []))
        if elems:
            self._write_xyz_path(workdir / "neb_path.xyz", elems, path_coords, energies)
            self._write_single_xyz(
                workdir / "neb_climbing.xyz", elems, path_coords[ts_idx],
                title=f"qmmmkit NEB climbing image (E = {energies[ts_idx]:.8f} Ha)",
            )

        log.info("neb.done", ts_index=ts_idx, ts_energy=float(energies[ts_idx]),
                 barrier_forward=float(energies[ts_idx] - energies[0]),
                 barrier_reverse=float(energies[ts_idx] - energies[-1]))

        return TaskResult(
            name=self.name,
            converged=bool(opt.is_converged),
            energy=float(energies[ts_idx]),
            nsteps=int(opt.cur_cycle),
            extra={
                "energies": energies.tolist(),
                "ts_index": ts_idx,
                "barrier_forward_hartree": float(energies[ts_idx] - energies[0]),
                "barrier_reverse_hartree": float(energies[ts_idx] - energies[-1]),
                "path_coords_path": str(workdir / "neb_path.npy"),
                "path_xyz": str(workdir / "neb_path.xyz") if elems else None,
                "climbing_xyz": str(workdir / "neb_climbing.xyz") if elems else None,
            },
        )

    # ------------------------------------------------------------
    @staticmethod
    def _load_endpoints(image_paths: list[str]):
        from pysisyphus.helpers import geom_loader  # type: ignore

        return [geom_loader(p) for p in image_paths]

    @staticmethod
    def _interpolate_band(endpoints, n_images: int):
        """Linear interpolation between the supplied endpoints.

        If the user supplied k images and asks for N total, intermediates are
        distributed proportionally between adjacent supplied images.
        """
        from pysisyphus.helpers import interpolate as _interpolate  # type: ignore

        if len(endpoints) >= n_images:
            return endpoints[:n_images]
        # Distribute the remaining N-k slots between the k-1 segments.
        gap = n_images - len(endpoints)
        segments = len(endpoints) - 1
        per_segment = [gap // segments] * segments
        for i in range(gap % segments):
            per_segment[i] += 1

        out = [endpoints[0]]
        for i, n_extra in enumerate(per_segment):
            inter = _interpolate(endpoints[i], endpoints[i + 1], n_extra)
            out.extend(inter)
            out.append(endpoints[i + 1])
        return out

    @staticmethod
    def _write_xyz_path(path: Path, elems: list[str], coords: np.ndarray,
                        energies: np.ndarray) -> None:
        with path.open("w", encoding="utf-8") as f:
            for i, (frame, e) in enumerate(zip(coords, energies)):
                f.write(f"{len(elems)}\nimage {i:3d}  E = {e:.8f} Ha\n")
                for el, (x, y, z) in zip(elems, frame):
                    f.write(f"{el:<2}  {x: .8f}  {y: .8f}  {z: .8f}\n")

    @staticmethod
    def _write_single_xyz(path: Path, elems: list[str], coords: np.ndarray,
                          *, title: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write(f"{len(elems)}\n{title}\n")
            for el, (x, y, z) in zip(elems, coords):
                f.write(f"{el:<2}  {x: .8f}  {y: .8f}  {z: .8f}\n")


class _PysisAshAdapter:
    """Adapter exposing our QMMMCalculator as a pysisyphus calculator."""

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
            printlevel=0,
        )
        e = float(getattr(result, "energy"))
        g = np.asarray(getattr(result, "gradient")).flatten()
        return {"energy": e, "forces": -g}

    def get_energy(self, atoms, coords):
        result = self.get_forces(atoms, coords)
        return {"energy": result["energy"]}
