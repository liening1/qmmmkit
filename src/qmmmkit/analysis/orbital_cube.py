"""Orbital cube generator.

Writes Gaussian cube files for selected orbitals so they can be visualised
in VMD / Multiwfn / Avogadro / UCSF Chimera. By default we dump HOMO and
LUMO (with a configurable window of frontier orbitals); the user can also
ask for explicit indices.

For UHF/UKS we dump the alpha set unless ``spin="beta"`` is requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_analysis
from ..schemas.result import AnalysisResult
from ._base import BaseAnalysis
from ._cube import (
    ANGSTROM_PER_BOHR,
    evaluate_orbital,
    make_grid,
    write_cube,
)


class OrbitalCubeOptions(BaseModel):
    indices: list[int] | None = Field(
        default=None,
        description="Explicit 0-indexed MO indices. If None, uses ``frontier``.",
    )
    frontier: int = Field(
        default=2, ge=0,
        description="When ``indices`` is None: dump HOMO-(frontier-1) ... LUMO+(frontier-1).",
    )
    spin: Literal["alpha", "beta"] = "alpha"
    spacing: float = Field(default=0.20, gt=0)
    margin: float = Field(default=4.0, gt=0)


@register_analysis("orbital_cube")
class OrbitalCubeAnalysis(BaseAnalysis):
    name = "orbital_cube"
    options_schema = OrbitalCubeOptions

    def run(self, calc, *, workdir: Path, log) -> AnalysisResult:
        from .natural_orbitals import _resolve_mf

        opts: OrbitalCubeOptions = self.options  # type: ignore[assignment]
        log.info("orbital_cube.start", frontier=opts.frontier, spin=opts.spin)

        mf = _resolve_mf(calc)
        mol = mf.mol
        mo_coeff, mo_occ = self._get_orbitals(mf, opts.spin)
        idxs = self._select_indices(mo_occ, opts)
        log.info("orbital_cube.indices", indices=idxs)

        coords_ang = mol.atom_coords() * ANGSTROM_PER_BOHR
        grid = make_grid(coords_ang, spacing=opts.spacing, margin=opts.margin)
        pts = grid.points()
        atom_numbers = list(mol.atom_charges())
        atom_coords = mol.atom_coords()

        artifacts: dict[str, str] = {}
        for idx in idxs:
            label = self._label_for(idx, mo_occ)
            psi = evaluate_orbital(mol, mo_coeff[:, idx], pts).reshape(grid.shape)
            out_path = Path(workdir) / f"orbital_{idx:04d}_{label}.cube"
            write_cube(
                out_path, psi, grid,
                atom_numbers=atom_numbers, atom_coords_bohr=atom_coords,
                title=f"qmmmkit orbital {idx} ({label})",
                subtitle=f"spin={opts.spin}; occ={mo_occ[idx]:.3f}",
            )
            artifacts[f"orbital_{idx:04d}_{label}"] = str(out_path)
            log.info("orbital_cube.wrote", index=idx, label=label, path=str(out_path))

        return AnalysisResult(
            kind=self.name,
            method=f"orbitals on cube grid (spin={opts.spin})",
            summary={
                "indices": list(idxs),
                "spacing_angstrom": opts.spacing,
                "voxels_per_orbital": int(grid.n_voxels),
            },
            artifacts=artifacts,
        )

    # ------------------------------------------------------------
    @staticmethod
    def _get_orbitals(mf, spin: str):
        c = getattr(mf, "mo_coeff")
        o = getattr(mf, "mo_occ")
        if isinstance(c, (list, tuple)):
            return (c[0], o[0]) if spin == "alpha" else (c[1], o[1])
        return c, o

    @staticmethod
    def _select_indices(mo_occ: np.ndarray, opts: OrbitalCubeOptions) -> list[int]:
        if opts.indices is not None:
            return sorted({int(i) for i in opts.indices})
        homo = int(np.where(mo_occ > 1e-6)[0].max())
        lo = max(homo - opts.frontier + 1, 0)
        hi = min(homo + opts.frontier + 1, len(mo_occ))
        return list(range(lo, hi))

    @staticmethod
    def _label_for(idx: int, mo_occ: np.ndarray) -> str:
        homo = int(np.where(mo_occ > 1e-6)[0].max()) if (mo_occ > 1e-6).any() else -1
        if idx == homo:
            return "HOMO"
        if idx == homo + 1:
            return "LUMO"
        if idx < homo:
            return f"HOMO-{homo - idx}"
        return f"LUMO+{idx - homo - 1}"
