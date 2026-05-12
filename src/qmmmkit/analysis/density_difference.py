"""3D density-difference / charge-depletion cube analysis.

Given a converged QM/MM mean-field for the full QM region (A∪B) and two
fragment mean-fields (A with B as ghosts; B with A as ghosts), compute

    Δρ(r) = ρ_AB(r) - ρ_A(r) - ρ_B(r)

on a Gaussian cube grid covering the QM region. Positive ``Δρ`` means
electrons have accumulated in that voxel after fragments come together;
negative means depletion. This is the 3D companion to the 1D charge
displacement function in ``charge_displacement.py`` and is the standard
"charge depletion / accumulation map" used in metal-ligand and donor-
acceptor analyses.

The three mean-fields must use compatible basis sets and identical
geometries (fragment SCFs use ghost atoms for the partner so the basis
matches). We do NOT enforce that here — passing inconsistent inputs
will give meaningless cubes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_analysis
from ..schemas.result import AnalysisResult
from ._base import BaseAnalysis
from ._cube import (
    ANGSTROM_PER_BOHR,
    CubeGrid,
    evaluate_density,
    make_grid,
    write_cube,
)


@dataclass
class DensityDifferenceResult:
    cube_path: Path
    grid: CubeGrid
    integrated_positive: float    # electrons accumulated (∫Δρ⁺ dr)
    integrated_negative: float    # electrons depleted (∫Δρ⁻ dr)
    net_transfer: float           # ∫Δρ dr (should be ~0 for closed shells)


def density_difference_cube(
    mf_full,
    mf_A,
    mf_B,
    *,
    out_path: str | Path,
    spacing: float = 0.20,        # Angstrom
    margin: float = 4.0,          # Angstrom
) -> DensityDifferenceResult:
    """Compute and write Δρ(r) for the (A∪B - A - B) fragmentation."""
    out_path = Path(out_path)
    coords_ang = mf_full.mol.atom_coords() * ANGSTROM_PER_BOHR
    grid = make_grid(coords_ang, spacing=spacing, margin=margin)
    pts = grid.points()

    rho_full = evaluate_density(mf_full.mol, _spinsum(mf_full.make_rdm1()), pts)
    rho_A = evaluate_density(mf_A.mol, _spinsum(mf_A.make_rdm1()), pts)
    rho_B = evaluate_density(mf_B.mol, _spinsum(mf_B.make_rdm1()), pts)
    delta = (rho_full - rho_A - rho_B).reshape(grid.shape)

    voxel_volume = abs(np.linalg.det(grid.axes))
    integrated_pos = float(delta[delta > 0].sum() * voxel_volume)
    integrated_neg = float(delta[delta < 0].sum() * voxel_volume)
    net = float(delta.sum() * voxel_volume)

    write_cube(
        out_path, delta, grid,
        atom_numbers=list(mf_full.mol.atom_charges()),
        atom_coords_bohr=mf_full.mol.atom_coords(),
        title="qmmmkit density difference",
        subtitle="Δρ = ρ(AB) - ρ(A) - ρ(B); positive = accumulation",
    )
    return DensityDifferenceResult(
        cube_path=out_path,
        grid=grid,
        integrated_positive=integrated_pos,
        integrated_negative=integrated_neg,
        net_transfer=net,
    )


def _spinsum(rdm):
    if isinstance(rdm, tuple):
        return rdm[0] + rdm[1]
    return rdm


# ---------------------------------------------------------------------------
# Registered analysis plugin
# ---------------------------------------------------------------------------
class DensityDifferenceOptions(BaseModel):
    fragment_a_atoms: list[int] = Field(
        ..., min_length=1,
        description="0-indexed atom indices of fragment A (within the QM region).",
    )
    fragment_b_atoms: list[int] = Field(
        ..., min_length=1,
        description="0-indexed atom indices of fragment B (within the QM region).",
    )
    spacing: float = Field(default=0.20, gt=0)
    margin: float = Field(default=4.0, gt=0)


@register_analysis("density_difference")
class DensityDifferenceAnalysis(BaseAnalysis):
    name = "density_difference"
    options_schema = DensityDifferenceOptions

    def run(self, calc, *, workdir: Path, log) -> AnalysisResult:
        opts: DensityDifferenceOptions = self.options  # type: ignore[assignment]
        log.info("density_diff.start",
                 a_atoms=len(opts.fragment_a_atoms), b_atoms=len(opts.fragment_b_atoms))

        try:
            mf_full = self._scf_full(calc)
            mf_A = self._scf_fragment(calc, opts.fragment_a_atoms, ghost_atoms=opts.fragment_b_atoms)
            mf_B = self._scf_fragment(calc, opts.fragment_b_atoms, ghost_atoms=opts.fragment_a_atoms)
        except NotImplementedError as e:
            log.warning("density_diff.skipped", reason=str(e))
            return AnalysisResult(
                kind=self.name,
                method="skipped",
                summary={"reason": str(e)},
            )

        out_cube = Path(workdir) / "density_difference.cube"
        result = density_difference_cube(
            mf_full, mf_A, mf_B,
            out_path=out_cube, spacing=opts.spacing, margin=opts.margin,
        )

        log.info("density_diff.done", cube=str(out_cube),
                 net_transfer=result.net_transfer,
                 integrated_positive=result.integrated_positive)

        return AnalysisResult(
            kind=self.name,
            method="rho(AB) - rho(A) - rho(B) on cube grid",
            summary={
                "voxels": int(result.grid.n_voxels),
                "spacing_angstrom": opts.spacing,
                "integrated_positive_electrons": result.integrated_positive,
                "integrated_negative_electrons": result.integrated_negative,
                "net_transfer_electrons": result.net_transfer,
            },
            artifacts={"cube": str(result.cube_path)},
        )

    # ------------------------------------------------------------
    @staticmethod
    def _scf_full(calc):
        """Run the full QM/MM single point and return its PySCF mf."""
        return calc.get_qm_mf()

    @staticmethod
    def _scf_fragment(calc, atoms, *, ghost_atoms):
        """Run a fragment SCF: ``atoms`` real, ``ghost_atoms`` ghost, in MM env."""
        return calc.fragment_scf(
            real_atoms=list(atoms),
            ghost_atoms=list(ghost_atoms),
            embed_in_mm=True,
        )
