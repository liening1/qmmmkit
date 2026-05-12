"""Non-Covalent Interactions (NCI) analysis (Yang/Cohen/Johnson 2010).

Maps weak interactions — H-bonds, van der Waals contacts, steric clashes —
by combining two scalar fields evaluated on the QM region:

  RDG(r)  = (1 / (2 (3π²)^(1/3))) * |∇ρ(r)| / ρ(r)^(4/3)
  s(r)    = sign(λ₂(r)) * ρ(r)

where λ₂ is the *middle* eigenvalue of the density Hessian. RDG isosurfaces
near 0.5 a.u. mark interaction regions; colouring those isosurfaces by
sign(λ₂)·ρ distinguishes attractive (λ₂<0, blue), van der Waals (λ₂≈0,
green) and repulsive (λ₂>0, red) contacts.

Two cubes are written: ``rdg.cube`` and ``sign_lambda2_rho.cube``. They
are designed to be visualised together (RDG isosurface coloured by
sign-density) in VMD / Multiwfn / UCSF Chimera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_analysis
from ..schemas.result import AnalysisResult
from ._base import BaseAnalysis
from ._cube import (
    ANGSTROM_PER_BOHR,
    evaluate_density_hessian,
    make_grid,
    write_cube,
)


_RDG_PREFACTOR = 1.0 / (2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0))


class NCIOptions(BaseModel):
    spacing: float = Field(default=0.10, gt=0, description="Grid spacing in Angstrom.")
    margin: float = Field(default=2.0, gt=0, description="Margin around the molecule (Angstrom).")
    rho_cutoff: float = Field(
        default=0.05, gt=0,
        description="Density (a.u.) above which RDG is set to a large dummy value, so "
                    "core / bonded regions don't drown the interaction iso-surface.",
    )
    rdg_cutoff: float = Field(
        default=2.0, gt=0,
        description="RDG above which voxels are masked (also pushed to a dummy value).",
    )


@register_analysis("nci")
class NCIAnalysis(BaseAnalysis):
    name = "nci"
    options_schema = NCIOptions

    def run(self, calc, *, workdir: Path, log) -> AnalysisResult:
        from .natural_orbitals import _resolve_mf

        opts: NCIOptions = self.options  # type: ignore[assignment]
        log.info("nci.start", spacing=opts.spacing, margin=opts.margin)

        mf = _resolve_mf(calc)
        mol = mf.mol
        rdm = mf.make_rdm1()
        if isinstance(rdm, tuple):
            rdm = rdm[0] + rdm[1]

        coords_ang = mol.atom_coords() * ANGSTROM_PER_BOHR
        grid = make_grid(coords_ang, spacing=opts.spacing, margin=opts.margin)
        pts = grid.points()

        log.info("nci.evaluating", voxels=int(grid.n_voxels))
        rho, grad, hess = evaluate_density_hessian(mol, rdm, pts)
        rho = np.maximum(rho, 1e-30)

        # RDG; mask high-density (covalent) and high-RDG voxels for clarity.
        gnorm = np.linalg.norm(grad, axis=0)
        rdg = _RDG_PREFACTOR * gnorm / np.power(rho, 4.0 / 3.0)
        masked = (rho > opts.rho_cutoff) | (rdg > opts.rdg_cutoff)
        rdg_display = np.where(masked, 100.0, rdg)

        # sign(λ₂) · ρ — middle eigenvalue of the (3,3) density Hessian.
        eigvals = np.linalg.eigvalsh(hess)             # (N, 3), ascending
        lambda2 = eigvals[:, 1]
        sign_l2_rho = np.sign(lambda2) * rho

        rdg_path = Path(workdir) / "rdg.cube"
        sign_path = Path(workdir) / "sign_lambda2_rho.cube"
        atom_numbers = list(mol.atom_charges())
        atom_coords = mol.atom_coords()
        write_cube(
            rdg_path, rdg_display, grid,
            atom_numbers=atom_numbers, atom_coords_bohr=atom_coords,
            title="qmmmkit NCI / reduced density gradient",
            subtitle="RDG (a.u.); core/bonded voxels masked at 100",
        )
        write_cube(
            sign_path, sign_l2_rho, grid,
            atom_numbers=atom_numbers, atom_coords_bohr=atom_coords,
            title="qmmmkit NCI / sign(lambda_2) * rho",
            subtitle="negative=attractive, ~0=vdW, positive=repulsive",
        )

        # Crude classification of interaction voxels: those with low RDG and small ρ.
        interaction = (rdg < 0.5) & (rho < opts.rho_cutoff)
        attractive = int(np.sum(interaction & (lambda2 < 0)))
        vdw = int(np.sum(interaction & (np.abs(lambda2) < 1e-3)))
        repulsive = int(np.sum(interaction & (lambda2 > 0)))
        log.info("nci.done", attractive=attractive, vdw=vdw, repulsive=repulsive,
                 rdg_cube=str(rdg_path), sign_cube=str(sign_path))

        return AnalysisResult(
            kind=self.name,
            method="reduced density gradient + sign(lambda_2) * rho on cube grid",
            summary={
                "voxels": int(grid.n_voxels),
                "spacing_angstrom": opts.spacing,
                "interaction_voxels_total": int(np.sum(interaction)),
                "interaction_voxels_attractive": attractive,
                "interaction_voxels_vdw": vdw,
                "interaction_voxels_repulsive": repulsive,
                "rho_cutoff": opts.rho_cutoff,
                "rdg_cutoff": opts.rdg_cutoff,
            },
            artifacts={
                "rdg_cube": str(rdg_path),
                "sign_lambda2_rho_cube": str(sign_path),
            },
        )
