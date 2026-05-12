"""Helpers for evaluating densities/orbitals on a regular grid and writing
Gaussian "cube" files.

This is a thin shim over PySCF: we build the grid here (so users can
control extent/spacing without learning PySCF's cubegen flags) and
delegate AO evaluation to ``pyscf.dft.numint``. Cube format follows the
Gaussian convention (one file per scalar field).

Bohr is the canonical unit for cube files; we accept Angstrom inputs
and convert at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ANGSTROM_PER_BOHR = 0.529177210903
BOHR_PER_ANGSTROM = 1.0 / ANGSTROM_PER_BOHR


@dataclass
class CubeGrid:
    origin: np.ndarray            # (3,) Bohr
    axes: np.ndarray              # (3,3) Bohr per voxel
    shape: tuple[int, int, int]   # (nx, ny, nz)

    def points(self) -> np.ndarray:
        """Return (N,3) array of grid points in Bohr, in cube-file order
        (slowest axis: x, then y, fastest: z)."""
        nx, ny, nz = self.shape
        i = np.arange(nx)[:, None, None]
        j = np.arange(ny)[None, :, None]
        k = np.arange(nz)[None, None, :]
        ax, ay, az = self.axes
        pts = (
            self.origin
            + i[..., None] * ax
            + j[..., None] * ay
            + k[..., None] * az
        )
        return pts.reshape(-1, 3)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))


def make_grid(
    coords_angstrom: np.ndarray,
    *,
    spacing: float = 0.20,        # Angstrom
    margin: float = 4.0,          # Angstrom around the molecule
) -> CubeGrid:
    """Build an axis-aligned cube grid that covers the molecule plus a margin."""
    pts = np.asarray(coords_angstrom, dtype=float)
    if pts.size == 0:
        raise ValueError("Cannot build a cube grid from an empty molecule.")
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    extent = hi - lo
    n = np.maximum(np.round(extent / spacing).astype(int) + 1, 4)
    origin = lo * BOHR_PER_ANGSTROM
    axes = np.eye(3) * (spacing * BOHR_PER_ANGSTROM)
    return CubeGrid(origin=origin, axes=axes, shape=tuple(int(x) for x in n))


def write_cube(
    path: str | Path,
    field: np.ndarray,
    grid: CubeGrid,
    *,
    atom_numbers: Sequence[int],
    atom_coords_bohr: np.ndarray,
    title: str = "qmmmkit cube",
    subtitle: str = "scalar field",
) -> Path:
    """Write a Gaussian cube file. ``field`` has shape ``grid.shape``."""
    path = Path(path)
    field = np.asarray(field, dtype=float).reshape(grid.shape)
    natoms = len(atom_numbers)
    nx, ny, nz = grid.shape
    ax, ay, az = grid.axes

    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n{subtitle}\n")
        f.write(f"{natoms:5d} {grid.origin[0]:12.6f} {grid.origin[1]:12.6f} {grid.origin[2]:12.6f}\n")
        for n, vec in ((nx, ax), (ny, ay), (nz, az)):
            f.write(f"{n:5d} {vec[0]:12.6f} {vec[1]:12.6f} {vec[2]:12.6f}\n")
        for z, xyz in zip(atom_numbers, atom_coords_bohr):
            f.write(f"{z:5d} {float(z):12.6f} {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n")
        flat = field.ravel()
        for i in range(0, flat.size, 6):
            chunk = flat[i:i + 6]
            f.write(" ".join(f"{x:13.5e}" for x in chunk) + "\n")
    return path


def evaluate_density(mol, rdm: np.ndarray, points_bohr: np.ndarray) -> np.ndarray:
    """ρ(r) on the given grid points using PySCF AO evaluators."""
    from pyscf import dft

    ao = dft.numint.eval_ao(mol, points_bohr, deriv=0)
    return dft.numint.eval_rho(mol, ao, rdm, xctype="LDA")


def evaluate_density_with_gradient(mol, rdm: np.ndarray, points_bohr: np.ndarray):
    """Return (rho, grad_rho) where grad_rho has shape (3, N). Needed for NCI/RDG."""
    from pyscf import dft

    ao = dft.numint.eval_ao(mol, points_bohr, deriv=1)  # (4, N, nao): val + d/dx,dy,dz
    rho = dft.numint.eval_rho(mol, ao, rdm, xctype="GGA")  # (4, N): rho, dx, dy, dz
    return rho[0], rho[1:4]


def evaluate_orbital(mol, mo_coeff_one: np.ndarray, points_bohr: np.ndarray) -> np.ndarray:
    """ψ(r) for a single MO (column vector in AO basis)."""
    from pyscf import dft

    ao = dft.numint.eval_ao(mol, points_bohr, deriv=0)  # (N, nao)
    return ao @ mo_coeff_one


def evaluate_density_hessian(
    mol,
    rdm: np.ndarray,
    points_bohr: np.ndarray,
    *,
    chunk: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(rho, grad_rho, hess_rho)`` evaluated on ``points_bohr``.

    Shapes:
        rho       (N,)
        grad_rho  (3, N)            -- (∂_a ρ)
        hess_rho  (N, 3, 3)         -- ∂_a ∂_b ρ  (symmetric)

    Uses PySCF's deriv=2 AO evaluator. RDM must be the *spin-summed* AO-basis
    1-RDM (closed-shell or sum of α+β). Internally we exploit the symmetry of P
    to halve the floating-point work.

    Points are processed in chunks of ``chunk`` to keep peak memory bounded.
    """
    from pyscf import dft

    P = np.asarray(rdm, dtype=float)
    pts = np.asarray(points_bohr, dtype=float)
    N = pts.shape[0]
    rho = np.zeros(N)
    grad = np.zeros((3, N))
    hess = np.zeros((N, 3, 3))

    # AO derivative ordering used by PySCF for deriv=2:
    #   0:val, 1:x, 2:y, 3:z, 4:xx, 5:yy, 6:zz, 7:xy, 8:xz, 9:yz
    HESS_INDEX = {
        (0, 0): 4, (1, 1): 5, (2, 2): 6,
        (0, 1): 7, (1, 0): 7,
        (0, 2): 8, (2, 0): 8,
        (1, 2): 9, (2, 1): 9,
    }

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        ao = dft.numint.eval_ao(mol, pts[start:stop], deriv=2)  # (10, n, nao)
        phi = ao[0]
        phi_a = ao[1:4]
        # rho = phi P phi
        phi_P = phi @ P                                  # (n, nao)
        rho_chunk = np.einsum("ni,ni->n", phi_P, phi)    # (n,)
        rho[start:stop] = rho_chunk
        for a in range(3):
            grad[a, start:stop] = 2.0 * np.einsum("ni,ni->n", phi_P, phi_a[a])
        for a in range(3):
            for b in range(a, 3):
                phi_ab = ao[HESS_INDEX[(a, b)]]
                term1 = 2.0 * np.einsum("ni,ni->n", phi_P, phi_ab)
                term2 = 2.0 * np.einsum(
                    "ni,ni->n", phi_a[a] @ P, phi_a[b],
                )
                val = term1 + term2
                hess[start:stop, a, b] = val
                if a != b:
                    hess[start:stop, b, a] = val
    return rho, grad, hess


__all__ = [
    "CubeGrid",
    "make_grid",
    "write_cube",
    "evaluate_density",
    "evaluate_density_with_gradient",
    "evaluate_density_hessian",
    "evaluate_orbital",
    "BOHR_PER_ANGSTROM",
    "ANGSTROM_PER_BOHR",
]
