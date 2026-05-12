"""Charge displacement analysis (CDA / charge displacement curve, CDC).

Following Belpassi, Infante, Tarantelli, Tarantelli (J. Am. Chem. Soc. 2008,
130, 1048) and the analogous "charge depletion" maps used in metal-ligand
bonding analyses. For two fragments A and B making up the QM region:

    Δρ(r) = ρ_AB(r) - ρ_A(r) - ρ_B(r)

where each ρ is evaluated with ghost atoms for the partner so the basis
matches and BSSE is treated consistently. The charge displacement function

    Δq(z) = ∫_{-∞}^{z} dz' ∫∫ Δρ(x, y, z') dx dy

is integrated electron flux across the plane at z; positive Δq means
electrons accumulated on the negative-z side. The bond axis (z) is taken
as the user-specified axis, defaulting to the line joining the two
fragment centers of mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class CDAResult:
    axis_values: np.ndarray  # (Nz,) along bond axis, Bohr
    delta_rho_planar: np.ndarray  # (Nz,) integrated Δρ in slabs
    cd_function: np.ndarray  # (Nz,) cumulative ∫ Δρ
    isodensity_boundary: tuple[float, float] | None = None  # z values bracketing the bond
    net_transfer: float | None = None  # cd at the isodensity boundary
    grid: dict | None = None


def charge_displacement(
    mf_full,
    mf_A,
    mf_B,
    *,
    axis: np.ndarray | None = None,
    n_along: int = 400,
    n_perp: int = 80,
    extent: float = 8.0,  # +/- extent along axis from midpoint, Bohr
    perp_extent: float = 6.0,  # half-width transverse to axis, Bohr
    fragment_A_atoms: Sequence[int] | None = None,
    fragment_B_atoms: Sequence[int] | None = None,
) -> CDAResult:
    """Compute the charge displacement curve for fragments A and B.

    Args:
        mf_full: converged PySCF mean-field for the full A∪B system.
        mf_A: converged PySCF mean-field for fragment A with B as ghosts.
        mf_B: converged PySCF mean-field for fragment B with A as ghosts.
        axis: bond axis as a 3-vector. If None, uses the vector between
            centers of mass of fragment_A_atoms and fragment_B_atoms.
        n_along, n_perp: grid resolution along and perpendicular to axis.
        extent, perp_extent: half-extents of the integration box (Bohr).
    """
    from pyscf import gto

    mol = mf_full.mol
    rdm_full = _spinsum(mf_full.make_rdm1())
    rdm_A = _spinsum(mf_A.make_rdm1())
    rdm_B = _spinsum(mf_B.make_rdm1())

    # Build axis frame
    coords_full = mol.atom_coords()  # Bohr
    if axis is None:
        if fragment_A_atoms is None or fragment_B_atoms is None:
            raise ValueError("Provide either `axis` or both fragment_A_atoms / fragment_B_atoms.")
        ca = coords_full[list(fragment_A_atoms)].mean(axis=0)
        cb = coords_full[list(fragment_B_atoms)].mean(axis=0)
        axis_vec = cb - ca
        midpoint = 0.5 * (ca + cb)
    else:
        axis_vec = np.asarray(axis, dtype=float)
        midpoint = coords_full.mean(axis=0)

    if np.linalg.norm(axis_vec) < 1e-8:
        raise ValueError("Bond axis is zero-length; pass a non-degenerate axis.")
    e_z = axis_vec / np.linalg.norm(axis_vec)
    e_x, e_y = _orthonormal_frame(e_z)

    z_vals = np.linspace(-extent, extent, n_along)
    p_vals = np.linspace(-perp_extent, perp_extent, n_perp)
    dx = p_vals[1] - p_vals[0]
    dy = dx
    dz = z_vals[1] - z_vals[0]

    # Build grid points for slabs and evaluate three densities
    delta_planar = np.zeros(n_along)
    for k, z in enumerate(z_vals):
        xx, yy = np.meshgrid(p_vals, p_vals, indexing="ij")
        pts = (
            midpoint[None, :]
            + xx.reshape(-1, 1) * e_x[None, :]
            + yy.reshape(-1, 1) * e_y[None, :]
            + z * e_z[None, :]
        )  # (n_perp^2, 3)
        rho_full = _eval_rho(mol, rdm_full, pts)
        rho_A = _eval_rho(mf_A.mol, rdm_A, pts)
        rho_B = _eval_rho(mf_B.mol, rdm_B, pts)
        delta = rho_full - rho_A - rho_B
        delta_planar[k] = delta.sum() * dx * dy

    cd = np.cumsum(delta_planar) * dz

    # Locate "isodensity boundary": the plane between the two fragments where
    # the total density of either fragment falls to ~10% of its peak. Used as
    # the reference plane for net transfer.
    boundary = _find_isodensity_boundary(z_vals, cd)
    net = float(cd[boundary[2]]) if boundary is not None else None

    return CDAResult(
        axis_values=z_vals,
        delta_rho_planar=delta_planar,
        cd_function=cd,
        isodensity_boundary=(float(z_vals[boundary[0]]), float(z_vals[boundary[1]])) if boundary else None,
        net_transfer=net,
        grid={"e_z": e_z, "e_x": e_x, "e_y": e_y, "midpoint": midpoint, "dz": dz, "dxdy": dx * dy},
    )


def _spinsum(rdm):
    if isinstance(rdm, tuple):
        return rdm[0] + rdm[1]
    return rdm


def _eval_rho(mol, rdm, points: np.ndarray) -> np.ndarray:
    from pyscf import dft

    ao = dft.numint.eval_ao(mol, points, deriv=0)
    return dft.numint.eval_rho(mol, ao, rdm, xctype="LDA")


def _orthonormal_frame(e_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, e_z)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e_x = helper - np.dot(helper, e_z) * e_z
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(e_z, e_x)
    return e_x, e_y


def _find_isodensity_boundary(z_vals: np.ndarray, cd: np.ndarray):
    """Heuristic: bracket where the CD function plateaus between the fragments.

    Returns (left_idx, right_idx, midpoint_idx) or None if not found.
    """
    grad = np.gradient(cd, z_vals)
    centre = len(z_vals) // 2
    # widen until |grad| drops below 5% of its peak, on both sides of centre
    peak = float(np.abs(grad).max())
    if peak == 0:
        return None
    threshold = 0.05 * peak
    left = centre
    while left > 0 and abs(grad[left]) > threshold:
        left -= 1
    right = centre
    while right < len(z_vals) - 1 and abs(grad[right]) > threshold:
        right += 1
    mid = (left + right) // 2
    return left, right, mid
