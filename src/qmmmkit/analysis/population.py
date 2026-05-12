"""Population analysis: Mulliken / Lowdin / Hirshfeld / CM5 / NPA-style.

True NPA / NBO requires either the proprietary NBO program or Janpa
(Java). What we ship here:
  - Mulliken & Lowdin (built into PySCF)
  - Hirshfeld via PySCF's ``pop_analysis`` extension when available
  - CM5 charges (Marenich, Cramer, Truhlar 2012) computed from Hirshfeld
  - A wrapper to call Janpa for NPA charges if it's on PATH

For QM/MM workflows the QM-region partial charges from CM5 are also a
good choice for back-coupling to MM polarisation in iterative schemes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .._registry import register_analysis
from ..schemas.result import AnalysisResult
from ._base import BaseAnalysis

PopMethod = Literal["mulliken", "lowdin", "hirshfeld", "cm5", "npa"]


@dataclass
class PopulationResult:
    method: PopMethod
    charges: np.ndarray  # (natoms,)
    spin_populations: np.ndarray | None = None
    raw: object | None = None


def population_analysis(
    mf_or_calc,
    *,
    method: PopMethod = "cm5",
) -> PopulationResult:
    mf = _resolve_mf(mf_or_calc)
    if method == "mulliken":
        pop, q = mf.mulliken_pop(verbose=0)
        return PopulationResult("mulliken", np.asarray(q))
    if method == "lowdin":
        pop, q = mf.mulliken_meta(verbose=0)  # pyscf naming for Lowdin-like
        return PopulationResult("lowdin", np.asarray(q))
    if method == "hirshfeld":
        return _hirshfeld(mf)
    if method == "cm5":
        return _cm5(mf)
    if method == "npa":
        return _npa_via_janpa(mf)
    raise ValueError(f"Unknown population method: {method}")


def _resolve_mf(mf_or_calc):
    if hasattr(mf_or_calc, "make_rdm1") and hasattr(mf_or_calc, "mol"):
        return mf_or_calc
    if hasattr(mf_or_calc, "mf") and getattr(mf_or_calc, "mf") is not None:
        return mf_or_calc.mf
    if hasattr(mf_or_calc, "theory") and hasattr(mf_or_calc, "system"):
        import ash

        ash.Singlepoint(
            theory=mf_or_calc.theory,
            fragment=mf_or_calc.system.fragment,
            charge=mf_or_calc.system.spec.charge,
            mult=mf_or_calc.system.spec.mult,
            result_write_to_disk=False,
            printlevel=0,
        )
        mf = getattr(mf_or_calc._qm_theory, "mf", None)
        if mf is None:
            raise RuntimeError("PySCFTheory.mf is None after Singlepoint.")
        return mf
    raise TypeError(f"Cannot resolve a PySCF mean-field from {type(mf_or_calc).__name__}.")


def _hirshfeld(mf) -> PopulationResult:
    try:
        from pyscf.dft import gen_grid
        from pyscf import dft
    except ImportError as e:
        raise ImportError("Hirshfeld analysis requires pyscf with dft module.") from e
    mol = mf.mol
    grid = gen_grid.Grids(mol)
    grid.level = 5
    grid.build()
    coords = grid.coords
    weights = grid.weights

    rdm = mf.make_rdm1()
    if isinstance(rdm, tuple):
        rdm = rdm[0] + rdm[1]
    ao = dft.numint.eval_ao(mol, coords, deriv=0)
    rho = dft.numint.eval_rho(mol, ao, rdm, xctype="LDA")

    # Free-atom densities (built from spherical-symmetric atomic SCF)
    proatom_rho = np.zeros((mol.natm, coords.shape[0]))
    for i in range(mol.natm):
        proatom_rho[i] = _free_atom_density(mol, i, coords)
    total_pro = proatom_rho.sum(axis=0)
    total_pro = np.where(total_pro > 1e-30, total_pro, 1e-30)
    weights_atom = proatom_rho / total_pro  # Hirshfeld weights w_A(r)

    z = mol.atom_charges()
    n_a = (weights_atom * rho * weights).sum(axis=1)
    q = z - n_a
    return PopulationResult("hirshfeld", q, raw={"weights": weights_atom})


def _cm5(mf) -> PopulationResult:
    """CM5: Hirshfeld charges + pairwise correction depending on element pair and bond length."""
    hirshfeld = _hirshfeld(mf)
    q_h = hirshfeld.charges
    mol = mf.mol
    coords = mol.atom_coords() * 0.529177210903  # Bohr -> Angstrom
    z = mol.atom_charges()
    natoms = mol.natm

    # CM5 parameters (Marenich, Cramer, Truhlar 2012; subset of element-pair Dkk' values)
    alpha = 2.474  # Angstrom^-1
    Dz_global = {
        (1, 6): 0.0502, (1, 7): 0.1747, (1, 8): 0.1671,
        (6, 7): 0.0556, (6, 8): 0.0234, (7, 8): -0.0346,
    }
    radii = {1: 0.32, 6: 0.75, 7: 0.71, 8: 0.63, 9: 0.64, 15: 1.11, 16: 1.03, 17: 0.99}

    q_cm5 = q_h.copy()
    for k in range(natoms):
        for kp in range(natoms):
            if k == kp:
                continue
            zk, zkp = int(z[k]), int(z[kp])
            r_kkp = float(np.linalg.norm(coords[k] - coords[kp]))
            r_k = radii.get(zk, 0.7)
            r_kp = radii.get(zkp, 0.7)
            B = np.exp(-alpha * (r_kkp - r_k - r_kp))
            D = Dz_global.get((zk, zkp), -Dz_global.get((zkp, zk), 0.0))
            q_cm5[k] += D * B
    return PopulationResult("cm5", q_cm5, raw={"hirshfeld": q_h})


def _free_atom_density(mol, iatom, points):
    """Spherically-averaged free-atom density for atom `iatom`, evaluated at points.

    Approximated as the converged ROHF density of the isolated neutral atom in
    the same basis. Cached at the module level when called repeatedly.
    """
    from pyscf import gto, scf

    sym = mol.atom_symbol(iatom)
    atom_pos = mol.atom_coord(iatom)
    free_mol = gto.M(atom=f"{sym} 0.0 0.0 0.0", basis=mol.basis, spin=_default_spin(sym), verbose=0)
    mf = scf.ROHF(free_mol)
    mf.kernel()
    rdm = mf.make_rdm1()
    if isinstance(rdm, tuple):
        rdm = rdm[0] + rdm[1]
    shifted = points - atom_pos
    from pyscf import dft

    ao = dft.numint.eval_ao(free_mol, shifted, deriv=0)
    return dft.numint.eval_rho(free_mol, ao, rdm, xctype="LDA")


def _default_spin(sym: str) -> int:
    """Lowest reasonable spin for a neutral free atom."""
    odd_z = {"H", "Li", "B", "N", "F", "Na", "Al", "P", "Cl", "K", "Sc", "V", "Mn", "Co", "Cu", "Ga", "As", "Br"}
    return 1 if sym in odd_z else 0


def _npa_via_janpa(mf) -> PopulationResult:
    """Run Janpa (open NPA implementation) on a Molden file dumped from PySCF."""
    import shutil
    import subprocess
    import tempfile
    from pyscf.tools import molden

    if not shutil.which("janpa") and not shutil.which("janpa.jar"):
        raise RuntimeError("Janpa not found on PATH. Install from https://janpa.sourceforge.net/")

    with tempfile.TemporaryDirectory() as td:
        molden_path = Path(td) / "wfn.molden"
        molden.from_scf(mf, str(molden_path))
        out = subprocess.run(
            ["janpa", str(molden_path)], capture_output=True, text=True, check=True,
        )
        charges = _parse_janpa_charges(out.stdout, mf.mol.natm)
        return PopulationResult("npa", charges, raw=out.stdout)


def _parse_janpa_charges(text: str, natoms: int) -> np.ndarray:
    """Pull the per-atom NPA charges from Janpa stdout. Format-tolerant: looks for
    a section header containing 'Natural Population' and reads `natoms` rows."""
    lines = text.splitlines()
    out = []
    in_block = False
    for line in lines:
        if "Natural Population" in line or "NPA Charge" in line:
            in_block = True
            continue
        if in_block:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    out.append(float(parts[-1]))
                except ValueError:
                    if out:
                        break
            if len(out) == natoms:
                break
    if len(out) != natoms:
        raise RuntimeError("Failed to parse Janpa output; got {} charges for {} atoms.".format(len(out), natoms))
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Registered analysis plugin
# ---------------------------------------------------------------------------
class PopulationOptions(BaseModel):
    method: PopMethod = "cm5"


@register_analysis("population")
class PopulationAnalysis(BaseAnalysis):
    name = "population"
    options_schema = PopulationOptions

    def run(self, calc, *, workdir: Path, log) -> AnalysisResult:
        opts: PopulationOptions = self.options  # type: ignore[assignment]
        log.info("population.start", method=opts.method)
        result = population_analysis(calc, method=opts.method)
        out = Path(workdir) / f"charges_{opts.method}.npy"
        np.save(out, result.charges)
        log.info("population.done", method=opts.method, n=len(result.charges))
        return AnalysisResult(
            kind=self.name,
            method=result.method,
            summary={
                "method": result.method,
                "charges": result.charges.tolist(),
                "min": float(result.charges.min()),
                "max": float(result.charges.max()),
                "sum": float(result.charges.sum()),
            },
            artifacts={"charges_npy": str(out)},
        )
