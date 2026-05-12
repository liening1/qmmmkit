"""Natural orbital analysis (registered as the ``natural_orbitals`` plugin).

Three flavours are supported:

1. **Canonical natural orbitals** from a correlated 1-RDM (MP2/CCSD/CASSCF).
   Diagonalises the spin-summed 1-RDM in the AO basis to give NOs and
   occupation numbers.
2. **UNO (UHF natural orbitals)**: NOs from an unrestricted SCF, useful for
   detecting biradical character via fractional occupations.
3. **Spin natural orbitals**: diagonalise the spin-density matrix to identify
   which orbitals carry unpaired spin. Useful for radicals in QM/MM.

Output is a ``NaturalOrbitalResult`` and an optional Molden file that
Multiwfn / Avogadro / Jmol can read for visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from .._registry import register_analysis
from ..schemas.result import AnalysisResult
from ._base import BaseAnalysis


@dataclass
class NaturalOrbitalResult:
    occupations: np.ndarray  # (nmo,)
    coefficients: np.ndarray  # (nao, nmo) in AO basis
    method: str
    flavour: Literal["canonical", "uno", "spin"]
    effective_unpaired: float | None = None  # n_eff = sum n_i (2 - n_i) for canonical/UNO
    raw_rdm: np.ndarray | None = None
    mol: Any | None = None  # PySCF Mole, retained for molden export


def natural_orbitals(
    mf_or_calc,
    *,
    flavour: Literal["canonical", "uno", "spin"] = "canonical",
    correlated_method: str | None = None,
) -> NaturalOrbitalResult:
    """Compute natural orbitals from a (PySCF) mean-field or QMMMCalculator.

    Args:
        mf_or_calc: Either a PySCF mean-field object (``mf``) whose ``make_rdm1`` /
            ``make_rdm1s`` is callable, or a QMMMCalculator we should drive forward.
        flavour: Which type of NOs to compute.
        correlated_method: If set (e.g. ``"MP2"``, ``"CCSD"``), build the 1-RDM at
            this correlated level rather than from the SCF density. Ignored for
            ``flavour="uno"`` and ``flavour="spin"``.
    """
    mf = _resolve_mf(mf_or_calc)
    mol = mf.mol
    s_ao = mol.intor("int1e_ovlp")

    if flavour == "uno":
        rdm_a, rdm_b = mf.make_rdm1()  # UHF returns a tuple
        rdm = rdm_a + rdm_b
        method = "UHF"
    elif flavour == "spin":
        rdm_a, rdm_b = mf.make_rdm1()
        rdm = rdm_a - rdm_b
        method = "spin-NO"
    elif correlated_method:
        rdm = _correlated_rdm(mf, correlated_method)
        method = correlated_method
    else:
        rdm = mf.make_rdm1()
        if isinstance(rdm, tuple):
            rdm = rdm[0] + rdm[1]
        method = "SCF"

    occ, coeff = _diagonalise_in_ao(rdm, s_ao)
    n_eff = None
    if flavour in ("canonical", "uno"):
        # Yamaguchi's "effective number of unpaired electrons"
        n_eff = float(np.sum(occ * (2.0 - occ)))

    return NaturalOrbitalResult(
        occupations=occ,
        coefficients=coeff,
        method=method,
        flavour=flavour,
        effective_unpaired=n_eff,
        raw_rdm=rdm,
        mol=mol,
    )


def write_molden(result: NaturalOrbitalResult, path: str | Path) -> Path:
    """Write the natural orbitals to a Molden file."""
    from pyscf.tools import molden

    if result.mol is None:
        raise ValueError("Result is missing the PySCF Mole; cannot write Molden.")
    path = Path(path)
    energies = np.zeros_like(result.occupations)  # NOs have no canonical orbital energies
    molden.from_mo(result.mol, str(path), result.coefficients,
                   ene=energies, occ=result.occupations)
    return path


def _resolve_mf(mf_or_calc):
    """Accept either a PySCF mf, our QMMMCalculator, or an ASH PySCFTheory.

    For a QMMMCalculator we drive a Singlepoint to ensure the SCF has
    converged, then pull ``mf`` off the underlying ash PySCFTheory.
    """
    # Direct PySCF mean-field
    if hasattr(mf_or_calc, "make_rdm1") and hasattr(mf_or_calc, "mol"):
        return mf_or_calc
    # ash.PySCFTheory exposes ``mf`` directly after Singlepoint runs
    if hasattr(mf_or_calc, "mf") and getattr(mf_or_calc, "mf") is not None:
        return mf_or_calc.mf
    # QMMMCalculator: ensure SCF has run, then dig into the wrapped theory
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
        qm_theory = mf_or_calc._qm_theory
        mf = getattr(qm_theory, "mf", None)
        if mf is None:
            raise RuntimeError(
                "PySCFTheory.mf is None after Singlepoint; SCF likely did not converge."
            )
        return mf
    raise TypeError(
        f"Cannot resolve a PySCF mean-field from {type(mf_or_calc).__name__}."
    )


def _correlated_rdm(mf, method: str) -> np.ndarray:
    """Build the AO-basis 1-RDM at the requested correlated level."""
    method = method.upper()
    if method == "MP2":
        from pyscf import mp

        mp2 = mp.MP2(mf).run()
        rdm_mo = mp2.make_rdm1()
        c = mf.mo_coeff
        return c @ rdm_mo @ c.T
    if method == "CCSD":
        from pyscf import cc

        ccobj = cc.CCSD(mf).run()
        ccobj.solve_lambda()
        rdm_mo = ccobj.make_rdm1()
        c = mf.mo_coeff
        return c @ rdm_mo @ c.T
    if method == "CASSCF":
        # Caller is expected to have run mf as a CASSCF object already
        if not hasattr(mf, "make_rdm1"):
            raise ValueError("For CASSCF flavour pass the converged CASSCF object as mf.")
        return mf.make_rdm1()
    raise ValueError(f"Unsupported correlated method: {method}")


def _diagonalise_in_ao(rdm: np.ndarray, s_ao: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalise a 1-RDM in the AO basis using the symmetric S^(1/2) approach."""
    # P S transforms RDM into a non-symmetric matrix whose eigenvalues are the NO occupations.
    # The standard fix is to symmetrise via S^(1/2): solve  S^(1/2) P S^(1/2) C' = C' n
    s_evals, s_evecs = np.linalg.eigh(s_ao)
    s_half = s_evecs @ np.diag(np.sqrt(np.maximum(s_evals, 0.0))) @ s_evecs.T
    s_inv_half = s_evecs @ np.diag(1.0 / np.sqrt(np.maximum(s_evals, 1e-12))) @ s_evecs.T
    m = s_half @ rdm @ s_half
    occ, c_orth = np.linalg.eigh(m)
    # sort in descending occupation
    order = np.argsort(occ)[::-1]
    occ = occ[order]
    c_orth = c_orth[:, order]
    coeff_ao = s_inv_half @ c_orth
    return occ, coeff_ao


# ---------------------------------------------------------------------------
# Registered analysis plugin
# ---------------------------------------------------------------------------
class NaturalOrbitalsOptions(BaseModel):
    flavour: Literal["canonical", "uno", "spin"] = "canonical"
    correlated_method: Literal["MP2", "CCSD", "CASSCF", None] | None = None
    write_molden: bool = True


@register_analysis("natural_orbitals")
class NaturalOrbitalsAnalysis(BaseAnalysis):
    name = "natural_orbitals"
    options_schema = NaturalOrbitalsOptions

    def run(self, calc, *, workdir: Path, log) -> AnalysisResult:
        opts: NaturalOrbitalsOptions = self.options  # type: ignore[assignment]
        log.info("nat_orb.start", flavour=opts.flavour, correlated=opts.correlated_method)
        result = natural_orbitals(
            calc, flavour=opts.flavour, correlated_method=opts.correlated_method,
        )
        artifacts: dict[str, str] = {}
        if opts.write_molden:
            target = Path(workdir) / f"natural_orbitals_{result.flavour}.molden"
            try:
                write_molden(result, target)
                artifacts["molden"] = str(target)
                log.info("nat_orb.wrote", path=str(target))
            except Exception as e:  # noqa: BLE001
                log.warning("nat_orb.molden_failed", error=str(e))

        return AnalysisResult(
            kind=self.name,
            method=result.method,
            summary={
                "flavour": result.flavour,
                "occupations": result.occupations.tolist(),
                "effective_unpaired": result.effective_unpaired,
                "n_orbitals": int(result.coefficients.shape[1]),
            },
            artifacts=artifacts,
        )
