"""Energy Decomposition Analysis (EDA) for QM/MM.

State-of-the-art EDA splits the total QM/MM energy into a small set of
physically meaningful components, so the user can answer questions like
"how much does the protein environment stabilise the substrate?"

This first iteration emits a defensible 3-term decomposition that does
not require any QM density partitioning beyond what's already converged:

    E_total      = E_QM_polarised + E_MM_internal               (electrostatic embedding identity)
    ΔE_pol       = E_QM_polarised - E_QM_gas                    (cost of polarising the QM density)
    ΔE_es        = E_total - E_QM_gas - E_MM_internal           (classical electrostatic + Pauli)

with::

    E_QM_polarised  - QM-region energy with the MM point-charge field present
                      (= ``ash.QMMMTheory.QMenergy`` after Singlepoint).
    E_MM_internal   - MM-region classical energy excluding QM-MM electrostatics
                      (= ``ash.QMMMTheory.MMenergy`` in ASH's bookkeeping).
    E_QM_gas        - the same QM region in vacuum (no MM charges), at the
                      same level of theory and basis.

A full Morokuma-style decomposition (electrostatic / exchange / pol /
charge-transfer) needs an unperturbed-density matrix element evaluation;
that's left for a follow-up.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .._registry import register_task
from ..calculator import QMMMCalculator
from ..schemas.result import TaskResult
from ._base import BaseTask


class EDAOptions(BaseModel):
    write_summary: bool = Field(default=True, description="Write eda_summary.txt next to result.json.")


_HARTREE_TO_KCAL = 627.5094740631


@register_task("eda")
class EDATask(BaseTask):
    name = "eda"
    options_schema = EDAOptions

    def run(self, calc: QMMMCalculator, *, workdir: Path, log) -> TaskResult:
        import ash

        opts: EDAOptions = self.options  # type: ignore[assignment]

        log.info("eda.start", qm_atoms=len(calc.system.qm_atoms))

        # ----- Full QM/MM single point ---------------------------------
        result_full = ash.Singlepoint(
            theory=calc.theory,
            fragment=calc.system.fragment,
            charge=calc.system.spec.charge,
            mult=calc.system.spec.mult,
            result_write_to_disk=False,
            printlevel=0,
        )
        e_total = float(getattr(result_full, "energy"))
        e_qm_polarised = _maybe_float(
            getattr(result_full, "qm_energy", None) or getattr(calc.theory, "QMenergy", None)
        )
        e_mm_internal = _maybe_float(
            getattr(result_full, "mm_energy", None) or getattr(calc.theory, "MMenergy", None)
        )
        log.info("eda.full", e_total=e_total, e_qm_polarised=e_qm_polarised, e_mm_internal=e_mm_internal)

        # ----- Gas-phase QM via fragment_scf ---------------------------
        mf_gas = calc.fragment_scf(
            real_atoms=list(calc.system.qm_atoms),
            ghost_atoms=[],
            embed_in_mm=False,
        )
        e_qm_gas = float(getattr(mf_gas, "e_tot"))
        log.info("eda.gas", e_qm_gas=e_qm_gas)

        # ----- Decomposition -------------------------------------------
        delta_pol_ha = (e_qm_polarised - e_qm_gas) if e_qm_polarised is not None else None
        delta_es_ha = (
            (e_total - e_qm_gas - e_mm_internal)
            if (e_mm_internal is not None) else None
        )

        components = {
            "E_total_hartree": e_total,
            "E_QM_polarised_hartree": e_qm_polarised,
            "E_QM_gas_hartree": e_qm_gas,
            "E_MM_internal_hartree": e_mm_internal,
            "delta_E_polarisation_hartree": delta_pol_ha,
            "delta_E_polarisation_kcalmol": (
                delta_pol_ha * _HARTREE_TO_KCAL if delta_pol_ha is not None else None
            ),
            "delta_E_environment_hartree": delta_es_ha,
            "delta_E_environment_kcalmol": (
                delta_es_ha * _HARTREE_TO_KCAL if delta_es_ha is not None else None
            ),
        }

        if opts.write_summary:
            self._write_summary(workdir, components)

        log.info("eda.done", **{k: v for k, v in components.items() if v is not None})
        return TaskResult(
            name=self.name,
            converged=True,
            energy=e_total,
            qm_energy=e_qm_polarised,
            mm_energy=e_mm_internal,
            extra=components,
        )

    @staticmethod
    def _write_summary(workdir: Path, components: dict) -> Path:
        out = Path(workdir) / "eda_summary.txt"
        with out.open("w", encoding="utf-8") as f:
            f.write("qmmmkit QM/MM energy decomposition\n")
            f.write("=" * 50 + "\n\n")
            for label, val in components.items():
                f.write(f"  {label:<35s} : {val if val is not None else 'n/a'}\n")
            f.write("\nDelta_E_polarisation = E_QM_polarised - E_QM_gas (cost of polarising the QM density)\n")
            f.write("Delta_E_environment  = E_total - E_QM_gas - E_MM_internal "
                    "(electrostatic + Pauli interaction with environment)\n")
        return out


def _maybe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
