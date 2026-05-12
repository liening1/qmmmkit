"""QM region configuration schema (PySCF, mapped via ASH PySCFTheory)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCFType = Literal["RHF", "UHF", "ROHF", "RKS", "UKS"]
WaveFunctionMethod = Literal[
    "HF", "DFT", "MP2", "CCSD", "CCSD(T)", "CASSCF", "CASCI", "TDDFT",
]


class QMSpec(BaseModel):
    """QM region settings.

    The combination of ``scf_type`` + ``functional`` controls the SCF level.
    ``method`` selects the post-SCF treatment (HF and DFT have method=HF/DFT).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    method: WaveFunctionMethod = "DFT"
    functional: str | None = Field(
        default="B3LYP",
        description="DFT functional name (PySCF naming). Required when method == 'DFT'.",
    )
    basis: str = Field(default="def2-SVP", description="AO basis. PySCF basis-set name or path.")
    scf_type: SCFType = Field(
        default="RKS",
        description="Wavefunction class. RKS/UKS for DFT, RHF/UHF/ROHF for HF and post-HF.",
    )
    aux_basis: str | None = Field(
        default=None, description="Density-fitting auxiliary basis (RI-J, RI-MP2, ...). Optional."
    )
    density_fitting: bool = Field(default=False, description="Enable density fitting for SCF.")
    dispersion: Literal["d3", "d3bj", "d4", None] | None = Field(default="d3bj")

    nprocs: int = Field(default=1, ge=1)
    memory: int = Field(default=4000, ge=256, description="MB of memory passed to PySCF.")

    # Convergence + numerical
    scf_maxiter: int = Field(default=100, ge=10)
    conv_tol: float = Field(default=1e-8, gt=0, description="SCF density convergence threshold.")
    grid_level: int = Field(default=5, ge=0, le=9)

    # CASSCF / CAS
    cas_active_space: tuple[int, int] | None = Field(
        default=None, description="(n_electrons, n_orbitals) for CAS methods."
    )
    cas_state_average: int = Field(default=1, ge=1)

    # CC tweaks
    frozen_core: bool = Field(default=True)

    # Free-form pyscf kwargs we forward verbatim. Use sparingly.
    pyscf_kwargs: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> "QMSpec":
        if self.method == "DFT" and not self.functional:
            raise ValueError("DFT requested but no functional supplied.")
        if self.method != "DFT" and self.scf_type in ("RKS", "UKS"):
            raise ValueError(
                f"scf_type={self.scf_type} requires method='DFT'; got method={self.method!r}."
            )
        if self.method in ("CASSCF", "CASCI") and self.cas_active_space is None:
            raise ValueError(f"{self.method} requires cas_active_space=(nelec, norb).")
        return self
