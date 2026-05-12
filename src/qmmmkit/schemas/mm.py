"""MM configuration schema (OpenMM, mapped via ASH OpenMMTheory).

Note: ASH's OpenMMTheory takes ``periodic_nonbonded_cutoff`` in *Angstrom*,
not nanometers (despite OpenMM's native nm convention). We follow ASH here
to avoid a unit-conversion bug at the wrapper boundary.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Platform = Literal["Reference", "CPU", "OpenCL", "CUDA"]
NonbondedPBC = Literal["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "Ewald", "PME", "LJPME"]
Constraints = Literal[None, "HBonds", "AllBonds", "HAngles"]


class MMSpec(BaseModel):
    """OpenMM-side parameters."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    platform: Platform = "CPU"
    numcores: int = Field(default=1, ge=1)

    # Nonbonded settings - we keep them split so non-periodic and periodic
    # systems get the right defaults without silent overrides at runtime.
    nonbonded_method_pbc: NonbondedPBC = "PME"
    periodic_nonbonded_cutoff: float = Field(
        default=12.0, gt=0, description="Cutoff in Angstrom (ASH convention)."
    )
    nonbonded_method_no_pbc: Literal["NoCutoff", "CutoffNonPeriodic"] = "NoCutoff"
    nonbonded_cutoff_no_pbc: float = Field(default=20.0, gt=0)

    # Constraints + waters
    constraints: Constraints = None
    rigid_water: bool = False
    autoconstraints: Literal[None, "HBonds", "AllBonds", "HAngles"] = "HBonds"
    hydrogen_mass: float = Field(default=1.5, gt=0, description="Effective H mass (HMR).")

    # Free-form OpenMM kwargs we forward verbatim.
    openmm_kwargs: dict = Field(default_factory=dict)
