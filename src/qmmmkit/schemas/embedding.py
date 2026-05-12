"""QM/MM embedding configuration.

ASH uses string codes ``elstat`` (electrostatic) and ``mech`` (mechanical)
for the ``embedding`` kwarg. We surface the more readable names but
serialise to the wrapper-friendly form on the way out.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EmbeddingScheme = Literal["electrostatic", "mechanical"]
LinkAtomMethod = Literal["simple", "ratio"]
ForceProjection = Literal["adv", "lever", "chain", "none"]
ChargeBoundary = Literal["shift", "rcd", "z1", "none"]


class EmbeddingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    scheme: EmbeddingScheme = "electrostatic"
    use_link_atoms: bool = True
    link_atom_method: LinkAtomMethod = "simple"
    link_atom_type: str = Field(default="H", description="Element symbol used to cap broken bonds.")
    link_atom_ratio: float = Field(default=0.723, gt=0, description="Used when link_atom_method='ratio'.")
    link_atom_force_proj: ForceProjection = "adv"
    charge_boundary_method: ChargeBoundary = Field(
        default="shift",
        description="How to redistribute MM partial charges near the boundary.",
    )
    truncate_pc_radius: float | None = Field(
        default=None,
        gt=0,
        description="Optional cutoff (Angstrom) for distant point charges polarising the QM region.",
    )
    dipole_correction: bool = True

    def to_ash_kwargs(self) -> dict:
        """Translate to the kwargs ash.QMMMTheory actually accepts."""
        out = {
            "embedding": "elstat" if self.scheme == "electrostatic" else "mech",
            "linkatom_method": self.link_atom_method,
            "linkatom_type": self.link_atom_type,
            "linkatom_ratio": self.link_atom_ratio,
            "linkatom_forceproj_method": self.link_atom_force_proj,
            "chargeboundary_method": self.charge_boundary_method,
            "dipole_correction": self.dipole_correction,
        }
        if self.truncate_pc_radius is not None:
            out["TruncatedPC"] = True
            out["TruncPCRadius"] = self.truncate_pc_radius
        return out
