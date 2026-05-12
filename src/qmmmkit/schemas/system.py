"""System (topology + QM region) schema."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


TopologyKind = Literal["pdb", "amber", "charmm", "gromacs", "xyz"]


class SystemSpec(BaseModel):
    """Topology + QM region. Coordinate units are Angstrom throughout."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: TopologyKind = Field(
        default="pdb",
        description="Topology source. 'pdb' uses an OpenMM forcefield XML; 'amber' uses prmtop+inpcrd.",
    )
    structure: str = Field(..., description="Path to PDB / inpcrd / gro / xyz (depending on kind).")
    topology: str | None = Field(
        default=None,
        description="Path to prmtop / psf / top when kind != 'pdb' / 'xyz'.",
    )
    forcefield: list[str] = Field(
        default_factory=lambda: ["amber14-all.xml", "amber14/tip3p.xml"],
        description="OpenMM forcefield XMLs (used only when kind == 'pdb').",
    )
    qm_atoms: list[Annotated[int, Field(ge=0)]] = Field(
        ..., min_length=1, description="0-indexed atom indices in the QM region."
    )
    active_atoms: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        description="Atoms allowed to move during optimisation. Default: QM + 6 A shell.",
    )
    active_shell: float | None = Field(
        default=6.0, ge=0, description="Angstrom shell around QM region used to fill active_atoms when None."
    )
    charge: int = Field(default=0, description="Net charge of the QM region.")
    mult: int = Field(default=1, ge=1, description="Spin multiplicity (2S+1) of the QM region.")
    name: str | None = None
    periodic: bool = Field(default=False, description="Use PBC in the MM forcefield.")

    @field_validator("qm_atoms", "active_atoms")
    @classmethod
    def _unique_sorted(cls, v):
        if v is None:
            return v
        if len(v) != len(set(v)):
            raise ValueError("Atom indices must be unique.")
        return sorted(v)

    def resolve_paths(self, base: Path) -> None:
        """Resolve relative ``structure`` / ``topology`` paths against ``base``."""
        for attr in ("structure", "topology"):
            v = getattr(self, attr, None)
            if v and not Path(v).is_absolute():
                setattr(self, attr, str((base / v).resolve()))

    def input_files(self) -> list[Path]:
        out: list[Path] = []
        for attr in ("structure", "topology"):
            v = getattr(self, attr, None)
            if v and Path(v).exists():
                out.append(Path(v))
        return out
