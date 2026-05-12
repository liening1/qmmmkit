"""Runtime QM/MM system: an ash.Fragment plus the validated SystemSpec.

``QMMMSystem`` is the runtime object the rest of the package consumes.
It is built from a ``SystemSpec`` (pydantic-validated) and an
``ash.Fragment`` (lazily constructed from the spec).

Selection helpers (residue / shell expansion) operate on the loaded
fragment and are pure-numpy/pure-Python so they unit-test cleanly
without ASH installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .schemas import SystemSpec


@dataclass
class QMMMSystem:
    """A runtime QM/MM system: spec + ash.Fragment."""

    spec: SystemSpec
    fragment: Any  # ash.Fragment

    # ------------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: SystemSpec) -> "QMMMSystem":
        """Construct an ``ash.Fragment`` from a SystemSpec.

        For PDB systems we hand the path to ``Fragment(pdbfile=...)``.
        For Amber systems Fragment still loads coords from the PDB
        (the prmtop is consumed by OpenMMTheory later, not by Fragment).
        """
        try:
            import ash
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ASH is required at runtime. Install with `pip install qmmmkit[mm]` "
                "and `pip install git+https://github.com/RagnarB83/ash.git`."
            ) from e

        if spec.kind in ("pdb", "amber", "charmm"):
            frag = ash.Fragment(
                pdbfile=spec.structure,
                charge=spec.charge,
                mult=spec.mult,
            )
        elif spec.kind == "xyz":
            frag = ash.Fragment(xyzfile=spec.structure, charge=spec.charge, mult=spec.mult)
        elif spec.kind == "gromacs":
            frag = ash.Fragment(grofile=spec.structure, charge=spec.charge, mult=spec.mult)
        else:  # pragma: no cover - validated at the schema layer
            raise ValueError(f"Unsupported topology kind: {spec.kind}")

        sys_obj = cls(spec=spec, fragment=frag)
        if spec.active_atoms is None and spec.active_shell is not None:
            sys_obj.spec.active_atoms = sys_obj.shell(spec.active_shell)
        return sys_obj

    # ------------------------------------------------------------
    @property
    def qm_atoms(self) -> list[int]:
        return list(self.spec.qm_atoms)

    @property
    def active_atoms(self) -> list[int] | None:
        return list(self.spec.active_atoms) if self.spec.active_atoms is not None else None

    @property
    def coords(self) -> np.ndarray:
        return np.asarray(getattr(self.fragment, "coords"))

    @coords.setter
    def coords(self, value: np.ndarray) -> None:
        self.fragment.coords = np.asarray(value)

    @property
    def n_atoms(self) -> int:
        return int(getattr(self.fragment, "numatoms", len(self.coords)))

    # ------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------
    def shell(self, radius: float, *, from_atoms: Iterable[int] | None = None) -> list[int]:
        """Atoms within ``radius`` Angstrom of ``from_atoms`` (defaults to QM region)."""
        seeds = list(from_atoms) if from_atoms is not None else self.qm_atoms
        return _atoms_within(self.coords, seeds, radius)

    def select_residues(self, resids: Iterable[int]) -> list[int]:
        rids = set(int(r) for r in resids)
        atom_resids = _residue_ids(self.fragment, self.n_atoms)
        return [i for i, r in enumerate(atom_resids) if r in rids]

    def expand_qm_by_residue(self) -> None:
        atom_resids = _residue_ids(self.fragment, self.n_atoms)
        included = {atom_resids[i] for i in self.qm_atoms}
        self.spec.qm_atoms = sorted(i for i, r in enumerate(atom_resids) if r in included)

    # ------------------------------------------------------------
    def summary(self) -> str:
        return (
            f"QMMMSystem '{self.spec.name or 'unnamed'}': {self.n_atoms} atoms total, "
            f"{len(self.qm_atoms)} QM, {len(self.active_atoms or [])} active, "
            f"charge={self.spec.charge}, mult={self.spec.mult}"
        )


# ---------------------------------------------------------------------------
def _residue_ids(fragment, natoms: int) -> list[int]:
    for attr in ("atom_residues", "residues", "resids"):
        v = getattr(fragment, attr, None)
        if v is not None and len(v):
            return [int(x) for x in v]
    return [0] * natoms


def _atoms_within(coords: np.ndarray, seeds: Sequence[int], radius: float) -> list[int]:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be (N,3).")
    if not len(seeds):
        return []
    seed_xyz = coords[list(seeds)]
    diff = coords[:, None, :] - seed_xyz[None, :, :]
    dmin = np.linalg.norm(diff, axis=-1).min(axis=1)
    return sorted(set(np.where(dmin <= radius)[0].tolist()) | set(seeds))


__all__ = ["QMMMSystem"]
