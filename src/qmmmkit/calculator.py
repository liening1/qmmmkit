"""High-level QM/MM calculator wrapping ash.QMMMTheory.

Translates our pydantic specs (QMSpec / MMSpec / EmbeddingSpec) into the
*real* ASH constructor kwargs. Discrepancies caught during the audit:

- ``QMMMTheory(embedding=...)`` accepts ``"elstat"`` / ``"mech"``, not
  ``"electrostatic"`` / ``"mechanical"``.
- ``linkatom_method`` is ``"simple"`` / ``"ratio"`` (an algorithm), not the
  capping atom symbol; the symbol goes into ``linkatom_type``.
- ``OpenMMTheory`` for a PDB+forcefield combo expects ``xmlfiles=[...]``
  and ``pdbfile=...`` (not ``forcefield=...``).
- Cutoffs in OpenMMTheory are in Angstrom (``periodic_nonbonded_cutoff``),
  not nanometers despite OpenMM's native unit.
- Charge / mult flow through ``Singlepoint``/``Optimizer`` calls, not
  through the QMMMTheory constructor (we still pass them as a hint).

Beyond the canonical ``build()`` path, the calculator also provides
``fragment_scf(real_atoms, ghost_atoms, ...)``: a direct PySCF SCF on a
subset of the QM region with the partner fragment treated as ghost
atoms (basis only, no nucleus) and the MM environment as point charges.
This is the building block for fragment-based analyses (charge
displacement, EDA, QTAIM-on-fragments).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import EmbeddingSpec, MMSpec, QMSpec, SystemSpec
from .system import QMMMSystem


@dataclass
class QMMMCalculator:
    """Bundle a system + the three specs and lazily build ash theories."""

    system: QMMMSystem
    qm: QMSpec = field(default_factory=QMSpec)
    mm: MMSpec = field(default_factory=MMSpec)
    embedding: EmbeddingSpec = field(default_factory=EmbeddingSpec)

    _qm_theory: Any | None = field(default=None, init=False, repr=False)
    _mm_theory: Any | None = field(default=None, init=False, repr=False)
    _theory: Any | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------
    @property
    def theory(self) -> Any:
        if self._theory is None:
            self.build()
        return self._theory

    def build(self) -> Any:
        try:
            import ash
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ASH is required at runtime. `pip install git+https://github.com/RagnarB83/ash.git`."
            ) from e

        self._qm_theory = self._build_qm_theory(ash)
        self._mm_theory = self._build_mm_theory(ash)
        self._theory = ash.QMMMTheory(
            qm_theory=self._qm_theory,
            mm_theory=self._mm_theory,
            fragment=self.system.fragment,
            qmatoms=list(self.system.qm_atoms),
            qm_charge=self.system.spec.charge,
            qm_mult=self.system.spec.mult,
            actatoms=self.system.active_atoms,
            **self.embedding.to_ash_kwargs(),
        )
        return self._theory

    # ------------------------------------------------------------
    def _build_qm_theory(self, ash):
        functional = self.qm.functional if self.qm.method == "DFT" else None
        kwargs: dict[str, Any] = {
            "scf_type": self.qm.scf_type,
            "basis": self.qm.basis,
            "functional": functional,
            "auxbasis": self.qm.aux_basis,
            "densityfit": self.qm.density_fitting,
            "dispersion": self.qm.dispersion,
            "numcores": self.qm.nprocs,
            "memory": self.qm.memory,
            "scf_maxiter": self.qm.scf_maxiter,
            "conv_tol": self.qm.conv_tol,
            "gridlevel": self.qm.grid_level,
        }

        if self.qm.method == "MP2":
            kwargs.update(MP2=True, MP2_DF=self.qm.density_fitting, frozen_core_setting="Auto" if self.qm.frozen_core else None)
        elif self.qm.method.startswith("CCSD"):
            kwargs.update(CC=True, CCmethod=self.qm.method, frozen_core_setting="Auto" if self.qm.frozen_core else None)
        elif self.qm.method == "CASSCF":
            kwargs.update(CASSCF=True, active_space=list(self.qm.cas_active_space or ()))
        elif self.qm.method == "CASCI":
            kwargs.update(CAS=True, active_space=list(self.qm.cas_active_space or ()))
        elif self.qm.method == "TDDFT":
            kwargs.update(TDDFT=True)

        kwargs.update(self.qm.pyscf_kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return ash.PySCFTheory(**kwargs)

    # ------------------------------------------------------------
    def _build_mm_theory(self, ash):
        spec = self.system.spec
        common: dict[str, Any] = {
            "platform": self.mm.platform,
            "numcores": self.mm.numcores,
            "rigidwater": self.mm.rigid_water,
            "autoconstraints": self.mm.autoconstraints,
            "hydrogenmass": self.mm.hydrogen_mass,
            "periodic": spec.periodic,
            "fragment": self.system.fragment,
        }
        if spec.periodic:
            common.update(
                nonbondedMethod_PBC=self.mm.nonbonded_method_pbc,
                periodic_nonbonded_cutoff=self.mm.periodic_nonbonded_cutoff,
            )
        else:
            common.update(
                nonbondedMethod_noPBC=self.mm.nonbonded_method_no_pbc,
                nonbonded_cutoff_noPBC=self.mm.nonbonded_cutoff_no_pbc,
            )

        if self.mm.constraints:
            common["constraints"] = self.mm.constraints
        common.update(self.mm.openmm_kwargs)

        if spec.kind == "amber":
            return ash.OpenMMTheory(
                Amberfiles=True,
                amberprmtopfile=spec.topology or _missing(spec, "topology", "Amber prmtop"),
                pdbfile=spec.structure,
                **common,
            )
        if spec.kind == "charmm":
            return ash.OpenMMTheory(
                CHARMMfiles=True,
                psffile=spec.topology or _missing(spec, "topology", "CHARMM PSF"),
                pdbfile=spec.structure,
                **common,
            )
        if spec.kind == "gromacs":
            return ash.OpenMMTheory(
                GROMACSfiles=True,
                gromacstopfile=spec.topology or _missing(spec, "topology", "GROMACS top"),
                grofile=spec.structure,
                **common,
            )
        # PDB + forcefield XMLs (default)
        return ash.OpenMMTheory(
            xmlfiles=list(spec.forcefield),
            pdbfile=spec.structure,
            **common,
        )

    # ------------------------------------------------------------
    def fragment_scf(
        self,
        real_atoms: list[int],
        ghost_atoms: list[int] | None = None,
        *,
        charge: int | None = None,
        mult: int | None = None,
        embed_in_mm: bool = True,
    ) -> Any:
        """Run a standalone PySCF SCF on a subset of the QM region.

        ``real_atoms`` carry both basis and nucleus; ``ghost_atoms`` carry
        only basis (PySCF "ghost-X" prefix). Indices reference the *full
        system* indexing. When ``embed_in_mm=True`` and we can recover the
        MM partial charges from the underlying ash QMMMTheory, the SCF is
        polarised by those point charges via ``pyscf.qmmm.mm_charge``.

        Returns the converged PySCF mean-field. Raises ImportError if PySCF
        is not available.
        """
        try:
            from pyscf import dft, gto, scf
        except ImportError as e:  # pragma: no cover
            raise ImportError("fragment_scf needs PySCF.") from e

        ghost_atoms = list(ghost_atoms or [])
        real_atoms = list(real_atoms)
        if not real_atoms:
            raise ValueError("fragment_scf needs at least one real atom.")
        overlap = set(real_atoms) & set(ghost_atoms)
        if overlap:
            raise ValueError(f"Atoms cannot be both real and ghost: {sorted(overlap)}")

        elems = list(getattr(self.system.fragment, "elems", []))
        coords = self.system.coords  # Angstrom
        if not elems:
            raise RuntimeError("System fragment has no elems; cannot build a Mole.")

        atom_spec = []
        for i in real_atoms:
            atom_spec.append([elems[i], tuple(map(float, coords[i]))])
        for i in ghost_atoms:
            atom_spec.append([f"ghost-{elems[i]}", tuple(map(float, coords[i]))])

        eff_charge = self.system.spec.charge if charge is None else int(charge)
        eff_mult = self.system.spec.mult if mult is None else int(mult)
        mol = gto.M(
            atom=atom_spec,
            basis=self.qm.basis,
            charge=eff_charge,
            spin=eff_mult - 1,
            unit="Angstrom",
            verbose=0,
        )

        mf = self._make_pyscf_mf(mol, dft, scf)

        if embed_in_mm:
            mm_coords, mm_charges = self._extract_mm_environment(real_atoms + ghost_atoms)
            if len(mm_coords):
                from pyscf import qmmm as pyscf_qmmm

                mf = pyscf_qmmm.mm_charge(mf, mm_coords, mm_charges)

        mf.kernel()
        return mf

    # ------------------------------------------------------------
    def _make_pyscf_mf(self, mol, dft_mod, scf_mod):
        """Build a PySCF mean-field object matching our QMSpec."""
        if self.qm.method == "DFT":
            cls = dft_mod.UKS if self.qm.scf_type == "UKS" else dft_mod.RKS
            mf = cls(mol)
            mf.xc = self.qm.functional
            mf.grids.level = self.qm.grid_level
        else:
            cls_map = {"RHF": scf_mod.RHF, "UHF": scf_mod.UHF, "ROHF": scf_mod.ROHF}
            cls = cls_map.get(self.qm.scf_type, scf_mod.RHF)
            mf = cls(mol)
        mf.max_cycle = self.qm.scf_maxiter
        mf.conv_tol = self.qm.conv_tol
        if self.qm.density_fitting and self.qm.aux_basis:
            mf = mf.density_fit(auxbasis=self.qm.aux_basis)
        return mf

    def _extract_mm_environment(self, qm_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Try to recover (coords, charges) of all MM atoms from the underlying theory.

        Different ASH versions store this differently; we probe a few likely
        attribute names and fall back to an empty environment (i.e. gas-phase
        fragments) if none works. Coords returned in Angstrom because that's
        what we use everywhere; we don't multiply by Bohr conversion here.
        """
        if self._theory is None:
            return np.empty((0, 3)), np.empty(0)

        coords = self.system.coords
        natoms = coords.shape[0]
        charges_full = None
        for attr in ("charges", "MMcharges", "mm_charges", "atom_charges"):
            cand = getattr(self._theory, attr, None)
            if cand is None:
                cand = getattr(self._mm_theory, attr, None) if self._mm_theory else None
            if cand is not None and len(cand) == natoms:
                charges_full = np.asarray(cand, dtype=float)
                break

        if charges_full is None:
            return np.empty((0, 3)), np.empty(0)

        qm_set = set(qm_indices)
        mm_idx = [i for i in range(natoms) if i not in qm_set]
        if not mm_idx:
            return np.empty((0, 3)), np.empty(0)
        return coords[mm_idx].copy(), charges_full[mm_idx].copy()

    # ------------------------------------------------------------
    def get_qm_mf(self) -> Any:
        """Return the converged PySCF mean-field of the full QM region.

        Triggers an ash.Singlepoint if the SCF hasn't run yet, then digs
        into the underlying PySCFTheory for ``mf``.
        """
        import ash

        if self._theory is None:
            self.build()
        ash.Singlepoint(
            theory=self._theory,
            fragment=self.system.fragment,
            charge=self.system.spec.charge,
            mult=self.system.spec.mult,
            result_write_to_disk=False,
            printlevel=0,
        )
        mf = getattr(self._qm_theory, "mf", None)
        if mf is None:
            raise RuntimeError(
                "ash.PySCFTheory.mf is None after Singlepoint; SCF likely did not converge."
            )
        return mf

    # ------------------------------------------------------------
    def describe(self) -> str:
        scf = self.qm.scf_type + (
            f"/{self.qm.functional}" if self.qm.method == "DFT" else f"/{self.qm.method}"
        )
        lines = [
            self.system.summary(),
            f"QM: {scf}/{self.qm.basis}"
            + (f" + {self.qm.dispersion}" if self.qm.dispersion else "")
            + (" (DF)" if self.qm.density_fitting else ""),
            f"MM: OpenMM platform={self.mm.platform} "
            + (f"PBC/{self.mm.nonbonded_method_pbc} cutoff={self.mm.periodic_nonbonded_cutoff} A"
               if self.system.spec.periodic else
               f"no-PBC/{self.mm.nonbonded_method_no_pbc} cutoff={self.mm.nonbonded_cutoff_no_pbc} A"),
            f"Embedding: {self.embedding.scheme} "
            + (f"(link atoms: {self.embedding.link_atom_method}/{self.embedding.link_atom_type})"
               if self.embedding.use_link_atoms else "(no link atoms)"),
        ]
        return "\n".join(lines)


def _missing(spec, attr: str, what: str):
    raise ValueError(
        f"system.kind={spec.kind!r} requires '{attr}' to be set in the manifest "
        f"(path to the {what} file)."
    )


# Back-compat aliases for code that imported the old dataclass names.
QMConfig = QMSpec
MMConfig = MMSpec
EmbeddingConfig = EmbeddingSpec


__all__ = [
    "QMMMCalculator",
    "QMConfig", "QMSpec",
    "MMConfig", "MMSpec",
    "EmbeddingConfig", "EmbeddingSpec",
]
