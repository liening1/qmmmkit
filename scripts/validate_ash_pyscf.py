#!/usr/bin/env python
"""Live-engine validation for qmmmkit.

Exercises every code path against an actually-installed ASH / PySCF / OpenMM
stack and prints a per-step pass/fail report. Designed to be the *first*
thing you run after `conda env create -f environment.yml; pip install -e .`
on a workstation with the real engines available.

Test system: a 3-water cluster. The first water is QM; the other two are MM
(TIP3P). All atoms in residue 1 are in the QM region, the rest are MM —
this exercises QM/MM electrostatic embedding without invoking link atoms.

Each step is wrapped in try/except so a single failure doesn't stop the
rest. Exit code is 0 if every non-optional step passes, 1 otherwise.

Usage::

    conda activate qmmmkit
    python scripts/validate_ash_pyscf.py
    python scripts/validate_ash_pyscf.py --keep-workdir   # don't delete on success
    python scripts/validate_ash_pyscf.py --skip optimize  # skip a slow step
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Test fixture: 3-water cluster as TIP3P-compatible PDB.
# Residue 1 is the QM water; residues 2 and 3 are MM.
# ---------------------------------------------------------------------------
WATER_TRIMER_PDB = """\
HEADER    qmmmkit validation: water trimer
CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1
ATOM      1  O   HOH A   1      -1.500   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1      -2.150   0.700   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -2.150  -0.700   0.000  1.00  0.00           H
ATOM      4  O   HOH A   2       1.500   0.000   0.000  1.00  0.00           O
ATOM      5  H1  HOH A   2       2.150   0.700   0.000  1.00  0.00           H
ATOM      6  H2  HOH A   2       2.150  -0.700   0.000  1.00  0.00           H
ATOM      7  O   HOH A   3       0.000   2.500   0.000  1.00  0.00           O
ATOM      8  H1  HOH A   3       0.700   3.150   0.000  1.00  0.00           H
ATOM      9  H2  HOH A   3      -0.700   3.150   0.000  1.00  0.00           H
END
"""
QM_ATOMS = [0, 1, 2]                # residue 1
MM_ATOMS = list(range(3, 9))        # residues 2 and 3


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    label: str
    status: str           # "PASS" / "FAIL" / "SKIP"
    duration_s: float
    detail: str = ""
    optional: bool = False


@dataclass
class Validator:
    skip: list[str] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)

    def run(self, label: str, fn: Callable[[], Any], *, optional: bool = False) -> Any:
        if any(s in label.lower() for s in self.skip):
            print(f"  [SKIP] {label}  (--skip)")
            self.results.append(StepResult(label, "SKIP", 0.0, optional=optional))
            return None
        t0 = time.perf_counter()
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - validator catches everything
            dt = time.perf_counter() - t0
            tb = traceback.format_exc().splitlines()
            short = " | ".join(tb[-3:])[:240]
            tag = "FAIL" if not optional else "FAIL*"
            print(f"  [{tag}] {label} ({dt:.1f}s)  {type(exc).__name__}: {exc}")
            self.results.append(
                StepResult(label, "FAIL", dt, detail=short, optional=optional)
            )
            return None
        dt = time.perf_counter() - t0
        print(f"  [PASS] {label} ({dt:.1f}s)")
        self.results.append(StepResult(label, "PASS", dt, optional=optional))
        return value

    def report(self) -> int:
        print()
        print("=" * 70)
        passed = [r for r in self.results if r.status == "PASS"]
        failed_required = [r for r in self.results if r.status == "FAIL" and not r.optional]
        failed_optional = [r for r in self.results if r.status == "FAIL" and r.optional]
        skipped = [r for r in self.results if r.status == "SKIP"]
        print(
            f"qmmmkit validation: "
            f"{len(passed)} passed, "
            f"{len(failed_required)} failed (required), "
            f"{len(failed_optional)} failed (optional), "
            f"{len(skipped)} skipped"
        )
        for r in failed_required + failed_optional:
            tag = "FAIL" if not r.optional else "fail-optional"
            print(f"  {tag}: {r.label}")
            if r.detail:
                print(f"    -> {r.detail}")
        print("=" * 70)
        return 0 if not failed_required else 1


# ---------------------------------------------------------------------------
def run_validation(workdir: Path, *, skip: list[str]) -> int:
    v = Validator(skip=skip)

    pdb_path = workdir / "trimer.pdb"
    pdb_path.write_text(WATER_TRIMER_PDB)
    print(f"workdir: {workdir}")
    print(f"PDB: {pdb_path}")

    # ---- imports + versions --------------------------------------
    print("\n--- environment ---")
    qmmmkit_module = v.run("import qmmmkit", lambda: __import__("qmmmkit"))
    if qmmmkit_module is not None:
        print(f"        qmmmkit {getattr(qmmmkit_module, '__version__', '?')}")

    ash_module = v.run("import ash", lambda: __import__("ash"))
    if ash_module is not None:
        print(f"        ash {getattr(ash_module, '__version__', '?')}")

    pyscf_module = v.run("import pyscf", lambda: __import__("pyscf"))
    if pyscf_module is not None:
        print(f"        pyscf {getattr(pyscf_module, '__version__', '?')}")

    openmm_module = v.run("import openmm", lambda: __import__("openmm"), optional=True)
    if openmm_module is not None:
        print(f"        openmm {getattr(openmm_module, 'version', None) and openmm_module.version.short_version}")

    # We can stop early if the core stack isn't there.
    if qmmmkit_module is None or ash_module is None or pyscf_module is None:
        print("\nMissing core dependency, aborting validation.")
        return v.report()

    # ---- spec + system + calculator ------------------------------
    print("\n--- system + calculator ---")
    from qmmmkit.calculator import QMMMCalculator
    from qmmmkit.schemas import (
        EmbeddingSpec, JobManifest, MMSpec, QMSpec, SystemSpec, TaskSpec,
    )
    from qmmmkit.system import QMMMSystem

    def _build_spec():
        return SystemSpec(
            kind="pdb",
            structure=str(pdb_path),
            qm_atoms=QM_ATOMS,
            forcefield=["amber14-all.xml", "amber14/tip3p.xml"],
            charge=0,
            mult=1,
            active_shell=None,
            periodic=False,
        )

    spec = v.run("build SystemSpec", _build_spec)
    if spec is None:
        return v.report()

    sys_obj = v.run("QMMMSystem.from_spec (ash.Fragment)", lambda: QMMMSystem.from_spec(spec))
    if sys_obj is None:
        return v.report()

    print(f"        system has {sys_obj.n_atoms} atoms, {len(sys_obj.qm_atoms)} QM")

    qm_cfg = QMSpec(
        method="DFT", scf_type="RKS", functional="PBE",
        basis="STO-3G", dispersion=None, density_fitting=False,
        scf_maxiter=50, conv_tol=1e-7, grid_level=3, nprocs=1, memory=2000,
    )
    mm_cfg = MMSpec(platform="CPU", periodic_nonbonded_cutoff=12.0, rigid_water=False)
    emb_cfg = EmbeddingSpec(scheme="electrostatic", use_link_atoms=False)

    calc = v.run(
        "QMMMCalculator.build (ash.PySCFTheory + OpenMMTheory + QMMMTheory)",
        lambda: _build_calculator(sys_obj, qm_cfg, mm_cfg, emb_cfg),
    )
    if calc is None:
        return v.report()

    print(f"        {calc.describe().splitlines()[1]}")

    # ---- single point --------------------------------------------
    print("\n--- single point ---")

    def _singlepoint():
        import ash
        result = ash.Singlepoint(
            theory=calc.theory, fragment=calc.system.fragment,
            charge=spec.charge, mult=spec.mult,
            result_write_to_disk=False, printlevel=0,
        )
        return float(getattr(result, "energy"))

    e_full = v.run("Singlepoint(QMMMTheory)", _singlepoint)
    if e_full is not None:
        print(f"        E_total = {e_full:.8f} Ha")

    # ---- get_qm_mf -----------------------------------------------
    mf_full = v.run("calc.get_qm_mf (PySCFTheory.mf retrieval)", calc.get_qm_mf)
    if mf_full is not None:
        n_orb = int(getattr(mf_full, "mo_coeff").shape[1])
        nelec = sum(getattr(mf_full, "mo_occ"))
        print(f"        mol: {mf_full.mol.nao} AOs, {nelec:.0f} electrons, {n_orb} MOs")

    # ---- fragment_scf gas-phase ----------------------------------
    print("\n--- fragment SCFs ---")
    mf_gas = v.run(
        "fragment_scf (gas-phase, no MM)",
        lambda: calc.fragment_scf(QM_ATOMS, ghost_atoms=[], embed_in_mm=False),
    )
    if mf_gas is not None and mf_full is not None:
        e_gas = float(mf_gas.e_tot)
        e_qm_pol = _maybe_float(getattr(calc.theory, "QMenergy", None))
        if e_qm_pol is not None:
            print(f"        E_QM_gas       = {e_gas:.8f} Ha")
            print(f"        E_QM_polarised = {e_qm_pol:.8f} Ha")
            print(f"        ΔE_pol         = {(e_qm_pol - e_gas)*627.5:.2f} kcal/mol")

    # ---- fragment_scf embedded -----------------------------------
    mf_embedded = v.run(
        "fragment_scf (embedded in MM point charges)",
        lambda: calc.fragment_scf(QM_ATOMS, ghost_atoms=[], embed_in_mm=True),
        optional=True,  # depends on ASH exposing MM charges in a known attribute
    )
    if mf_embedded is not None:
        print(f"        E_embedded = {float(mf_embedded.e_tot):.8f} Ha")

    # ---- analyses -------------------------------------------------
    print("\n--- analyses ---")
    from qmmmkit.analysis import (
        natural_orbitals, population_analysis, write_molden,
    )
    from qmmmkit.analysis.nci import NCIAnalysis, NCIOptions
    from qmmmkit.analysis.orbital_cube import (
        OrbitalCubeAnalysis, OrbitalCubeOptions,
    )

    class _LogStub:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def exception(self, *a, **k): pass

    log_stub = _LogStub()

    if mf_full is not None:
        v.run(
            "Mulliken population analysis",
            lambda: population_analysis(mf_full, method="mulliken"),
        )
        v.run(
            "CM5 population analysis",
            lambda: population_analysis(mf_full, method="cm5"),
            optional=True,  # CM5 needs Hirshfeld which needs free-atom SCFs
        )
        no_result = v.run(
            "Natural orbitals (canonical, MP2)",
            lambda: natural_orbitals(mf_full, flavour="canonical", correlated_method="MP2"),
            optional=True,  # MP2 with the small basis may struggle
        )
        if no_result is not None:
            molden_path = workdir / "validation_no.molden"
            v.run("write_molden (NO file)", lambda: write_molden(no_result, molden_path))
            print(f"        n_eff(unpaired) = {no_result.effective_unpaired:.4f}")

        nci_analysis = NCIAnalysis(NCIOptions(spacing=0.20, margin=2.0))
        v.run(
            "NCI cube generation",
            lambda: nci_analysis.run(calc, workdir=workdir, log=log_stub),
        )

        orb_cube = OrbitalCubeAnalysis(OrbitalCubeOptions(frontier=1, spacing=0.30, margin=3.0))
        v.run(
            "Orbital cube (HOMO + LUMO)",
            lambda: orb_cube.run(calc, workdir=workdir, log=log_stub),
        )

    # ---- EDA task -------------------------------------------------
    print("\n--- tasks ---")
    from qmmmkit.tasks.eda import EDAOptions, EDATask

    v.run(
        "EDATask",
        lambda: EDATask(EDAOptions(write_summary=True)).run(
            calc, workdir=workdir, log=log_stub,
        ),
    )

    # ---- optimize task --------------------------------------------
    from qmmmkit.tasks.optimize import OptimizeTask, OptOptions

    v.run(
        "OptimizeTask (5 cycles, GAU_LOOSE)",
        lambda: OptimizeTask(OptOptions(maxiter=5, convergence="GAU_LOOSE")).run(
            calc, workdir=workdir, log=log_stub,
        ),
        optional=True,  # may not converge in 5 cycles; we just want it to run
    )

    # ---- end-to-end manifest pipeline -----------------------------
    print("\n--- end-to-end manifest run ---")
    e2e_workdir = workdir / "e2e"
    e2e_workdir.mkdir(exist_ok=True)
    manifest = JobManifest(
        name="validation",
        system=spec,
        qm=qm_cfg, mm=mm_cfg, embedding=emb_cfg,
        task=TaskSpec(name="single_point", options={}),
        analyses=[],
    )
    manifest_path = e2e_workdir / "manifest.yaml"
    manifest.save(manifest_path)

    def _e2e_run():
        from qmmmkit.runner import execute_manifest

        result = execute_manifest(manifest, workdir=e2e_workdir)
        return result

    e2e = v.run("execute_manifest (full pipeline)", _e2e_run)
    if e2e is not None:
        rj = e2e_workdir / "result.json"
        payload = json.loads(rj.read_text())
        print(f"        result.json wrote {sum(1 for _ in rj.read_text().splitlines())} lines")
        print(f"        engines: {payload['provenance']['engines']}")
        print(f"        E_total = {payload['task']['energy']:.8f} Ha")

    return v.report()


# ---------------------------------------------------------------------------
def _build_calculator(sys_obj, qm_cfg, mm_cfg, emb_cfg):
    from qmmmkit.calculator import QMMMCalculator

    calc = QMMMCalculator(system=sys_obj, qm=qm_cfg, mm=mm_cfg, embedding=emb_cfg)
    calc.build()
    return calc


def _maybe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@contextmanager
def temp_workdir(keep: bool):
    if keep:
        wd = Path(tempfile.mkdtemp(prefix="qmmmkit-validate-"))
        try:
            yield wd
        finally:
            print(f"\nKept workdir: {wd}")
    else:
        with tempfile.TemporaryDirectory(prefix="qmmmkit-validate-") as td:
            yield Path(td)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--keep-workdir", action="store_true",
                   help="Keep the temporary workdir on success (always kept on failure).")
    p.add_argument("--skip", action="append", default=[],
                   help="Skip any step whose label contains this substring (case-insensitive). Repeatable.")
    args = p.parse_args(argv)
    skip = [s.lower() for s in args.skip]

    print("qmmmkit live validation\n=======================\n"
          f"python: {sys.version.split()[0]}  ({sys.executable})\n")

    keep = args.keep_workdir
    rc = 1
    try:
        with temp_workdir(keep) as wd:
            try:
                rc = run_validation(wd, skip=skip)
            except KeyboardInterrupt:
                print("\nInterrupted; preserving workdir.")
                keep = True
                rc = 130
            if rc != 0:
                # Always preserve the workdir on a real failure so the user can poke at it.
                if not keep:
                    print(f"\nFailure: preserving workdir {wd} for debugging.")
                    # The TemporaryDirectory cleanup happens on context exit; copy out.
                    import shutil

                    persistent = Path(tempfile.mkdtemp(prefix="qmmmkit-validate-failed-"))
                    shutil.copytree(wd, persistent / "workdir", dirs_exist_ok=True)
                    print(f"Copied to: {persistent / 'workdir'}")
    finally:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
