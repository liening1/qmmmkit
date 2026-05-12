"""End-to-end example using the manifest-driven runner.

Builds a JobManifest in Python (rather than from YAML), then runs it
through the same ``execute_manifest`` the CLI / GUI / SLURM workers use.

Run:
    conda env create -f environment.yml
    conda activate qmmmkit
    pip install -e .
    python examples/example_qmmm_singlepoint.py path/to/system.pdb
"""

from __future__ import annotations

import sys
from pathlib import Path

from qmmmkit.runner import execute_manifest
from qmmmkit.schemas import (
    AnalysisSpec,
    EmbeddingSpec,
    JobManifest,
    MMSpec,
    QMSpec,
    SystemSpec,
    TaskSpec,
)


def main(pdb_path: str) -> None:
    manifest = JobManifest(
        name=Path(pdb_path).stem + "_sp",
        system=SystemSpec(
            kind="pdb",
            structure=str(Path(pdb_path).resolve()),
            qm_atoms=[],  # fill in real atom indices for your QM region
            forcefield=["amber14-all.xml", "amber14/tip3p.xml"],
            charge=0,
            mult=1,
            active_shell=6.0,
        ),
        qm=QMSpec(
            method="DFT", scf_type="RKS", functional="B3LYP",
            basis="def2-SVP", dispersion="d3bj",
        ),
        mm=MMSpec(platform="CPU"),
        embedding=EmbeddingSpec(scheme="electrostatic", use_link_atoms=True),
        task=TaskSpec(name="single_point", options={"gradient": False}),
        analyses=[
            AnalysisSpec(kind="population", options={"method": "cm5"}),
            AnalysisSpec(
                kind="natural_orbitals",
                options={"flavour": "canonical", "correlated_method": "MP2"},
            ),
        ],
    )

    if not manifest.system.qm_atoms:
        sys.exit(
            "Edit examples/example_qmmm_singlepoint.py: SystemSpec.qm_atoms is empty. "
            "Replace it with the 0-indexed atom indices that should be in the QM region."
        )

    workdir = Path("./qmmmkit_run").resolve()
    result = execute_manifest(manifest, workdir=workdir)

    print(f"\nE = {result.task.energy:.8f} Ha")
    print(f"Engines: {result.provenance.engines}")
    print(f"Duration: {result.provenance.duration_seconds:.1f} s")
    print(f"Result file: {workdir / 'result.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python example_qmmm_singlepoint.py <system.pdb>")
    main(sys.argv[1])
