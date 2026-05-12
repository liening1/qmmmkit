# qmmmkit

State-of-the-art QM/MM toolkit for molecular and biomolecular systems.

PySCF + OpenMM coupled via [ASH](https://github.com/RagnarB83/ash), with
density-based analyses, reaction-path methods, structured provenance, and
dispatch to remote HPC SLURM clusters from a workstation GUI.

[![tests](https://github.com/liening1/qmmmkit/actions/workflows/test.yml/badge.svg)](https://github.com/liening1/qmmmkit/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Status: alpha.** The framework compiles cleanly and the mocked-engine
> test suite passes in CI. Real-engine validation runs via
> `scripts/validate_ash_pyscf.py` on any host with PySCF + OpenMM + ASH
> installed.

## Highlights

- **Manifest-first.** A single YAML reproducibly describes a calculation
  end-to-end. The same manifest runs locally or via SLURM on an HPC.
- **Tasks.** Single point, geometry optimisation, transition state
  (`TSOpt=True`), IRC, N-D PES scans, Nudged Elastic Band (climbing
  image), energy decomposition (EDA).
- **Analyses.** Mulliken / Löwdin / Hirshfeld / CM5 / NPA-via-Janpa
  charges; canonical / UNO / spin natural orbitals with Molden export;
  charge displacement (1D CDA); 3D density-difference cubes; NCI
  (Yang–Cohen–Johnson reduced density gradient + sign(λ₂)·ρ); orbital
  cubes (HOMO/LUMO and arbitrary windows).
- **QM/MM embeddings.** Electrostatic with link atoms (`linkatom_method`
  = `simple`/`ratio`); ghost-atom fragment SCFs in the MM environment.
- **HPC-native.** SLURM submit/poll/sync over SSH, cluster profiles in
  `~/.config/qmmmkit/clusters.yaml`, Apptainer + Spack recipes,
  checkpoint/restart contracts on `OptimizeTask` and `PESScanTask`.
- **Reproducibility.** `result.json` and JSON-lines `events.log` carry
  full provenance (qmmmkit version + git SHA, Python interpreter, every
  engine version, input file SHA-256s, SLURM env, timing).
- **Workstation GUI.** PySide6 + NGL.js — atom-click QM region selection
  in a real 3D viewer, with cluster dispatch built in.

## Install

### From source (recommended for development)

```bash
git clone https://github.com/liening1/qmmmkit.git
cd qmmmkit
conda env create -f environment.yml
conda activate qmmmkit
pip install -e .
```

### From PyPI (not yet released)

```bash
pip install qmmmkit                 # core
pip install "qmmmkit[gui,slurm]"    # + workstation GUI + remote dispatch
```

Optional extras: `mm` (OpenMM stack), `optimizers` (pysisyphus, Sella),
`viz` (py3Dmol, nglview), `gui` (PySide6), `slurm` (paramiko, fabric).

## Quick start

### CLI

```bash
qmmmkit run examples/example_manifest.yaml
qmmmkit submit examples/example_manifest.yaml --cluster hpc --handle ./handle.json
qmmmkit logs ./handle.json --follow
qmmmkit fetch ./handle.json --to ./results
```

### Python

```python
from qmmmkit.runner import execute_manifest
from qmmmkit.schemas import JobManifest

manifest = JobManifest.load("examples/example_manifest.yaml")
result = execute_manifest(manifest, workdir="./run")
print(f"E_total = {result.task.energy:.8f} Ha")
```

### GUI

```bash
qmmmkit-gui
# or:  python -m qmmmkit.gui
```

Load a PDB, click atoms to toggle them in/out of the QM region, configure
the QM/MM levels in the sidebar, pick `Local` or any cluster profile from
the "Run on" dropdown, hit Run. Job status panel polls remote state every
5 s and tails the log.

## HPC deployment

```bash
ssh user@hpc.example.edu
bash <(curl -sSL https://raw.githubusercontent.com/liening1/qmmmkit/main/scripts/clone_on_hpc.sh) \
     --repo liening1/qmmmkit
```

This clones the repo, creates a conda env, `pip install -e .`, and runs
the live-engine validation suite. After it passes, configure a cluster
profile on your **workstation** (`~/.config/qmmmkit/clusters.yaml` — see
[`examples/example_clusters.yaml`](examples/example_clusters.yaml)) and
dispatch jobs from the GUI or CLI.

Three install shapes are supported:
- **conda env** (default; see `scripts/deploy_hpc.sh`),
- **Apptainer image** (`scripts/build_apptainer.sh`),
- **Spack package** (`packaging/spack/package.py`).

## Architecture

```
                         ┌────────────────────────────────┐
                         │  Workstation GUI (PySide6 +    │
                         │  NGL.js)  /  CLI               │
                         └───────────────┬────────────────┘
                                         │  JobManifest (YAML)
                                         ▼
        ┌────────────────────────────────┴────────────────────────────────┐
        │  Scheduler abstraction                                           │
        │      LocalScheduler  (subprocess)                                │
        │      SlurmScheduler  (SSH → sbatch → squeue/sacct → sftp sync)   │
        └────────────────────────────────┬────────────────────────────────┘
                                         │  python -m qmmmkit.runner
                                         ▼
                  ┌──────────────────────┴──────────────────────┐
                  │  qmmmkit.runner.execute_manifest             │
                  │  - registry-driven task / analysis dispatch  │
                  │  - structured logging (loguru, JSON-lines)   │
                  │  - provenance (engine versions, hashes, SHA) │
                  └──────────────────────┬──────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  │  Tasks                │  Analyses           │
                  │  single_point         │  population         │
                  │  optimize             │  natural_orbitals   │
                  │  transition_state     │  charge_displacement│
                  │  irc                  │  density_difference │
                  │  pes_scan             │  nci                │
                  │  neb                  │  orbital_cube       │
                  │  eda                  │                     │
                  └──────────────────────┬──────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  │  QMMMCalculator                              │
                  │  - ash.PySCFTheory + OpenMMTheory + QMMMTheory │
                  │  - fragment_scf (ghost-atom subsets)         │
                  └──────────────────────────────────────────────┘
```

Plugins are registered via decorators (`@register_task`, `@register_analysis`,
`@register_scheduler`) and discovered via Python entry points, so external
packages can extend qmmmkit without forking the source.

## Repository layout

```
src/qmmmkit/
  schemas/        pydantic models: SystemSpec, QMSpec, MMSpec, ...
  tasks/          registered tasks (single_point, optimize, neb, eda, ...)
  analysis/       registered analyses (cubes, NOs, NCI, populations, ...)
  scheduler/      LocalScheduler, SlurmScheduler, transport, profiles
  gui/            PySide6 desktop GUI + NGL.js viewer + bridge
  runner.py       headless manifest runner
  cli.py          `qmmmkit` typer-based CLI
  calculator.py   ash.QMMMTheory wrapper + fragment_scf
  system.py       runtime system (ash.Fragment + SystemSpec)
  _registry.py    plugin registries
  _provenance.py  versions, hashes, SLURM env, timing
  _logging.py     loguru sinks: log.txt + events.log
examples/         manifests + cluster profile template
scripts/          deploy_hpc.sh, clone_on_hpc.sh, build_apptainer.sh,
                  sync_to_hpc.{sh,ps1}, validate_ash_pyscf.py
packaging/        Apptainer.def, spack/package.py
tests/            mocked-engine suite (runs in CI without PySCF/OpenMM)
```

## Reproducibility & provenance

Every successful run writes a `result.json` validated against the
`JobResult` pydantic schema. It carries:

- the verbatim `manifest` that ran (so you can replay it),
- `provenance.engines` — versions of ash, pyscf, openmm, geometric,
  pysisyphus, numpy, scipy, ...,
- `provenance.qmmmkit_git_sha` — short SHA of this checkout,
- `provenance.inputs` — SHA-256 of each input file (truncated),
- `provenance.slurm` — `SLURM_JOB_ID`, `SLURM_JOB_NODELIST`,
  `SLURM_JOB_PARTITION`, ... captured automatically on compute nodes,
- timing (wall + CPU) and host details.

Tag releases (`git tag v0.1.0 && git push --tags`) and your results
become bit-for-bit traceable to the commit that produced them.

## Roadmap

- Real-engine validation on a maintained HPC profile.
- GUI cube/orbital visualisation in the same NGL canvas.
- Full Morokuma-style EDA decomposition (electrostatic / exchange /
  polarisation / charge-transfer).
- NEB convergence dashboards.
- Polarisable embedding (PE library).

## License

MIT — see [LICENSE](LICENSE).

## Citing

If you publish results produced with qmmmkit, please also cite the
backends it leverages: ASH, PySCF, OpenMM, geomeTRIC, pysisyphus.

## Contributing

Issues and pull requests are welcome at
<https://github.com/liening1/qmmmkit>. Run the mocked-engine test suite
locally with `pytest tests/`; full real-engine validation requires the
conda environment and runs via `python scripts/validate_ash_pyscf.py`.
