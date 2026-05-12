"""Headless manifest runner.

Single dispatcher: turn a JobManifest into a JobResult on disk. Same
code path on workstation (LocalScheduler) and on a SLURM compute node.

Outputs in workdir:
- ``log.txt``       human-readable log
- ``events.log``    JSON-lines structured log
- ``result.json``   JobResult (validated pydantic schema)
- ``state.json``    coarse state for the scheduler to poll without parsing result.json

Registry-driven: tasks and analyses are looked up by name in
``qmmmkit._registry``; new ones can be added via entry points without
touching this file.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from pathlib import Path

from . import __version__ as _qmmmkit_version
from ._logging import run_logger
from ._provenance import (
    Provenance,
    collect_static,
    hash_inputs,
    start_timer,
    stop_timer,
)
from ._registry import ANALYSES, TASKS
from .calculator import QMMMCalculator
from .schemas import JobManifest
from .schemas.result import (
    AnalysisResult,
    JobResult,
    ProvenanceModel,
    TaskResult,
)
from .system import QMMMSystem


def execute_manifest(manifest: JobManifest, *, workdir: Path) -> JobResult:
    """Run the calculation described by ``manifest`` in ``workdir``."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]

    state_path = workdir / "state.json"
    _write_state(state_path, "running", started_at=time.time())

    # Import the registry-populating subpackages so all built-in plugins are visible.
    from . import analysis as _analysis  # noqa: F401
    from . import tasks as _tasks  # noqa: F401

    # IMPORTANT: resolve relative paths BEFORE hashing inputs. ``input_files()``
    # checks file existence, and a manifest with `structure: ./input.pdb`
    # relative to its own directory would otherwise be invisible from a
    # different cwd, silently dropping provenance hashes.
    manifest.resolve_paths()

    provenance = collect_static()
    provenance.inputs = hash_inputs(manifest.input_files())
    start_timer(provenance)

    with run_logger(workdir, run_id=run_id) as log:
        log.info("run.start", name=manifest.name, task=manifest.task.name,
                 qmmmkit_version=_qmmmkit_version, host=provenance.host)
        try:
            log.info("system.load", structure=manifest.system.structure,
                     qm=len(manifest.system.qm_atoms), kind=manifest.system.kind)
            system = QMMMSystem.from_spec(manifest.system)
            log.info("system.ready", **{
                "n_atoms": system.n_atoms,
                "qm_atoms": len(system.qm_atoms),
                "active_atoms": len(system.active_atoms or []),
            })

            calc = QMMMCalculator(
                system=system, qm=manifest.qm, mm=manifest.mm, embedding=manifest.embedding,
            )
            log.info("calc.describe", text=calc.describe())
            calc.build()

            task_result = _run_task(manifest, calc, workdir, log)
            analysis_results = _run_analyses(manifest, calc, workdir, log)

            stop_timer(provenance)
            prov_model = ProvenanceModel.model_validate(provenance.to_dict())

            job_result = JobResult(
                manifest=manifest,
                provenance=prov_model,
                task=task_result,
                analyses=analysis_results,
                workdir=str(workdir),
                log_path=str(workdir / "log.txt"),
                events_path=str(workdir / "events.log"),
            )
            (workdir / "result.json").write_text(
                json.dumps(job_result.model_dump(mode="json"), indent=2, default=str)
            )
            _write_state(state_path, "completed", finished_at=time.time())
            log.info("run.done", duration=provenance.duration_seconds())
            return job_result

        except Exception as exc:  # noqa: BLE001
            stop_timer(provenance)
            log.exception("run.failed", error=str(exc))
            _write_state(
                state_path, "failed", finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
            )
            raise


# ---------------------------------------------------------------------------
def _run_task(manifest: JobManifest, calc: QMMMCalculator, workdir: Path, log) -> TaskResult:
    cls = TASKS.get(manifest.task.name)
    task = cls(manifest.task.options)
    log.info("task.start", name=cls.name)
    result = task.run(calc, workdir=workdir, log=log)
    if not isinstance(result, TaskResult):
        raise TypeError(f"Task {cls.name!r} returned {type(result).__name__}, expected TaskResult.")
    return result


def _run_analyses(manifest: JobManifest, calc: QMMMCalculator, workdir: Path, log) -> list[AnalysisResult]:
    out: list[AnalysisResult] = []
    for spec in manifest.analyses:
        if not ANALYSES.has(spec.kind):
            log.warning("analysis.unknown", kind=spec.kind, known=ANALYSES.names())
            continue
        cls = ANALYSES.get(spec.kind)
        analysis = cls(spec.options)
        log.info("analysis.start", kind=cls.name)
        try:
            r = analysis.run(calc, workdir=workdir, log=log)
        except Exception as exc:  # noqa: BLE001 - one analysis must not kill the rest
            log.exception("analysis.failed", kind=cls.name, error=str(exc))
            r = AnalysisResult(kind=cls.name, summary={"error": f"{type(exc).__name__}: {exc}"})
        out.append(r)
    return out


# ---------------------------------------------------------------------------
def _write_state(path: Path, status: str, **fields) -> None:
    payload = {"status": status, **{k: v for k, v in fields.items() if v is not None}}
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """SLURM workers / LocalScheduler invoke this: ``python -m qmmmkit.runner manifest.yaml [workdir]``."""
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m qmmmkit.runner <manifest.yaml> [workdir]", file=sys.stderr)
        return 2
    manifest = JobManifest.load(args[0])
    workdir = Path(args[1]) if len(args) > 1 else Path.cwd()
    try:
        execute_manifest(manifest, workdir=workdir)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
