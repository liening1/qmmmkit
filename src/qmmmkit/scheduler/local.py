"""Local scheduler: run a manifest in a subprocess on the current machine.

A subprocess is used (rather than in-process) so the worker can be killed
cleanly via ``cancel`` and so that PySCF/OpenMM crashes don't take down
the GUI. Output flows into ``workdir/log.txt`` like the SLURM backend.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

from .._registry import register_scheduler
from ..manifest import JobManifest
from .base import JobHandle, JobState, Scheduler, SchedulerError


@register_scheduler("local")
class LocalScheduler(Scheduler):
    name = "local"

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.home() / ".qmmmkit" / "jobs").expanduser()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    # ------------------------------------------------------------
    def submit(self, manifest: JobManifest) -> JobHandle:
        manifest.resolve_paths()
        job_id = f"local-{uuid.uuid4().hex[:8]}"
        workdir = self.base_dir / job_id
        workdir.mkdir(parents=True, exist_ok=False)

        manifest_copy = workdir / "manifest.yaml"
        manifest.save(manifest_copy)

        cmd = [sys.executable, "-m", "qmmmkit.runner", str(manifest_copy), str(workdir)]
        log_path = workdir / "stdio.log"
        env = os.environ.copy()
        log_fp = log_path.open("wb")
        proc = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env,
            cwd=str(workdir), close_fds=True,
        )
        self._procs[job_id] = proc

        return JobHandle(
            id=job_id,
            state=JobState.RUNNING,
            workdir=workdir,
            cluster=None,
            submitted_at=time.time(),
            started_at=time.time(),
            extra={"pid": proc.pid, "stdio_log": str(log_path)},
        )

    # ------------------------------------------------------------
    def poll(self, handle: JobHandle) -> JobHandle:
        proc = self._procs.get(handle.id)
        # If we lost the proc handle (e.g., GUI restart) fall back to state.json
        if proc is None:
            return self._poll_from_state(handle)
        rc = proc.poll()
        if rc is None:
            handle.state = JobState.RUNNING
            return handle
        handle.exit_code = rc
        handle.finished_at = time.time()
        handle.state = JobState.COMPLETED if rc == 0 else JobState.FAILED
        return handle

    def _poll_from_state(self, handle: JobHandle) -> JobHandle:
        state_file = handle.workdir / "state.json"
        if not state_file.exists():
            return handle
        import json
        d = json.loads(state_file.read_text())
        status = d.get("status", "unknown")
        handle.state = {
            "running": JobState.RUNNING,
            "completed": JobState.COMPLETED,
            "failed": JobState.FAILED,
        }.get(status, JobState.UNKNOWN)
        handle.finished_at = d.get("finished_at")
        return handle

    # ------------------------------------------------------------
    def cancel(self, handle: JobHandle) -> JobHandle:
        proc = self._procs.get(handle.id)
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        handle.state = JobState.CANCELLED
        handle.finished_at = time.time()
        return handle

    # ------------------------------------------------------------
    def stream_log(self, handle: JobHandle, *, follow: bool = True) -> Iterator[str]:
        log_path = handle.workdir / "log.txt"
        # Prefer the structured log; fall back to stdio
        if not log_path.exists():
            log_path = handle.workdir / "stdio.log"
        # Tail the file
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip("\n")
                    continue
                if not follow:
                    return
                proc = self._procs.get(handle.id)
                if proc is None or proc.poll() is not None:
                    rest = f.read()
                    for r in rest.splitlines():
                        yield r
                    return
                time.sleep(0.2)

    # ------------------------------------------------------------
    def fetch_results(self, handle: JobHandle, dest: Path) -> Path:
        # Local: results already on disk; just resolve the workdir.
        dest = Path(dest)
        if dest.resolve() == handle.workdir.resolve():
            return handle.workdir
        import shutil

        dest.mkdir(parents=True, exist_ok=True)
        for item in handle.workdir.iterdir():
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
        return dest
