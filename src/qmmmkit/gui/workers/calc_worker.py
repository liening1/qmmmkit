"""GUI workers that drive the Scheduler abstraction.

Two flavours, both running on a QThread:

- ``DispatchWorker`` calls ``scheduler.submit(manifest)`` and then poll-loops
  ``scheduler.poll(handle)`` until terminal, emitting ``state_changed`` each
  time the handle moves. Works equally well for ``LocalScheduler`` (subprocess)
  and ``SlurmScheduler`` (SSH+sbatch).

The GUI builds whichever ``Scheduler`` is appropriate (Local or SLURM with a
chosen profile) and hands it to a fresh worker.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal


@dataclass
class DispatchRequest:
    manifest_path: str
    workdir: str
    cluster: str | None = None     # None / "Local" -> LocalScheduler


class DispatchWorker(QObject):
    progress = Signal(str)
    state_changed = Signal(dict)   # JobHandle.to_dict()
    finished = Signal(dict)        # summary payload
    failed = Signal(str)
    log_line = Signal(str)         # streamed log line

    POLL_INTERVAL = 5.0  # seconds

    def __init__(self, request: DispatchRequest) -> None:
        super().__init__()
        self.request = request
        self._abort = False
        self._handle: Any = None
        self._scheduler: Any = None

    def abort(self) -> None:
        self._abort = True
        try:
            if self._scheduler is not None and self._handle is not None:
                self._scheduler.cancel(self._handle)
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------
    def run(self) -> None:
        try:
            from ...schemas import JobManifest
            from ...scheduler import LocalScheduler, SlurmScheduler, load_profile

            self.progress.emit("Loading manifest...")
            manifest = JobManifest.load(self.request.manifest_path)

            cluster = self.request.cluster
            if cluster and cluster.lower() != "local":
                self.progress.emit(f"Loading cluster profile: {cluster}")
                profile = load_profile(cluster)
                self._scheduler = SlurmScheduler(profile)
            else:
                self._scheduler = LocalScheduler()

            self.progress.emit("Submitting job...")
            self._handle = self._scheduler.submit(manifest)
            self._save_handle()
            self.state_changed.emit(self._handle.to_dict())
            self.progress.emit(
                f"Submitted: id={self._handle.id} cluster={self._handle.cluster or 'local'}"
            )

            # ----- Poll until terminal --------------------------------
            last_log_size = 0
            while not self._handle.state.is_terminal():
                if self._abort:
                    self._scheduler.cancel(self._handle)
                    break
                time.sleep(self.POLL_INTERVAL)
                try:
                    self._handle = self._scheduler.poll(self._handle)
                except Exception as exc:  # noqa: BLE001 - poll failures shouldn't crash worker
                    self.log_line.emit(f"poll error: {exc}")
                    continue
                self.state_changed.emit(self._handle.to_dict())
                last_log_size = self._stream_new_log_lines(last_log_size)

            self._save_handle()
            self.state_changed.emit(self._handle.to_dict())
            payload = self._build_summary()
            self.finished.emit(payload)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    # ----------------------------------------------------------------
    def _save_handle(self) -> None:
        if not self._handle:
            return
        path = Path(self.request.workdir) / "handle.json"
        path.write_text(json.dumps(self._handle.to_dict(), indent=2))

    def _stream_new_log_lines(self, last_size: int) -> int:
        """Tail ``log.txt`` from the workdir if accessible (LocalScheduler always;
        SlurmScheduler tail uses stream_log)."""
        try:
            log_path = Path(self.request.workdir) / "log.txt"
            if log_path.exists():
                content = log_path.read_text(encoding="utf-8", errors="replace")
                if len(content) > last_size:
                    new_chunk = content[last_size:]
                    for line in new_chunk.splitlines():
                        self.log_line.emit(line)
                    return len(content)
        except OSError:
            pass
        return last_size

    def _build_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "task": "?", "energy": None, "qm_energy": None, "mm_energy": None,
            "converged": None, "duration_seconds": None, "result_json": None,
            "log_path": str(Path(self.request.workdir) / "log.txt"),
            "n_analyses": 0,
            "state": self._handle.state.value if self._handle else "unknown",
        }
        result_json = Path(self.request.workdir) / "result.json"
        if result_json.exists():
            try:
                payload = json.loads(result_json.read_text())
                task = payload.get("task", {})
                prov = payload.get("provenance", {})
                summary.update(
                    task=task.get("name", "?"),
                    energy=task.get("energy"),
                    qm_energy=task.get("qm_energy"),
                    mm_energy=task.get("mm_energy"),
                    converged=task.get("converged"),
                    duration_seconds=prov.get("duration_seconds"),
                    result_json=str(result_json),
                    n_analyses=len(payload.get("analyses", [])),
                )
            except (OSError, json.JSONDecodeError):
                pass
        return summary


def start_worker(request: DispatchRequest) -> tuple[QThread, DispatchWorker]:
    thread = QThread()
    worker = DispatchWorker(request)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread, worker


# ---- Back-compat shim for callers that imported the old names --------------
CalcRequest = DispatchRequest
CalcWorker = DispatchWorker
