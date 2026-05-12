"""Structured logging built on loguru.

Two sinks for every run:
- ``log.txt`` — human-readable, single line per event.
- ``events.log`` — JSON-lines, one record per event, with structured fields.

The GUI tails ``log.txt`` (cheap, human-friendly), automated tooling
parses ``events.log`` (richer fields including engine versions and timings).

Use ``with run_logger(workdir, run_id=...) as log:`` to bind a fresh sink
for one job. Outside that context, ``get_logger()`` returns a process-wide
default logger that writes only to stderr.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    from loguru import logger as _loguru_logger
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "loguru is required. Install with `pip install loguru` or `pip install qmmmkit`."
    ) from e


_HUMAN_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{extra[run_id]}</cyan> "
    "{message}"
)


def _json_sink(record: dict) -> None:
    """Loguru-compatible sink that emits JSON-lines to ``record['extra']['_jsonl_path']``."""
    path = record["extra"].get("_jsonl_path")
    if not path:
        return
    payload = {
        "time": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    extras = {k: v for k, v in record["extra"].items() if not k.startswith("_") and k != "run_id"}
    if extras:
        payload["fields"] = _safe(extras)
    if record["exception"]:
        payload["exception"] = str(record["exception"])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _safe(o: Any) -> Any:
    if isinstance(o, (str, int, float, bool, type(None))):
        return o
    if isinstance(o, dict):
        return {str(k): _safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_safe(v) for v in o]
    return repr(o)


def get_logger():
    """Return the process-wide logger. Safe to import everywhere."""
    return _loguru_logger.bind(run_id="-")


@contextmanager
def run_logger(workdir: Path, *, run_id: str, level: str = "INFO") -> Iterator[Any]:
    """Bind a per-run logger that writes to ``workdir/log.txt`` + ``workdir/events.log``."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    human_path = workdir / "log.txt"
    jsonl_path = workdir / "events.log"

    _loguru_logger.remove()
    sink_stderr = _loguru_logger.add(sys.stderr, level=level, format=_HUMAN_FORMAT, enqueue=False)
    sink_human = _loguru_logger.add(
        human_path, level=level, format=_HUMAN_FORMAT, enqueue=False, rotation=None,
    )
    sink_jsonl = _loguru_logger.add(_json_sink, level=level, enqueue=False)

    bound = _loguru_logger.bind(run_id=run_id, _jsonl_path=str(jsonl_path))
    try:
        yield bound
    finally:
        for sink in (sink_stderr, sink_human, sink_jsonl):
            try:
                _loguru_logger.remove(sink)
            except ValueError:
                pass
        # Restore a basic stderr sink so subsequent imports keep working.
        _loguru_logger.add(sys.stderr, level="INFO", format=_HUMAN_FORMAT)


__all__ = ["get_logger", "run_logger"]
