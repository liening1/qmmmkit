"""SLURM scheduler over SSH.

Workflow:
  1. ``submit`` opens an SSH session, mkdirs ``<workdir>/<job-id>`` on the
     cluster, uploads the manifest + any input files, renders an sbatch
     script from the template + cluster profile, submits with ``sbatch``,
     and parses the SLURM job id from stdout.
  2. ``poll`` queries ``squeue -j <id>`` then falls back to ``sacct`` once
     the job leaves the queue.
  3. ``stream_log`` tails ``log.txt`` (and ``slurm-<id>.out``) over SFTP.
  4. ``fetch_results`` walks the remote workdir and downloads everything
     with sftp.
"""

from __future__ import annotations

import re
import shlex
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Iterator

from .._registry import register_scheduler
from ..manifest import JobManifest
from .base import JobHandle, JobState, Scheduler, SchedulerError
from .profile import ClusterProfile, SbatchDefaults
from .transport import SSHTransport

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "sbatch.sh.tmpl"

_STATE_MAP = {
    "PD": JobState.PENDING,
    "CF": JobState.PENDING,
    "R": JobState.RUNNING,
    "CG": JobState.RUNNING,
    "S": JobState.PENDING,
    "ST": JobState.PENDING,
    "RS": JobState.PENDING,
    "CD": JobState.COMPLETED,
    "CA": JobState.CANCELLED,
    "F": JobState.FAILED,
    "TO": JobState.FAILED,
    "NF": JobState.FAILED,
    "BF": JobState.FAILED,
    "OOM": JobState.FAILED,
}
_LONG_STATE_MAP = {
    "PENDING": JobState.PENDING,
    "CONFIGURING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "COMPLETED": JobState.COMPLETED,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "TIMEOUT": JobState.FAILED,
    "NODE_FAIL": JobState.FAILED,
    "BOOT_FAIL": JobState.FAILED,
    "OUT_OF_MEMORY": JobState.FAILED,
}


@register_scheduler("slurm")
class SlurmScheduler(Scheduler):
    name = "slurm"

    def __init__(self, profile: ClusterProfile) -> None:
        self.profile = profile

    # ------------------------------------------------------------
    def submit(self, manifest: JobManifest) -> JobHandle:
        manifest.resolve_paths()
        local_id = uuid.uuid4().hex[:10]
        remote_workdir = PurePosixPath(self.profile.workdir) / f"{manifest.name}-{local_id}"

        transport = SSHTransport(self.profile)
        with transport.session() as t:
            t.mkdir_p(remote_workdir)

            # Stage manifest (with file paths rewritten to remote workdir)
            remote_manifest_path = remote_workdir / "manifest.yaml"
            staged_manifest = self._stage_manifest(manifest, t, remote_workdir)
            t.put_text(staged_manifest, remote_manifest_path)

            # Stage sbatch script
            sbatch_text = self._render_sbatch(manifest, remote_workdir)
            sbatch_path = remote_workdir / "submit.sbatch"
            t.put_text(sbatch_text, sbatch_path, mode=0o755)

            # Submit
            res = t.run(f"sbatch {shlex.quote(str(sbatch_path))}").check("sbatch")
            slurm_id = _parse_sbatch_jobid(res.stdout)
            if slurm_id is None:
                raise SchedulerError(f"Could not parse SLURM job id from: {res.stdout!r}")

        return JobHandle(
            id=slurm_id,
            state=JobState.SUBMITTED,
            workdir=Path(str(remote_workdir)),  # remote path stored as PurePosix-flavoured Path
            cluster=self.profile.name,
            submitted_at=time.time(),
            extra={
                "remote_workdir": str(remote_workdir),
                "sbatch_path": str(sbatch_path),
                "manifest_path": str(remote_manifest_path),
                "host": self.profile.host,
                "user": self.profile.user,
            },
        )

    # ------------------------------------------------------------
    def poll(self, handle: JobHandle) -> JobHandle:
        transport = SSHTransport(self.profile)
        with transport.session() as t:
            squeue = t.run(f"squeue -h -j {shlex.quote(handle.id)} -o '%T'")
            if squeue.rc == 0 and squeue.stdout.strip():
                state_long = squeue.stdout.strip().splitlines()[0].strip()
                handle.state = _LONG_STATE_MAP.get(state_long, JobState.UNKNOWN)
                if handle.state == JobState.RUNNING and handle.started_at is None:
                    handle.started_at = time.time()
                return handle

            # Not in queue: fall back to sacct
            sacct = t.run(
                f"sacct -j {shlex.quote(handle.id)} -X -n -o State,ExitCode,Start,End -P"
            )
            if sacct.rc == 0 and sacct.stdout.strip():
                line = sacct.stdout.strip().splitlines()[0]
                parts = line.split("|")
                if len(parts) >= 4:
                    state_long = parts[0].strip().split()[0]  # 'CANCELLED by 12345' -> 'CANCELLED'
                    exit_field = parts[1].strip()
                    handle.state = _LONG_STATE_MAP.get(state_long, JobState.UNKNOWN)
                    handle.exit_code = _parse_exit_code(exit_field)
                    handle.finished_at = time.time()
                    return handle

            handle.state = JobState.UNKNOWN
        return handle

    # ------------------------------------------------------------
    def cancel(self, handle: JobHandle) -> JobHandle:
        transport = SSHTransport(self.profile)
        with transport.session() as t:
            t.run(f"scancel {shlex.quote(handle.id)}")
        handle.state = JobState.CANCELLED
        handle.finished_at = time.time()
        return handle

    # ------------------------------------------------------------
    def stream_log(self, handle: JobHandle, *, follow: bool = True) -> Iterator[str]:
        transport = SSHTransport(self.profile)
        remote_log = handle.extra.get("remote_workdir", str(handle.workdir)) + "/log.txt"
        seen = 0
        with transport.session() as t:
            while True:
                if not t.exists(remote_log):
                    if not follow:
                        return
                    time.sleep(2.0)
                    continue
                content = t.read_text(remote_log)
                if len(content) > seen:
                    chunk = content[seen:]
                    seen = len(content)
                    for line in chunk.splitlines():
                        yield line
                if not follow:
                    return
                # Check whether job has finished; if so, drain once more then stop
                handle = self.poll(handle)
                if handle.state.is_terminal():
                    content = t.read_text(remote_log)
                    if len(content) > seen:
                        for line in content[seen:].splitlines():
                            yield line
                    return
                time.sleep(3.0)

    # ------------------------------------------------------------
    def fetch_results(self, handle: JobHandle, dest: Path) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        remote_root = handle.extra.get("remote_workdir", str(handle.workdir))
        transport = SSHTransport(self.profile)
        with transport.session() as t:
            for here, _dirs, files in t.walk(remote_root):
                rel = PurePosixPath(here).relative_to(remote_root)
                local_dir = dest / Path(*rel.parts) if rel.parts else dest
                local_dir.mkdir(parents=True, exist_ok=True)
                for fname in files:
                    t.get_file(f"{here}/{fname}", local_dir / fname)
        return dest

    # ------------------------------------------------------------
    # internals
    # ------------------------------------------------------------
    def _stage_manifest(self, manifest: JobManifest, t: SSHTransport, remote_workdir: PurePosixPath) -> str:
        """Upload input files referenced by the manifest, then return a YAML
        text whose paths are rewritten to point inside ``remote_workdir``."""
        for local_path in manifest.input_files():
            t.put_file(local_path, remote_workdir / local_path.name)

        # Rewrite ``system.structure`` (and optional ``system.topology``) to point
        # inside the remote workdir before emitting YAML.
        d = manifest.to_dict()
        for key in ("structure", "topology"):
            local_val = d.get("system", {}).get(key)
            if local_val:
                d["system"][key] = str(remote_workdir / Path(local_val).name)
        import yaml

        return yaml.safe_dump(d, sort_keys=False)

    def _render_sbatch(self, manifest: JobManifest, remote_workdir: PurePosixPath) -> str:
        s = self.profile.sbatch
        # Allow per-manifest overrides via task.options.sbatch
        overrides = manifest.task.options.get("sbatch", {})
        time_str = overrides.get("time", s.time)
        nodes = int(overrides.get("nodes", s.nodes))
        ntasks = int(overrides.get("ntasks", s.ntasks))
        cpus = int(overrides.get("cpus_per_task", max(s.cpus_per_task, manifest.qm.nprocs)))
        mem = overrides.get("mem", s.mem)
        partition = overrides.get("partition", s.partition)
        account = overrides.get("account", s.account)
        qos = overrides.get("qos", s.qos)
        gres = overrides.get("gres", s.gres)
        constraint = overrides.get("constraint", s.constraint)

        run_command = self._compose_run_command(remote_workdir)

        replacements = {
            "job_name": _safe_jobname(manifest.name),
            "workdir": str(remote_workdir),
            "time": time_str,
            "nodes": str(nodes),
            "ntasks": str(ntasks),
            "cpus_per_task": str(cpus),
            "mem_line": f"#SBATCH --mem={mem}" if mem else "",
            "partition_line": f"#SBATCH --partition={partition}" if partition else "",
            "account_line": f"#SBATCH --account={account}" if account else "",
            "qos_line": f"#SBATCH --qos={qos}" if qos else "",
            "gres_line": f"#SBATCH --gres={gres}" if gres else "",
            "constraint_line": f"#SBATCH --constraint={constraint}" if constraint else "",
            "extra_directives": "\n".join(s.extra),
            "module_loads": self._module_loads_block(),
            "env_exports": self._env_block(),
            "conda_activate": self._conda_block(),
            "run_command": run_command,
        }
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
        # Plain-string substitution (NOT str.format, to avoid bash brace clashes).
        out = template
        for k, v in replacements.items():
            out = out.replace("{" + k + "}", str(v))
        return out

    def _module_loads_block(self) -> str:
        if not self.profile.modules:
            return "# (no modules)"
        lines = ["module purge || true"]
        for m in self.profile.modules:
            lines.append(f"module load {m}")
        return "\n".join(lines)

    def _env_block(self) -> str:
        if not self.profile.env:
            return "# (no extra env)"
        return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in self.profile.env.items())

    def _conda_block(self) -> str:
        if self.profile.conda_env is None:
            return "# (no conda env)"
        return (
            "# Activate conda env\n"
            'if [ -n "${CONDA_EXE:-}" ]; then\n'
            '  source "$(dirname $(dirname $CONDA_EXE))/etc/profile.d/conda.sh"\n'
            "fi\n"
            f"conda activate {shlex.quote(self.profile.conda_env)}"
        )

    def _compose_run_command(self, remote_workdir: PurePosixPath) -> str:
        manifest_remote = str(remote_workdir / "manifest.yaml")
        runner = (
            f"{self.profile.runner_python} -m qmmmkit.runner "
            f"{shlex.quote(manifest_remote)} {shlex.quote(str(remote_workdir))}"
        )
        if self.profile.apptainer_image:
            return (
                f"apptainer exec --bind {shlex.quote(str(remote_workdir))} "
                f"{shlex.quote(self.profile.apptainer_image)} {runner}"
            )
        return runner


# ------------------------------------------------------------------
# parsers
# ------------------------------------------------------------------
_SBATCH_OK = re.compile(r"Submitted batch job (\d+)")


def _parse_sbatch_jobid(stdout: str) -> str | None:
    for line in stdout.splitlines():
        m = _SBATCH_OK.search(line)
        if m:
            return m.group(1)
    return None


def _parse_exit_code(field: str) -> int | None:
    # sacct gives 'M:N' where M is exit code and N is signal
    if not field:
        return None
    parts = field.split(":")
    try:
        return int(parts[0])
    except ValueError:
        return None


def _safe_jobname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)[:60] or "qmmmkit"
