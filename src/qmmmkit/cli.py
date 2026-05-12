"""qmmmkit command-line entry point.

Subcommands::

    qmmmkit run job.yaml                  # run locally (subprocess), tail log
    qmmmkit submit job.yaml --cluster hpc # submit to a cluster, return job id
    qmmmkit status <handle.json>          # poll a saved handle
    qmmmkit logs <handle.json>            # tail/show the remote log
    qmmmkit fetch <handle.json> --to ./out
    qmmmkit cancel <handle.json>
    qmmmkit clusters list
    qmmmkit clusters add <name> --host ... --user ... --workdir ...
    qmmmkit clusters show <name>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import typer
    from rich import box
    from rich.console import Console
    from rich.table import Table
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "qmmmkit CLI needs `typer` and `rich`. Install with `pip install qmmmkit`."
    ) from e

from .manifest import JobManifest
from .scheduler import (
    ClusterProfile,
    JobHandle,
    JobState,
    LocalScheduler,
    SlurmScheduler,
    list_profiles,
    load_profile,
    save_profile,
)
from .scheduler.profile import SbatchDefaults

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="QM/MM toolkit: run a JobManifest locally or dispatch to SLURM.",
)
clusters_app = typer.Typer(no_args_is_help=True, help="Manage HPC cluster profiles.")
app.add_typer(clusters_app, name="clusters")
console = Console()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _save_handle(handle: JobHandle, path: Path) -> Path:
    path.write_text(json.dumps(handle.to_dict(), indent=2))
    return path


def _load_handle(path: Path) -> JobHandle:
    return JobHandle.from_dict(json.loads(path.read_text()))


def _scheduler_for(handle: JobHandle):
    if handle.cluster:
        return SlurmScheduler(load_profile(handle.cluster))
    return LocalScheduler()


# ----------------------------------------------------------------------
# run / submit / status / logs / fetch / cancel
# ----------------------------------------------------------------------
@app.command()
def run(
    manifest_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Tail log until done."),
) -> None:
    """Run a manifest locally in a subprocess."""
    manifest = JobManifest.load(manifest_path)
    sched = LocalScheduler()
    handle = sched.submit(manifest)
    handle_path = handle.workdir / "handle.json"
    _save_handle(handle, handle_path)
    console.print(f"[green]Submitted local job[/] [bold]{handle.id}[/] (workdir: {handle.workdir})")

    if follow:
        for line in sched.stream_log(handle, follow=True):
            console.print(line, highlight=False)
        handle = sched.poll(handle)
        _save_handle(handle, handle_path)
        _print_terminal(handle)


@app.command()
def submit(
    manifest_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    cluster: str = typer.Option(..., "--cluster", "-c", help="Cluster profile name from clusters.yaml."),
    handle_out: Path = typer.Option(Path("handle.json"), "--handle", help="Where to store the JobHandle."),
) -> None:
    """Submit a manifest to SLURM via SSH."""
    manifest = JobManifest.load(manifest_path)
    profile = load_profile(cluster)
    sched = SlurmScheduler(profile)
    handle = sched.submit(manifest)
    _save_handle(handle, handle_out)
    console.print(
        f"[green]Submitted SLURM job[/] [bold]{handle.id}[/] on [cyan]{profile.name}[/]\n"
        f"  remote workdir: {handle.workdir}\n  handle: {handle_out}"
    )


@app.command()
def status(handle_path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Refresh a job handle and print its state."""
    handle = _load_handle(handle_path)
    sched = _scheduler_for(handle)
    handle = sched.poll(handle)
    _save_handle(handle, handle_path)
    _print_terminal(handle)


@app.command()
def logs(
    handle_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    follow: bool = typer.Option(True, "--follow/--no-follow"),
) -> None:
    """Tail (or print) the job's log."""
    handle = _load_handle(handle_path)
    sched = _scheduler_for(handle)
    for line in sched.stream_log(handle, follow=follow):
        console.print(line, highlight=False)


@app.command()
def fetch(
    handle_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    to: Path = typer.Option(Path("./results"), "--to", "-o", help="Local destination directory."),
) -> None:
    """Download the remote workdir to a local directory."""
    handle = _load_handle(handle_path)
    sched = _scheduler_for(handle)
    out = sched.fetch_results(handle, to)
    console.print(f"[green]Fetched results to[/] {out}")


@app.command()
def cancel(handle_path: Path = typer.Argument(..., exists=True, dir_okay=False)) -> None:
    """Cancel a running job."""
    handle = _load_handle(handle_path)
    sched = _scheduler_for(handle)
    handle = sched.cancel(handle)
    _save_handle(handle, handle_path)
    console.print(f"[yellow]Cancelled[/] {handle.id} ({handle.cluster or 'local'})")


# ----------------------------------------------------------------------
# clusters
# ----------------------------------------------------------------------
@clusters_app.command("list")
def clusters_list() -> None:
    names = list_profiles()
    if not names:
        console.print("[dim]No clusters configured.[/] Use `qmmmkit clusters add <name>`.")
        return
    t = Table(box=box.SIMPLE_HEAVY)
    t.add_column("name", style="bold cyan")
    t.add_column("host")
    t.add_column("user")
    t.add_column("workdir")
    for name in names:
        p = load_profile(name)
        t.add_row(name, p.host, p.user, p.workdir)
    console.print(t)


@clusters_app.command("show")
def clusters_show(name: str) -> None:
    p = load_profile(name)
    console.print(p.to_dict())


@clusters_app.command("add")
def clusters_add(
    name: str,
    host: str = typer.Option(..., "--host"),
    user: str = typer.Option(..., "--user"),
    workdir: str = typer.Option(..., "--workdir"),
    port: int = typer.Option(22, "--port"),
    identity_file: Optional[str] = typer.Option(None, "--identity-file"),
    partition: Optional[str] = typer.Option(None, "--partition"),
    account: Optional[str] = typer.Option(None, "--account"),
    time_limit: str = typer.Option("04:00:00", "--time"),
    cpus_per_task: int = typer.Option(4, "--cpus-per-task"),
    mem: Optional[str] = typer.Option("16G", "--mem"),
    gres: Optional[str] = typer.Option(None, "--gres"),
    conda_env: Optional[str] = typer.Option(None, "--conda-env"),
    apptainer_image: Optional[str] = typer.Option(None, "--apptainer-image"),
) -> None:
    """Create or overwrite a cluster profile in ~/.config/qmmmkit/clusters.yaml."""
    profile = ClusterProfile(
        name=name, host=host, user=user, workdir=workdir, port=port,
        identity_file=identity_file,
        sbatch=SbatchDefaults(
            account=account, partition=partition, time=time_limit,
            cpus_per_task=cpus_per_task, mem=mem, gres=gres,
        ),
        conda_env=conda_env, apptainer_image=apptainer_image,
    )
    path = save_profile(profile)
    console.print(f"[green]Saved cluster profile[/] [bold]{name}[/] -> {path}")


# ----------------------------------------------------------------------
def _print_terminal(handle: JobHandle) -> None:
    color = {
        JobState.COMPLETED: "green",
        JobState.RUNNING: "yellow",
        JobState.PENDING: "yellow",
        JobState.SUBMITTED: "yellow",
        JobState.FAILED: "red",
        JobState.CANCELLED: "magenta",
        JobState.UNKNOWN: "dim",
    }.get(handle.state, "white")
    console.print(
        f"\n[{color}]state[/]: [bold]{handle.state.value}[/]  "
        f"id={handle.id}  cluster={handle.cluster or 'local'}  "
        f"exit={handle.exit_code}"
    )


if __name__ == "__main__":
    app()
