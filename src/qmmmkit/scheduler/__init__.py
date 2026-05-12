"""Scheduler abstraction: run a JobManifest locally or dispatch to SLURM/etc.

Standard usage:

    from qmmmkit.manifest import JobManifest
    from qmmmkit.scheduler import LocalScheduler, SlurmScheduler, ClusterProfile

    manifest = JobManifest.load("job.yaml")
    handle = LocalScheduler().submit(manifest)            # in-process
    # OR
    profile = ClusterProfile.load("hpc")
    handle = SlurmScheduler(profile).submit(manifest)     # SSH + sbatch

    while not handle.state.is_terminal():
        handle = scheduler.poll(handle)
    scheduler.fetch_results(handle, dest="./results")
"""

from .base import JobHandle, JobState, Scheduler, SchedulerError
from .local import LocalScheduler
from .profile import ClusterProfile, list_profiles, load_profile, save_profile
from .slurm import SlurmScheduler

__all__ = [
    "JobHandle",
    "JobState",
    "Scheduler",
    "SchedulerError",
    "LocalScheduler",
    "SlurmScheduler",
    "ClusterProfile",
    "list_profiles",
    "load_profile",
    "save_profile",
]
