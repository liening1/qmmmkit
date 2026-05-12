"""SSH/SFTP transport used by SlurmScheduler.

Thin wrapper around paramiko so the scheduler stays readable. Uses the
user's ssh-agent or an explicit identity file from ClusterProfile.
"""

from __future__ import annotations

import io
import os
import shlex
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .profile import ClusterProfile


@dataclass
class CommandResult:
    rc: int
    stdout: str
    stderr: str

    def check(self, message: str = "") -> "CommandResult":
        if self.rc != 0:
            raise RuntimeError(
                f"{message or 'remote command failed'} (rc={self.rc}): {self.stderr.strip() or self.stdout.strip()}"
            )
        return self


class SSHTransport:
    """A thin facade over paramiko.SSHClient + sftp."""

    def __init__(self, profile: ClusterProfile) -> None:
        self.profile = profile
        self._client = None
        self._sftp = None

    # ------------------------------------------------------------
    def connect(self) -> None:
        try:
            import paramiko
        except ImportError as e:
            raise ImportError(
                "SLURM submission needs paramiko. Install with `pip install qmmmkit[slurm]`."
            ) from e

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {
            "hostname": self.profile.host,
            "username": self.profile.user,
            "port": self.profile.port,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if self.profile.identity_file:
            kwargs["key_filename"] = str(Path(self.profile.identity_file).expanduser())
        client.connect(**kwargs)
        self._client = client
        self._sftp = client.open_sftp()

    def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            finally:
                self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    @contextmanager
    def session(self):
        self.connect()
        try:
            yield self
        finally:
            self.close()

    # ------------------------------------------------------------
    def run(self, command: str, *, timeout: float | None = None) -> CommandResult:
        if self._client is None:
            raise RuntimeError("SSHTransport.run called before connect().")
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return CommandResult(rc=rc, stdout=out, stderr=err)

    # ------------------------------------------------------------
    def mkdir_p(self, path: str | PurePosixPath) -> None:
        self.run(f"mkdir -p {shlex.quote(str(path))}").check(f"mkdir -p {path}")

    def put_text(self, content: str, remote_path: str | PurePosixPath, *, mode: int = 0o644) -> None:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        with self._sftp.open(str(remote_path), "w") as f:
            f.write(content)
        self._sftp.chmod(str(remote_path), mode)

    def put_file(self, local: Path, remote_path: str | PurePosixPath) -> None:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        self._sftp.put(str(local), str(remote_path))

    def get_file(self, remote_path: str | PurePosixPath, local: Path) -> None:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        local.parent.mkdir(parents=True, exist_ok=True)
        self._sftp.get(str(remote_path), str(local))

    def read_text(self, remote_path: str | PurePosixPath) -> str:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        with self._sftp.open(str(remote_path), "r") as f:
            return f.read().decode("utf-8", errors="replace")

    def listdir(self, remote_path: str | PurePosixPath) -> list[str]:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        return list(self._sftp.listdir(str(remote_path)))

    def walk(self, remote_root: str | PurePosixPath) -> Iterable[tuple[str, list[str], list[str]]]:
        """Recursive listdir, yielding (path, dirs, files) like os.walk."""
        if self._sftp is None:
            raise RuntimeError("connect() first")
        stack = [str(remote_root)]
        while stack:
            here = stack.pop()
            dirs, files = [], []
            for entry in self._sftp.listdir_attr(here):
                full = f"{here}/{entry.filename}"
                if stat.S_ISDIR(entry.st_mode):
                    dirs.append(entry.filename)
                    stack.append(full)
                else:
                    files.append(entry.filename)
            yield here, dirs, files

    def exists(self, remote_path: str | PurePosixPath) -> bool:
        if self._sftp is None:
            raise RuntimeError("connect() first")
        try:
            self._sftp.stat(str(remote_path))
            return True
        except IOError:
            return False
