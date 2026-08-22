"""Web-process supervisor for the isolated resident VoxCPM2 service."""

from __future__ import annotations

import os
import secrets
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from audiobook_worker.model_settings import voxcpm2_paths
from audiobook_worker.voxcpm2_service_client import (
    VoxCPM2ServiceEndpoint,
    request_service,
)
from audiobook_worker.voxcpm2_service_protocol import BATCH_ADAPTER_VERSION


_START_TIMEOUT_SECONDS = 15.0
_STOP_TIMEOUT_SECONDS = 5.0
# Darwin reserves one byte for the terminating NUL in ``sockaddr_un.sun_path``.
# Keep a margin below the 104-byte platform limit instead of relying on a
# potentially long ``TMPDIR`` (which is commonly under ``/var/folders``).
_UNIX_SOCKET_MAX_PATH_BYTES = 103
_SHORT_SOCKET_DIRECTORY_PREFIX = "audiobook-voxcpm2-"


class VoxCPM2ServiceSupervisorError(RuntimeError):
    """The web process cannot safely provide a resident local model service."""


@dataclass(frozen=True)
class VoxCPM2ServiceRuntime:
    socket_path: Path
    token: str
    process_id: int


class VoxCPM2ServiceSupervisor:
    """Own exactly one lazily started service child for one ``ServerState``."""

    def __init__(self, data_directory: Path, *, configuration_root: Path | None = None) -> None:
        self.data_directory = data_directory.resolve()
        self.configuration_root = configuration_root.resolve() if configuration_root else None
        self.runtime_directory = self.data_directory / ".runtime"
        self.log_path = self.data_directory / "voxcpm2-batch-service.log"
        self.selection_path = self.runtime_directory / "voxcpm2-benchmark-selection.json"
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._endpoint: VoxCPM2ServiceEndpoint | None = None
        self._socket_path: Path | None = None
        self._token_path: Path | None = None

    def worker_environment(self) -> dict[str, str]:
        """Start/reuse the service and return only worker-child capabilities."""

        runtime = self.ensure_running()
        return {
            "AUDIOBOOK_VOXCPM_SERVICE_SOCKET": str(runtime.socket_path),
            "AUDIOBOOK_VOXCPM_SERVICE_TOKEN": runtime.token,
        }

    def ensure_running(self) -> VoxCPM2ServiceRuntime:
        with self._lock:
            if self._is_healthy_locked():
                assert self._endpoint is not None
                assert self._process is not None
                return VoxCPM2ServiceRuntime(
                    self._endpoint.socket_path,
                    self._endpoint.token,
                    int(self._process.pid),
                )
            self._discard_stale_locked()
            paths = voxcpm2_paths(self.configuration_root)
            runner_python = paths["python"]
            model_path = paths["model"]
            if not runner_python.is_file():
                raise VoxCPM2ServiceSupervisorError(
                    f"VoxCPM2 isolated Python is missing: {runner_python}"
                )
            if not model_path.is_dir():
                raise VoxCPM2ServiceSupervisorError(
                    f"VoxCPM2 model directory is missing: {model_path}"
                )
            service_path = Path(__file__).with_name("voxcpm2_batch_service.py")
            if not service_path.is_file():
                raise VoxCPM2ServiceSupervisorError(
                    f"VoxCPM2 batch service is missing: {service_path}"
                )
            self.runtime_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            token = secrets.token_urlsafe(32)
            suffix = secrets.token_hex(5)
            socket_path = self._new_socket_path(suffix)
            token_path = self.runtime_directory / f".voxcpm2-{suffix}.token"
            self._write_token_file(token_path, token)
            self._token_path = token_path
            endpoint = VoxCPM2ServiceEndpoint(socket_path, token, _START_TIMEOUT_SECONDS)
            command = [
                str(runner_python),
                str(service_path),
                "--socket",
                str(socket_path),
                "--token-file",
                str(token_path),
                "--idle-seconds",
                str(self._idle_seconds()),
                "--selection-path",
                str(self.selection_path),
            ]
            try:
                with self.log_path.open("ab") as log_file:
                    process = subprocess.Popen(
                        command,
                        cwd=Path(__file__).resolve().parents[1],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                    )
            except Exception:
                token_path.unlink(missing_ok=True)
                self._token_path = None
                raise
            self._process = process
            self._endpoint = endpoint
            self._socket_path = socket_path
            try:
                self._wait_until_healthy_locked()
            except Exception:
                self._terminate_locked()
                raise
            return VoxCPM2ServiceRuntime(socket_path, token, int(process.pid))

    def close(self) -> None:
        """Stop only the child process this supervisor started, if still alive."""

        with self._lock:
            endpoint = self._endpoint
            process = self._process
            if endpoint is not None and process is not None and process.poll() is None:
                try:
                    request_service(endpoint, "shutdown")
                except Exception:
                    pass
                deadline = time.monotonic() + _STOP_TIMEOUT_SECONDS
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
            self._terminate_locked()

    def _is_healthy_locked(self) -> bool:
        process = self._process
        endpoint = self._endpoint
        if process is None or endpoint is None or process.poll() is not None:
            return False
        try:
            response = request_service(endpoint, "ping")
        except Exception:
            return False
        return response.get("adapterVersion") == BATCH_ADAPTER_VERSION

    def _wait_until_healthy_locked(self) -> None:
        assert self._process is not None
        assert self._endpoint is not None
        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                detail = self._log_tail()
                raise VoxCPM2ServiceSupervisorError(
                    "VoxCPM2 batch service exited during startup"
                    + (f": {detail}" if detail else "")
                )
            try:
                response = request_service(self._endpoint, "ping")
                if response.get("adapterVersion") == BATCH_ADAPTER_VERSION:
                    return
                last_error = VoxCPM2ServiceSupervisorError(
                    "VoxCPM2 batch service reported an incompatible adapter version."
                )
            except Exception as error:
                last_error = error
            time.sleep(0.05)
        detail = self._log_tail()
        suffix = f": {detail}" if detail else ""
        raise VoxCPM2ServiceSupervisorError(
            f"VoxCPM2 batch service did not become ready within {_START_TIMEOUT_SECONDS:g} seconds{suffix}"
        ) from last_error

    def _discard_stale_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._terminate_locked()
        else:
            self._terminate_locked()

    def _terminate_locked(self) -> None:
        process = self._process
        socket_path = self._socket_path
        token_path = self._token_path
        self._process = None
        self._endpoint = None
        self._socket_path = None
        self._token_path = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=_STOP_TIMEOUT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=_STOP_TIMEOUT_SECONDS)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        if socket_path is not None:
            self._unlink_owned_socket(socket_path)
        if token_path is not None:
            token_path.unlink(missing_ok=True)

    @staticmethod
    def _short_socket_directory() -> Path:
        """Return a private, short path suitable for macOS Unix sockets.

        The normal audiobook data directory intentionally has a descriptive
        name under ``~/Library/Application Support``.  That is too long for a
        Darwin Unix-domain socket once the per-service random name is added.
        ``/tmp`` itself is world-writable, so use an owner-only child directory
        and refuse a path owned by anyone else.
        """

        directory = Path("/tmp") / f"{_SHORT_SOCKET_DIRECTORY_PREFIX}{os.getuid()}"
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        try:
            status = directory.lstat()
        except OSError as error:
            raise VoxCPM2ServiceSupervisorError(
                f"Cannot inspect VoxCPM2 socket directory: {directory}"
            ) from error
        if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
            raise VoxCPM2ServiceSupervisorError(
                f"VoxCPM2 socket path is not a private directory: {directory}"
            )
        if status.st_uid != os.getuid():
            raise VoxCPM2ServiceSupervisorError(
                f"VoxCPM2 socket directory is not owned by the current user: {directory}"
            )
        if stat.S_IMODE(status.st_mode) & 0o077:
            try:
                directory.chmod(0o700)
            except OSError as error:
                raise VoxCPM2ServiceSupervisorError(
                    f"Cannot secure VoxCPM2 socket directory: {directory}"
                ) from error
        return directory

    def _new_socket_path(self, suffix: str) -> Path:
        directory = self._short_socket_directory()
        for _ in range(16):
            candidate = directory / f"service-{os.getpid()}-{suffix}.sock"
            if len(os.fsencode(candidate)) >= _UNIX_SOCKET_MAX_PATH_BYTES:
                raise VoxCPM2ServiceSupervisorError(
                    f"VoxCPM2 Unix socket path is too long: {candidate}"
                )
            try:
                candidate.lstat()
            except FileNotFoundError:
                return candidate
            suffix = secrets.token_hex(5)
        raise VoxCPM2ServiceSupervisorError("Cannot allocate a unique VoxCPM2 socket path.")

    @staticmethod
    def _write_token_file(path: Path, token: str) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, token.encode("utf-8"))
        finally:
            os.close(descriptor)

    @staticmethod
    def _unlink_owned_socket(path: Path) -> None:
        try:
            status = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(status.st_mode):
            path.unlink(missing_ok=True)

    def _idle_seconds(self) -> int:
        try:
            value = int(str(os.environ.get("AUDIOBOOK_VOXCPM_IDLE_SECONDS", "300")).strip())
        except ValueError:
            value = 300
        return max(0, min(3600, value))

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-1000:].strip().splitlines()[-1]
        except (OSError, IndexError):
            return ""
