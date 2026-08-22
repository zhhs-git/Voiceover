import os
import stat
import subprocess
from pathlib import Path

import pytest

from audiobook_worker import voxcpm2_service_supervisor as supervisor_module
from audiobook_worker.voxcpm2_service_protocol import BATCH_ADAPTER_VERSION
from audiobook_worker.voxcpm2_service_supervisor import VoxCPM2ServiceSupervisor


class _FakeProcess:
    def __init__(self, process_id: int) -> None:
        self.pid = process_id
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-voxcpm2", timeout)
        return self.returncode


def _short_test_socket_directory() -> Path:
    directory = Path(__file__).resolve().parents[4] / ".test-voxcpm2-supervisor"
    directory.mkdir(mode=0o700, exist_ok=True)
    return directory


def _configure_supervisor_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[_FakeProcess], list[tuple[str, str]]]:
    isolated_python = tmp_path / "isolated-python"
    isolated_python.write_text("#! fake", encoding="utf-8")
    model_path = tmp_path / "VoxCPM2"
    model_path.mkdir()
    monkeypatch.setattr(
        supervisor_module,
        "voxcpm2_paths",
        lambda _root=None: {"python": isolated_python, "model": model_path},
    )
    socket_directory = _short_test_socket_directory()
    monkeypatch.setattr(
        VoxCPM2ServiceSupervisor,
        "_short_socket_directory",
        staticmethod(lambda: socket_directory),
    )
    monkeypatch.setattr(supervisor_module, "_STOP_TIMEOUT_SECONDS", 0.0)

    processes: list[_FakeProcess] = []

    def fake_popen(*_args, **_kwargs):
        process = _FakeProcess(10_000 + len(processes))
        processes.append(process)
        return process

    calls: list[tuple[str, str]] = []

    def fake_request_service(endpoint, operation, payload=None):
        del payload
        calls.append((str(endpoint.socket_path), operation))
        return {"status": "succeeded", "adapterVersion": BATCH_ADAPTER_VERSION}

    monkeypatch.setattr(supervisor_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(supervisor_module, "request_service", fake_request_service)
    return processes, calls


def test_supervisor_reuses_one_healthy_service_and_cleans_its_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    processes, calls = _configure_supervisor_fakes(tmp_path, monkeypatch)
    supervisor = VoxCPM2ServiceSupervisor(tmp_path / "data", configuration_root=tmp_path)
    try:
        first = supervisor.worker_environment()
        second = supervisor.worker_environment()

        assert first == second
        assert len(processes) == 1
        socket_path = Path(first["AUDIOBOOK_VOXCPM_SERVICE_SOCKET"])
        assert len(os.fsencode(socket_path)) < supervisor_module._UNIX_SOCKET_MAX_PATH_BYTES
        assert socket_path.parent == _short_test_socket_directory()
        assert supervisor._token_path is not None
        assert stat.S_IMODE(supervisor._token_path.stat().st_mode) == 0o600
        assert "ping" in [operation for _socket, operation in calls]
        token_path = supervisor._token_path
    finally:
        supervisor.close()
    assert processes[0].terminate_calls == 1
    assert not token_path.exists()


def test_supervisor_restarts_after_its_owned_service_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    processes, _calls = _configure_supervisor_fakes(tmp_path, monkeypatch)
    supervisor = VoxCPM2ServiceSupervisor(tmp_path / "data", configuration_root=tmp_path)
    try:
        first = supervisor.worker_environment()
        processes[0].returncode = 1
        second = supervisor.worker_environment()

        assert len(processes) == 2
        assert first["AUDIOBOOK_VOXCPM_SERVICE_SOCKET"] != second["AUDIOBOOK_VOXCPM_SERVICE_SOCKET"]
        assert first["AUDIOBOOK_VOXCPM_SERVICE_TOKEN"] != second["AUDIOBOOK_VOXCPM_SERVICE_TOKEN"]
    finally:
        supervisor.close()
