"""Client transport for the private resident VoxCPM2 Unix-socket service."""

from __future__ import annotations

import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiobook_worker.voxcpm2_service_protocol import (
    PROTOCOL_VERSION,
    VoxCPM2ServiceProtocolError,
    encode_json_line,
    read_json_line,
)


_SOCKET_ENV = "AUDIOBOOK_VOXCPM_SERVICE_SOCKET"
_TOKEN_ENV = "AUDIOBOOK_VOXCPM_SERVICE_TOKEN"
_TIMEOUT_ENV = "AUDIOBOOK_VOXCPM_SERVICE_TIMEOUT_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 60 * 60


class VoxCPM2ServiceClientError(RuntimeError):
    """The isolated service cannot satisfy a regular-worker request."""


@dataclass(frozen=True)
class VoxCPM2ServiceEndpoint:
    socket_path: Path
    token: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


def endpoint_from_environment(
    environment: dict[str, str] | None = None,
) -> VoxCPM2ServiceEndpoint | None:
    """Return the ephemeral endpoint injected by :class:`ServerState`.

    A standalone worker never receives both values and therefore keeps the
    existing one-shot runner fallback.
    """

    values = environment if environment is not None else os.environ
    raw_path = str(values.get(_SOCKET_ENV) or "").strip()
    token = str(values.get(_TOKEN_ENV) or "").strip()
    if not raw_path and not token:
        return None
    if not raw_path or not token:
        raise VoxCPM2ServiceClientError("VoxCPM2 service endpoint is incomplete.")
    try:
        configured_timeout = float(values.get(_TIMEOUT_ENV, _DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        configured_timeout = _DEFAULT_TIMEOUT_SECONDS
    timeout = max(60.0, min(float(_DEFAULT_TIMEOUT_SECONDS), configured_timeout))
    return VoxCPM2ServiceEndpoint(Path(raw_path), token, timeout)


def request_service(
    endpoint: VoxCPM2ServiceEndpoint,
    operation: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform one correlated request without exposing the capability token."""

    request_id = uuid.uuid4().hex
    message: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "token": endpoint.token,
        "requestId": request_id,
        "operation": operation,
    }
    if payload:
        message.update(payload)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(endpoint.timeout_seconds)
            client.connect(str(endpoint.socket_path))
            client.sendall(encode_json_line(message))
            with client.makefile("rb") as stream:
                response = read_json_line(stream)
    except (OSError, TimeoutError, VoxCPM2ServiceProtocolError) as error:
        raise VoxCPM2ServiceClientError(f"VoxCPM2 service is unavailable: {error}") from error
    if response.get("requestId") != request_id:
        raise VoxCPM2ServiceClientError("VoxCPM2 service returned a mismatched response id.")
    if response.get("status") != "succeeded":
        error = response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            raise VoxCPM2ServiceClientError(f"VoxCPM2 service failed: {error['message']}")
        raise VoxCPM2ServiceClientError("VoxCPM2 service failed without an error message.")
    return response


def synthesize_with_service(
    endpoint: VoxCPM2ServiceEndpoint,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Submit one chapter's missing segment subset to the resident model."""

    return request_service(endpoint, "synthesize", payload)
