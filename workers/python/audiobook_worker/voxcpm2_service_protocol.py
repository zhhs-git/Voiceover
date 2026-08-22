"""Private JSON-line protocol for the resident VoxCPM2 service.

The regular audiobook worker and the model service intentionally run in
different Python environments.  This module contains only standard-library
code so both environments can validate the same small, versioned contract
without importing ``voxcpm`` into the regular worker.
"""

from __future__ import annotations

import hmac
import json
from typing import Any, BinaryIO


PROTOCOL_VERSION = 1
PROMPT_FORMAT_VERSION = 2
BATCH_ADAPTER_VERSION = 2
MAX_MESSAGE_BYTES = 8 * 1024 * 1024
_MAX_REQUEST_ID_LENGTH = 160


class VoxCPM2ServiceProtocolError(RuntimeError):
    """A service message is malformed, unauthorized, or unsupported."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VoxCPM2ServiceProtocolError(f"{name} must be an object.")
    return value


def _text(value: object, *, name: str, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise VoxCPM2ServiceProtocolError(f"{name} is required.")
    return result


def _list_of_objects(value: object, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise VoxCPM2ServiceProtocolError(f"{name} must be an array.")
    return [_object(item, name=f"{name} item") for item in value]


def decode_json_line(raw: bytes) -> dict[str, Any]:
    """Decode one bounded newline-delimited JSON object."""

    if not raw:
        raise VoxCPM2ServiceProtocolError("service connection closed", code="connection_closed")
    if len(raw) > MAX_MESSAGE_BYTES:
        raise VoxCPM2ServiceProtocolError("service message is too large", code="message_too_large")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VoxCPM2ServiceProtocolError("service message is not valid JSON") from error
    return _object(decoded, name="service message")


def read_json_line(stream: BinaryIO) -> dict[str, Any]:
    """Read exactly one bounded protocol record from a socket file object."""

    raw = stream.readline(MAX_MESSAGE_BYTES + 1)
    if len(raw) > MAX_MESSAGE_BYTES:
        raise VoxCPM2ServiceProtocolError("service message is too large", code="message_too_large")
    return decode_json_line(raw)


def encode_json_line(payload: dict[str, Any]) -> bytes:
    """Serialize a compact protocol record without allowing multi-line frames."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VoxCPM2ServiceProtocolError("service response is not JSON serializable") from error
    if len(encoded) > MAX_MESSAGE_BYTES:
        raise VoxCPM2ServiceProtocolError("service message is too large", code="message_too_large")
    return encoded + b"\n"


def validate_request(
    payload: object,
    *,
    expected_token: str | None = None,
) -> dict[str, Any]:
    """Validate and return a copy of one client request.

    The token is validated only inside the local service and deliberately never
    copied into logs or responses.  Validation here is structural; the service
    owns runtime checks such as readable files and model compatibility.
    """

    request = dict(_object(payload, name="service request"))
    version = request.get("version")
    if version != PROTOCOL_VERSION:
        raise VoxCPM2ServiceProtocolError(
            f"unsupported VoxCPM2 service protocol version: {version!r}",
            code="unsupported_version",
        )
    request_id = _text(request.get("requestId"), name="requestId")
    if len(request_id) > _MAX_REQUEST_ID_LENGTH:
        raise VoxCPM2ServiceProtocolError("requestId is too long")
    request["requestId"] = request_id

    token = _text(request.get("token"), name="token")
    if expected_token is not None and not hmac.compare_digest(token, expected_token):
        raise VoxCPM2ServiceProtocolError("invalid service token", code="invalid_token")

    operation = _text(request.get("operation"), name="operation").casefold()
    if operation not in {"ping", "shutdown", "cancel", "synthesize"}:
        raise VoxCPM2ServiceProtocolError(
            f"unsupported service operation: {operation}", code="unsupported_operation"
        )
    request["operation"] = operation
    if operation != "synthesize":
        return request

    if request.get("promptFormatVersion") != PROMPT_FORMAT_VERSION:
        raise VoxCPM2ServiceProtocolError(
            f"promptFormatVersion must be {PROMPT_FORMAT_VERSION}",
            code="invalid_prompt_format",
        )
    _text(request.get("modelPath"), name="modelPath")
    device = _text(request.get("device"), name="device").casefold()
    if device not in {"auto", "mps", "cpu"}:
        raise VoxCPM2ServiceProtocolError("unsupported VoxCPM2 device")
    chapter = request.get("chapter", {})
    _object(chapter, name="chapter")
    profiles = _list_of_objects(request.get("profiles", []), name="profiles")
    segments = _list_of_objects(request.get("segments", []), name="segments")
    if not profiles and not segments:
        raise VoxCPM2ServiceProtocolError("synthesize request has no profiles or segments")

    profile_paths: set[str] = set()
    for profile in profiles:
        _text(profile.get("voiceId"), name="profile voiceId")
        profile_path = _text(profile.get("profilePath"), name="profilePath")
        _text(profile.get("metadataPath"), name="metadataPath")
        _text(profile.get("lockPath"), name="lockPath")
        _text(profile.get("signature"), name="profile signature")
        _text(profile.get("voiceDesign"), name="voiceDesign")
        _text(profile.get("profileControl"), name="profileControl")
        _text(profile.get("referenceText"), name="referenceText")
        _text(profile.get("language"), name="profile language")
        if profile.get("promptFormatVersion") != PROMPT_FORMAT_VERSION:
            raise VoxCPM2ServiceProtocolError("profile promptFormatVersion is invalid")
        if profile_path in profile_paths:
            raise VoxCPM2ServiceProtocolError("duplicate profilePath in request")
        profile_paths.add(profile_path)

    segment_ids: set[str] = set()
    output_paths: set[str] = set()
    for segment in segments:
        segment_id = _text(segment.get("id"), name="segment id")
        _text(segment.get("text"), name=f"segment text for {segment_id}")
        output_path = _text(segment.get("outputPath"), name=f"outputPath for {segment_id}")
        _text(segment.get("referenceWavPath"), name=f"referenceWavPath for {segment_id}")
        _text(segment.get("language"), name=f"segment language for {segment_id}")
        if segment.get("promptFormatVersion") != PROMPT_FORMAT_VERSION:
            raise VoxCPM2ServiceProtocolError(
                f"segment promptFormatVersion is invalid for {segment_id}"
            )
        if segment_id in segment_ids:
            raise VoxCPM2ServiceProtocolError("duplicate segment id in request")
        if output_path in output_paths:
            raise VoxCPM2ServiceProtocolError("duplicate segment outputPath in request")
        raw_position = segment.get("sourcePosition")
        if isinstance(raw_position, bool):
            raise VoxCPM2ServiceProtocolError(
                f"sourcePosition must be a non-negative integer for {segment_id}"
            )
        try:
            source_position = int(raw_position)
        except (TypeError, ValueError) as error:
            raise VoxCPM2ServiceProtocolError(
                f"sourcePosition must be a non-negative integer for {segment_id}"
            ) from error
        if source_position < 0:
            raise VoxCPM2ServiceProtocolError(
                f"sourcePosition must be a non-negative integer for {segment_id}"
            )
        raw_source_ids = segment.get("sourceSegmentIds")
        if not isinstance(raw_source_ids, list) or not raw_source_ids:
            raise VoxCPM2ServiceProtocolError(
                f"sourceSegmentIds must be a non-empty array for {segment_id}"
            )
        for source_id in raw_source_ids:
            _text(source_id, name=f"sourceSegmentIds item for {segment_id}")
        # The direct-TTS compatibility route has no durable segment cache, so
        # it may use an empty signature.  The field remains part of every
        # payload and cache-backed chapter work always supplies the real one.
        _text(segment.get("cacheSignature"), name=f"cacheSignature for {segment_id}", required=False)
        segment_ids.add(segment_id)
        output_paths.add(output_path)
    return request


def success_response(request_id: str, **payload: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "requestId": request_id,
        "status": "succeeded",
        **payload,
    }


def error_response(request_id: str, error: Exception | str, *, code: str = "service_error") -> dict[str, Any]:
    if isinstance(error, VoxCPM2ServiceProtocolError):
        code = error.code
    message = str(error).replace("\n", " ").strip()[:1000] or "VoxCPM2 service failed"
    return {
        "version": PROTOCOL_VERSION,
        "requestId": request_id,
        "status": "failed",
        "error": {"code": code, "message": message},
    }
