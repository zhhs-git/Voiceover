"""Lazy resident VoxCPM2 service for cross-chapter batched synthesis.

This program is launched only by :class:`audiobook_worker.web_server.ServerState`
through the isolated VoxCPM2 virtual environment.  It accepts authenticated
local Unix-socket requests from regular worker children, owns exactly one model
instance, and returns each result under its original request and segment id.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import stat
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__:
    from .voxcpm2_batch_adapter import (
        BATCH_ADAPTER_VERSION,
        MAX_BATCH_SIZE,
        BatchAudioResult,
        VoxCPM2BatchAdapter,
        VoxCPM2BatchAdapterError,
    )
    from .voxcpm2_runner import _atomic_write_wav, _ensure_profile
    from .voxcpm2_service_protocol import (
        encode_json_line,
        error_response,
        read_json_line,
        success_response,
        validate_request,
    )
else:  # pragma: no cover - executed by the isolated direct-script runtime.
    from voxcpm2_batch_adapter import (  # type: ignore[no-redef]
        BATCH_ADAPTER_VERSION,
        MAX_BATCH_SIZE,
        BatchAudioResult,
        VoxCPM2BatchAdapter,
        VoxCPM2BatchAdapterError,
    )
    from voxcpm2_runner import _atomic_write_wav, _ensure_profile  # type: ignore[no-redef]
    from voxcpm2_service_protocol import (  # type: ignore[no-redef]
        encode_json_line,
        error_response,
        read_json_line,
        success_response,
        validate_request,
    )


_DISPATCH_WINDOW_SECONDS = 0.05
_MAX_IDLE_SECONDS = 60 * 60
# A cache-only chapter can cause the web process to hand a worker an endpoint
# even though that worker never submits ``synthesize``. Keep a zero-idle
# service alive long enough for the supervisor ping and worker handoff, but do
# not leave such a no-work process resident forever.
_NO_SYNTHESIS_STARTUP_GRACE_SECONDS = 10.0


def _bounded_idle_seconds(value: object) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = 300
    return max(0, min(_MAX_IDLE_SECONDS, parsed))


def _readable_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise VoxCPM2BatchAdapterError(f"VoxCPM2 wrote an unreadable WAV: {path}") from error
    if frames <= 0 or sample_rate <= 0:
        raise VoxCPM2BatchAdapterError(f"VoxCPM2 wrote an empty WAV: {path}")
    return frames / sample_rate


def _length_bucket(segment: dict[str, Any]) -> int:
    """Use a stable coarse bucket before collating tensors with left padding."""

    text = str(segment.get("text") or "")
    # Chinese characters and words both produce roughly proportional token
    # counts here; exact model tokenization stays inside the adapter.
    return min(12, max(0, len(text.strip()) // 12))


@dataclass
class _ServiceRequest:
    payload: dict[str, Any]
    received_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    profile_results: list[dict[str, Any]] = field(default_factory=list)
    ready: list["_ReadySegment"] = field(default_factory=list)
    results: dict[int, dict[str, Any]] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)
    prepared: bool = False
    fairness_skips: int = 0
    dispatches: int = 0
    fallback_count: int = 0
    maximum_effective_batch: int = 0

    @property
    def request_id(self) -> str:
        return str(self.payload["requestId"])

    @property
    def original_segments(self) -> list[dict[str, Any]]:
        raw = self.payload.get("segments")
        return raw if isinstance(raw, list) else []

    @property
    def chapter_key(self) -> str:
        chapter = self.payload.get("chapter")
        if isinstance(chapter, dict):
            book_id = str(chapter.get("bookId") or "")
            chapter_id = str(chapter.get("chapterId") or "")
            if book_id or chapter_id:
                return f"{book_id}\u0000{chapter_id}"
        return self.request_id

    def is_complete(self) -> bool:
        return len(self.results) + len(self.failures) >= len(self.original_segments)


@dataclass(eq=False)
class _ReadySegment:
    request: _ServiceRequest
    position: int
    segment: dict[str, Any]
    retry_count: int = 0
    force_single: bool = False

    @property
    def segment_id(self) -> str:
        return str(self.segment["id"])

    @property
    def bucket(self) -> int:
        return _length_bucket(self.segment)


class VoxCPM2BatchService:
    """Own one isolated model process and a bounded fair inference queue."""

    def __init__(
        self,
        socket_path: Path,
        token: str,
        *,
        idle_seconds: int = 300,
        selection_path: Path | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.token = token
        self.idle_seconds = _bounded_idle_seconds(idle_seconds)
        self.selection_path = selection_path
        self._listener: socket.socket | None = None
        self._incoming: queue.Queue[_ServiceRequest] = queue.Queue()
        self._requests: dict[str, _ServiceRequest] = {}
        # Connection threads only enqueue synthesis work, except cancellation.
        # Guard request-state mutation so a cancellation can take effect only
        # at a batch boundary, never while the adapter is walking a ready list.
        self._state_lock = threading.RLock()
        self._shutdown_requested = threading.Event()
        self._last_activity = time.monotonic()
        # An explicit idle value of zero means "release immediately after the
        # first completed request", not "exit before the supervisor can send
        # its startup health probe and hand the endpoint to a worker". A
        # cache-only worker may never submit a synthesis request, so that path
        # has a small bounded startup grace rather than an infinite lifetime.
        self._has_accepted_synthesis_request = False
        self._next_dispatch_at: float | None = None
        self._pipeline: Any | None = None
        self._adapter: VoxCPM2BatchAdapter | None = None
        self._model_path: Path | None = None
        self._device: str | None = None
        self._sample_rate: int | None = None
        self._model_loads = 0
        self._batches_completed = 0

    def serve_forever(self) -> None:
        """Bind the owner-only socket and dispatch until idle or shutdown."""

        self._bind_listener()
        try:
            while not self._shutdown_requested.is_set():
                self._accept_one_connection()
                with self._state_lock:
                    self._drain_incoming()
                    # On a cold service, model loading itself takes seconds.
                    # Do not start that work after accepting only the first
                    # client, otherwise concurrent chapters that already
                    # connected cannot join its initial batch. Gather one
                    # short window of authenticated requests first, then
                    # prepare profiles and dispatch together.
                    if self._round_is_due():
                        self._prepare_waiting_requests()
                        if self._should_dispatch():
                            self._dispatch_one_batch()
                        else:
                            self._next_dispatch_at = None
                    self._finalize_ready_requests()
                    idle = self._is_idle()
                if idle:
                    break
        finally:
            with self._state_lock:
                self._fail_remaining("VoxCPM2 resident service stopped before synthesis completed.")
            self._close_listener()

    def _bind_listener(self) -> None:
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            existing = self.socket_path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if not stat.S_ISSOCK(existing.st_mode):
                raise RuntimeError(f"Refusing to replace non-socket runtime path: {self.socket_path}")
            self.socket_path.unlink()
        old_umask = os.umask(0o077)
        try:
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(self.socket_path))
        finally:
            os.umask(old_umask)
        self.socket_path.chmod(0o600)
        listener.listen(MAX_BATCH_SIZE * 4)
        listener.settimeout(0.05)
        self._listener = listener

    def _close_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        try:
            if self.socket_path.exists() or self.socket_path.is_symlink():
                self.socket_path.unlink()
        except OSError:
            pass

    def _accept_one_connection(self) -> None:
        listener = self._listener
        if listener is None:
            return
        try:
            connection, _ = listener.accept()
        except TimeoutError:
            return
        except OSError:
            if not self._shutdown_requested.is_set():
                raise
            return
        thread = threading.Thread(
            target=self._handle_connection,
            args=(connection,),
            name="voxcpm2-service-client",
            daemon=True,
        )
        thread.start()

    def _handle_connection(self, connection: socket.socket) -> None:
        request_id = ""
        with connection:
            try:
                with connection.makefile("rb") as stream:
                    raw = read_json_line(stream)
                request_id = str(raw.get("requestId") or "") if isinstance(raw, dict) else ""
                request = validate_request(raw, expected_token=self.token)
                request_id = str(request["requestId"])
                operation = request["operation"]
                if operation == "ping":
                    response = success_response(request_id, adapterVersion=BATCH_ADAPTER_VERSION)
                elif operation == "shutdown":
                    self._shutdown_requested.set()
                    response = success_response(request_id, adapterVersion=BATCH_ADAPTER_VERSION)
                elif operation == "cancel":
                    response = self._cancel_request(request_id, request)
                else:
                    service_request = _ServiceRequest(request)
                    self._incoming.put(service_request)
                    service_request.done.wait()
                    response = service_request.response or error_response(
                        request_id,
                        "VoxCPM2 service did not produce a response.",
                    )
            except Exception as error:
                response = error_response(request_id, error)
            try:
                connection.sendall(encode_json_line(response))
            except OSError:
                # A disconnected worker is allowed to abandon a request. Its
                # result remains an atomic cache artifact if it completes.
                pass

    def _cancel_request(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            target = str(payload.get("targetRequestId") or "").strip()
            if not target:
                return error_response(request_id, "targetRequestId is required for cancel.")
            candidate = self._requests.get(target)
            if candidate is None or candidate.is_complete():
                return success_response(request_id, cancelled=False)
            for ready in list(candidate.ready):
                candidate.ready.remove(ready)
                candidate.failures[ready.position] = "VoxCPM2 request was cancelled before inference."
            self._finalize_ready_requests()
            return success_response(request_id, cancelled=True)

    def _drain_incoming(self) -> None:
        accepted = False
        while True:
            try:
                request = self._incoming.get_nowait()
            except queue.Empty:
                break
            if request.request_id in self._requests:
                request.response = error_response(
                    request.request_id,
                    "duplicate VoxCPM2 service request id",
                )
                request.done.set()
                continue
            self._requests[request.request_id] = request
            accepted = True
        if accepted:
            self._has_accepted_synthesis_request = True
            self._last_activity = time.monotonic()
            if self._next_dispatch_at is None:
                self._next_dispatch_at = self._last_activity + _DISPATCH_WINDOW_SECONDS

    def _ensure_model(self, payload: dict[str, Any]) -> None:
        requested_path = Path(str(payload["modelPath"])).resolve()
        requested_device = str(payload.get("device") or "auto").strip().casefold()
        if self._pipeline is not None:
            if requested_path != self._model_path or requested_device != self._device:
                raise VoxCPM2BatchAdapterError(
                    "Resident VoxCPM2 service already owns a different model/device configuration."
                )
            return
        if not requested_path.is_dir():
            raise VoxCPM2BatchAdapterError(f"VoxCPM2 model directory is missing: {requested_path}")
        # This import is intentionally delayed until a real local request is
        # accepted, so starting the web service never loads or even imports the
        # ML runtime.
        from voxcpm import VoxCPM

        pipeline = VoxCPM.from_pretrained(
            str(requested_path),
            load_denoiser=False,
            optimize=False,
            device=requested_device,
        )
        sample_rate = int(pipeline.tts_model.sample_rate)
        if sample_rate <= 0:
            raise VoxCPM2BatchAdapterError("VoxCPM2 reported an invalid sample rate.")
        self._pipeline = pipeline
        self._adapter = VoxCPM2BatchAdapter(pipeline)
        self._model_path = requested_path
        self._device = requested_device
        self._sample_rate = sample_rate
        self._model_loads += 1

    def _prepare_waiting_requests(self) -> None:
        for request in list(self._requests.values()):
            if request.prepared or request.response is not None:
                continue
            try:
                self._ensure_model(request.payload)
                assert self._pipeline is not None
                assert self._sample_rate is not None
                profiles = request.payload.get("profiles")
                if isinstance(profiles, list):
                    request.profile_results = [
                        _ensure_profile(self._pipeline, profile, self._sample_rate)
                        for profile in profiles
                    ]
                request.ready = [
                    _ReadySegment(request, position, segment)
                    for position, segment in enumerate(request.original_segments)
                ]
                request.prepared = True
                self._last_activity = time.monotonic()
            except Exception as error:
                request.failures = {
                    position: str(error) for position, _segment in enumerate(request.original_segments)
                }
                request.prepared = True
                request.ready.clear()

    def _configured_batch_size(self) -> int:
        raw = str(os.environ.get("AUDIOBOOK_VOXCPM_BATCH_SIZE", "auto")).strip().casefold()
        if raw == "auto" or not raw:
            return self._benchmark_selected_batch_size()
        try:
            configured = int(raw)
        except ValueError:
            return 1
        return max(1, min(MAX_BATCH_SIZE, configured))

    def _benchmark_selected_batch_size(self) -> int:
        if self.selection_path is None or not self.selection_path.is_file():
            # Benchmark evidence is required before auto enables multi-item
            # batches. Explicit 2/4 remains available for controlled tests.
            return 1
        try:
            value = json.loads(self.selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 1
        if not isinstance(value, dict) or value.get("adapterVersion") != BATCH_ADAPTER_VERSION:
            return 1
        try:
            selected = int(value.get("selectedBatchSize", 1))
        except (TypeError, ValueError):
            return 1
        return max(1, min(MAX_BATCH_SIZE, selected))

    def _round_is_due(self) -> bool:
        return self._next_dispatch_at is not None and time.monotonic() >= self._next_dispatch_at

    def _should_dispatch(self) -> bool:
        if not any(request.ready for request in self._requests.values()):
            return False
        return self._round_is_due()

    def _select_batch(self) -> list[_ReadySegment]:
        active = [request for request in self._requests.values() if request.ready]
        if not active:
            return []
        forced = [ready for request in active for ready in request.ready if ready.force_single]
        if forced:
            forced.sort(key=lambda ready: (-ready.request.fairness_skips, ready.request.received_at))
            chosen = forced[0]
            chosen.request.ready.remove(chosen)
            return [chosen]
        capacity = self._configured_batch_size()
        active.sort(key=lambda request: (-request.fairness_skips, request.received_at))
        forced_requests = [request for request in active if request.fairness_skips >= 2]
        anchor = forced_requests[0] if forced_requests else active[0]
        anchor_item = anchor.ready[0]
        selected: list[_ReadySegment] = [anchor_item]
        selected_requests: set[str] = {anchor.request_id}
        # First reserve at most one head segment per active chapter.  A forced
        # chapter may break the bucket restriction after being skipped twice.
        for request in active:
            if len(selected) >= capacity or request.request_id in selected_requests:
                continue
            candidate = request.ready[0]
            compatible = abs(candidate.bucket - anchor_item.bucket) <= 1
            if compatible or request.fairness_skips >= 2:
                selected.append(candidate)
                selected_requests.add(request.request_id)
        # Then fill remaining capacity only from already selected chapters;
        # their immediately following source entries are eligible, but source
        # order remains metadata/assembly order rather than completion order.
        for request in active:
            if len(selected) >= capacity or request.request_id not in selected_requests:
                continue
            for candidate in request.ready[1:]:
                if len(selected) >= capacity:
                    break
                if abs(candidate.bucket - anchor_item.bucket) <= 1:
                    selected.append(candidate)
        for item in selected:
            item.request.ready.remove(item)
        selected_chapters = {item.request.request_id for item in selected}
        for request in active:
            if request.request_id in selected_chapters:
                request.fairness_skips = 0
            else:
                request.fairness_skips = min(3, request.fairness_skips + 1)
        return selected

    def _dispatch_one_batch(self) -> None:
        batch = self._select_batch()
        self._next_dispatch_at = None
        if not batch:
            return
        self._last_activity = time.monotonic()
        for item in batch:
            item.request.dispatches += 1
            item.request.maximum_effective_batch = max(
                item.request.maximum_effective_batch,
                len(batch),
            )
        try:
            if self._adapter is None or self._sample_rate is None:
                raise VoxCPM2BatchAdapterError("VoxCPM2 model did not initialize.")
            results = self._adapter.generate_batch([item.segment for item in batch])
            if len(results) != len(batch):
                raise VoxCPM2BatchAdapterError("VoxCPM2 batch did not return every requested segment.")
        except Exception as error:
            for item in batch:
                self._retry_or_fail(item, error)
            self._last_activity = time.monotonic()
            if any(request.ready for request in self._requests.values()):
                # A failed multi-item batch requeues only its failed items as
                # forced B=1 work. The collection window must be rearmed or
                # the new round would never become due.
                self._next_dispatch_at = self._last_activity + _DISPATCH_WINDOW_SECONDS
            return
        for item, result in zip(batch, results, strict=True):
            try:
                self._commit_result(item, result)
            except Exception as error:
                self._retry_or_fail(item, error)
        self._batches_completed += 1
        self._last_activity = time.monotonic()
        if any(request.ready for request in self._requests.values()):
            self._next_dispatch_at = self._last_activity + _DISPATCH_WINDOW_SECONDS

    def _commit_result(self, item: _ReadySegment, result: BatchAudioResult) -> None:
        if result.segment_id != item.segment_id:
            raise VoxCPM2BatchAdapterError(
                f"VoxCPM2 batch result mapped {result.segment_id} to {item.segment_id}."
            )
        assert self._sample_rate is not None
        output_path = Path(str(item.segment["outputPath"]))
        _atomic_write_wav(output_path, result.waveform, self._sample_rate)
        duration = _readable_duration(output_path)
        item.request.results[item.position] = {
            "id": item.segment_id,
            "path": str(output_path),
            "durationSeconds": duration,
            "sampleRate": self._sample_rate,
            "generatedPatches": result.generated_patches,
        }

    def _retry_or_fail(self, item: _ReadySegment, error: Exception) -> None:
        if item.retry_count == 0 and not item.force_single:
            item.retry_count = 1
            item.force_single = True
            item.request.fallback_count += 1
            item.request.ready.append(item)
            return
        item.request.failures[item.position] = str(error)

    def _finalize_ready_requests(self) -> None:
        completed: list[str] = []
        for request_id, request in self._requests.items():
            if request.response is not None or not request.prepared or not request.is_complete():
                continue
            ordered_results = [request.results[index] for index in sorted(request.results)]
            failures = [
                {
                    "id": str(request.original_segments[index].get("id") or ""),
                    "message": request.failures[index],
                }
                for index in sorted(request.failures)
            ]
            common = {
                "modelLoads": self._model_loads,
                "device": str(getattr(getattr(self._pipeline, "tts_model", None), "device", "")),
                "sampleRate": self._sample_rate,
                "adapterVersion": BATCH_ADAPTER_VERSION,
                "profiles": request.profile_results,
                "segments": ordered_results,
                "failures": failures,
                "metrics": {
                    "batches": request.dispatches,
                    "configuredBatchSize": self._configured_batch_size(),
                    "maxEffectiveBatchSize": request.maximum_effective_batch,
                    "fallbackCount": request.fallback_count,
                },
            }
            if failures:
                first = failures[0]
                request.response = error_response(
                    request.request_id,
                    f"VoxCPM2 failed segment {first['id']}: {first['message']}",
                    code="segment_failed",
                )
                request.response.update(common)
            else:
                request.response = success_response(request.request_id, **common)
            request.done.set()
            completed.append(request_id)
        for request_id in completed:
            self._requests.pop(request_id, None)

    def _is_idle(self) -> bool:
        if self._requests or not self._incoming.empty():
            return False
        elapsed = time.monotonic() - self._last_activity
        if not self._has_accepted_synthesis_request:
            # ServerState starts the service before the worker knows whether
            # every segment is a cache hit. Retaining it for the configured
            # idle period (or this minimum handoff grace for zero) lets a real
            # request connect safely without pinning an unused process.
            return elapsed >= max(float(self.idle_seconds), _NO_SYNTHESIS_STARTUP_GRACE_SECONDS)
        return elapsed >= self.idle_seconds

    def _fail_remaining(self, message: str) -> None:
        while True:
            try:
                request = self._incoming.get_nowait()
            except queue.Empty:
                break
            self._requests[request.request_id] = request
        for request in self._requests.values():
            if request.response is None:
                request.response = error_response(request.request_id, message)
                request.done.set()
        self._requests.clear()


def _read_token(path: Path) -> str:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise RuntimeError("VoxCPM2 service token file permissions are not owner-only.")
        token = path.read_text(encoding="utf-8").strip()
    finally:
        path.unlink(missing_ok=True)
    if not token:
        raise RuntimeError("VoxCPM2 service token file is empty.")
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the resident VoxCPM2 batch service")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--idle-seconds", type=int, default=300)
    parser.add_argument("--selection-path", type=Path, default=None)
    arguments = parser.parse_args(argv)
    try:
        service = VoxCPM2BatchService(
            arguments.socket,
            _read_token(arguments.token_file),
            idle_seconds=arguments.idle_seconds,
            selection_path=arguments.selection_path,
        )
        service.serve_forever()
    except Exception as error:
        print(f"VoxCPM2 batch service failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
