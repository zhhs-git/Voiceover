import json
import tempfile
import threading
import time
import wave
from pathlib import Path

import pytest

from audiobook_worker import voxcpm2_batch_service as service_module
from audiobook_worker.voxcpm2_batch_adapter import BatchAudioResult
from audiobook_worker.voxcpm2_batch_service import VoxCPM2BatchService
from audiobook_worker.voxcpm2_service_client import (
    VoxCPM2ServiceClientError,
    VoxCPM2ServiceEndpoint,
    request_service,
    synthesize_with_service,
)
from audiobook_worker.voxcpm2_service_protocol import (
    BATCH_ADAPTER_VERSION,
    PROMPT_FORMAT_VERSION,
    VoxCPM2ServiceProtocolError,
    validate_request,
)


def _wait_for_socket(path: Path) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"service socket did not appear: {path}")


def _short_socket_path(label: str) -> Path:
    """Use a private short ``/tmp`` path accepted by Darwin Unix sockets."""

    # The sandbox permits normal files under the repository but rejects Unix
    # socket binds there.  Production uses a short owner-only child of /tmp for
    # the same Darwin limitation, so keep the protocol test on that boundary.
    directory = Path(tempfile.mkdtemp(prefix="abvx-test-", dir="/tmp"))
    return directory / f"{label}.sock"


def _start_service(service: VoxCPM2BatchService) -> threading.Thread:
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    _wait_for_socket(service.socket_path)
    return thread


def _stop_service(endpoint: VoxCPM2ServiceEndpoint, thread: threading.Thread) -> None:
    try:
        request_service(endpoint, "shutdown")
    finally:
        thread.join(timeout=2)
    assert not thread.is_alive()
    endpoint.socket_path.unlink(missing_ok=True)
    try:
        endpoint.socket_path.parent.rmdir()
    except OSError:
        pass


def _write_wav(path: Path, _waveform, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams((1, 2, sample_rate, 120, "NONE", "not compressed"))
        wav_file.writeframes(b"\x00\x02" * 120)


def _synthesis_payload(
    chapter_id: str,
    output_path: Path,
    *,
    segment_id: str = "seg_0001",
) -> dict[str, object]:
    return {
        "promptFormatVersion": PROMPT_FORMAT_VERSION,
        "modelPath": "/fake/VoxCPM2",
        "device": "cpu",
        "chapter": {"bookId": "book_1", "chapterId": chapter_id},
        "profiles": [],
        "segments": [
            {
                "id": segment_id,
                "text": f"第{chapter_id}章的测试台词。",
                "delivery": "情绪自然克制，语速适中",
                "language": "zh",
                "promptFormatVersion": PROMPT_FORMAT_VERSION,
                "referenceWavPath": "/fake/reference.wav",
                "outputPath": str(output_path),
                "sourcePosition": 0,
                "sourceSegmentIds": [segment_id],
                "cacheSignature": f"signature-{chapter_id}",
            }
        ],
    }


def _ready_request(request_id: str, chapter_id: str, texts: list[str]):
    payload = {
        "requestId": request_id,
        "chapter": {"bookId": "book_1", "chapterId": chapter_id},
        "segments": [
            {
                "id": f"{request_id}_seg_{position}",
                "text": text,
            }
            for position, text in enumerate(texts)
        ],
    }
    request = service_module._ServiceRequest(payload)
    request.prepared = True
    request.ready = [
        service_module._ReadySegment(request, position, segment)
        for position, segment in enumerate(request.original_segments)
    ]
    return request


def test_service_ping_requires_its_ephemeral_token(tmp_path: Path):
    socket_path = _short_socket_path("token")
    service = VoxCPM2BatchService(socket_path, "correct-token", idle_seconds=60)
    thread = _start_service(service)
    endpoint = VoxCPM2ServiceEndpoint(socket_path, "correct-token", timeout_seconds=2)
    try:
        assert request_service(endpoint, "ping")["adapterVersion"] == BATCH_ADAPTER_VERSION
        with pytest.raises(VoxCPM2ServiceClientError, match="invalid service token"):
            request_service(
                VoxCPM2ServiceEndpoint(socket_path, "wrong-token", timeout_seconds=2),
                "ping",
            )
    finally:
        _stop_service(endpoint, thread)


def test_protocol_rejects_a_segment_without_source_order_metadata(tmp_path: Path):
    payload = {
        "version": 1,
        "token": "test-token",
        "requestId": "request-1",
        "operation": "synthesize",
        **_synthesis_payload("chapter_001", tmp_path / "segment.wav"),
    }
    del payload["segments"][0]["sourcePosition"]

    with pytest.raises(VoxCPM2ServiceProtocolError, match="sourcePosition"):
        validate_request(payload, expected_token="test-token")


def test_service_batches_two_chapters_and_preserves_request_output_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUDIOBOOK_VOXCPM_BATCH_SIZE", "2")
    monkeypatch.setattr(service_module, "_DISPATCH_WINDOW_SECONDS", 0.1)
    socket_path = _short_socket_path("mapping")
    service = VoxCPM2BatchService(socket_path, "test-token", idle_seconds=60)
    batches: list[list[str]] = []

    class FakePipeline:
        class TtsModel:
            device = "cpu"

        tts_model = TtsModel()

    class FakeAdapter:
        def generate_batch(self, items):
            batches.append([str(item["id"]) for item in items])
            return [
                BatchAudioResult(str(item["id"]), [0.0, 0.1, 0.0], 3)
                for item in items
            ]

    def fake_ensure_model(_payload):
        if service._pipeline is None:
            # A real cold model load is much slower than the service's short
            # request-collection window. The service must collect both client
            # requests before entering this blocking operation.
            time.sleep(0.15)
        service._pipeline = FakePipeline()
        service._adapter = FakeAdapter()
        service._sample_rate = 24_000
        service._model_loads = 1

    monkeypatch.setattr(service, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(service_module, "_atomic_write_wav", _write_wav)
    thread = _start_service(service)
    endpoint = VoxCPM2ServiceEndpoint(socket_path, "test-token", timeout_seconds=3)
    responses: dict[str, dict[str, object]] = {}

    def synthesize(chapter_id: str) -> None:
        output = tmp_path / f"{chapter_id}.wav"
        responses[chapter_id] = synthesize_with_service(
            endpoint,
            _synthesis_payload(chapter_id, output, segment_id=f"seg_{chapter_id}"),
        )

    first = threading.Thread(target=synthesize, args=("chapter_001",))
    second = threading.Thread(target=synthesize, args=("chapter_002",))
    first.start()
    second.start()
    first.join(timeout=4)
    second.join(timeout=4)
    try:
        assert not first.is_alive()
        assert not second.is_alive()
        assert len(batches) == 1
        assert set(batches[0]) == {"seg_chapter_001", "seg_chapter_002"}
        for chapter_id, response in responses.items():
            segment = response["segments"][0]
            assert segment["id"] == f"seg_{chapter_id}"
            assert Path(segment["path"]).name == f"{chapter_id}.wav"
            assert Path(segment["path"]).is_file()
            assert response["metrics"] == {
                "batches": 1,
                "configuredBatchSize": 2,
                "maxEffectiveBatchSize": 2,
                "fallbackCount": 0,
            }
    finally:
        _stop_service(endpoint, thread)


def test_failed_multi_item_batch_retries_only_as_single_item_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUDIOBOOK_VOXCPM_BATCH_SIZE", "2")
    monkeypatch.setattr(service_module, "_DISPATCH_WINDOW_SECONDS", 0.1)
    socket_path = _short_socket_path("retry")
    service = VoxCPM2BatchService(socket_path, "test-token", idle_seconds=60)
    attempted_sizes: list[int] = []

    class FakePipeline:
        class TtsModel:
            device = "cpu"

        tts_model = TtsModel()

    class FakeAdapter:
        def generate_batch(self, items):
            attempted_sizes.append(len(items))
            if len(items) > 1:
                raise RuntimeError("synthetic mixed-batch failure")
            item = items[0]
            return [BatchAudioResult(str(item["id"]), [0.0, 0.1, 0.0], 3)]

    def fake_ensure_model(_payload):
        service._pipeline = FakePipeline()
        service._adapter = FakeAdapter()
        service._sample_rate = 24_000
        service._model_loads = 1

    monkeypatch.setattr(service, "_ensure_model", fake_ensure_model)
    monkeypatch.setattr(service_module, "_atomic_write_wav", _write_wav)
    thread = _start_service(service)
    endpoint = VoxCPM2ServiceEndpoint(socket_path, "test-token", timeout_seconds=4)
    responses: list[dict[str, object]] = []

    def synthesize(chapter_id: str) -> None:
        responses.append(
            synthesize_with_service(
                endpoint,
                _synthesis_payload(
                    chapter_id,
                    tmp_path / f"{chapter_id}.wav",
                    segment_id=f"seg_{chapter_id}",
                ),
            )
        )

    first = threading.Thread(target=synthesize, args=("chapter_001",))
    second = threading.Thread(target=synthesize, args=("chapter_002",))
    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    try:
        assert not first.is_alive()
        assert not second.is_alive()
        assert attempted_sizes == [2, 1, 1]
        assert all(response["failures"] == [] for response in responses)
        assert all(response["metrics"]["fallbackCount"] == 1 for response in responses)
    finally:
        _stop_service(endpoint, thread)


def test_scheduler_prefers_one_compatible_head_from_each_chapter(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIOBOOK_VOXCPM_BATCH_SIZE", "2")
    service = VoxCPM2BatchService(tmp_path / "service.sock", "test-token", idle_seconds=60)
    first = _ready_request("request-a", "chapter-a", ["短句。", "第二句。"])
    second = _ready_request("request-b", "chapter-b", ["另一句。"])
    service._requests = {first.request_id: first, second.request_id: second}

    selected = service._select_batch()

    assert {item.request.chapter_key for item in selected} == {first.chapter_key, second.chapter_key}
    assert [item.position for item in selected] == [0, 0]


def test_scheduler_serves_a_skipped_chapter_on_the_next_round(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIOBOOK_VOXCPM_BATCH_SIZE", "1")
    service = VoxCPM2BatchService(tmp_path / "service.sock", "test-token", idle_seconds=60)
    first = _ready_request("request-a", "chapter-a", ["短句。", "第二句。"])
    second = _ready_request("request-b", "chapter-b", ["另一句。", "还有一句。"])
    service._requests = {first.request_id: first, second.request_id: second}

    round_one = service._select_batch()
    round_two = service._select_batch()

    assert round_one[0].request is first
    assert round_two[0].request is second
    assert second.fairness_skips == 0


def test_cancel_discards_only_unstarted_items_at_a_batch_boundary(tmp_path: Path):
    service = VoxCPM2BatchService(tmp_path / "service.sock", "test-token", idle_seconds=60)
    request = _ready_request("request-a", "chapter-a", ["第一句。", "第二句。"])
    service._requests = {request.request_id: request}

    response = service._cancel_request(
        "cancel-request",
        {"targetRequestId": request.request_id},
    )

    assert response["status"] == "succeeded"
    assert response["cancelled"] is True
    assert request.ready == []
    assert request.failures == {
        0: "VoxCPM2 request was cancelled before inference.",
        1: "VoxCPM2 request was cancelled before inference.",
    }


def test_auto_batch_size_requires_a_matching_benchmark_selection(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AUDIOBOOK_VOXCPM_BATCH_SIZE", "auto")
    selection_path = tmp_path / "selection.json"
    service = VoxCPM2BatchService(
        tmp_path / "service.sock",
        "test-token",
        idle_seconds=60,
        selection_path=selection_path,
    )

    assert service._configured_batch_size() == 1
    selection_path.write_text(
        json.dumps({"adapterVersion": BATCH_ADAPTER_VERSION, "selectedBatchSize": 4}),
        encoding="utf-8",
    )
    assert service._configured_batch_size() == 4
    selection_path.write_text(
        json.dumps({"adapterVersion": BATCH_ADAPTER_VERSION - 1, "selectedBatchSize": 4}),
        encoding="utf-8",
    )
    assert service._configured_batch_size() == 1


def test_zero_idle_keeps_a_short_handoff_grace_without_a_synthesis_request(tmp_path: Path):
    service = VoxCPM2BatchService(tmp_path / "service.sock", "test-token", idle_seconds=0)

    assert not service._is_idle()
    service._last_activity -= service_module._NO_SYNTHESIS_STARTUP_GRACE_SECONDS
    assert service._is_idle()


def test_cache_only_service_releases_after_its_configured_idle_period(tmp_path: Path):
    service = VoxCPM2BatchService(tmp_path / "service.sock", "test-token", idle_seconds=60)
    service._last_activity -= 59

    assert not service._is_idle()
    assert service._pipeline is None

    service._last_activity -= 2

    assert service._is_idle()
