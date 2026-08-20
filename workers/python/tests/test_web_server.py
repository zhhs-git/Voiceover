import json
import threading
import time
import wave
import zipfile
from pathlib import Path

from audiobook_worker.web_server import (
    BatchConcurrencyConfig,
    ExternalAudiobookArchive,
    FinalAudioArchive,
    FinalAudioExportError,
    ServerState,
    WebHandler,
    safe_filename,
)
from audiobook_worker import model_settings as model_settings_module


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 44100, 4410, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 4410)


def _available_llm_options(*model_ids: str) -> list[dict[str, object]]:
    return [
        {
            "id": model_id,
            "provider": model_id.split("/", 1)[0],
            "displayName": model_id,
            "family": "default",
            "available": True,
        }
        for model_id in model_ids
    ]


def _available_voxcpm2_capability() -> dict[str, object]:
    return {
        "id": "voxcpm2",
        "modelId": "VoxCPM2",
        "displayName": "VoxCPM2（本地）",
        "available": True,
        "reason": "已检测到本地模型和独立运行环境。",
    }


def test_voxcpm2_paths_preserves_the_venv_python_launcher(tmp_path: Path):
    launcher = tmp_path / "data" / "voxcpm2" / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    target = tmp_path / "base-python"
    target.touch()
    launcher.symlink_to(target)

    paths = model_settings_module.voxcpm2_paths(tmp_path)

    assert paths["python"] == launcher
    assert paths["python"].is_symlink()
    assert paths["python"].resolve() == target.resolve()


def _wait_for_batch(
    state: ServerState, batch_id: str, predicate
) -> dict[str, object]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = state.batch_generation_status(batch_id)
        if predicate(response):
            return response
        time.sleep(0.01)
    raise AssertionError(f"Batch did not reach the expected state: {batch_id}")


def test_safe_filename_preserves_extension_after_sanitizing_unicode_name():
    assert safe_filename("三体.epub") == "三体.epub"
    assert safe_filename("我的书.PDF") == "我的书.PDF"


def test_web_state_uses_shared_sqlite_and_book_directories(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = state.work_directory("book_123")
    work_dir.mkdir(parents=True)

    state.create_book(
        {
            "id": "book_123",
            "title": "Shared Book",
            "sourcePath": str(tmp_path / "uploads" / "book.txt"),
            "workDir": str(work_dir),
        }
    )
    state.upsert_chapter(
        {
            "id": "chapter_001",
            "bookId": "book_123",
            "title": "Chapter 1",
            "status": "pending",
            "scriptPath": None,
        }
    )

    assert state.list_books()[0]["title"] == "Shared Book"
    assert state.chapters("book_123")[0]["id"] == "chapter_001"
    assert state.run_worker(
        "_write_file",
        {"path": str(work_dir / "sample.json"), "content": '{"ok": true}'},
    )["status"] == "succeeded"
    assert state.run_worker(
        "_read_file", {"path": str(work_dir / "sample.json")}
    ) == {"ok": True}


def test_model_settings_payload_projects_only_safe_llm_metadata(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("AUDIOBOOK_LLM_MODEL", raising=False)
    monkeypatch.delenv("MODEL_ID", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        model_settings_module,
        "read_models_json",
        lambda: {
            "default": "openai/gpt-safe",
            "providers": {
                "openai": {
                    "baseUrl": "https://private.example/v1",
                    "apiKey": "super-secret-api-key",
                    "apiKeyEnv": "PRIVATE_API_KEY",
                    "models": [
                        {
                            "id": "gpt-safe",
                            "name": "Safe GPT",
                            "token": "another-secret",
                        }
                    ],
                }
            },
        },
    )
    monkeypatch.setattr(
        model_settings_module,
        "probe_voxcpm2",
        lambda _root=None: _available_voxcpm2_capability(),
    )
    state = ServerState(tmp_path)

    payload = state.model_settings_payload()
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["current"] == {
        "llmModelId": "openai/gpt-safe",
        "ttsBackend": "mimo",
        "ttsModelId": "mimo-v2.5-tts-voiceclone",
    }
    assert payload["llmOptions"] == [
        {
            "id": "openai/gpt-safe",
            "provider": "openai",
            "displayName": "Safe GPT",
            "family": "default",
            "available": True,
        },
        {
            "id": "mock",
            "provider": "local",
            "displayName": "离线 Mock（仅测试）",
            "family": "mock",
            "available": True,
        },
    ]
    for forbidden in (
        "super-secret-api-key",
        "another-secret",
        "https://private.example/v1",
        "PRIVATE_API_KEY",
        "apiKey",
        "token",
    ):
        assert forbidden not in serialized
    state.close()


def test_model_settings_persist_and_invalid_update_keeps_last_valid_value(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        model_settings_module,
        "discover_llm_options",
        lambda: _available_llm_options("openai/gpt-first", "openai/gpt-second"),
    )
    monkeypatch.setattr(
        model_settings_module,
        "probe_voxcpm2",
        lambda _root=None: _available_voxcpm2_capability(),
    )
    state = ServerState(tmp_path)
    saved = state.update_model_settings(
        {
            "llmModelId": "openai/gpt-first",
            "ttsBackend": "mimo",
            "ttsModelId": "mimo-v2.5-tts-voiceclone",
        }
    )

    try:
        state.update_model_settings(
            {
                "llmModelId": "missing/model",
                "ttsBackend": "mimo",
                "ttsModelId": "mimo-v2.5-tts-voiceclone",
            }
        )
    except ValueError as error:
        assert "LLM 模型不可用或不存在" in str(error)
    else:
        raise AssertionError("an unavailable LLM must be rejected")

    assert state.current_model_settings().to_dict() == saved["current"]
    state.close()

    reopened = ServerState(tmp_path)
    assert reopened.current_model_settings().to_dict() == saved["current"]
    reopened.close()


def test_model_settings_reject_unavailable_voxcpm2_without_replacing_current_value(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        model_settings_module,
        "discover_llm_options",
        lambda: _available_llm_options("openai/gpt-safe"),
    )
    monkeypatch.setattr(
        model_settings_module,
        "probe_voxcpm2",
        lambda _root=None: {
            **_available_voxcpm2_capability(),
            "available": False,
            "reason": "缺少 VoxCPM2 模型目录。",
        },
    )
    state = ServerState(tmp_path)
    previous = state.current_model_settings().to_dict()

    try:
        state.update_model_settings(
            {
                "llmModelId": "openai/gpt-safe",
                "ttsBackend": "voxcpm2",
                "ttsModelId": "VoxCPM2",
            }
        )
    except ValueError as error:
        assert "缺少 VoxCPM2 模型目录" in str(error)
    else:
        raise AssertionError("an unavailable VoxCPM2 environment must be rejected")

    assert state.current_model_settings().to_dict() == previous
    state.close()


def test_delete_book_removes_managed_upload_and_generated_work_directory(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = state.work_directory("book_123")
    work_dir.mkdir(parents=True)
    source_path = state.uploads_directory / "book.txt"
    source_path.write_text("原始内容", encoding="utf-8")
    state.create_book(
        {
            "id": "book_123",
            "title": "Shared Book",
            "sourcePath": str(source_path),
            "workDir": str(work_dir),
        }
    )

    state.delete_book("book_123")

    assert not source_path.exists()
    assert not work_dir.exists()
    assert state.list_books() == []


def test_delete_book_does_not_remove_external_original_file(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = state.work_directory("book_123")
    work_dir.mkdir(parents=True)
    source_path = tmp_path.parent / f"{tmp_path.name}-original.txt"
    source_path.write_text("用户自己的原始文件", encoding="utf-8")
    state.create_book(
        {
            "id": "book_123",
            "title": "External Source Book",
            "sourcePath": str(source_path),
            "workDir": str(work_dir),
        }
    )

    state.delete_book("book_123")

    assert source_path.exists()
    assert not work_dir.exists()
    source_path.unlink()


def test_web_state_persists_narrator_voice_and_invalidates_assembled_audio(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = state.work_directory("book_123")
    audio_dir = work_dir / "audio"
    audio_dir.mkdir(parents=True)
    state.create_book(
        {
            "id": "book_123",
            "title": "Shared Book",
            "sourcePath": str(tmp_path / "uploads" / "book.txt"),
            "workDir": str(work_dir),
        }
    )
    _write_wav(audio_dir / "chapter_001.wav")
    _write_wav(audio_dir / "chapter_001_mixed.wav")

    state.set_narrator_voice("book_123", "narrator_male")

    assert state.get_book(str(tmp_path / "uploads" / "book.txt"))["narratorVoiceId"] == "narrator_male"
    assert not (audio_dir / "chapter_001.wav").exists()
    assert not (audio_dir / "chapter_001_mixed.wav").exists()


def test_web_state_keeps_chapters_in_order_after_analysis_replaces_a_row(tmp_path: Path):
    state = ServerState(tmp_path)
    for chapter_id in ("chapter_001", "chapter_002", "chapter_003"):
        state.upsert_chapter(
            {
                "id": chapter_id,
                "bookId": "book_123",
                "title": chapter_id,
                "status": "pending",
                "scriptPath": None,
            }
        )

    state.upsert_chapter(
        {
            "id": "chapter_001",
            "bookId": "book_123",
            "title": "chapter_001",
            "status": "succeeded",
            "scriptPath": "/data/books/book_123/scripts/chapter_001.json",
        }
    )

    assert [chapter["id"] for chapter in state.chapters("book_123")] == [
        "chapter_001",
        "chapter_002",
        "chapter_003",
    ]


def test_web_state_rejects_paths_outside_data_directory(tmp_path: Path):
    state = ServerState(tmp_path)

    try:
        state.path_in_data("/tmp/not-audiobook-data.txt")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("outside paths must be rejected")


def test_web_state_only_resolves_text_for_a_chapter_owned_by_the_book(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = state.work_directory("book_123")
    chapters_dir = work_dir / "chapters"
    chapters_dir.mkdir(parents=True)

    state.create_book(
        {
            "id": "book_123",
            "title": "Shared Book",
            "sourcePath": str(tmp_path / "uploads" / "book.txt"),
            "workDir": str(work_dir),
        }
    )
    state.upsert_chapter(
        {
            "id": "chapter_001",
            "bookId": "book_123",
            "title": "Chapter 1",
            "status": "pending",
            "scriptPath": None,
        }
    )
    text_path = chapters_dir / "chapter_001.txt"
    text_path.write_text("第一行\n第二行", encoding="utf-8")

    assert state.chapter_text_path("book_123", "chapter_001") == text_path.resolve()
    assert state.chapter_text_path("book_123", "chapter_001").read_text(encoding="utf-8") == "第一行\n第二行"

    try:
        state.chapter_text_path("book_123", "chapter_002")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("chapters from another or unknown id must not be resolved")

    try:
        state.chapter_text_path("book_123", "../book.txt")
    except ValueError:
        pass
    else:
        raise AssertionError("chapter ids must not allow path traversal")


def _seed_batch_book(state: ServerState, chapter_ids: tuple[str, ...]) -> Path:
    work_dir = state.work_directory("book_123")
    state.create_book(
        {
            "id": "book_123",
            "title": "Shared Book",
            "sourcePath": str(state.uploads_directory / "book.txt"),
            "workDir": str(work_dir),
        }
    )
    for chapter_id in chapter_ids:
        script_path = work_dir / "scripts" / f"{chapter_id}.json"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("{}", encoding="utf-8")
        chapter_path = work_dir / "chapters" / f"{chapter_id}.txt"
        chapter_path.parent.mkdir(parents=True, exist_ok=True)
        chapter_path.write_text("正文", encoding="utf-8")
        state.upsert_chapter(
            {
                "id": chapter_id,
                "bookId": "book_123",
                "title": chapter_id,
                "status": "succeeded",
                "scriptPath": str(script_path),
            }
        )
    return work_dir


def test_external_audiobook_archive_automates_analysis_generation_and_chapter_mp3s(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    upload_path = state.uploads_directory / "novel.txt"
    upload_path.write_text("第一章", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    analysis_requests: list[dict[str, object]] = []

    def fake_worker(command: str, request: dict[str, object]):
        calls.append((command, str(request.get("chapterId") or "")))
        if command == "extract_book":
            chapters_dir = Path(str(request["outputDirectory"]))
            chapters_dir.mkdir(parents=True, exist_ok=True)
            chapters = []
            for chapter_id, title in (("chapter_001", "第一章"), ("chapter_002", "第二章")):
                text_path = chapters_dir / f"{chapter_id}.txt"
                text_path.write_text(title, encoding="utf-8")
                chapters.append(
                    {
                        "id": chapter_id,
                        "title": title,
                        "textPath": str(text_path),
                    }
                )
            return {
                "status": "succeeded",
                "warnings": [],
                "artifacts": [{"kind": "book_extraction", "metadata": {"title": "示例书", "chapters": chapters}}],
            }
        if command == "analyze_chapter":
            analysis_requests.append(request)
            chapter_id = str(request["chapterId"])
            scripts_dir = Path(str(request["outputDirectory"]))
            scripts_dir.mkdir(parents=True, exist_ok=True)
            script_path = scripts_dir / f"{chapter_id}.json"
            script_path.write_text(
                json.dumps({"chapterId": chapter_id, "characters": [{"id": "hero", "canonicalName": "主角"}]}),
                encoding="utf-8",
            )
            return {
                "status": "succeeded",
                "warnings": [],
                "artifacts": [{"kind": "chapter_script", "path": str(script_path)}],
            }
        if command == "convert_to_mp3":
            output_path = Path(str(request["outputPath"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"ID3fake-mp3")
            return {"status": "succeeded", "warnings": [], "artifacts": [{"kind": "mp3", "path": str(output_path)}]}
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)

    def fake_start_batch(request: dict[str, object]):
        book_id = str(request["bookId"])
        work_dir = state.work_directory(book_id)
        chapters = []
        for position, chapter_id in enumerate(request["chapterIds"]):
            mixed_path = work_dir / "audio" / f"{chapter_id}_mixed.wav"
            _write_wav(mixed_path)
            chapters.append(
                {
                    "chapterId": chapter_id,
                    "title": "第一章" if chapter_id == "chapter_001" else "第二章",
                    "position": position,
                    "status": "succeeded",
                    "mixedAudioPath": str(mixed_path),
                }
            )
        return {"batchId": "batch_api", "status": "queued", "chapters": chapters}

    monkeypatch.setattr(state, "start_batch_generation", fake_start_batch)
    monkeypatch.setattr(
        state,
        "_wait_for_external_batch",
        lambda batch_id: {
            "batchId": batch_id,
            "status": "succeeded",
            "chapters": fake_start_batch({"bookId": state.list_books()[0]["id"], "chapterIds": ["chapter_001", "chapter_002"]})["chapters"],
        },
    )

    archive = state.create_external_audiobook_archive(
        filename="novel.txt",
        upload_path=upload_path,
    )

    assert archive.chapter_count == 2
    assert archive.archive_path.is_file()
    assert archive.download_filename == "示例书-chapters-mp3.zip"
    with zipfile.ZipFile(archive.archive_path) as zipped:
        assert zipped.namelist() == ["001-第一章.mp3", "002-第二章.mp3"]
        assert zipped.read("001-第一章.mp3") == b"ID3fake-mp3"
    assert [command for command, _ in calls] == [
        "extract_book",
        "analyze_chapter",
        "analyze_chapter",
        "convert_to_mp3",
        "convert_to_mp3",
    ]
    assert analysis_requests[0]["knownCharacters"] is None
    assert analysis_requests[1]["knownCharacters"] == [
        {"id": "hero", "canonicalName": "主角"}
    ]
    assert analysis_requests[0]["llmModelId"] == state.current_model_settings().llm_model_id
    state.close()


def test_external_audiobook_archive_rejects_unsupported_upload(tmp_path: Path):
    state = ServerState(tmp_path)
    upload_path = state.uploads_directory / "novel.docx"
    upload_path.write_bytes(b"not a supported book")

    try:
        state.create_external_audiobook_archive(
            filename="novel.docx",
            upload_path=upload_path,
        )
    except Exception as error:
        assert getattr(error, "code", None) == "unsupported_book_format"
    else:
        raise AssertionError("unsupported uploads must fail")
    state.close()


def test_external_audiobook_http_endpoint_returns_a_zip_download(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    upload_path = state.uploads_directory / "novel.txt"
    upload_path.write_bytes(b"book body")
    archive_path = tmp_path / "chapter-audio.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("001-第一章.mp3", b"ID3fake-mp3")
    captured: dict[str, object] = {}

    def fake_create_archive(*, filename: str, upload_path: Path, narrator_voice_id: str):
        captured.update(
            {
                "filename": filename,
                "upload": upload_path.read_bytes(),
                "narratorVoiceId": narrator_voice_id,
            }
        )
        return ExternalAudiobookArchive(
            book_id="book_api",
            archive_path=archive_path,
            download_filename="示例书-chapters-mp3.zip",
            chapter_count=1,
        )

    monkeypatch.setattr(state, "create_external_audiobook_archive", fake_create_archive)
    served: dict[str, object] = {}

    class FakeHandler:
        path = "/api/external/audiobook/chapters.mp3.zip?narratorVoiceId=narrator_male"

        def __init__(self, handler_state, handler_upload):
            self.state = handler_state
            self.upload = handler_upload

        def save_external_audiobook_upload(self):
            return self.upload, "novel.txt"

        def serve_file(self, path: Path, **kwargs):
            served["path"] = path
            served.update(kwargs)

    WebHandler.external_audiobook_chapters_mp3(FakeHandler(state, upload_path))

    assert served["path"] == archive_path
    assert served["download_filename"] == "示例书-chapters-mp3.zip"
    assert served["extra_headers"] == {
        "X-Audiobook-Book-Id": "book_api",
        "X-Audiobook-Chapter-Count": "1",
    }
    assert captured == {
        "filename": "novel.txt",
        "upload": b"book body",
        "narratorVoiceId": "narrator_male",
    }
    state.close()


def _seed_final_audio_export(
    state: ServerState,
    chapter_ids: tuple[str, ...],
) -> Path:
    work_dir = _seed_batch_book(state, chapter_ids)
    now = time.time()
    state.db.execute(
        "INSERT INTO generation_batches "
        "(id, book_id, status, force, cache_segments, cancel_requested, created_at, updated_at) "
        "VALUES ('batch_export', 'book_123', 'succeeded', 0, 1, 0, ?, ?)",
        (now, now),
    )
    for position, chapter_id in enumerate(chapter_ids):
        mixed_path = work_dir / "audio" / f"{chapter_id}_mixed.wav"
        _write_wav(mixed_path)
        state.db.execute(
            "INSERT INTO generation_batch_chapters "
            "(batch_id, chapter_id, position, status, current_stage, next_stage, stage_state, "
            "mixed_audio_path, updated_at) "
            "VALUES ('batch_export', ?, ?, 'succeeded', NULL, 'complete', 'complete', ?, ?)",
            (chapter_id, position, str(mixed_path), now),
        )
    state.db.commit()
    return work_dir


def test_final_audio_archive_packages_selected_mp3s_with_requested_bitrate(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    _seed_final_audio_export(state, ("chapter_001", "chapter_002"))
    conversion_requests: list[dict[str, object]] = []

    def fake_worker(command: str, request: dict[str, object]):
        assert command == "convert_to_mp3"
        conversion_requests.append(request)
        output_path = Path(str(request["outputPath"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"ID3fake-mp3")
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)

    archive = state.create_final_audio_archive(
        book_id="book_123",
        chapter_ids=["chapter_002", "chapter_001"],
        output_format="mp3",
        bitrate_kbps=128,
    )

    assert archive.chapter_count == 2
    assert archive.skipped_count == 0
    assert archive.download_filename == "Shared_Book-final-audio-mp3-128kbps.zip"
    assert [request["bitrateKbps"] for request in conversion_requests] == [128, 128]
    with zipfile.ZipFile(archive.archive_path) as zipped:
        assert zipped.namelist() == ["001-chapter_001.mp3", "002-chapter_002.mp3"]
        assert zipped.read("001-chapter_001.mp3") == b"ID3fake-mp3"
    state.close()


def test_final_audio_archive_copies_wav_and_skips_a_mix_that_disappears(tmp_path: Path):
    state = ServerState(tmp_path)
    work_dir = _seed_final_audio_export(state, ("chapter_001", "chapter_002"))
    second_mix_path = work_dir / "audio" / "chapter_002_mixed.wav"
    second_mix_path.unlink()

    archive = state.create_final_audio_archive(
        book_id="book_123",
        chapter_ids=["chapter_001", "chapter_002"],
        output_format="wav",
        bitrate_kbps=None,
    )

    assert archive.chapter_count == 1
    assert archive.skipped_count == 1
    with zipfile.ZipFile(archive.archive_path) as zipped:
        assert zipped.namelist() == ["001-chapter_001.wav"]
        assert zipped.read("001-chapter_001.wav") == (work_dir / "audio" / "chapter_001_mixed.wav").read_bytes()
    state.close()


def test_final_audio_archive_rejects_invalid_selection_and_bitrate(tmp_path: Path):
    state = ServerState(tmp_path)
    _seed_final_audio_export(state, ("chapter_001",))

    for request, code in (
        (
            {"chapter_ids": [], "output_format": "mp3", "bitrate_kbps": 128},
            "empty_chapter_selection",
        ),
        (
            {"chapter_ids": ["chapter_001"], "output_format": "mp3", "bitrate_kbps": 96},
            "invalid_mp3_bitrate",
        ),
        (
            {"chapter_ids": ["chapter_missing"], "output_format": "wav", "bitrate_kbps": None},
            "unknown_chapter",
        ),
        (
            {"chapter_ids": ["chapter_001"], "output_format": "wav", "bitrate_kbps": 128},
            "unexpected_wav_bitrate",
        ),
    ):
        try:
            state.create_final_audio_archive(book_id="book_123", **request)
        except FinalAudioExportError as error:
            assert error.code == code
        else:
            raise AssertionError(f"expected {code}")
    state.close()


def test_final_audio_http_endpoint_streams_zip_with_export_counts(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    archive_path = tmp_path / "final-audio.zip"
    archive_path.write_bytes(b"zip")
    captured: dict[str, object] = {}

    def fake_create_archive(**kwargs):
        captured.update(kwargs)
        return FinalAudioArchive(
            archive_path=archive_path,
            download_filename="Shared_Book-final-audio-mp3-192kbps.zip",
            chapter_count=2,
            skipped_count=1,
        )

    monkeypatch.setattr(state, "create_final_audio_archive", fake_create_archive)
    served: dict[str, object] = {}

    class FakeHandler:
        def __init__(self, handler_state):
            self.state = handler_state

        def read_json(self):
            return {
                "chapterIds": ["chapter_001", "chapter_002"],
                "format": "mp3",
                "bitrateKbps": 192,
            }

        def serve_file(self, path: Path, **kwargs):
            served["path"] = path
            served.update(kwargs)

    WebHandler.final_audio_archive(FakeHandler(state), "book_123")

    assert captured == {
        "book_id": "book_123",
        "chapter_ids": ["chapter_001", "chapter_002"],
        "output_format": "mp3",
        "bitrate_kbps": 192,
    }
    assert served["path"] == archive_path
    assert served["download_filename"] == "Shared_Book-final-audio-mp3-192kbps.zip"
    assert served["extra_headers"] == {
        "X-Audiobook-Chapter-Count": "2",
        "X-Audiobook-Skipped-Chapter-Count": "1",
    }
    state.close()


def test_post_route_dispatches_to_the_book_scoped_final_audio_export():
    captured: list[str] = []

    class FakeHandler:
        path = "/api/books/book_123/final-audio.zip"

        def final_audio_archive(self, book_id: str):
            captured.append(book_id)

        def send_error_json(self, *_args, **_kwargs):
            raise AssertionError("the final-audio export route should be recognized")

    WebHandler.do_POST(FakeHandler())

    assert captured == ["book_123"]


def test_batch_generation_preserves_each_chapter_stage_order_and_persists_outputs(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    work_dir = _seed_batch_book(state, ("chapter_001", "chapter_002"))
    calls: list[tuple[str, str]] = []

    def fake_worker(command: str, request: dict[str, object]):
        chapter_id = str(request.get("chapterId"))
        calls.append((chapter_id, command))
        if command == "generate_audio_assets":
            return {
                "status": "succeeded",
                "warnings": [],
                "artifacts": [
                    {
                        "kind": "stable_audio_music",
                        "path": str(work_dir / "audio-assets" / chapter_id / "music" / "scene_001.wav"),
                        "metadata": {"assetId": "scene_001", "sceneId": "scene_001"},
                    }
                ],
            }
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )
    finished = _wait_for_batch(
        state, str(started["batchId"]), lambda response: response["status"] == "succeeded"
    )

    expected_stage_commands = [
        "synthesize_chapter_audio",
        "assemble_chapter_audio",
        "transcribe_chapter_audio",
        "plan_chapter_audio",
        "generate_audio_assets",
        "mix_chapter_audio",
    ]
    # Chapter pipelines may now overlap. The durable contract is that each
    # chapter itself keeps this stage order, not that one whole chapter must
    # finish before the scheduler starts the next one.
    assert {chapter_id for chapter_id, _ in calls} == {
        "chapter_001",
        "chapter_002",
    }
    for chapter_id in ("chapter_001", "chapter_002"):
        assert [
            command for called_chapter_id, command in calls if called_chapter_id == chapter_id
        ] == expected_stage_commands
    assert finished["succeededCount"] == 2
    assert all(chapter["status"] == "succeeded" for chapter in finished["chapters"])
    assert finished["chapters"][0]["mixedAudioPath"] == str(
        work_dir / "audio" / "chapter_001_mixed.wav"
    )
    state.close()


def test_batch_model_snapshot_remains_immutable_after_global_settings_change(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        model_settings_module,
        "discover_llm_options",
        lambda: _available_llm_options("openai/gpt-first", "openai/gpt-second"),
    )
    monkeypatch.setattr(
        model_settings_module,
        "probe_voxcpm2",
        lambda _root=None: _available_voxcpm2_capability(),
    )
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001",))
    monkeypatch.setattr(state, "_launch_batch_generation_locked", lambda _batch_id: None)
    state.update_model_settings(
        {
            "llmModelId": "openai/gpt-first",
            "ttsBackend": "mimo",
            "ttsModelId": "mimo-v2.5-tts-voiceclone",
        }
    )
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001"]}
    )
    batch_id = str(started["batchId"])

    state.update_model_settings(
        {
            "llmModelId": "openai/gpt-second",
            "ttsBackend": "voxcpm2",
            "ttsModelId": "VoxCPM2",
        }
    )
    status = state.batch_generation_status(batch_id)
    definition = state._batch_stage_definition("voice_synthesize")
    _context, voice_request = state._batch_stage_request(
        batch_id,
        "chapter_001",
        definition,
    )
    _context, plan_request = state._batch_stage_request(
        batch_id,
        "chapter_001",
        state._batch_stage_definition("audio_plan"),
    )

    assert status["modelSettings"] == {
        "llmModelId": "openai/gpt-first",
        "ttsBackend": "mimo",
        "ttsModelId": "mimo-v2.5-tts-voiceclone",
    }
    assert voice_request["backend"] == "mimo"
    assert voice_request["modelId"] == "mimo-v2.5-tts-voiceclone"
    assert plan_request["llmModelId"] == "openai/gpt-first"
    assert state._batch_stage_resource(batch_id, definition) == "mimo"
    assert state.current_model_settings().to_dict() == {
        "llmModelId": "openai/gpt-second",
        "ttsBackend": "voxcpm2",
        "ttsModelId": "VoxCPM2",
    }
    state.close()


def test_voxcpm2_batch_voice_stage_uses_the_capacity_one_local_resource(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(
        model_settings_module,
        "discover_llm_options",
        lambda: _available_llm_options("openai/gpt-safe"),
    )
    monkeypatch.setattr(
        model_settings_module,
        "probe_voxcpm2",
        lambda _root=None: _available_voxcpm2_capability(),
    )
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001", "chapter_002"))
    state.update_model_settings(
        {
            "llmModelId": "openai/gpt-safe",
            "ttsBackend": "voxcpm2",
            "ttsModelId": "VoxCPM2",
        }
    )
    first_voice_started = threading.Event()
    release_voice = threading.Event()
    active = 0
    maximum = 0
    voice_calls: list[str] = []
    lock = threading.Lock()

    def fake_worker(command: str, request: dict[str, object]):
        nonlocal active, maximum
        if command == "synthesize_chapter_audio":
            with lock:
                active += 1
                maximum = max(maximum, active)
                voice_calls.append(str(request["chapterId"]))
            first_voice_started.set()
            try:
                release_voice.wait(timeout=2)
            finally:
                with lock:
                    active -= 1
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )
    batch_id = str(started["batchId"])
    assert state._batch_stage_resource(
        batch_id,
        state._batch_stage_definition("voice_synthesize"),
    ) == "voxcpm"
    assert first_voice_started.wait(timeout=1)
    time.sleep(0.05)
    assert voice_calls == ["chapter_001"]
    assert maximum == 1

    state.cancel_batch_generation(batch_id)
    release_voice.set()
    finished = _wait_for_batch(
        state,
        batch_id,
        lambda response: response["status"] == "cancelled",
    )
    assert all(chapter["status"] == "cancelled" for chapter in finished["chapters"])
    state.close()


def test_batch_generation_releases_mimo_for_the_next_chapter_while_later_stages_run(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001", "chapter_002"))
    first_voice_started = threading.Event()
    allow_first_voice_to_finish = threading.Event()
    first_later_stage_started = threading.Event()
    second_voice_started = threading.Event()
    release_workers = threading.Event()
    active_mimo = 0
    maximum_mimo = 0
    calls: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fake_worker(command: str, request: dict[str, object]):
        nonlocal active_mimo, maximum_mimo
        chapter_id = str(request["chapterId"])
        with lock:
            calls.append((chapter_id, command))
        if command == "synthesize_chapter_audio":
            with lock:
                active_mimo += 1
                maximum_mimo = max(maximum_mimo, active_mimo)
            try:
                if chapter_id == "chapter_001":
                    first_voice_started.set()
                    allow_first_voice_to_finish.wait(timeout=2)
                else:
                    second_voice_started.set()
                    release_workers.wait(timeout=2)
            finally:
                with lock:
                    active_mimo -= 1
        elif chapter_id == "chapter_001" and command == "assemble_chapter_audio":
            first_later_stage_started.set()
            release_workers.wait(timeout=2)
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started_batch = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )

    assert first_voice_started.wait(timeout=1)
    time.sleep(0.05)
    assert not second_voice_started.is_set()
    allow_first_voice_to_finish.set()
    assert second_voice_started.wait(timeout=1)
    assert first_later_stage_started.wait(timeout=1)
    assert maximum_mimo == 1
    release_workers.set()
    finished = _wait_for_batch(
        state,
        str(started_batch["batchId"]),
        lambda response: response["status"] == "succeeded",
    )

    assert finished["succeededCount"] == 2
    assert calls.index(("chapter_002", "synthesize_chapter_audio")) > calls.index(
        ("chapter_001", "synthesize_chapter_audio")
    )
    state.close()


def test_batch_generation_never_marks_a_batch_done_while_a_chapter_is_running(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001", "chapter_002"))
    first_finished = threading.Event()
    second_mix_started = threading.Event()
    release_second = threading.Event()

    def fake_worker(command: str, request: dict[str, object]):
        chapter_id = str(request["chapterId"])
        if command == "mix_chapter_audio" and chapter_id == "chapter_002":
            second_mix_started.set()
            release_second.wait(timeout=2)
        if command == "mix_chapter_audio" and chapter_id == "chapter_001":
            first_finished.set()
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )
    assert first_finished.wait(timeout=1)
    assert second_mix_started.wait(timeout=1)
    _wait_for_batch(
        state,
        str(started["batchId"]),
        lambda response: response["succeededCount"] == 1,
    )

    assert state._finish_batch_if_done(str(started["batchId"])) is False
    assert state.batch_generation_status(str(started["batchId"]))["status"] == "running"
    release_second.set()
    _wait_for_batch(
        state,
        str(started["batchId"]),
        lambda response: response["status"] == "succeeded",
    )
    state.close()


def test_batch_generation_claim_is_atomic_under_competing_schedulers(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001",))
    now = time.time()
    state.db.execute(
        "INSERT INTO generation_batches "
        "(id, book_id, status, force, cache_segments, cancel_requested, created_at, updated_at) "
        "VALUES ('batch_claim', 'book_123', 'queued', 0, 1, 0, ?, ?)",
        (now, now),
    )
    state.db.execute(
        "INSERT INTO generation_batch_chapters "
        "(batch_id, chapter_id, position, status, next_stage, stage_state, updated_at) "
        "VALUES ('batch_claim', 'chapter_001', 0, 'queued', 'voice_synthesize', 'ready', ?)",
        (now,),
    )
    state.db.commit()
    barrier = threading.Barrier(2)
    claims: list[bool] = []
    definition = state._batch_stage_definition("voice_synthesize")

    def claim():
        barrier.wait(timeout=1)
        claims.append(
            state._claim_ready_batch_stage("batch_claim", "chapter_001", definition)
        )

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert claims.count(True) == 1
    assert claims.count(False) == 1
    chapter = state.batch_generation_status("batch_claim")["chapters"][0]
    assert chapter["status"] == "running"
    assert chapter["nextStage"] == "voice_synthesize"
    assert chapter["stageState"] == "running"
    state.close()


def test_batch_concurrency_configuration_has_safe_defaults_and_hard_ceilings(
    monkeypatch
):
    for name in (
        "AUDIOBOOK_BATCH_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIMO_TOTAL_CONCURRENCY",
        "AUDIOBOOK_MIMO_CONCURRENCY",
        "AUDIOBOOK_LLM_WORKER_CONCURRENCY",
        "AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY",
        "AUDIOBOOK_MLX_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIX_WORKER_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = BatchConcurrencyConfig.from_environment()
    assert defaults.chapter_workers == 5
    assert defaults.mimo_total == 1
    assert defaults.mimo_per_process == 1
    assert defaults.mimo_process_slots == 1
    assert defaults.mimo_process_concurrency == 1
    assert defaults.llm_workers == 2
    assert defaults.local_audio_workers == 4
    assert defaults.mix_workers == 2
    assert defaults.non_mimo_workers_while_mimo_pending == 4

    for name in (
        "AUDIOBOOK_BATCH_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIMO_TOTAL_CONCURRENCY",
        "AUDIOBOOK_MIMO_CONCURRENCY",
        "AUDIOBOOK_LLM_WORKER_CONCURRENCY",
        "AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY",
        "AUDIOBOOK_MLX_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIX_WORKER_CONCURRENCY",
    ):
        monkeypatch.setenv(name, "99")
    capped = BatchConcurrencyConfig.from_environment()
    assert capped.chapter_workers == 5
    assert capped.mimo_total == 1
    assert capped.mimo_per_process == 1
    assert capped.mimo_process_slots == 1
    assert capped.llm_workers == 2
    assert capped.local_audio_workers == 4
    assert capped.mix_workers == 2

    monkeypatch.delenv("AUDIOBOOK_MLX_WORKER_CONCURRENCY", raising=False)
    monkeypatch.setenv("AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY", "2")
    new_setting = BatchConcurrencyConfig.from_environment()
    assert new_setting.local_audio_workers == 2

    monkeypatch.setenv("AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY", "4")
    monkeypatch.setenv("AUDIOBOOK_MLX_WORKER_CONCURRENCY", "1")
    legacy_capped = BatchConcurrencyConfig.from_environment()
    assert legacy_capped.local_audio_workers == 1

    monkeypatch.setenv("AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY", "2")
    monkeypatch.setenv("AUDIOBOOK_MLX_WORKER_CONCURRENCY", "3")
    new_setting_wins_when_lower = BatchConcurrencyConfig.from_environment()
    assert new_setting_wins_when_lower.local_audio_workers == 2

    monkeypatch.setenv("AUDIOBOOK_MIMO_TOTAL_CONCURRENCY", "1")
    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "4")
    minimum_mimo = BatchConcurrencyConfig.from_environment()
    assert minimum_mimo.mimo_total == 1
    assert minimum_mimo.mimo_per_process == 1
    assert minimum_mimo.mimo_process_concurrency == 1
    assert minimum_mimo.mimo_process_slots == 1


def test_batch_status_exposes_the_shared_mimo_rate_cooldown(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001",))
    monkeypatch.setattr(state, "_launch_batch_generation_locked", lambda _batch_id: None)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001"]}
    )
    state.mimo_rate_state_path.write_text(
        json.dumps({"cooldownUntilMonotonic": time.monotonic() + 5.0}),
        encoding="utf-8",
    )

    cooldown = state.batch_generation_status(str(started["batchId"]))["mimoCooldownSeconds"]

    assert isinstance(cooldown, float)
    assert 4.0 <= cooldown <= 5.0
    state.close()


def test_batch_stage_admission_keeps_one_worker_slot_for_waiting_mimo(
    tmp_path: Path,
    monkeypatch,
):
    state = ServerState(tmp_path)
    chapter_ids = (
        "chapter_001",
        "chapter_002",
        "chapter_003",
        "chapter_004",
        "chapter_005",
    )
    _seed_batch_book(state, chapter_ids)
    now = time.time()
    state.db.execute(
        "INSERT INTO generation_batches "
        "(id, book_id, status, force, cache_segments, cancel_requested, created_at, updated_at) "
        "VALUES ('batch_reservation', 'book_123', 'queued', 0, 1, 0, ?, ?)",
        (now, now),
    )
    state.db.executemany(
        "INSERT INTO generation_batch_chapters "
        "(batch_id, chapter_id, position, status, next_stage, stage_state, updated_at) "
        "VALUES ('batch_reservation', ?, ?, 'queued', ?, 'ready', ?)",
        [
            ("chapter_001", 0, "voice_synthesize", now),
            ("chapter_002", 1, "audio_plan", now),
            ("chapter_003", 2, "audio_plan", now),
            ("chapter_004", 3, "voice_assemble", now),
            ("chapter_005", 4, "mix", now),
        ],
    )
    state.db.commit()

    entered_four_non_mimo_stages = threading.Event()
    release_workers = threading.Event()
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def fake_worker(command: str, request: dict[str, object]):
        with calls_lock:
            calls.append((str(request["chapterId"]), command))
            if len(calls) == 4:
                entered_four_non_mimo_stages.set()
        release_workers.wait(timeout=2)
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    # Model a direct MiMo request already holding the one resource permit.
    # The queued voice stage must then reserve the fifth worker position
    # instead of allowing a fifth later-stage worker to start.
    assert state.worker_resource_semaphores["mimo"].acquire(blocking=False)
    try:
        with state.batch_generation_lock:
            state._launch_batch_generation_locked("batch_reservation")
        assert entered_four_non_mimo_stages.wait(timeout=1)
        time.sleep(0.05)
        assert len(calls) == 4
        assert {command for _, command in calls} == {
            "plan_chapter_audio",
            "assemble_chapter_audio",
            "mix_chapter_audio",
        }
    finally:
        state.cancel_batch_generation("batch_reservation")
        release_workers.set()
        state.worker_resource_semaphores["mimo"].release()

    finished = _wait_for_batch(
        state,
        "batch_reservation",
        lambda response: response["status"] == "cancelled",
    )
    assert all(chapter["status"] == "cancelled" for chapter in finished["chapters"])
    state.close()


def test_batch_transcription_and_stable_audio_share_four_local_audio_slots(
    tmp_path: Path,
    monkeypatch,
):
    state = ServerState(tmp_path)
    chapter_ids = tuple(f"chapter_{index:03d}" for index in range(1, 6))
    _seed_batch_book(state, chapter_ids)
    now = time.time()
    state.db.execute(
        "INSERT INTO generation_batches "
        "(id, book_id, status, force, cache_segments, cancel_requested, created_at, updated_at) "
        "VALUES ('batch_local_audio', 'book_123', 'queued', 0, 1, 0, ?, ?)",
        (now, now),
    )
    state.db.executemany(
        "INSERT INTO generation_batch_chapters "
        "(batch_id, chapter_id, position, status, next_stage, stage_state, updated_at) "
        "VALUES ('batch_local_audio', ?, ?, 'queued', ?, 'ready', ?)",
        [
            ("chapter_001", 0, "transcript", now),
            ("chapter_002", 1, "stable_audio", now),
            ("chapter_003", 2, "transcript", now),
            ("chapter_004", 3, "stable_audio", now),
            ("chapter_005", 4, "transcript", now),
        ],
    )
    state.db.commit()

    entered_four_local_audio_stages = threading.Event()
    release_workers = threading.Event()
    calls: list[str] = []
    calls_lock = threading.Lock()

    def fake_worker(command: str, _request: dict[str, object]):
        with calls_lock:
            calls.append(command)
            if len(calls) == 4:
                entered_four_local_audio_stages.set()
        release_workers.wait(timeout=2)
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    try:
        with state.batch_generation_lock:
            state._launch_batch_generation_locked("batch_local_audio")
        assert entered_four_local_audio_stages.wait(timeout=1)
        time.sleep(0.05)
        assert len(calls) == 4
        assert set(calls) == {
            "transcribe_chapter_audio",
            "generate_audio_assets",
        }
    finally:
        state.cancel_batch_generation("batch_local_audio")
        release_workers.set()

    finished = _wait_for_batch(
        state,
        "batch_local_audio",
        lambda response: response["status"] == "cancelled",
    )
    assert all(chapter["status"] == "cancelled" for chapter in finished["chapters"])
    state.close()


def test_worker_resource_limits_bound_mimo_local_audio_llm_and_mix_commands(
    tmp_path: Path, monkeypatch
):
    for name in (
        "AUDIOBOOK_BATCH_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIMO_TOTAL_CONCURRENCY",
        "AUDIOBOOK_MIMO_CONCURRENCY",
        "AUDIOBOOK_LLM_WORKER_CONCURRENCY",
        "AUDIOBOOK_LOCAL_AUDIO_WORKER_CONCURRENCY",
        "AUDIOBOOK_MLX_WORKER_CONCURRENCY",
        "AUDIOBOOK_MIX_WORKER_CONCURRENCY",
    ):
        monkeypatch.delenv(name, raising=False)
    state = ServerState(tmp_path)
    active: dict[str, int] = {"mimo": 0, "local_audio": 0, "llm": 0, "mix": 0}
    maximum: dict[str, int] = {"mimo": 0, "local_audio": 0, "llm": 0, "mix": 0}
    lock = threading.Lock()
    release = threading.Event()
    entered: dict[str, threading.Event] = {
        "mimo": threading.Event(),
        "local_audio": threading.Event(),
        "llm": threading.Event(),
        "mix": threading.Event(),
    }
    resource_by_command = {
        "synthesize_chapter_audio": "mimo",
        "synthesize_segment_audio": "mimo",
        "transcribe_chapter_audio": "local_audio",
        "generate_audio_assets": "local_audio",
        "plan_chapter_audio": "llm",
        "mix_chapter_audio": "mix",
        "convert_to_mp3": "mix",
    }

    mimo_process_concurrencies: list[str | None] = []
    mimo_rate_state_paths: list[str | None] = []

    def fake_subprocess_run(_args, **kwargs):
        command = str(_args[3])
        resource = resource_by_command[command]
        if resource == "mimo":
            environment = kwargs.get("env")
            mimo_process_concurrencies.append(
                environment.get("AUDIOBOOK_MIMO_CONCURRENCY")
                if isinstance(environment, dict)
                else None
            )
            mimo_rate_state_paths.append(
                environment.get("AUDIOBOOK_MIMO_RATE_STATE_PATH")
                if isinstance(environment, dict)
                else None
            )
        with lock:
            active[resource] += 1
            maximum[resource] = max(maximum[resource], active[resource])
            entered[resource].set()
        release.wait(timeout=2)
        with lock:
            active[resource] -= 1
        output_path = Path(_args[5])
        output_path.write_text(
            json.dumps({"status": "succeeded", "warnings": [], "artifacts": []}),
            encoding="utf-8",
        )
        return type("Completed", (), {"stderr": "", "returncode": 0})()

    monkeypatch.setattr("audiobook_worker.web_server.subprocess.run", fake_subprocess_run)
    requests = [
        ("synthesize_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("synthesize_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("synthesize_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("synthesize_segment_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("transcribe_chapter_audio", {"voiceAudioPath": str(tmp_path / "voice.wav")}),
        ("transcribe_chapter_audio", {"voiceAudioPath": str(tmp_path / "voice.wav")}),
        ("generate_audio_assets", {"scriptPath": str(tmp_path / "script.json"), "outputDirectory": str(tmp_path / "audio-assets")}),
        ("generate_audio_assets", {"scriptPath": str(tmp_path / "script.json"), "outputDirectory": str(tmp_path / "audio-assets")}),
        ("plan_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("plan_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("plan_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("mix_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("mix_chapter_audio", {"scriptPath": str(tmp_path / "script.json")}),
        ("convert_to_mp3", {"wavPath": str(tmp_path / "voice.wav")}),
    ]
    threads = [
        threading.Thread(target=state.run_worker, args=(command, request))
        for command, request in requests
    ]
    for thread in threads:
        thread.start()
    try:
        assert entered["mimo"].wait(timeout=1)
        assert entered["local_audio"].wait(timeout=1)
        assert entered["llm"].wait(timeout=1)
        assert entered["mix"].wait(timeout=1)
        time.sleep(0.05)
        assert maximum == {"mimo": 1, "local_audio": 4, "llm": 2, "mix": 2}
        assert mimo_process_concurrencies == ["1"]
        assert mimo_rate_state_paths == [str(state.mimo_rate_state_path)]
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        state.close()
    assert all(not thread.is_alive() for thread in threads)
    assert mimo_process_concurrencies == ["1", "1", "1", "1"]
    assert mimo_rate_state_paths == [str(state.mimo_rate_state_path)] * 4


def test_direct_voxcpm2_tts_requests_use_the_single_local_voxcpm_resource(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    script_path = tmp_path / "scripts" / "chapter_001.json"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("{}", encoding="utf-8")
    active = 0
    maximum = 0
    started = threading.Event()
    release = threading.Event()
    environments: list[dict[str, str]] = []
    lock = threading.Lock()

    def fake_subprocess_run(arguments, **kwargs):
        nonlocal active, maximum
        assert arguments[3] == "synthesize_chapter_audio"
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        with lock:
            environments.append(environment)
            active += 1
            maximum = max(maximum, active)
        started.set()
        try:
            release.wait(timeout=2)
        finally:
            with lock:
                active -= 1
        Path(arguments[5]).write_text(
            json.dumps({"status": "succeeded", "warnings": [], "artifacts": []}),
            encoding="utf-8",
        )
        return type("Completed", (), {"stderr": "", "returncode": 0})()

    monkeypatch.setattr(
        "audiobook_worker.web_server.subprocess.run",
        fake_subprocess_run,
    )
    request = {
        "scriptPath": str(script_path),
        "outputDirectory": str(tmp_path / "segments" / "voxcpm2"),
        "backend": "voxcpm2",
        "modelId": "VoxCPM2",
    }
    threads = [
        threading.Thread(
            target=state.run_worker,
            args=("synthesize_chapter_audio", dict(request)),
        )
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    try:
        assert started.wait(timeout=1)
        # While a direct local-model request is admitted, the independent
        # MiMo permit remains available and the other VoxCPM2 requests wait.
        assert state.worker_resource_semaphores["mimo"].acquire(blocking=False)
        state.worker_resource_semaphores["mimo"].release()
        time.sleep(0.05)
        assert maximum == 1
        assert len(environments) == 1
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)
        state.close()

    assert all(not thread.is_alive() for thread in threads)
    assert maximum == 1
    assert len(environments) == 3


def test_batch_generation_persists_stage_and_chapter_durations(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001",))

    def fake_worker(command: str, request: dict[str, object]):
        time.sleep(0.002)
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001"]}
    )
    finished = _wait_for_batch(
        state, str(started["batchId"]), lambda response: response["status"] == "succeeded"
    )

    chapter = finished["chapters"][0]
    assert isinstance(chapter["durationSeconds"], float)
    assert chapter["durationSeconds"] > 0
    assert set(chapter["stageTimings"]) == {
        "voice", "transcript", "audio_plan", "stable_audio", "mix"
    }
    # Voice has synthesis plus assembly, so it must accumulate both commands.
    assert chapter["stageTimings"]["voice"] >= 0.004
    state.close()


def test_batch_generation_upgrades_legacy_database_with_timing_columns(tmp_path: Path):
    database_path = tmp_path / "audiobook.db"
    connection = __import__("sqlite3").connect(database_path)
    connection.execute(
        "CREATE TABLE generation_batch_chapters ("
        "batch_id TEXT, chapter_id TEXT, position INTEGER, status TEXT, current_stage TEXT, "
        "error TEXT, voice_audio_path TEXT, mixed_audio_path TEXT, audio_assets_json TEXT, "
        "started_at REAL, completed_at REAL, updated_at REAL, PRIMARY KEY (batch_id, chapter_id))"
    )
    connection.commit()
    connection.close()

    state = ServerState(tmp_path)
    columns = {
        row[1]
        for row in state.db.execute("PRAGMA table_info(generation_batch_chapters)").fetchall()
    }
    assert {"duration_seconds", "stage_timings_json", "next_stage", "stage_state"} <= columns
    state.close()


def test_batch_generation_continues_after_one_chapter_failure(tmp_path: Path, monkeypatch):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001", "chapter_002"))
    calls: list[tuple[str, str]] = []

    def fake_worker(command: str, request: dict[str, object]):
        chapter_id = str(request.get("chapterId"))
        calls.append((chapter_id, command))
        if chapter_id == "chapter_001" and command == "plan_chapter_audio":
            return {
                "status": "failed",
                "warnings": [],
                "artifacts": [],
                "error": {"message": "planner unavailable"},
            }
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )
    finished = _wait_for_batch(
        state,
        str(started["batchId"]),
        lambda response: response["status"] == "completed_with_errors",
    )

    assert finished["failedCount"] == 1
    assert finished["succeededCount"] == 1
    assert finished["chapters"][0]["currentStage"] == "audio_plan"
    assert finished["chapters"][1]["status"] == "succeeded"
    assert ("chapter_002", "mix_chapter_audio") in calls
    state.close()


def test_batch_generation_cancel_marks_remaining_chapters_without_losing_completed(
    tmp_path: Path, monkeypatch
):
    state = ServerState(tmp_path)
    _seed_batch_book(state, ("chapter_001", "chapter_002"))
    entered = __import__("threading").Event()
    release = __import__("threading").Event()

    def fake_worker(command: str, request: dict[str, object]):
        if str(request.get("chapterId")) == "chapter_001" and command == "synthesize_chapter_audio":
            entered.set()
            release.wait(timeout=2)
        return {"status": "succeeded", "warnings": [], "artifacts": []}

    monkeypatch.setattr(state, "run_worker", fake_worker)
    started = state.start_batch_generation(
        {"bookId": "book_123", "chapterIds": ["chapter_001", "chapter_002"]}
    )
    assert entered.wait(timeout=1)
    state.cancel_batch_generation(str(started["batchId"]))
    release.set()
    finished = _wait_for_batch(
        state, str(started["batchId"]), lambda response: response["status"] == "cancelled"
    )

    assert all(chapter["status"] == "cancelled" for chapter in finished["chapters"])
    state.close()


def test_batch_generation_resumes_only_the_interrupted_stage_after_restart(tmp_path: Path, monkeypatch):
    first_state = ServerState(tmp_path)
    _seed_batch_book(first_state, ("chapter_001",))
    now = time.time()
    first_state.db.execute(
        "INSERT INTO generation_batches (id, book_id, status, force, cache_segments, cancel_requested, created_at, updated_at) "
        "VALUES ('batch_1', 'book_123', 'running', 0, 1, 0, ?, ?)",
        (now, now),
    )
    first_state.db.execute(
        "INSERT INTO generation_batch_chapters (batch_id, chapter_id, position, status, current_stage, updated_at) "
        "VALUES ('batch_1', 'chapter_001', 0, 'running', 'audio_plan', ?)",
        (now,),
    )
    first_state.db.execute(
        "UPDATE generation_batch_chapters SET next_stage = 'audio_plan', stage_state = 'running', "
        "voice_audio_path = ?, stage_timings_json = ? WHERE batch_id = 'batch_1' AND chapter_id = 'chapter_001'",
        ("/existing/chapter_001.wav", json.dumps({"voice": 12.5})),
    )
    first_state.db.commit()
    first_state.close()

    monkeypatch.setattr(ServerState, "_launch_batch_generation_locked", lambda self, batch_id: None)
    resumed = ServerState(tmp_path)
    payload = resumed.batch_generation_status("batch_1")

    assert payload["status"] == "queued"
    chapter = payload["chapters"][0]
    assert chapter["status"] == "running"
    assert chapter["currentStage"] is None
    assert chapter["nextStage"] == "audio_plan"
    assert chapter["stageState"] == "ready"
    assert chapter["voiceAudioPath"] == "/existing/chapter_001.wav"
    assert chapter["stageTimings"] == {"voice": 12.5}
    resumed.close()


def test_web_state_accepts_legacy_unicode_book_ids(tmp_path: Path):
    state = ServerState(tmp_path)
    book_id = "中国骑兵_1784689201114"
    work_dir = state.work_directory(book_id)
    chapters_dir = work_dir / "chapters"
    chapters_dir.mkdir(parents=True)

    state.create_book(
        {
            "id": book_id,
            "title": "中国骑兵",
            "sourcePath": str(tmp_path / "uploads" / "book.txt"),
            "workDir": str(work_dir),
        }
    )
    state.upsert_chapter(
        {
            "id": "chapter_001",
            "bookId": book_id,
            "title": "第一章",
            "status": "pending",
            "scriptPath": None,
        }
    )
    text_path = chapters_dir / "chapter_001.txt"
    text_path.write_text("上传的正文", encoding="utf-8")

    assert state.chapter_text_path(book_id, "chapter_001") == text_path.resolve()


def test_web_character_registry_keeps_system_id_aliases_and_manual_voice(tmp_path: Path):
    state = ServerState(tmp_path)
    state.upsert_character(
        {
            "id": "char_001",
            "bookId": "book_123",
            "canonicalName": "李怀玉",
            "gender": "female",
            "ageClass": "adult",
            "identityStatus": "confirmed",
            "voiceId": "character_auto_001",
            "voiceSource": "auto",
            "confidence": 0.9,
            "aliases": '["怀玉"]',
        }
    )
    state.upsert_character(
        {
            "id": "char_001",
            "bookId": "book_123",
            "canonicalName": "李怀玉",
            "identityStatus": "confirmed",
            "voiceId": "female_adult_05",
            "voiceSource": "manual",
            "confidence": 0.95,
            "aliases": '["怀玉", "长公主"]',
        }
    )
    state.upsert_character(
        {
            "id": "model_candidate_7",
            "bookId": "book_123",
            "canonicalName": "长公主",
            "identityStatus": "provisional",
            "voiceId": "character_auto_ignored",
            "voiceSource": "auto",
            "confidence": 0.4,
            "aliases": "[]",
        }
    )

    characters = state.characters("book_123")
    assert len(characters) == 1
    character = characters[0]
    assert character["id"] == "char_001"
    assert character["voiceId"] == "female_adult_05"
    assert character["voiceSource"] == "manual"
    assert character["identityStatus"] == "confirmed"
    assert set(json.loads(str(character["aliases"]))) == {"怀玉", "长公主"}
    alias_rows = state.db.execute(
        "SELECT alias_key FROM character_aliases WHERE book_id = ? ORDER BY alias_key",
        ("book_123",),
    ).fetchall()
    assert [row["alias_key"] for row in alias_rows] == ["怀玉", "李怀玉", "长公主"]
