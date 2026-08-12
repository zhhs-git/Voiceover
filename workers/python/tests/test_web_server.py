import json
import time
import wave
from pathlib import Path

from audiobook_worker.web_server import ServerState, safe_filename


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 44100, 4410, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 4410)


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


def test_batch_generation_runs_chapters_in_order_and_persists_outputs(
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

    assert [chapter for chapter, _ in calls] == [
        "chapter_001",
        "chapter_001",
        "chapter_001",
        "chapter_001",
        "chapter_001",
        "chapter_001",
        "chapter_002",
        "chapter_002",
        "chapter_002",
        "chapter_002",
        "chapter_002",
        "chapter_002",
    ]
    assert [command for _, command in calls[:6]] == [
        "synthesize_chapter_audio",
        "assemble_chapter_audio",
        "transcribe_chapter_audio",
        "plan_chapter_audio",
        "generate_audio_assets",
        "mix_chapter_audio",
    ]
    assert finished["succeededCount"] == 2
    assert all(chapter["status"] == "succeeded" for chapter in finished["chapters"])
    assert finished["chapters"][0]["mixedAudioPath"] == str(
        work_dir / "audio" / "chapter_001_mixed.wav"
    )
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


def test_batch_generation_resumes_queued_rows_after_restart(tmp_path: Path, monkeypatch):
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
        "VALUES ('batch_1', 'chapter_001', 0, 'running', 'mix', ?)",
        (now,),
    )
    first_state.db.commit()
    first_state.close()

    monkeypatch.setattr(ServerState, "_launch_batch_generation_locked", lambda self, batch_id: None)
    resumed = ServerState(tmp_path)
    payload = resumed.batch_generation_status("batch_1")

    assert payload["status"] == "queued"
    assert payload["chapters"][0]["status"] == "queued"
    assert payload["chapters"][0]["currentStage"] is None
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
