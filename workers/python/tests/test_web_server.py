import json
from pathlib import Path

from audiobook_worker.web_server import ServerState, safe_filename


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
