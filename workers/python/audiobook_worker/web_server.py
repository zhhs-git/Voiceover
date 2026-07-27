"""Small dependency-free HTTP server for the shared LAN web application.

The desktop application historically delegated all filesystem, SQLite, and
worker operations to Tauri commands.  This module exposes the same operations
over HTTP so the React application can be used by browsers on the LAN.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, unquote, urlparse


MAX_UPLOAD_BYTES = int(os.environ.get("AUDIOBOOK_MAX_UPLOAD_BYTES", str(2 * 1024**3)))
WORKER_TIMEOUT_SECONDS = int(os.environ.get("AUDIOBOOK_WORKER_TIMEOUT_SECONDS", str(24 * 60 * 60)))
SAFE_ID_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
GENERIC_CHARACTER_LABELS = {
    "小姐", "少爷", "姑娘", "公子", "夫人", "太太", "老爷", "殿下", "陛下",
    "皇上", "皇后", "公主", "王爷", "世子", "大人", "先生", "女士", "母亲",
    "父亲", "娘", "爹", "妈妈", "爸爸", "mother", "father", "wife", "husband",
    "miss", "mrs", "ms", "mr", "sir", "madam", "lady", "lord", "girl", "boy",
    "woman", "man",
}


def default_data_directory() -> Path:
    configured = os.environ.get("AUDIOBOOK_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "audiobook-generator"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "audiobook-generator"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "audiobook-generator"


def safe_filename(name: str) -> str:
    raw_name = Path(unquote(name)).name
    suffix = Path(raw_name).suffix
    stem = raw_name[: -len(suffix)] if suffix else raw_name
    safe_stem = re.sub(r"[^\w-]+", "_", stem, flags=re.UNICODE).strip("._") or "book"
    safe_suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix)
    return f"{safe_stem}{safe_suffix}"


def character_name_key(value: object) -> str:
    """Normalize a character name for the book-scoped alias index."""
    return "".join(
        character
        for character in str(value or "").strip().casefold()
        if character.isalnum()
    )


GENERIC_CHARACTER_KEYS = {
    character_name_key(label) for label in GENERIC_CHARACTER_LABELS
}


def is_generic_character_key(value: object) -> bool:
    return character_name_key(value) in GENERIC_CHARACTER_KEYS


def character_aliases(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        alias = str(item or "").strip()
        key = character_name_key(alias)
        if alias and key and key not in seen:
            result.append(alias)
            seen.add(key)
    return result


class ServerState:
    def __init__(self, data_directory: Path, frontend_directory: Path | None = None) -> None:
        self.data_directory = data_directory.resolve()
        self.frontend_directory = frontend_directory.resolve() if frontend_directory else None
        self.books_directory = self.data_directory / "books"
        self.uploads_directory = self.data_directory / "uploads"
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.books_directory.mkdir(parents=True, exist_ok=True)
        self.uploads_directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.data_directory / "audiobook.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db_lock = threading.RLock()
        self.worker_lock = threading.RLock()
        self.initialize_database()

    def initialize_database(self) -> None:
        with self.db_lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS books (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, source_path TEXT NOT NULL,
                    source_language TEXT NOT NULL, output_language TEXT NOT NULL, work_dir TEXT NOT NULL,
                    imported_at TEXT, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS chapters (
                    id TEXT NOT NULL, book_id TEXT NOT NULL, title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', script_path TEXT,
                    PRIMARY KEY (id, book_id)
                );
                CREATE TABLE IF NOT EXISTS characters (
                    id TEXT NOT NULL, book_id TEXT NOT NULL, canonical_name TEXT NOT NULL,
                    gender TEXT, age_class TEXT, identity_status TEXT DEFAULT 'confirmed',
                    voice_id TEXT, voice_source TEXT,
                    voice_assignment_version INTEGER, voice_profile TEXT, fallback_voice_id TEXT,
                    voice_description TEXT, confidence REAL DEFAULT 0.0,
                    aliases TEXT DEFAULT '[]', updated_at TEXT,
                    PRIMARY KEY (id, book_id)
                );
                CREATE TABLE IF NOT EXISTS character_aliases (
                    book_id TEXT NOT NULL, character_id TEXT NOT NULL,
                    alias_key TEXT NOT NULL, alias TEXT NOT NULL, updated_at TEXT,
                    PRIMARY KEY (book_id, character_id, alias_key)
                );
                CREATE INDEX IF NOT EXISTS idx_books_source_path ON books(source_path);
                CREATE INDEX IF NOT EXISTS idx_characters_book_id ON characters(book_id);
                CREATE INDEX IF NOT EXISTS idx_character_aliases_lookup
                    ON character_aliases(book_id, alias_key);
                CREATE INDEX IF NOT EXISTS idx_chapters_book_id ON chapters(book_id);
                """
            )
            self._ensure_columns(
                "books",
                {"imported_at": "TEXT", "updated_at": "TEXT"},
            )
            self._ensure_columns(
                "characters",
                {
                    "age_class": "TEXT",
                    "identity_status": "TEXT",
                    "voice_source": "TEXT",
                    "voice_assignment_version": "INTEGER",
                    "voice_profile": "TEXT",
                    "fallback_voice_id": "TEXT",
                    "voice_description": "TEXT",
                },
            )
            self.db.commit()

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        existing = {
            row[1]
            for row in self.db.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, definition in columns.items():
            if name not in existing:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def path_in_data(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.data_directory / candidate
        candidate = candidate.resolve()
        if candidate != self.data_directory and self.data_directory not in candidate.parents:
            raise ValueError("path is outside the audiobook data directory")
        return candidate

    def work_directory(self, book_id: str) -> Path:
        if not SAFE_ID_PATTERN.fullmatch(book_id):
            raise ValueError("invalid book id")
        return self.books_directory / book_id

    def row_to_dict(self, row: sqlite3.Row) -> dict[str, object]:
        return dict(row)

    def list_books(self) -> list[dict[str, object]]:
        with self.db_lock:
            rows = self.db.execute(
                "SELECT id, title, source_path, work_dir, imported_at "
                "FROM books ORDER BY imported_at DESC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "sourcePath": row["source_path"],
                "workDir": row["work_dir"],
                "importedAt": row["imported_at"],
            }
            for row in rows
        ]

    def get_book(self, source_path: str) -> dict[str, object] | None:
        with self.db_lock:
            row = self.db.execute(
                "SELECT id, title, source_path, work_dir, imported_at FROM books WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "title": row["title"],
            "sourcePath": row["source_path"],
            "workDir": row["work_dir"],
            "importedAt": row["imported_at"],
        }

    def create_book(self, record: dict[str, object]) -> None:
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        with self.db_lock:
            self.db.execute(
                "INSERT OR REPLACE INTO books "
                "(id, title, source_path, source_language, output_language, work_dir, imported_at, updated_at) "
                "VALUES (?, ?, ?, 'en', 'en', ?, ?, ?)",
                (record["id"], record["title"], record["sourcePath"], record["workDir"], now, now),
            )
            self.db.commit()

    def delete_book(self, book_id: str) -> None:
        with self.db_lock:
            row = self.db.execute("SELECT work_dir FROM books WHERE id = ?", (book_id,)).fetchone()
            if row is None:
                return
            work_dir = self.path_in_data(row["work_dir"])
            with self.db:
                self.db.execute("DELETE FROM characters WHERE book_id = ?", (book_id,))
                self.db.execute("DELETE FROM character_aliases WHERE book_id = ?", (book_id,))
                self.db.execute("DELETE FROM chapters WHERE book_id = ?", (book_id,))
                self.db.execute("DELETE FROM books WHERE id = ?", (book_id,))
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def upsert_chapter(self, record: dict[str, object]) -> None:
        with self.db_lock:
            self.db.execute(
                "INSERT OR REPLACE INTO chapters (id, book_id, title, status, script_path) VALUES (?, ?, ?, ?, ?)",
                (record["id"], record["bookId"], record["title"], record["status"], record.get("scriptPath")),
            )
            self.db.commit()

    def chapters(self, book_id: str, with_scripts_only: bool = False) -> list[dict[str, object]]:
        query = "SELECT id, title, status, script_path FROM chapters WHERE book_id = ?"
        params: tuple[object, ...] = (book_id,)
        if with_scripts_only:
            query += " AND script_path IS NOT NULL"
        query += " ORDER BY id ASC"
        with self.db_lock:
            rows = self.db.execute(query, params).fetchall()
        return [
            {"id": row["id"], "title": row["title"], "status": row["status"], "scriptPath": row["script_path"]}
            for row in rows
        ]

    def chapter_text_path(self, book_id: str, chapter_id: str) -> Path:
        if not SAFE_ID_PATTERN.fullmatch(book_id):
            raise ValueError("invalid book id")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", chapter_id):
            raise ValueError("invalid chapter id")
        with self.db_lock:
            row = self.db.execute(
                "SELECT 1 FROM chapters WHERE book_id = ? AND id = ?",
                (book_id, chapter_id),
            ).fetchone()
        if row is None:
            raise FileNotFoundError(f"chapter not found: {chapter_id}")
        return self.path_in_data(
            self.work_directory(book_id) / "chapters" / f"{chapter_id}.txt"
        )

    def upsert_character(self, record: dict[str, object]) -> None:
        incoming_aliases = character_aliases(record.get("aliases"))
        incoming_canonical = str(record.get("canonicalName") or "").strip()
        incoming_names = {
            character_name_key(name)
            for name in [incoming_canonical, *incoming_aliases]
            if character_name_key(name) and not is_generic_character_key(name)
        }
        with self.db_lock:
            rows = self.db.execute(
                "SELECT id, canonical_name, aliases, gender, age_class, identity_status, "
                "voice_id, voice_source, voice_assignment_version, voice_profile, "
                "fallback_voice_id, voice_description, confidence "
                "FROM characters WHERE book_id = ?",
                (record["bookId"],),
            ).fetchall()
            existing = next((row for row in rows if row["id"] == record["id"]), None)
            if existing is None:
                for row in rows:
                    existing_names = {
                        character_name_key(name)
                        for name in [row["canonical_name"], *character_aliases(row["aliases"])]
                        if character_name_key(name) and not is_generic_character_key(name)
                    }
                    if incoming_names & existing_names:
                        existing = row
                        break

            target_id = existing["id"] if existing is not None else record["id"]
            canonical_name = (
                existing["canonical_name"]
                if existing is not None and existing["canonical_name"]
                else incoming_canonical
            )
            merged_aliases = character_aliases(
                [
                    *(character_aliases(existing["aliases"]) if existing is not None else []),
                    *incoming_aliases,
                    incoming_canonical,
                ]
            )
            canonical_key = character_name_key(canonical_name)
            merged_aliases = [
                alias for alias in merged_aliases
                if character_name_key(alias) != canonical_key
            ]

            def value(name: str):
                incoming = record.get(name)
                if existing is None or incoming is not None:
                    return incoming
                existing_name = {
                    "canonicalName": "canonical_name",
                    "ageClass": "age_class",
                    "identityStatus": "identity_status",
                    "voiceId": "voice_id",
                    "voiceSource": "voice_source",
                    "voiceAssignmentVersion": "voice_assignment_version",
                    "voiceProfile": "voice_profile",
                    "fallbackVoiceId": "fallback_voice_id",
                    "voiceDescription": "voice_description",
                }.get(name, name)
                return existing[existing_name]

            incoming_voice_source = record.get("voiceSource")
            existing_manual = existing is not None and existing["voice_source"] == "manual"
            preserve_manual_voice = existing_manual and incoming_voice_source != "manual"
            voice_id = existing["voice_id"] if preserve_manual_voice else value("voiceId")
            voice_source = existing["voice_source"] if preserve_manual_voice else value("voiceSource")
            voice_assignment_version = (
                existing["voice_assignment_version"]
                if preserve_manual_voice
                else value("voiceAssignmentVersion")
            )
            voice_profile = existing["voice_profile"] if preserve_manual_voice else value("voiceProfile")
            fallback_voice_id = existing["fallback_voice_id"] if preserve_manual_voice else value("fallbackVoiceId")
            voice_description = existing["voice_description"] if preserve_manual_voice else value("voiceDescription")
            existing_status = existing["identity_status"] if existing is not None else None
            incoming_status = record.get("identityStatus")
            identity_status = (
                "confirmed"
                if "confirmed" in {existing_status, incoming_status}
                else incoming_status or existing_status or "provisional"
            )
            confidence_values = [
                float(record.get("confidence") or 0.0),
                float(existing["confidence"] or 0.0) if existing is not None else 0.0,
            ]
            aliases_json = json.dumps(merged_aliases, ensure_ascii=False)
            self.db.execute(
                """INSERT INTO characters
                (id, book_id, canonical_name, gender, age_class, identity_status, voice_id, voice_source,
                 voice_assignment_version, voice_profile, fallback_voice_id, voice_description,
                 confidence, aliases, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(id, book_id) DO UPDATE SET
                  canonical_name=excluded.canonical_name,
                  gender=COALESCE(excluded.gender, characters.gender),
                  age_class=COALESCE(excluded.age_class, characters.age_class),
                  identity_status=CASE WHEN excluded.identity_status = 'confirmed' THEN 'confirmed'
                    ELSE COALESCE(excluded.identity_status, characters.identity_status) END,
                  voice_id=excluded.voice_id,
                  voice_source=excluded.voice_source,
                  voice_assignment_version=excluded.voice_assignment_version,
                  voice_profile=excluded.voice_profile,
                  fallback_voice_id=excluded.fallback_voice_id,
                  voice_description=excluded.voice_description,
                  confidence=excluded.confidence,
                  aliases=excluded.aliases, updated_at=excluded.updated_at""",
                (
                    target_id, record["bookId"], canonical_name, value("gender"), value("ageClass"),
                    identity_status, voice_id, voice_source, voice_assignment_version, voice_profile,
                    fallback_voice_id, voice_description, max(confidence_values), aliases_json,
                ),
            )
            self.db.execute(
                "DELETE FROM character_aliases WHERE book_id = ? AND character_id = ?",
                (record["bookId"], target_id),
            )
            for alias in [canonical_name, *merged_aliases]:
                alias_key = character_name_key(alias)
                if alias_key:
                    self.db.execute(
                        "INSERT OR IGNORE INTO character_aliases "
                        "(book_id, character_id, alias_key, alias, updated_at) VALUES (?, ?, ?, ?, datetime('now'))",
                        (record["bookId"], target_id, alias_key, alias),
                    )
            self.db.commit()

    def characters(self, book_id: str) -> list[dict[str, object]]:
        with self.db_lock:
            rows = self.db.execute(
                "SELECT id, canonical_name, gender, age_class, identity_status, voice_id, voice_source, "
                "voice_assignment_version, voice_profile, fallback_voice_id, voice_description, "
                "confidence, aliases FROM characters WHERE book_id = ?",
                (book_id,),
            ).fetchall()
        return [
            {
                "id": row["id"], "canonicalName": row["canonical_name"], "gender": row["gender"],
                "ageClass": row["age_class"], "identityStatus": row["identity_status"],
                "voiceId": row["voice_id"], "voiceSource": row["voice_source"],
                "voiceAssignmentVersion": row["voice_assignment_version"], "voiceProfile": row["voice_profile"],
                "fallbackVoiceId": row["fallback_voice_id"], "voiceDescription": row["voice_description"],
                "confidence": row["confidence"], "aliases": row["aliases"],
            }
            for row in rows
        ]

    def run_worker(self, command: str, request: dict[str, object]) -> dict[str, object]:
        if command == "_read_file":
            path = self.path_in_data(str(request["path"]))
            return json.loads(path.read_text(encoding="utf-8"))
        if command == "_write_file":
            path = self.path_in_data(str(request["path"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(request["content"]), encoding="utf-8")
            return {"status": "succeeded", "warnings": [], "artifacts": []}

        allowed = {
            "extract_book", "analyze_chapter", "synthesize_segment_audio", "synthesize_chapter_audio",
            "assemble_chapter_audio", "apply_corrections", "refresh_voice_assignments", "check_rights",
            "list_voices", "convert_to_mp3",
        }
        if command not in allowed:
            return {"status": "failed", "warnings": [], "artifacts": [], "error": {"code": "unknown_command", "message": command}}
        self.validate_request_paths(request)
        with self.worker_lock:
            with tempfile.TemporaryDirectory(prefix="audiobook-web-", dir=self.data_directory) as temp:
                input_path = Path(temp) / "input.json"
                output_path = Path(temp) / "output.json"
                input_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
                environment = os.environ.copy()
                environment.setdefault("AUDIOBOOK_TTS_DEVICE", "auto")
                environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                try:
                    completed = subprocess.run(
                        [sys.executable, "-m", "audiobook_worker.cli", command, str(input_path), str(output_path)],
                        cwd=Path(__file__).resolve().parents[1],
                        env=environment,
                        capture_output=True,
                        text=True,
                        timeout=WORKER_TIMEOUT_SECONDS,
                    )
                except subprocess.TimeoutExpired:
                    return {"status": "failed", "warnings": [], "artifacts": [], "error": {"code": "worker_timeout", "message": f"Worker command timed out: {command}"}}
                if output_path.exists():
                    try:
                        return json.loads(output_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                message = completed.stderr.strip() or f"worker exited with code {completed.returncode}"
                return {"status": "failed", "warnings": [], "artifacts": [], "error": {"code": "worker_exit_failed", "message": message}}

    def validate_request_paths(self, value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                self.validate_request_paths(child_value, child_key)
        elif isinstance(value, list):
            for child in value:
                self.validate_request_paths(child, key)
        elif isinstance(value, str) and (key.endswith("Path") or key.endswith("Directory") or key in {"path", "outputPath", "wavPath"}):
            self.path_in_data(value)


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class WebHandler(BaseHTTPRequestHandler):
    server_version = "AudiobookGeneratorWeb/1.0"

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    def send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status)

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("request body is too large")
        body = self.rfile.read(length)
        parsed = json.loads(body or b"{}")
        if not isinstance(parsed, dict):
            raise ValueError("request body must be a JSON object")
        return parsed

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-File-Name")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/health":
                self.send_json({"ok": True})
                return
            if path == "/api/invoke":
                self.send_error_json("use POST", HTTPStatus.METHOD_NOT_ALLOWED)
                return
            if path == "/api/files":
                query = parse_qs(parsed.query)
                self.serve_file(self.state.path_in_data(query.get("path", [""])[0]))
                return
            if path.startswith("/api/audio/"):
                parts = path.split("/")
                if len(parts) != 5:
                    raise ValueError("invalid audio path")
                book_id, chapter_id = unquote(parts[3]), unquote(parts[4])
                if not re.fullmatch(r"[A-Za-z0-9_-]+", chapter_id):
                    raise ValueError("invalid chapter id")
                self.serve_file(self.state.work_directory(book_id) / "audio" / f"{chapter_id}.wav")
                return
            if path == "/api/books":
                self.send_json(self.state.list_books())
                return
            parts = path.split("/")
            if (
                len(parts) == 7
                and parts[1] == "api"
                and parts[2] == "books"
                and parts[4] == "chapters"
                and parts[6] == "text"
            ):
                book_id, chapter_id = unquote(parts[3]), unquote(parts[5])
                self.serve_file(self.state.chapter_text_path(book_id, chapter_id))
                return
            if path.startswith("/api/books/"):
                book_id = unquote(path.split("/")[3])
                self.send_json({
                    "chapters": self.state.chapters(book_id),
                    "characters": self.state.characters(book_id),
                })
                return
            self.serve_frontend(path)
        except FileNotFoundError:
            self.send_error_json("file not found", HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error_json(str(error), HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - safety net for server requests
            self.send_error_json(str(error), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path.startswith("/api/books/"):
                book_id = unquote(parsed.path.split("/")[3])
                self.state.delete_book(book_id)
                self.send_json({"ok": True})
                return
            self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_error_json(str(error), HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/books/import":
                self.import_book()
                return
            if parsed.path == "/api/invoke":
                payload = self.read_json()
                command = str(payload.get("command", ""))
                args = payload.get("args", {})
                if not isinstance(args, dict):
                    raise ValueError("invoke args must be an object")
                self.send_json({"value": self.invoke(command, args)})
                return
            self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        except FileNotFoundError:
            self.send_error_json("file not found", HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error_json(str(error), HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover
            self.send_error_json(str(error), HTTPStatus.INTERNAL_SERVER_ERROR)

    def import_book(self) -> None:
        query_filename = parse_qs(urlparse(self.path).query).get("filename", [""])[0]
        content_type = self.headers.get("Content-Type", "").lower()
        if content_type.startswith("multipart/form-data"):
            upload_path, multipart_filename = self.save_multipart_upload()
            requested_filename = (
                self.headers.get("X-File-Name")
                or query_filename
                or multipart_filename
            )
            if not requested_filename:
                upload_path.unlink(missing_ok=True)
                raise ValueError("uploaded file name is missing")
            filename = safe_filename(requested_filename)
            final_upload_path = self.state.uploads_directory / f"{uuid.uuid4().hex}-{filename}"
            upload_path.replace(final_upload_path)
            upload_path = final_upload_path
        else:
            requested_filename = self.headers.get("X-File-Name") or query_filename
            if not requested_filename:
                requested_filename = {
                    "application/pdf": "book.pdf",
                    "application/epub+zip": "book.epub",
                    "text/plain": "book.txt",
                }.get(content_type.split(";", 1)[0], "")
            filename = safe_filename(requested_filename)
            upload_path = self.state.uploads_directory / f"{uuid.uuid4().hex}-{filename}"
            with upload_path.open("wb") as destination:
                self.copy_upload_body(destination)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".epub", ".pdf", ".txt"}:
            upload_path.unlink(missing_ok=True)
            raise ValueError("unsupported book format")
        if upload_path.stat().st_size == 0:
            upload_path.unlink(missing_ok=True)
            raise ValueError(
                "empty upload: browser sent 0 bytes "
                f"(content-length={self.headers.get('Content-Length', 'none')}, "
                f"transfer-encoding={self.headers.get('Transfer-Encoding', 'none')})"
            )

        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "book"
        book_id = f"{stem[:40]}_{secrets.token_hex(4)}"
        work_dir = self.state.work_directory(book_id)
        result = self.state.run_worker("extract_book", {
            "bookPath": str(upload_path),
            "outputDirectory": str(work_dir / "chapters"),
        })
        if result.get("status") != "succeeded":
            self.send_json(result, HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        artifact = (result.get("artifacts") or [{}])[0]
        metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
        if suffix == ".txt":
            metadata = {**metadata, "title": Path(filename).stem}
        self.state.create_book({
            "id": book_id,
            "title": metadata.get("title", Path(filename).stem),
            "sourcePath": str(upload_path),
            "workDir": str(work_dir),
        })
        chapters = metadata.get("chapters", [])
        for chapter in chapters:
            self.state.upsert_chapter({
                "id": chapter["id"], "bookId": book_id, "title": chapter["title"],
                "status": "pending", "scriptPath": None,
            })
        metadata = dict(metadata)
        metadata.update({"bookId": book_id, "workDir": str(work_dir), "sourcePath": str(upload_path)})
        (work_dir / "book-extraction.json").write_text(
            json.dumps(
                {
                    "title": metadata.get("title", Path(filename).stem),
                    "bookId": book_id,
                    "workDir": str(work_dir),
                    "chapters": metadata.get("chapters", []),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        response = dict(result)
        response["artifacts"] = [{**artifact, "metadata": metadata}]
        self.send_json(response)

    def save_multipart_upload(self) -> tuple[Path, str | None]:
        """Extract the first file part from a multipart browser upload."""
        match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", self.headers.get("Content-Type", ""), re.I)
        if not match:
            raise ValueError("multipart boundary is missing")
        boundary = (match.group(1) or match.group(2)).encode("utf-8")
        marker = b"--" + boundary
        raw_path = self.state.data_directory / f".multipart-{uuid.uuid4().hex}.tmp"
        output_path = self.state.uploads_directory / f".upload-{uuid.uuid4().hex}.part"
        try:
            with raw_path.open("wb") as raw:
                self.copy_upload_body(raw)
            with raw_path.open("rb") as raw:
                line = raw.readline()
                while line and not line.rstrip(b"\r\n") == marker:
                    line = raw.readline()
                if not line:
                    raise ValueError("multipart boundary not found")

                headers: dict[str, str] = {}
                line = raw.readline()
                while line not in {b"\r\n", b"\n", b""}:
                    key, separator, value = line.decode("latin-1").partition(":")
                    if separator:
                        headers[key.lower().strip()] = value.strip()
                    line = raw.readline()
                disposition = headers.get("content-disposition", "")
                filename_match = re.search(r'filename="([^"]*)"', disposition)
                multipart_filename = filename_match.group(1) if filename_match else None
                if not multipart_filename:
                    raise ValueError("multipart file name is missing")

                with output_path.open("wb") as output:
                    pending = b""
                    while True:
                        line = raw.readline()
                        if not line:
                            raise ValueError("incomplete multipart upload")
                        if line.startswith(marker):
                            if pending.endswith(b"\r\n"):
                                pending = pending[:-2]
                            elif pending.endswith(b"\n"):
                                pending = pending[:-1]
                            output.write(pending)
                            break
                        if pending:
                            output.write(pending)
                        pending = line
            return output_path, multipart_filename
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        finally:
            raw_path.unlink(missing_ok=True)

    def copy_upload_body(self, destination: BinaryIO) -> None:
        """Copy both normal and chunked browser request bodies to a file."""
        transfer_encoding = self.headers.get("Transfer-Encoding", "").lower()
        if "chunked" in transfer_encoding:
            total = 0
            while True:
                line = self.rfile.readline(128).strip()
                if not line:
                    continue
                try:
                    chunk_size = int(line.split(b";", 1)[0], 16)
                except ValueError as error:
                    raise ValueError("invalid chunked upload") from error
                if chunk_size == 0:
                    self.rfile.readline()
                    break
                total += chunk_size
                if total > MAX_UPLOAD_BYTES:
                    raise ValueError("uploaded file is too large")
                chunk = self.rfile.read(chunk_size)
                if len(chunk) != chunk_size:
                    raise ValueError("incomplete upload")
                destination.write(chunk)
                if self.rfile.read(2) != b"\r\n":
                    raise ValueError("invalid chunked upload terminator")
            return

        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("missing upload length")
        length = int(length_header)
        if length <= 0:
            raise ValueError("empty upload")
        if length > MAX_UPLOAD_BYTES:
            raise ValueError("uploaded file is too large")
        remaining = length
        while remaining:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("incomplete upload")
            destination.write(chunk)
            remaining -= len(chunk)

    def invoke(self, command: str, args: dict[str, object]) -> object:
        if command == "run_worker":
            input_json = args.get("inputJson", "{}")
            request = json.loads(str(input_json))
            return json.dumps(self.state.run_worker(str(args.get("command", "")), request), ensure_ascii=False)
        if command == "file_exists":
            paths = args.get("paths", [])
            if not isinstance(paths, list):
                raise ValueError("paths must be an array")
            return [str(self.state.path_in_data(str(path))) for path in paths if self.state.path_in_data(str(path)).exists()]
        if command == "copy_file":
            source = self.state.path_in_data(str(args["from"]))
            destination = self.state.path_in_data(str(args["to"]))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return str(destination)
        if command == "book_work_dir":
            return str(self.state.work_directory(str(args["bookId"])))
        if command == "db_list_books":
            return self.state.list_books()
        if command == "db_get_book":
            return self.state.get_book(str(args["sourcePath"]))
        if command == "db_create_book":
            self.state.create_book(args)
            return None
        if command == "db_delete_book":
            self.state.delete_book(str(args["bookId"]))
            return None
        if command == "db_upsert_chapter":
            self.state.upsert_chapter(args)
            return None
        if command == "db_get_chapters":
            return self.state.chapters(str(args["bookId"]))
        if command == "db_get_chapters_with_scripts":
            rows = self.state.chapters(str(args["bookId"]), with_scripts_only=True)
            return [{"id": row["id"], "scriptPath": row["scriptPath"]} for row in rows]
        if command == "db_upsert_character":
            self.state.upsert_character(args)
            return None
        if command == "db_get_characters":
            return self.state.characters(str(args["bookId"]))
        raise ValueError(f"unknown invoke command: {command}")

    def serve_file(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        total = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start, end = 0, total - 1
        status = HTTPStatus.OK
        if range_header and range_header.startswith("bytes="):
            start_text, _, end_text = range_header[6:].partition("-")
            start = int(start_text or 0)
            end = int(end_text) if end_text else total - 1
            end = min(end, total - 1)
            status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()
        with path.open("rb") as source:
            source.seek(start)
            remaining = length
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def serve_frontend(self, path: str) -> None:
        if self.state.frontend_directory is None:
            self.send_error_json("frontend is not configured", HTTPStatus.NOT_FOUND)
            return
        requested = path.lstrip("/") or "index.html"
        candidate = (self.state.frontend_directory / requested).resolve()
        if self.state.frontend_directory not in candidate.parents and candidate != self.state.frontend_directory:
            raise ValueError("invalid frontend path")
        if not candidate.is_file():
            candidate = self.state.frontend_directory / "index.html"
        self.serve_file(candidate)


class AudiobookHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: ServerState):
        self.state = state
        super().__init__(address, WebHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the audiobook generator LAN web server")
    parser.add_argument("--host", default=os.environ.get("AUDIOBOOK_WEB_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AUDIOBOOK_WEB_PORT", "8000")))
    parser.add_argument("--data-dir", type=Path, default=default_data_directory())
    parser.add_argument("--frontend-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    state = ServerState(args.data_dir, args.frontend_dir)
    server = AudiobookHTTPServer((args.host, args.port), state)
    print(f"Audiobook Generator Web: http://{args.host}:{args.port}", flush=True)
    print(f"Data directory: {state.data_directory}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
