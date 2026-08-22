"""Project-local LLM environment configuration.

The web settings page is allowed to update the non-committed project ``.env``
file, but the API key is deliberately write-only at the browser boundary.  This
module contains the small dotenv reader/writer used by both the web server and
the isolated worker resolver so the two paths cannot drift.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


CANONICAL_LLM_MODEL_KEY = "AUDIOBOOK_LLM_MODEL"
CANONICAL_LLM_BASE_URL_KEY = "AUDIOBOOK_LLM_BASE_URL"
CANONICAL_LLM_API_KEY = "AUDIOBOOK_LLM_API_KEY"

_DOTENV_KEYS = (
    CANONICAL_LLM_MODEL_KEY,
    CANONICAL_LLM_BASE_URL_KEY,
    CANONICAL_LLM_API_KEY,
)
_KEY_LINE = re.compile(
    r"^(?P<prefix>\s*(?:export\s+)?)(?P<key>AUDIOBOOK_LLM_MODEL|"
    r"AUDIOBOOK_LLM_BASE_URL|AUDIOBOOK_LLM_API_KEY)(?P<separator>\s*=\s*).*$"
)
_DOTENV_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class LlmEnvironment:
    """Resolved provider values; the API key stays internal to Python."""

    model_id: str = ""
    base_url: str = ""
    api_key: str = ""

    @property
    def api_key_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class DotenvSnapshot:
    path: Path
    existed: bool
    content: bytes
    mode: int | None
    process_environment: dict[str, str | None]


def project_root() -> Path:
    """Locate the audiobook-generator repository root from this module."""

    return Path(__file__).resolve().parents[3]


def dotenv_path(root: Path | None = None) -> Path:
    return (root or project_root()).resolve() / ".env"


def _decode_dotenv_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value[1:-1]
        return str(decoded)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return value


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read simple KEY=value entries without logging or exposing values."""

    target = path or dotenv_path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if key in _DOTENV_KEYS:
            values[key] = _decode_dotenv_value(raw_value)
    return values


def _first_value(
    file_values: dict[str, str],
    environment: dict[str, str],
    canonical: str,
    aliases: tuple[str, ...] = (),
) -> str:
    if canonical in file_values:
        return file_values[canonical].strip()
    if canonical in environment:
        return str(environment[canonical] or "").strip()
    for alias in aliases:
        value = str(environment.get(alias) or "").strip()
        if value:
            return value
    return ""


def read_llm_environment(
    root: Path | None = None,
    *,
    environment: dict[str, str] | None = None,
) -> LlmEnvironment:
    """Resolve canonical project values, then legacy process aliases."""

    file_values = read_dotenv(dotenv_path(root))
    process_values = environment if environment is not None else os.environ
    return LlmEnvironment(
        model_id=_first_value(
            file_values,
            process_values,
            CANONICAL_LLM_MODEL_KEY,
            ("MODEL_ID", "OPENAI_MODEL"),
        ),
        base_url=_first_value(
            file_values,
            process_values,
            CANONICAL_LLM_BASE_URL_KEY,
            ("MODEL_BASE_URL", "OPENAI_BASE_URL"),
        ),
        api_key=_first_value(
            file_values,
            process_values,
            CANONICAL_LLM_API_KEY,
            ("MODEL_API_KEY", "OPENAI_API_KEY"),
        ),
    )


def validate_llm_base_url(value: object) -> str:
    """Validate and normalize an OpenAI-compatible HTTP(S) endpoint."""

    base_url = str(value or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("LLM 服务 URL 不能为空。")
    if any(character.isspace() for character in base_url):
        raise ValueError("LLM 服务 URL 不能包含空白字符。")
    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        # Accessing ``port`` validates malformed bracketed hosts and ports.
        parsed.port
    except ValueError as error:
        raise ValueError("LLM 服务 URL 必须是带主机名的 http 或 https 地址。") from error
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("LLM 服务 URL 必须是带主机名的 http 或 https 地址。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("LLM 服务 URL 不能包含内嵌凭据、查询参数或 fragment。")
    return base_url


def validate_llm_model_id(value: object) -> str:
    """Validate the model identifier without imposing a provider catalog."""

    model_id = str(value or "").strip()
    if not model_id:
        raise ValueError("LLM 模型 ID 不能为空。")
    if any(character.isspace() for character in model_id):
        raise ValueError("LLM 模型 ID 不能包含空白字符。")
    return model_id


def _encode_dotenv_value(value: str) -> str:
    if value and re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def capture_dotenv(root: Path | None = None) -> DotenvSnapshot:
    path = dotenv_path(root)
    process_environment = {key: os.environ.get(key) for key in _DOTENV_KEYS}
    try:
        content = path.read_bytes()
        mode = stat.S_IMODE(path.stat().st_mode)
        return DotenvSnapshot(path, True, content, mode, process_environment)
    except FileNotFoundError:
        return DotenvSnapshot(path, False, b"", None, process_environment)


def _replace_dotenv(
    snapshot: DotenvSnapshot,
    content: bytes,
    *,
    mode: int = 0o600,
) -> None:
    snapshot.path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".env.", suffix=".tmp", dir=snapshot.path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, snapshot.path)
        os.chmod(snapshot.path, mode)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _apply_process_environment(config: LlmEnvironment) -> None:
    os.environ[CANONICAL_LLM_MODEL_KEY] = config.model_id
    os.environ[CANONICAL_LLM_BASE_URL_KEY] = config.base_url
    if config.api_key:
        os.environ[CANONICAL_LLM_API_KEY] = config.api_key
    else:
        os.environ.pop(CANONICAL_LLM_API_KEY, None)


def write_llm_environment(
    root: Path | None = None,
    *,
    model_id: str,
    base_url: str,
    api_key: str | None = None,
    clear_api_key: bool = False,
) -> DotenvSnapshot:
    """Atomically update canonical LLM keys and return the old file snapshot.

    ``api_key=None`` means retain the currently resolved key.  The explicit
    ``clear_api_key`` flag is required to remove it, preventing an accidental
    empty form field from deleting a working credential.
    """

    with _DOTENV_WRITE_LOCK:
        previous = capture_dotenv(root)
        current = read_llm_environment(root)
        normalized_model = validate_llm_model_id(model_id)
        normalized_url = validate_llm_base_url(base_url)

        if clear_api_key:
            next_api_key: str | None = None
        elif api_key is not None and str(api_key).strip():
            next_api_key = str(api_key).strip()
        elif current.api_key:
            # Migrate a legacy alias into the canonical project file when the
            # user saves other fields without typing the existing secret again.
            next_api_key = current.api_key
        else:
            next_api_key = None

        updates: dict[str, str | None] = {
            CANONICAL_LLM_MODEL_KEY: normalized_model,
            CANONICAL_LLM_BASE_URL_KEY: normalized_url,
            CANONICAL_LLM_API_KEY: next_api_key,
        }
        output: list[str] = []
        seen: set[str] = set()
        try:
            original = previous.content.decode("utf-8") if previous.existed else ""
            for line in original.splitlines(keepends=True):
                match = _KEY_LINE.match(line.rstrip("\r\n"))
                if not match:
                    output.append(line)
                    continue
                key = match.group("key")
                if key in seen:
                    continue
                seen.add(key)
                value = updates[key]
                if value is None:
                    continue
                ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                output.append(
                    f"{match.group('prefix')}{key}{match.group('separator')}"
                    f"{_encode_dotenv_value(value)}{ending}"
                )
            if output and not output[-1].endswith(("\n", "\r")):
                output.append("\n")
            for key in _DOTENV_KEYS:
                if key in seen or updates[key] is None:
                    continue
                output.append(f"{key}={_encode_dotenv_value(str(updates[key]))}\n")
            _replace_dotenv(previous, "".join(output).encode("utf-8"))
        except Exception:
            # The accepted file remains untouched because replacement is atomic.
            raise

        _apply_process_environment(
            LlmEnvironment(normalized_model, normalized_url, next_api_key or "")
        )
        return previous


def restore_dotenv(snapshot: DotenvSnapshot) -> None:
    """Restore a snapshot after a later SQLite transaction fails."""

    with _DOTENV_WRITE_LOCK:
        if snapshot.existed:
            _replace_dotenv(snapshot, snapshot.content, mode=snapshot.mode or 0o600)
        else:
            try:
                snapshot.path.unlink()
            except FileNotFoundError:
                pass
        for key, previous_value in snapshot.process_environment.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value
