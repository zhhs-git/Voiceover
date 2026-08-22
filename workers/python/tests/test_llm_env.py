import os
import stat
from pathlib import Path

import pytest

from audiobook_worker.llm_env import (
    CANONICAL_LLM_API_KEY,
    CANONICAL_LLM_BASE_URL_KEY,
    CANONICAL_LLM_MODEL_KEY,
    LlmEnvironment,
    capture_dotenv,
    read_llm_environment,
    restore_dotenv,
    validate_llm_base_url,
    write_llm_environment,
)


_PROVIDER_ENVIRONMENT_NAMES = (
    CANONICAL_LLM_MODEL_KEY,
    CANONICAL_LLM_BASE_URL_KEY,
    CANONICAL_LLM_API_KEY,
    "MODEL_ID",
    "MODEL_BASE_URL",
    "MODEL_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def isolate_provider_environment():
    previous = {name: os.environ.get(name) for name in _PROVIDER_ENVIRONMENT_NAMES}
    for name in _PROVIDER_ENVIRONMENT_NAMES:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _clear_provider_environment(monkeypatch) -> None:
    for name in _PROVIDER_ENVIRONMENT_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_dotenv_write_preserves_unrelated_content_and_retains_blank_key(tmp_path: Path, monkeypatch):
    _clear_provider_environment(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text("# existing comment\nOTHER_SETTING=unchanged\n", encoding="utf-8")

    write_llm_environment(
        tmp_path,
        model_id="provider/first-model",
        base_url="https://gateway.example/v1/",
        api_key="provider-test-secret",
    )
    write_llm_environment(
        tmp_path,
        model_id="provider/second-model",
        base_url="https://gateway.example/v1",
    )

    content = dotenv.read_text(encoding="utf-8")
    assert "# existing comment" in content
    assert "OTHER_SETTING=unchanged" in content
    assert "AUDIOBOOK_LLM_MODEL=provider/second-model" in content
    assert "AUDIOBOOK_LLM_BASE_URL=https://gateway.example/v1" in content
    assert "AUDIOBOOK_LLM_API_KEY=provider-test-secret" in content
    assert stat.S_IMODE(dotenv.stat().st_mode) == 0o600
    assert read_llm_environment(tmp_path) == LlmEnvironment(
        model_id="provider/second-model",
        base_url="https://gateway.example/v1",
        api_key="provider-test-secret",
    )


def test_dotenv_write_rejects_invalid_url_without_replacing_existing_file(tmp_path: Path, monkeypatch):
    _clear_provider_environment(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text("UNRELATED_VALUE=keep\n", encoding="utf-8")
    before = dotenv.read_bytes()

    with pytest.raises(ValueError, match="http 或 https"):
        write_llm_environment(
            tmp_path,
            model_id="provider/model",
            base_url="ftp://gateway.example/v1",
            api_key="provider-test-secret",
        )

    assert dotenv.read_bytes() == before


def test_project_dotenv_wins_over_process_aliases(tmp_path: Path, monkeypatch):
    _clear_provider_environment(monkeypatch)
    (tmp_path / ".env").write_text(
        "AUDIOBOOK_LLM_MODEL=project/model\n"
        "AUDIOBOOK_LLM_BASE_URL=https://project.example/v1\n"
        "AUDIOBOOK_LLM_API_KEY=project-secret\n",
        encoding="utf-8",
    )
    environment = {
        "MODEL_ID": "legacy/model",
        "MODEL_BASE_URL": "https://legacy.example/v1",
        "MODEL_API_KEY": "legacy-secret",
    }

    assert read_llm_environment(tmp_path, environment=environment) == LlmEnvironment(
        model_id="project/model",
        base_url="https://project.example/v1",
        api_key="project-secret",
    )


def test_restore_dotenv_restores_process_environment_after_later_failure(tmp_path: Path, monkeypatch):
    _clear_provider_environment(monkeypatch)
    monkeypatch.setenv("MODEL_ID", "legacy/model")
    monkeypatch.setenv("MODEL_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("MODEL_API_KEY", "legacy-secret")
    snapshot = capture_dotenv(tmp_path)

    write_llm_environment(
        tmp_path,
        model_id="provider/temporary-model",
        base_url="https://gateway.example/v1",
        api_key="provider-test-secret",
    )
    restore_dotenv(snapshot)

    assert not (tmp_path / ".env").exists()
    assert CANONICAL_LLM_MODEL_KEY not in os.environ
    assert CANONICAL_LLM_BASE_URL_KEY not in os.environ
    assert CANONICAL_LLM_API_KEY not in os.environ
    assert read_llm_environment(tmp_path) == LlmEnvironment(
        model_id="legacy/model",
        base_url="https://legacy.example/v1",
        api_key="legacy-secret",
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@gateway.example/v1",
        "https://gateway.example/v1?api_key=secret",
        "https://gateway.example/v1#fragment",
        "https://gateway.example /v1",
    ],
)
def test_provider_url_rejects_secret_bearing_or_malformed_parts(url: str):
    with pytest.raises(ValueError):
        validate_llm_base_url(url)
