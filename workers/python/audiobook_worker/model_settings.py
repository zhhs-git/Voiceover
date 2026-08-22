"""Durable, credential-free model selection shared by the web and workers.

The web server is the owner of persisted settings.  This module deliberately
only exposes model metadata that is safe to send to the browser; provider
URLs, API keys, and environment-variable names stay inside ``llm.py`` and the
process environment.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from audiobook_worker.llm import read_models_json
from audiobook_worker.llm_env import (
    project_root,
    read_llm_environment,
    validate_llm_base_url,
    validate_llm_model_id,
)


DEFAULT_TTS_BACKEND = "mimo"
DEFAULT_TTS_MODEL_ID = "mimo-v2.5-tts-voiceclone"
VOXCPM2_BACKEND = "voxcpm2"
VOXCPM2_MODEL_ID = "VoxCPM2"
MODEL_SETTINGS_VERSION = 2


@dataclass(frozen=True)
class ModelSettings:
    """The normalized settings that may be persisted or attached to a job."""

    llm_model_id: str
    tts_backend: str = DEFAULT_TTS_BACKEND
    tts_model_id: str = DEFAULT_TTS_MODEL_ID

    def to_dict(self) -> dict[str, str]:
        return {
            "llmModelId": self.llm_model_id,
            "ttsBackend": self.tts_backend,
            "ttsModelId": self.tts_model_id,
        }

    @classmethod
    def from_value(cls, value: object) -> "ModelSettings | None":
        if not isinstance(value, dict):
            return None
        llm_model_id = str(value.get("llmModelId") or "").strip()
        tts_backend = str(value.get("ttsBackend") or "").strip().casefold()
        tts_model_id = str(value.get("ttsModelId") or "").strip()
        if not llm_model_id or not tts_backend or not tts_model_id:
            return None
        return cls(llm_model_id, tts_backend, tts_model_id)


def voxcpm2_paths(root: Path | None = None) -> dict[str, Path]:
    configured = os.environ.get("AUDIOBOOK_VOXCPM2_ROOT", "").strip()
    base = (
        Path(configured).expanduser()
        if configured
        else (root or project_root()) / "data" / "voxcpm2"
    ).resolve()
    return {
        "root": base,
        # Do not resolve the launcher: venv/bin/python is normally a symlink
        # to a base interpreter. Executing the resolved target bypasses
        # pyvenv.cfg and loses the isolated voxcpm site-packages directory.
        "python": base / ".venv" / "bin" / "python",
        "model": base / "models" / "VoxCPM2",
    }


def _safe_model_id(provider: str, model_id: object) -> str:
    value = str(model_id or "").strip()
    if not value:
        return ""
    prefix = f"{provider}/"
    return value if value.startswith(prefix) else f"{prefix}{value}"


def _legacy_provider_config(model_id: str, root: Path | None = None) -> dict[str, object]:
    """Return catalog metadata for compatibility without projecting secrets."""

    del root  # The legacy catalog remains user-home scoped by design.
    try:
        config = read_models_json()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(config, dict) or not isinstance(config.get("providers"), dict):
        return {}
    for provider, raw_provider in config["providers"].items():
        if not isinstance(raw_provider, dict) or not isinstance(raw_provider.get("models"), list):
            continue
        for raw_model in raw_provider["models"]:
            if not isinstance(raw_model, dict):
                continue
            raw_id = str(raw_model.get("id") or "").strip()
            if model_id in {raw_id, _safe_model_id(str(provider), raw_id)}:
                return raw_provider
    return {}


def legacy_llm_base_url(model_id: str, root: Path | None = None) -> str:
    candidate = str(_legacy_provider_config(model_id, root).get("baseUrl") or "").strip()
    if not candidate:
        return ""
    try:
        return validate_llm_base_url(candidate)
    except ValueError:
        # A legacy catalog may contain an endpoint with embedded credentials.
        # Do not project it to the browser; the resolver still owns legacy
        # compatibility for actual requests.
        return ""


def legacy_llm_api_key(model_id: str, root: Path | None = None) -> str:
    provider = _legacy_provider_config(model_id, root)
    inline = str(provider.get("apiKey") or "").strip()
    if inline:
        return inline
    environment_name = str(provider.get("apiKeyEnv") or "").strip()
    return str(os.environ.get(environment_name) or "").strip() if environment_name else ""


def discover_llm_options(root: Path | None = None) -> list[dict[str, object]]:
    """Return safe LLM choices from the existing local Pi model config."""

    options: list[dict[str, object]] = []
    seen: set[str] = set()
    try:
        config = read_models_json()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config = None

    if isinstance(config, dict):
        providers = config.get("providers")
        if isinstance(providers, dict):
            for provider, raw_provider in providers.items():
                if not isinstance(raw_provider, dict):
                    continue
                models = raw_provider.get("models", [])
                if not isinstance(models, list):
                    continue
                for raw_model in models:
                    if not isinstance(raw_model, dict):
                        continue
                    model_id = _safe_model_id(str(provider), raw_model.get("id"))
                    if not model_id or model_id in seen:
                        continue
                    seen.add(model_id)
                    display_name = str(
                        raw_model.get("name")
                        or raw_model.get("displayName")
                        or raw_model.get("id")
                        or model_id
                    ).strip()
                    options.append(
                        {
                            "id": model_id,
                            "provider": str(provider),
                            "displayName": display_name,
                            "family": str(raw_provider.get("family") or "default"),
                            "available": True,
                        }
                    )

    # Config-free installations can still use the legacy environment-backed
    # resolver.  Present that model as a safe choice without exposing its URL
    # or the name/value of its credential environment variable.
    env_model = read_llm_environment(root).model_id
    if env_model and env_model != "mock" and env_model not in seen:
        options.append(
            {
                "id": env_model,
                "provider": "env",
                "displayName": env_model,
                "family": "default",
                "available": True,
            }
        )
        seen.add(env_model)

    if "mock" not in seen:
        options.append(
            {
                "id": "mock",
                "provider": "local",
                "displayName": "离线 Mock（仅测试）",
                "family": "mock",
                "available": True,
            }
        )
    return options


def default_llm_model_id(root: Path | None = None) -> str:
    configured = read_llm_environment(root).model_id
    if configured:
        return configured
    try:
        config = read_models_json()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        config = None
    if isinstance(config, dict):
        default = str(config.get("default") or "").strip()
        if default:
            return default
        providers = config.get("providers")
        if isinstance(providers, dict):
            for provider, raw_provider in providers.items():
                if not isinstance(raw_provider, dict):
                    continue
                models = raw_provider.get("models")
                if isinstance(models, list) and models:
                    first = models[0]
                    if isinstance(first, dict):
                        model_id = _safe_model_id(str(provider), first.get("id"))
                        if model_id:
                            return model_id
    return read_llm_environment(root).model_id or "mock"


def probe_voxcpm2(root: Path | None = None) -> dict[str, object]:
    """Check the isolated interpreter, import, and required local weights."""

    paths = voxcpm2_paths(root)
    missing: list[str] = []
    if not paths["python"].is_file():
        missing.append("独立 Python 环境")
    if not paths["model"].is_dir():
        missing.append("VoxCPM2 模型目录")
    for filename in ("config.json", "model.safetensors", "audiovae.pth"):
        if not (paths["model"] / filename).is_file():
            missing.append(f"模型文件 {filename}")

    if missing:
        return {
            "id": VOXCPM2_BACKEND,
            "modelId": VOXCPM2_MODEL_ID,
            "displayName": "VoxCPM2（本地）",
            "available": False,
            "reason": "缺少" + "、".join(missing),
        }

    try:
        completed = subprocess.run(
            [str(paths["python"]), "-c", "import voxcpm"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "id": VOXCPM2_BACKEND,
            "modelId": VOXCPM2_MODEL_ID,
            "displayName": "VoxCPM2（本地）",
            "available": False,
            "reason": f"无法检查 VoxCPM2 环境：{error}",
        }
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = detail[-1] if detail else "voxcpm 包无法导入"
        return {
            "id": VOXCPM2_BACKEND,
            "modelId": VOXCPM2_MODEL_ID,
            "displayName": "VoxCPM2（本地）",
            "available": False,
            "reason": f"VoxCPM2 环境不可用：{reason}",
        }
    return {
        "id": VOXCPM2_BACKEND,
        "modelId": VOXCPM2_MODEL_ID,
        "displayName": "VoxCPM2（本地）",
        "available": True,
        "reason": "已检测到本地模型和独立运行环境。",
    }


def tts_options(root: Path | None = None) -> list[dict[str, object]]:
    return [
        {
            "id": DEFAULT_TTS_BACKEND,
            "modelId": DEFAULT_TTS_MODEL_ID,
            "displayName": "MiMo Voice Clone（云端）",
            "available": True,
            "reason": "使用现有 MiMo voice-clone 流程。",
        },
        probe_voxcpm2(root),
    ]


def normalize_settings(
    value: object,
    *,
    root: Path | None = None,
    require_available_tts: bool = True,
    llm_base_url: str | None = None,
) -> ModelSettings:
    raw = ModelSettings.from_value(value)
    if raw is None:
        raise ValueError("模型配置必须包含 llmModelId、ttsBackend 和 ttsModelId。")
    validate_llm_model_id(raw.llm_model_id)
    llm_ids = {str(item["id"]) for item in discover_llm_options(root)}
    configured_base_url = str(llm_base_url or read_llm_environment(root).base_url).strip()
    if configured_base_url:
        try:
            validate_llm_base_url(configured_base_url)
        except ValueError:
            configured_base_url = ""
    if raw.llm_model_id not in llm_ids and not configured_base_url:
        raise ValueError(f"LLM 模型不可用或不存在：{raw.llm_model_id}")
    if raw.tts_backend == DEFAULT_TTS_BACKEND:
        if raw.tts_model_id != DEFAULT_TTS_MODEL_ID:
            raise ValueError(f"MiMo TTS 模型不受支持：{raw.tts_model_id}")
    elif raw.tts_backend == VOXCPM2_BACKEND:
        if raw.tts_model_id != VOXCPM2_MODEL_ID:
            raise ValueError(f"VoxCPM2 模型不受支持：{raw.tts_model_id}")
        capability = probe_voxcpm2(root)
        if require_available_tts and capability.get("available") is not True:
            raise ValueError(str(capability.get("reason") or "VoxCPM2 当前不可用。"))
    else:
        raise ValueError(f"不支持的 TTS 后端：{raw.tts_backend}")
    return raw


def legacy_default_settings(root: Path | None = None) -> ModelSettings:
    llm_id = default_llm_model_id(root)
    # A stale environment value should not prevent the service from starting.
    environment = read_llm_environment(root)
    if (
        llm_id not in {str(item["id"]) for item in discover_llm_options(root)}
        and not environment.base_url
    ):
        llm_id = "mock"
    return ModelSettings(llm_model_id=llm_id)


def effective_settings(
    stored_value: object,
    *,
    root: Path | None = None,
) -> ModelSettings:
    fallback = legacy_default_settings(root)
    try:
        normalized = normalize_settings(stored_value, root=root)
    except ValueError:
        return fallback
    environment = read_llm_environment(root)
    if environment.model_id and (
        environment.model_id in {str(item["id"]) for item in discover_llm_options(root)}
        or environment.base_url
    ):
        return replace(normalized, llm_model_id=environment.model_id)
    return normalized


def settings_payload(settings: ModelSettings, *, root: Path | None = None) -> dict[str, object]:
    environment = read_llm_environment(root)
    if environment.base_url:
        try:
            base_url = validate_llm_base_url(environment.base_url)
        except ValueError:
            # Do not ever project a manually edited credential-bearing or
            # malformed project URL. Returning a blank field lets the user
            # replace it through the validated settings endpoint.
            base_url = ""
    else:
        base_url = legacy_llm_base_url(settings.llm_model_id, root)
    # A project endpoint deliberately owns its credential boundary.  Once it
    # exists, an explicit clear must not make a legacy catalog key appear
    # configured or silently re-enter the project setting on a later save.
    api_key_configured = (
        environment.api_key_configured
        if environment.base_url
        else environment.api_key_configured or bool(legacy_llm_api_key(settings.llm_model_id, root))
    )
    return {
        "version": MODEL_SETTINGS_VERSION,
        "current": settings.to_dict(),
        "llmConfig": {
            "modelId": settings.llm_model_id,
            "baseUrl": base_url,
            "apiKeyConfigured": api_key_configured,
        },
        "llmOptions": discover_llm_options(root),
        "ttsOptions": tts_options(root),
    }
