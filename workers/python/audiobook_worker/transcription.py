"""Local Whisper transcription for the post-TTS audio-planning stage."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


class TranscriptionError(RuntimeError):
    pass


def _audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getframerate() <= 0 or wav_file.getnframes() <= 0:
                raise TranscriptionError(f"Audio file is empty or invalid: {path}")
            return wav_file.getnframes() / wav_file.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise TranscriptionError(f"Cannot read audio file: {path}") from error


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _normalize_segment(item: Any, index: int, duration: float) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    start = max(0.0, min(duration, _number(item.get("start"), 0.0)))
    end = max(start, min(duration, _number(item.get("end"), start)))
    return {
        "id": f"transcript_{index + 1:04d}",
        "start": round(start, 3),
        "end": round(end, 3),
        "text": text,
    }


def _local_model_reference(model: str) -> str:
    """Resolve a Hugging Face model id to an existing local snapshot when possible."""

    model_path = Path(model).expanduser()
    if model_path.is_dir():
        return str(model_path)
    if "/" not in model:
        return model

    cache_directory = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model.replace('/', '--')}"
    )
    ref_path = cache_directory / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return model
    snapshot = cache_directory / "snapshots" / revision
    return str(snapshot) if snapshot.is_dir() else model


def _whisper_python_candidates(explicit: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get("AUDIOBOOK_WHISPER_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    # The workstation already has mlx-whisper in this existing environment.
    # Keep this as a fallback so the audiobook worker does not need to install
    # a second MLX stack into its large TTS virtualenv.
    candidates.append(
        Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python"
    )
    # Also look at the Python commands visible to the desktop process.  GUI
    # applications do not always inherit the same PATH as a terminal, so
    # these are only additional fallbacks after the explicit/local paths.
    for command in ("python3", "python"):
        resolved = shutil.which(command)
        if resolved:
            candidates.append(Path(resolved))
    current = Path(sys.executable).resolve()
    unique: list[Path] = []
    seen_resolved: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        # Keep the original venv entry point for execution.  Resolving a
        # venv's ``bin/python`` symlink to its base interpreter drops the
        # venv site-packages, which would hide an installed mlx_whisper.
        if (
            resolved == current
            or resolved in seen_resolved
            or not candidate.is_file()
        ):
            continue
        seen_resolved.add(resolved)
        unique.append(candidate)
    return unique


_EXTERNAL_TRANSCRIPTION_SCRIPT = r'''
import json
import sys

import mlx_whisper

audio_path, model_reference, language = sys.argv[1:]
options = {"path_or_hf_repo": model_reference, "word_timestamps": False}
if language and not language.startswith("auto"):
    options["language"] = language
result = mlx_whisper.transcribe(audio_path, **options)
segments = []
for item in (result.get("segments", []) if isinstance(result, dict) else []):
    if not isinstance(item, dict):
        continue
    segments.append({
        "start": float(item.get("start", 0.0)),
        "end": float(item.get("end", item.get("start", 0.0))),
        "text": str(item.get("text", "")),
    })
print(json.dumps({"segments": segments}, ensure_ascii=False))
'''


_WHISPER_IMPORT_CHECK_SCRIPT = r'''
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("mlx_whisper") else 1)
'''


def _supports_mlx_whisper(python_executable: Path) -> bool:
    """Check module availability without importing MLX/initializing Metal."""

    try:
        completed = subprocess.run(
            [str(python_executable), "-c", _WHISPER_IMPORT_CHECK_SCRIPT],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _transcribe_with_existing_python(
    audio_path: Path,
    model_reference: str,
    language: str,
    python_executable: str | None,
) -> dict[str, Any]:
    candidates = _whisper_python_candidates(python_executable)
    available = [candidate for candidate in candidates if _supports_mlx_whisper(candidate)]
    if not available:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise TranscriptionError(
            "没有找到可调用本地 mlx-whisper 的 Python 环境。"
            + (f" 已检查：{searched}。" if searched else "")
            + " 请设置 AUDIOBOOK_WHISPER_PYTHON 指向已安装 mlx-whisper 的 Python。"
        )
    last_error: str | None = None
    for candidate in available:
        try:
            completed = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    _EXTERNAL_TRANSCRIPTION_SCRIPT,
                    str(audio_path),
                    model_reference,
                    language,
                ],
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            last_error = str(error)
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            last_error = (
                f"{candidate}: {detail[-1000:]}"
                if detail
                else f"{candidate}: exit code {completed.returncode}"
            )
            continue
        try:
            payload = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            last_error = f"Whisper 输出不是有效 JSON：{error}"
            continue
        if isinstance(payload, dict):
            return payload
        last_error = "Whisper 输出格式不是对象"
    raise TranscriptionError(f"Whisper 转录失败：{last_error or '未知错误'}")


def transcribe_audio(
    audio_path: Path | str,
    *,
    model: str | None = None,
    language: str | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Transcribe a generated voice track using local mlx-whisper.

    The dependency is imported only when this command is used, so regular
    chapter analysis and TTS do not require an ASR model to be installed.
    """
    path = Path(audio_path)
    duration = _audio_duration(path)
    model_name = model or DEFAULT_WHISPER_MODEL
    model_reference = _local_model_reference(model_name)
    normalized_language = str(language or "").strip()
    try:
        import mlx_whisper  # type: ignore[import-not-found]
    except ImportError:
        raw = _transcribe_with_existing_python(
            path,
            model_reference,
            normalized_language.split("-")[0] if normalized_language else "",
            python_executable,
        )
    else:
        options: dict[str, Any] = {
            "path_or_hf_repo": model_reference,
            "word_timestamps": False,
        }
        if normalized_language and not normalized_language.startswith("auto"):
            options["language"] = normalized_language.split("-")[0]

        try:
            raw = mlx_whisper.transcribe(str(path), **options)
        except TypeError:
            # Older mlx-whisper versions do not accept every keyword. Retry
            # with the stable subset while retaining the local model.
            fallback = {"path_or_hf_repo": options["path_or_hf_repo"]}
            if "language" in options:
                fallback["language"] = options["language"]
            try:
                raw = mlx_whisper.transcribe(str(path), **fallback)
            except Exception as error:
                raise TranscriptionError(f"Whisper 转录失败：{error}") from error
        except Exception as error:
            raise TranscriptionError(f"Whisper 转录失败：{error}") from error

    raw_segments = raw.get("segments", []) if isinstance(raw, dict) else []
    segments = [
        normalized
        for index, item in enumerate(raw_segments)
        if (normalized := _normalize_segment(item, index, duration)) is not None
    ]
    if not segments:
        raise TranscriptionError("Whisper 没有返回可用的转录片段。")
    return {
        "version": 1,
        "backend": "mlx-whisper",
        "model": model_name,
        "durationSeconds": round(duration, 3),
        "language": normalized_language or None,
        "segments": segments,
    }
