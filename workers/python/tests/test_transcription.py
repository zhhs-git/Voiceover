import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import audiobook_worker.transcription as transcription
from audiobook_worker.transcription import TranscriptionError, transcribe_audio


def _write_wav(path, *, frames=16000, sample_rate=16000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)


def test_transcribe_audio_uses_local_mlx_whisper_and_normalizes_timestamps(tmp_path, monkeypatch):
    audio_path = tmp_path / "voice.wav"
    _write_wav(audio_path)
    calls = []

    def transcribe(audio_path, **options):
        calls.append((audio_path, options))
        return {
            "segments": [
                {"start": -1.0, "end": 0.4, "text": "  第一段  "},
                {"start": 0.4, "end": 3.0, "text": "第二段"},
                {"start": 0.5, "end": 0.8, "text": ""},
            ]
        }

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=transcribe))
    result = transcribe_audio(audio_path, language="zh-CN")

    assert calls[0][0] == str(audio_path)
    assert "models--mlx-community--whisper-large-v3-turbo/snapshots/" in calls[0][1]["path_or_hf_repo"]
    assert calls[0][1]["language"] == "zh"
    assert result["durationSeconds"] == 1.0
    assert result["segments"] == [
        {"id": "transcript_0001", "start": 0.0, "end": 0.4, "text": "第一段"},
        {"id": "transcript_0002", "start": 0.4, "end": 1.0, "text": "第二段"},
    ]


def test_transcribe_audio_reports_missing_local_whisper(tmp_path, monkeypatch):
    audio_path = tmp_path / "voice.wav"
    _write_wav(audio_path, frames=8000)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    monkeypatch.setattr(transcription, "_whisper_python_candidates", lambda explicit=None: [])

    with pytest.raises(TranscriptionError, match="mlx-whisper"):
        transcribe_audio(audio_path)


def test_transcribe_audio_can_call_an_existing_whisper_python_environment(tmp_path, monkeypatch):
    audio_path = tmp_path / "voice.wav"
    _write_wav(audio_path, frames=8000)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    existing_python = Path("/existing/whisper/bin/python")
    monkeypatch.setattr(
        transcription,
        "_whisper_python_candidates",
        lambda explicit=None: [existing_python],
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"segments": [{"start": 0.0, "end": 0.5, "text": "本地 Whisper"}]},
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    result = transcribe_audio(audio_path, language="zh")

    transcription_call = calls[-1][0]
    assert transcription_call[0] == str(existing_python)
    assert "whisper-large-v3-turbo" in transcription_call[4]
    assert result["segments"][0]["text"] == "本地 Whisper"


def test_transcribe_audio_skips_existing_python_without_mlx_whisper(tmp_path, monkeypatch):
    audio_path = tmp_path / "voice.wav"
    _write_wav(audio_path, frames=8000)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    missing_python = Path("/missing/whisper/bin/python")
    working_python = Path("/existing/mlx-whisper/bin/python")
    monkeypatch.setattr(
        transcription,
        "_whisper_python_candidates",
        lambda explicit=None: [missing_python, working_python],
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == str(missing_python):
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if "importlib.util.find_spec" in command[2]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"segments": [{"start": 0.0, "end": 0.5, "text": "复用本地环境"}]},
                ensure_ascii=False,
            ),
            stderr="",
        )

    monkeypatch.setattr(transcription.subprocess, "run", fake_run)
    result = transcribe_audio(audio_path)

    assert all(command[0] != str(missing_python) for command in calls[1:])
    assert calls[-1][0] == str(working_python)
    assert result["segments"][0]["text"] == "复用本地环境"
