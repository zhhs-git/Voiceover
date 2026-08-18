import json
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from audiobook_worker import cli as cli_module


def run_worker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "audiobook_worker.cli", *args],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_prints_usage():
    result = run_worker("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "command" in result.stdout


def test_extract_book_command_imports_chinese_txt(tmp_path: Path):
    book_path = tmp_path / "测试小说.txt"
    book_path.write_text(
        "第一章 初见\n院子里很安静。\n\n第二章 重逢\n他们再次见面。",
        encoding="utf-8",
    )
    output_dir = tmp_path / "chapters"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"bookPath": str(book_path), "outputDirectory": str(output_dir)}),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"

    result = run_worker("extract_book", str(input_path), str(output_path))

    assert result.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = payload["artifacts"][0]["metadata"]
    assert payload["status"] == "succeeded"
    assert metadata["title"] == "测试小说"
    assert [chapter["title"] for chapter in metadata["chapters"]] == [
        "第一章 初见",
        "第二章 重逢",
    ]
    assert (output_dir / "chapter_001.txt").read_text(encoding="utf-8") == "院子里很安静。"


def test_unknown_command_returns_structured_error(tmp_path: Path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text("{}", encoding="utf-8")

    result = run_worker("unknown_command", str(input_path), str(output_path))

    assert result.returncode == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "status": "failed",
        "warnings": [],
        "artifacts": [],
        "error": {
            "code": "unknown_command",
            "message": "Unknown worker command: unknown_command",
        },
    }


def test_convert_to_mp3_uses_cbr_only_when_a_supported_export_bitrate_is_requested(
    tmp_path: Path, monkeypatch
):
    commands: list[list[str]] = []

    monkeypatch.setattr("shutil.which", lambda command: "/usr/bin/ffmpeg" if command == "ffmpeg" else None)

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"ID3")
        return type("Result", (), {"returncode": 0, "stderr": b""})()

    monkeypatch.setattr("subprocess.run", fake_run)

    cbr_result = cli_module._convert_to_mp3(
        {"wavPath": str(tmp_path / "source.wav"), "outputPath": str(tmp_path / "cbr.mp3"), "bitrateKbps": 128}
    )
    vbr_result = cli_module._convert_to_mp3(
        {"wavPath": str(tmp_path / "source.wav"), "outputPath": str(tmp_path / "vbr.mp3")}
    )

    assert cbr_result["status"] == "succeeded"
    assert vbr_result["status"] == "succeeded"
    assert ["-b:a", "128k"] == commands[0][commands[0].index("-b:a"):commands[0].index("-b:a") + 2]
    assert ["-q:a", "2"] == commands[1][commands[1].index("-q:a"):commands[1].index("-q:a") + 2]


def test_generate_audio_assets_returns_warning_for_empty_plan(tmp_path: Path):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(
        json.dumps({"bookId": "book_123", "chapterId": "chapter_001", "audioPlan": {"scenes": []}}),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({
            "bookId": "book_123",
            "chapterId": "chapter_001",
            "scriptPath": str(script_path),
            "outputDirectory": str(tmp_path / "audio-assets"),
        }),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"

    result = run_worker("generate_audio_assets", str(input_path), str(output_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    assert payload["warnings"] == ["no_audio_assets"]
    assert payload["artifacts"][0]["kind"] == "stable_audio_manifest"


def test_list_voices_returns_the_active_mimo_catalog(tmp_path: Path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"backend": "mimo"}), encoding="utf-8")

    result = run_worker("list_voices", str(input_path), str(output_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    voice_ids = {voice["id"] for voice in payload["voices"]}
    assert payload["status"] == "succeeded"
    assert len(voice_ids) > 4
    assert {"narrator_female", "female_adult_05", "male_adult_05"} <= voice_ids
    assert "female_british_01" not in voice_ids


def test_refresh_voice_assignments_upgrades_an_existing_script_in_place(tmp_path: Path):
    script_path = tmp_path / "chapter_001.json"
    script_path.write_text(
        json.dumps(
            {
                "bookId": "book_yun",
                "chapterId": "chapter_001",
                "language": "zh",
                "characters": [
                    {
                        "id": "父亲",
                        "canonicalName": "父亲",
                        "aliases": ["爹"],
                        "gender": "male",
                        "ageClass": "adult",
                        "voiceId": "male_adult_04",
                        "confidence": 0.9,
                    },
                    {
                        "id": "儿子",
                        "canonicalName": "儿子",
                        "aliases": [],
                        "gender": "male",
                        "ageClass": "young",
                        "voiceId": "male_adult_03",
                        "confidence": 0.9,
                    },
                ],
                "segments": [
                    {
                        "id": "seg_0001",
                        "text": "进来吧。",
                        "speakerId": "父亲",
                        "voiceId": "male_adult_04",
                    },
                    {
                        "id": "seg_0002",
                        "text": "知道了。",
                        "speakerId": "儿子",
                        "voiceId": "male_adult_03",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {"scriptPath": str(script_path), "forceLegacyAuto": True},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = run_worker("refresh_voice_assignments", str(input_path), str(output_path))

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    refreshed = json.loads(script_path.read_text(encoding="utf-8"))
    assert payload["status"] == "succeeded"
    characters = {character["canonicalName"]: character for character in refreshed["characters"]}
    assert characters["父亲"]["voiceSource"] == "auto"
    assert characters["儿子"]["voiceSource"] == "auto"
    assert characters["父亲"]["voiceId"].startswith("character_auto_")
    assert characters["儿子"]["voiceId"].startswith("character_auto_")
    assert characters["父亲"]["fallbackVoiceId"].startswith("male_adult_")
    assert refreshed["segments"][0]["voiceId"] == characters["父亲"]["voiceId"]


def test_synthesize_segment_audio_uses_kokoro_backend_when_requested(tmp_path: Path):
    """CLI still supports KokoroTTSBackend when explicitly requested."""
    import numpy as np
    import torch as _torch
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0001",
                "text": "In the beginning.",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "intensity": 0.2,
                "pace": "normal",
            }
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))

    request = {
        "scriptPath": str(script_path),
        "segmentId": "seg_0001",
        "outputDirectory": str(tmp_path / "audio"),
        "backend": "kokoro",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request))
    output_path = tmp_path / "output.json"

    fake_audio = _torch.tensor(np.zeros(24000, dtype=np.float32))
    mock_result = MagicMock()
    mock_result.audio = fake_audio
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [mock_result]

    with patch("audiobook_worker.tts.KPipeline") as mock_kp:
        mock_kp.return_value = mock_pipeline
        exit_code = main(["synthesize_segment_audio", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    mock_kp.assert_called_once()


def test_synthesize_segment_audio_uses_parler_backend(tmp_path: Path):
    """CLI synthesize_segment_audio command selects ParlerTTSBackend when backend=parler."""
    import numpy as np
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0001",
                "text": "In the beginning.",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "intensity": 0.2,
                "pace": "normal",
            }
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script))

    request = {
        "scriptPath": str(script_path),
        "segmentId": "seg_0001",
        "outputDirectory": str(tmp_path / "audio"),
        "backend": "parler",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request))
    output_path = tmp_path / "output.json"

    fake_audio = np.zeros(24000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24000
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: fake_audio.reshape(1, -1))
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    with patch("audiobook_worker.tts.ParlerTTSForConditionalGeneration") as mock_cls, \
         patch("audiobook_worker.tts.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        exit_code = main(["synthesize_segment_audio", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    # Verify Parler model was actually loaded, not the mock backend
    mock_cls.from_pretrained.assert_called_once()


def test_synthesize_segment_audio_uses_mimo_voiceclone_and_profile_directory(tmp_path: Path):
    """CLI selects MiMo voice cloning and forwards the book profile directory."""
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0001",
                "text": "夜色渐渐沉了下来。",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "pace": "normal",
            }
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    request = {
        "scriptPath": str(script_path),
        "segmentId": "seg_0001",
        "outputDirectory": str(tmp_path / "audio"),
        "backend": "mimo",
        "modelId": "mimo-v2.5-tts-voiceclone",
        "voiceProfileDirectory": str(tmp_path / "voice-profiles"),
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    fake_artifact = MagicMock(
        kind="segment_audio",
        path=tmp_path / "audio" / "seg_0001.wav",
        duration_seconds=1.5,
    )
    with patch("audiobook_worker.cli.MiMoTTSBackend") as backend_class:
        backend_class.return_value.synthesize_segment.return_value = fake_artifact
        exit_code = main(["synthesize_segment_audio", str(input_path), str(output_path)])

    assert exit_code == 0
    backend_class.assert_called_once_with(
        model_id="mimo-v2.5-tts-voiceclone",
        voice_profile_directory=str(tmp_path / "voice-profiles"),
    )
    backend_class.return_value.synthesize_segment.assert_called_once()
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "succeeded"


def test_synthesize_chapter_audio_merges_adjacent_compatible_segments_and_cleans_stale_audio(tmp_path: Path):
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello", "voiceId": "narrator_default", "emotion": "neutral", "pace": "normal"},
            {"id": "seg_0002", "text": "there.", "voiceId": "narrator_default", "emotion": "neutral", "pace": "normal"},
            {"id": "seg_0003", "text": "Stop.", "voiceId": "male_adult_01", "emotion": "angry", "pace": "fast"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "stale.wav").write_bytes(b"old")

    request = {
        "scriptPath": str(script_path),
        "outputDirectory": str(audio_dir),
        "backend": "mock",
        "mergeSegments": True,
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["synthesize_chapter_audio", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    assert result["metadata"]["originalSegmentCount"] == 3
    assert result["metadata"]["synthesizedSegmentCount"] == 2
    assert result["metadata"]["cachedSegmentCount"] == 0
    assert len(result["artifacts"]) == 2
    assert result["artifacts"][0]["metadata"]["sourceSegmentIds"] == ["seg_0001", "seg_0002"]
    assert not (audio_dir / "stale.wav").exists()


def test_synthesize_chapter_audio_reuses_cached_segments_without_loading_backend(tmp_path: Path):
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello.", "voiceId": "narrator_default", "emotion": "neutral", "pace": "normal"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    first_request = {
        "scriptPath": str(script_path),
        "outputDirectory": str(audio_dir),
        "backend": "mock",
        "mergeSegments": False,
        "cacheSegments": True,
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(first_request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 0
    first_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert first_result["artifacts"][0]["metadata"]["cacheHit"] is False
    cached_wav = audio_dir / "seg_0001.wav"
    original_mtime = cached_wav.stat().st_mtime_ns

    with patch("audiobook_worker.cli.MockTTSBackend") as mock_backend:
        mock_backend.side_effect = AssertionError("backend should not load on cache hit")
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 0

    second_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert second_result["artifacts"][0]["metadata"]["cacheHit"] is True
    assert second_result["artifacts"][0]["metadata"]["sourceSegmentIds"] == ["seg_0001"]
    assert cached_wav.stat().st_mtime_ns == original_mtime


def test_mimo_chapter_synthesis_prepares_profiles_then_runs_segments_serially_in_timeline_order(
    tmp_path: Path,
    monkeypatch,
):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import AudioArtifact

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "第一句。", "voiceId": "narrator_female"},
            {"id": "seg_0002", "text": "第二句。", "voiceId": "narrator_female"},
            {"id": "seg_0003", "text": "第三句。", "voiceId": "male_adult_01"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(audio_dir),
                "backend": "mimo",
                "modelId": "mimo-v2.5-tts-voiceclone",
                "mergeSegments": False,
                "cacheSegments": False,
                "voiceProfileDirectory": str(tmp_path / "voice-profiles"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    prepared = threading.Event()
    active = 0
    max_active = 0
    calls: list[str] = []
    lock = threading.Lock()

    class FakeMiMoBackend:
        def prepare_voice_profiles(self, segments, profile_directory):
            assert [segment["id"] for segment in segments] == [
                "seg_0001", "seg_0002", "seg_0003"
            ]
            prepared.set()

        def synthesize_segment(self, segment, output_directory):
            nonlocal active, max_active
            assert prepared.is_set()
            with lock:
                active += 1
                max_active = max(max_active, active)
                calls.append(segment["id"])
            try:
                time.sleep(0.02)
                output_path = Path(output_directory) / f"{segment['id']}.wav"
                with wave.open(str(output_path), "wb") as wav_file:
                    wav_file.setparams((1, 2, 24_000, 240, "NONE", "not compressed"))
                    wav_file.writeframes(b"\x00\x00" * 240)
                return AudioArtifact("segment_audio", output_path, 0.01)
            finally:
                with lock:
                    active -= 1

    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "99")
    with patch("audiobook_worker.cli.MiMoTTSBackend", return_value=FakeMiMoBackend()):
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert max_active == 1
    assert calls == ["seg_0001", "seg_0002", "seg_0003"]
    assert [artifact["metadata"]["segmentId"] for artifact in result["artifacts"]] == [
        "seg_0001", "seg_0002", "seg_0003"
    ]


def test_mimo_chapter_synthesis_leaves_request_retries_to_the_tts_backend(
    tmp_path: Path,
    monkeypatch,
):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import AudioArtifact

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "第一句。", "voiceId": "narrator_female"},
            {"id": "seg_0002", "text": "第二句。", "voiceId": "narrator_female"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(audio_dir),
                "backend": "mimo",
                "mergeSegments": False,
                "cacheSegments": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    calls: list[str] = []

    class FakeMiMoBackend:
        def prepare_voice_profiles(self, segments, profile_directory):
            return None

        def synthesize_segment(self, segment, output_directory):
            calls.append(segment["id"])
            if segment["id"] == "seg_0002" and calls.count("seg_0002") == 1:
                raise RuntimeError("transient test failure")
            segment_path = Path(output_directory) / f"{segment['id']}.wav"
            with wave.open(str(segment_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 24_000, 240, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x00" * 240)
            return AudioArtifact("segment_audio", segment_path, 0.01)

    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "2")
    with patch("audiobook_worker.cli.MiMoTTSBackend", return_value=FakeMiMoBackend()):
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 1

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["error"]["code"] == "tts_synthesis_failed"
    assert result["error"]["details"] == {"segmentId": "seg_0002"}
    assert calls.count("seg_0001") == 1
    assert calls.count("seg_0002") == 1


def test_mimo_chapter_synthesis_does_not_write_timeline_when_a_serial_segment_fails(
    tmp_path: Path,
    monkeypatch,
):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import AudioArtifact

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "第一句。", "voiceId": "narrator_female"},
            {"id": "seg_0002", "text": "第二句。", "voiceId": "narrator_female"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(audio_dir),
                "backend": "mimo",
                "mergeSegments": False,
                "cacheSegments": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    calls: list[str] = []

    class FakeMiMoBackend:
        def prepare_voice_profiles(self, segments, profile_directory):
            return None

        def synthesize_segment(self, segment, output_directory):
            calls.append(segment["id"])
            if segment["id"] == "seg_0002":
                raise RuntimeError("persistent test failure")
            segment_path = Path(output_directory) / f"{segment['id']}.wav"
            with wave.open(str(segment_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 24_000, 240, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x00" * 240)
            return AudioArtifact("segment_audio", segment_path, 0.01)

    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "2")
    with patch("audiobook_worker.cli.MiMoTTSBackend", return_value=FakeMiMoBackend()):
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 1

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["error"]["code"] == "tts_synthesis_failed"
    assert result["error"]["details"] == {"segmentId": "seg_0002"}
    assert calls.count("seg_0001") == 1
    assert calls.count("seg_0002") == 1
    assert not (audio_dir / "timeline.json").exists()


def test_mimo_chapter_synthesis_does_not_retry_a_permanent_mimo_request_error(
    tmp_path: Path,
    monkeypatch,
):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import AudioArtifact, MiMoRequestError

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "第一句。", "voiceId": "narrator_female"},
            {"id": "seg_0002", "text": "第二句。", "voiceId": "narrator_female"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script, ensure_ascii=False), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(audio_dir),
                "backend": "mimo",
                "mergeSegments": False,
                "cacheSegments": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    calls: list[str] = []

    class FakeMiMoBackend:
        def prepare_voice_profiles(self, segments, profile_directory):
            return None

        def synthesize_segment(self, segment, output_directory):
            calls.append(segment["id"])
            if segment["id"] == "seg_0002":
                raise MiMoRequestError("MiMo API request failed with HTTP 401", retryable=False)
            segment_path = Path(output_directory) / f"{segment['id']}.wav"
            with wave.open(str(segment_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 24_000, 240, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x00" * 240)
            return AudioArtifact("segment_audio", segment_path, 0.01)

    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "2")
    with patch("audiobook_worker.cli.MiMoTTSBackend", return_value=FakeMiMoBackend()):
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 1

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["error"]["code"] == "tts_synthesis_failed"
    assert result["error"]["details"] == {"segmentId": "seg_0002"}
    assert calls.count("seg_0001") == 1
    assert calls.count("seg_0002") == 1
    assert not (audio_dir / "timeline.json").exists()


def test_segment_cache_signature_changes_with_character_voice_description():
    from audiobook_worker.cli import _segment_cache_signature

    base_segment = {
        "id": "seg_0001",
        "text": "我知道了。",
        "voiceId": "male_adult_01",
        "emotion": "neutral",
        "pace": "normal",
    }
    father_signature = _segment_cache_signature(
        {**base_segment, "voiceDescription": "成熟父亲的固定音色"}, "mimo", None
    )
    son_signature = _segment_cache_signature(
        {**base_segment, "voiceDescription": "年轻儿子的固定音色"}, "mimo", None
    )
    fallback_signature = _segment_cache_signature(
        {**base_segment, "fallbackVoiceId": "male_adult_02"}, "mimo", None
    )

    assert father_signature != son_signature
    assert father_signature != fallback_signature


def test_synthesize_chapter_audio_splits_long_chinese_segment_and_cleans_old_cache(tmp_path: Path):
    from audiobook_worker.cli import main

    text = "她最后的记忆停留在飞云宫里的那一天，三月二十七，她饮下了御赐的鹤顶红，吐着大口大口的血，狼狈地趴在软榻上。"
    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0028",
                "text": text,
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "pace": "normal",
            },
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "seg_0028.wav").write_bytes(b"old")
    (audio_dir / "seg_0028.wav.json").write_text("{}", encoding="utf-8")

    request = {
        "scriptPath": str(script_path),
        "outputDirectory": str(audio_dir),
        "backend": "mock",
        "mergeSegments": True,
        "cacheSegments": True,
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 0

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["metadata"]["synthesizedSegmentCount"] == 2
    assert result["metadata"]["segmentIds"] == [
        "seg_0028_part_0001",
        "seg_0028_part_0002",
    ]
    assert not (audio_dir / "seg_0028.wav").exists()
    assert all(
        (audio_dir / f"seg_0028_part_{part:04d}.wav").exists()
        for part in (1, 2)
    )


def test_mix_repairs_old_assembly_without_resynthesizing_cached_segments(tmp_path: Path):
    """A voice-direction split must be shared by synthesis, assembly, and mix."""
    from audiobook_worker.audio import assemble_chapter_audio
    from audiobook_worker.cli import main

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {
                "id": "seg_0001",
                "text": "The first sentence.",
                "speakerId": "narrator",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "pace": "normal",
            },
            {
                "id": "seg_0002",
                "text": "The second sentence.",
                "speakerId": "narrator",
                "voiceId": "narrator_default",
                "emotion": "neutral",
                "pace": "normal",
            },
        ],
    }
    script_path = tmp_path / "scripts" / "script.json"
    script_path.parent.mkdir(parents=True)
    script_path.write_text(json.dumps(script), encoding="utf-8")
    direction_path = tmp_path / "analysis" / "ch01" / "voice_direction.json"
    direction_path.parent.mkdir(parents=True)
    direction_path.write_text(
        json.dumps(
            {
                "directions": [
                    {"segmentIndex": 0, "direction": "平稳，句尾收束"},
                    {"segmentIndex": 1, "direction": "稍快，句首短暂停顿"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    segment_dir = tmp_path / "segments"
    synth_input = tmp_path / "synth-input.json"
    synth_input.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(segment_dir),
                "backend": "mock",
                "modelId": "test-model",
                "mergeSegments": True,
                "cacheSegments": True,
            }
        ),
        encoding="utf-8",
    )
    synth_output = tmp_path / "synth-output.json"
    assert main(["synthesize_chapter_audio", str(synth_input), str(synth_output)]) == 0
    synth_result = json.loads(synth_output.read_text(encoding="utf-8"))
    assert synth_result["metadata"]["synthesizedSegmentCount"] == 2

    first_segment_mtime = (segment_dir / "seg_0001.wav").stat().st_mtime_ns
    second_segment_mtime = (segment_dir / "seg_0002.wav").stat().st_mtime_ns

    # Reproduce the old assembly bug: raw script merging kept only the first
    # file even though synthesis had correctly split the two directions.
    old_voice_path = tmp_path / "voice.wav"
    assemble_chapter_audio([segment_dir / "seg_0001.wav"], old_voice_path)

    mix_input = tmp_path / "mix-input.json"
    mix_input.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "segmentAudioDirectory": str(segment_dir),
                "voiceAudioPath": str(old_voice_path),
                "audioAssetsDirectory": str(tmp_path / "assets"),
                "outputPath": str(tmp_path / "mixed.wav"),
                "backend": "mock",
                "modelId": "test-model",
                "mergeSegments": True,
                "voiceGain": 1.0,
            }
        ),
        encoding="utf-8",
    )
    mix_output = tmp_path / "mix-output.json"
    assert main(["mix_chapter_audio", str(mix_input), str(mix_output)]) == 0

    result = json.loads(mix_output.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert "voice_timeline_reassembled_from_cached_segments" in result["warnings"]
    assert (segment_dir / "seg_0001.wav").stat().st_mtime_ns == first_segment_mtime
    assert (segment_dir / "seg_0002.wav").stat().st_mtime_ns == second_segment_mtime
    with wave.open(str(old_voice_path), "rb") as voice:
        assert voice.getnframes() > 0
    assert (tmp_path / "mixed.wav").exists()


def test_synthesize_chapter_audio_fails_when_backend_does_not_create_wav(tmp_path: Path):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import AudioArtifact

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello.", "voiceId": "narrator_default"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "outputDirectory": str(audio_dir),
                "backend": "mock",
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"

    with patch("audiobook_worker.cli.MockTTSBackend") as backend_class:
        backend_class.return_value.synthesize_segment.return_value = AudioArtifact(
            "segment_audio", audio_dir / "seg_0001.wav", 1.0
        )
        assert main(["synthesize_chapter_audio", str(input_path), str(output_path)]) == 1

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "invalid_segment_audio"


def test_assembly_fails_when_script_segment_audio_is_missing(tmp_path: Path):
    from audiobook_worker.cli import main
    from audiobook_worker.tts import MockTTSBackend

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello.", "voiceId": "narrator_default"},
            {"id": "seg_0002", "text": "World.", "voiceId": "narrator_default"},
        ],
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    audio_dir = tmp_path / "audio"
    MockTTSBackend().synthesize_segment(script["segments"][0], audio_dir)
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "segmentAudioDirectory": str(audio_dir),
                "outputPath": str(tmp_path / "chapter.wav"),
                "mergeSegments": False,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"

    assert main(["assemble_chapter_audio", str(input_path), str(output_path)]) == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error"]["code"] == "incomplete_segment_audio"
    assert not (tmp_path / "chapter.wav").exists()


def test_mix_recovers_legacy_segment_cache_when_voice_timeline_matches(tmp_path: Path):
    from audiobook_worker.audio import assemble_chapter_audio
    from audiobook_worker.cli import main
    from audiobook_worker.tts import MockTTSBackend

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello."},
            {"id": "seg_0002", "text": "Missing from the old cache."},
            {"id": "seg_0003", "text": "World."},
        ],
        "audioPlan": {"scenes": []},
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    segment_dir = tmp_path / "segments"
    backend = MockTTSBackend()
    first = backend.synthesize_segment(script["segments"][0], segment_dir)
    third = backend.synthesize_segment(script["segments"][2], segment_dir)
    voice_path = tmp_path / "voice.wav"
    assemble_chapter_audio([first.path, third.path], voice_path)

    output_path = tmp_path / "mixed.wav"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "segmentAudioDirectory": str(segment_dir),
                "voiceAudioPath": str(voice_path),
                "audioAssetsDirectory": str(tmp_path / "assets"),
                "outputPath": str(output_path),
                "mergeSegments": False,
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"

    assert main(["mix_chapter_audio", str(input_path), str(result_path)]) == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert "legacy_segment_cache_recovered_missing:seg_0002" in result["warnings"]
    assert output_path.exists()
    with wave.open(str(output_path), "rb") as mixed, wave.open(str(voice_path), "rb") as voice:
        assert mixed.getnframes() == voice.getnframes()


def test_mix_recovers_source_segment_ids_from_cache_sidecar(tmp_path: Path):
    from audiobook_worker.audio import assemble_chapter_audio
    from audiobook_worker.cli import main
    from audiobook_worker.tts import MockTTSBackend

    script = {
        "bookId": "book1",
        "chapterId": "ch01",
        "segments": [
            {"id": "seg_0001", "text": "Hello."},
            {"id": "seg_0002", "text": "World."},
        ],
        "audioPlan": {"scenes": []},
    }
    script_path = tmp_path / "script.json"
    script_path.write_text(json.dumps(script), encoding="utf-8")
    segment_dir = tmp_path / "segments"
    first = MockTTSBackend().synthesize_segment(script["segments"][0], segment_dir)
    (segment_dir / "seg_0001.wav.json").write_text(
        json.dumps(
            {
                "segmentId": "seg_0001",
                "sourceSegmentIds": ["seg_0001", "seg_0002"],
            }
        ),
        encoding="utf-8",
    )
    voice_path = tmp_path / "voice.wav"
    assemble_chapter_audio([first.path], voice_path)

    output_path = tmp_path / "mixed.wav"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "scriptPath": str(script_path),
                "segmentAudioDirectory": str(segment_dir),
                "voiceAudioPath": str(voice_path),
                "audioAssetsDirectory": str(tmp_path / "assets"),
                "outputPath": str(output_path),
                "mergeSegments": False,
            }
        ),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.json"

    assert main(["mix_chapter_audio", str(input_path), str(result_path)]) == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert "segment_cache_recovered_from_metadata" in result["warnings"]
    assert output_path.exists()
    with wave.open(str(output_path), "rb") as mixed, wave.open(str(voice_path), "rb") as voice:
        assert mixed.getnframes() == voice.getnframes()


def test_apply_corrections_command(tmp_path: Path):
    from audiobook_worker.cli import main

    chapter_path = tmp_path / "ch01.txt"
    chapter_path.write_text('"Hello," said Lizzy. "Hi," Elizabeth replied.', encoding="utf-8")
    output_dir = tmp_path / "scripts"
    output_dir.mkdir()

    request = {
        "bookId": "book1",
        "chapters": [
            {"chapterId": "ch01", "textPath": str(chapter_path), "title": "Chapter 1"}
        ],
        "corrections": {
            "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
            "genderOverrides": [{"characterId": "elizabeth", "gender": "female"}],
            "voiceOverrides": [],
        },
        "outputDirectory": str(output_dir),
        "language": "en",
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    with patch.dict(os.environ, {"AUDIOBOOK_LLM_MODEL": "mock"}):
        exit_code = main(["apply_corrections", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    assert len(result["artifacts"]) == 1
    assert result["artifacts"][0]["kind"] == "chapter_script"

    # verify alias merge was applied
    script = json.loads(Path(result["artifacts"][0]["path"]).read_text())
    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    assert len(speakers) == 1


def test_analyze_chapter_resumes_after_a_failed_llm_stage(tmp_path: Path):
    from audiobook_worker.cli import main
    from audiobook_worker.llm import ChapterAnalysisResult, SegmentAnnotation

    class FlakyAnalyzer:
        def __init__(self):
            self.calls = 0

        def analyze_chapter(self, request):
            self.calls += 1
            if self.calls == 1:
                request.stage_callback("characters", {"characters": []})
                request.stage_callback("voice_design", {"characters": []})
                request.stage_callback(
                    "speakers",
                    {
                        "segmentAnnotations": [
                            {
                                "segmentIndex": 0,
                                "speakerId": "narrator",
                                "confidence": 1.0,
                                "warnings": [],
                            }
                        ]
                    },
                )
                raise RuntimeError("delivery stage unavailable")
            assert request.resume_from_stage == "delivery"
            assert set(request.cached_stages) == {"characters", "voice_design", "speakers"}
            return ChapterAnalysisResult(
                characters=[],
                segment_annotations=[
                    SegmentAnnotation(
                        segment_index=0,
                        speaker_id="narrator",
                        emotion="neutral",
                        pace="normal",
                        confidence=1.0,
                    )
                ],
            )

    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text("院子里很安静。", encoding="utf-8")
    output_directory = tmp_path / "scripts"
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "bookId": "book_123",
                "chapterId": "chapter_001",
                "chapterTextPath": str(chapter_path),
                "outputDirectory": str(output_directory),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "output.json"
    analyzer = FlakyAnalyzer()

    with patch("audiobook_worker.cli.default_analyzer", return_value=analyzer):
        assert main(["analyze_chapter", str(input_path), str(output_path)]) == 1
        first_state = json.loads(
            (tmp_path / "analysis" / "chapter_001" / "state.json").read_text()
        )
        assert first_state["failedStage"] == "delivery"
        assert first_state["characters"]["status"] == "succeeded"
        assert first_state["voice_design"]["status"] == "succeeded"
        assert first_state["speakers"]["status"] == "succeeded"

        assert main(["analyze_chapter", str(input_path), str(output_path)]) == 0

    final_state = json.loads(
        (tmp_path / "analysis" / "chapter_001" / "state.json").read_text()
    )
    assert final_state["analysis"]["status"] == "succeeded"
    assert final_state["script"]["status"] == "succeeded"
    assert analyzer.calls == 2


def test_transcribe_and_plan_audio_commands_persist_post_tts_artifacts(tmp_path: Path):
    from audiobook_worker.cli import main
    from audiobook_worker.llm import ChapterAudioPlan

    script_path = tmp_path / "chapter_001.json"
    script_path.write_text(
        json.dumps(
            {
                "bookId": "book_123",
                "chapterId": "chapter_001",
                "language": "zh",
                "segments": [
                    {
                        "id": "seg_0001",
                        "type": "narration",
                        "text": "雨落在窗外。",
                        "speakerId": "active_character",
                        "emotion": "neutral",
                        "pace": "normal",
                    }
                ],
                "characters": [
                    {
                        "id": "active_character",
                        "canonicalName": "本章角色",
                        "gender": "male",
                        "ageClass": "adult",
                        "voiceId": "mimo_active",
                        "voiceDesign": "低沉、克制。",
                    },
                    {
                        "id": "inactive_character",
                        "canonicalName": "其他章节角色",
                        "gender": "female",
                        "ageClass": "adult",
                        "voiceId": "mimo_inactive",
                        "voiceDesign": "明亮。",
                    },
                ],
                "audioPlan": {"scenes": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    chapter_text_path = tmp_path / "chapter_001.txt"
    chapter_text_path.write_text("雨落在窗外。", encoding="utf-8")
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")
    analysis_directory = tmp_path / "analysis" / "chapter_001"
    transcript_path = analysis_directory / "transcript.json"
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"

    with patch(
        "audiobook_worker.cli.transcribe_audio",
        return_value={
            "version": 1,
            "segments": [{"start": 0.0, "end": 1.0, "text": "雨落在窗外。"}],
            "durationSeconds": 1.0,
            "model": "test-whisper",
        },
    ):
        input_path.write_text(
            json.dumps(
                {
                    "bookId": "book_123",
                    "chapterId": "chapter_001",
                    "scriptPath": str(script_path),
                    "voiceAudioPath": str(voice_path),
                    "analysisDirectory": str(analysis_directory),
                }
            ),
            encoding="utf-8",
        )
        assert main(["transcribe_chapter_audio", str(input_path), str(output_path)]) == 0

    assert transcript_path.exists()

    class Planner:
        def __init__(self):
            self.request = None

        def plan_audio(self, request):
            self.request = request
            return ChapterAudioPlan()

    planner = Planner()
    input_path.write_text(
        json.dumps(
            {
                "bookId": "book_123",
                "chapterId": "chapter_001",
                "scriptPath": str(script_path),
                "transcriptPath": str(transcript_path),
                "chapterTextPath": str(chapter_text_path),
                "analysisDirectory": str(analysis_directory),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with patch("audiobook_worker.cli.default_analyzer", return_value=planner):
        assert main(["plan_chapter_audio", str(input_path), str(output_path)]) == 0

    assert planner.request.text == "雨落在窗外。"
    assert planner.request.transcript[0]["start"] == 0.0
    assert [character["id"] for character in planner.request.characters] == [
        "active_character"
    ]
    assert (analysis_directory / "audio_plan.json").exists()
    saved_script = json.loads(script_path.read_text(encoding="utf-8"))
    assert saved_script["audioPlan"]["scenes"][0]["startSegmentIndex"] == 0
    assert saved_script["audioPlan"]["scenes"][0]["endSegmentIndex"] == 0
    assert saved_script["audioPlan"]["scenes"][0]["music"]["model"] == "sm-music"
    state = json.loads((analysis_directory / "state.json").read_text())
    assert state["transcript"]["status"] == "succeeded"
    assert state["audioPlan"]["status"] == "succeeded"


def test_read_file_command(tmp_path: Path):
    from audiobook_worker.cli import main

    data = {"key": "value", "nested": [1, 2]}
    file_path = tmp_path / "test.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps({"path": str(file_path)}), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["_read_file", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result == data


def test_check_rights_classifies_allowed_public_domain(tmp_path: Path):
    from audiobook_worker.cli import main

    book_path = tmp_path / "test.txt"
    book_path.write_text("Project Gutenberg public domain work", encoding="utf-8")

    request = {
        "bookPath": str(book_path),
        "metadata": {"title": "Test Book"},
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["check_rights", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "succeeded"
    assert result["classification"] == "allowed"
    assert result["reason"] == "public_domain_notice"
    assert not result["requiresAttestation"]
    assert result["evidence"] == ["public_domain_notice"]


def test_check_rights_classifies_blocked_drm(tmp_path: Path):
    from audiobook_worker.cli import main

    request = {
        "bookPath": str(tmp_path / "nonexistent.txt"),
        "metadata": {"drm": True},
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    output_path = tmp_path / "output.json"

    exit_code = main(["check_rights", str(input_path), str(output_path)])

    assert exit_code == 0
    result = json.loads(output_path.read_text())
    assert result["classification"] == "blocked"
    assert result["reason"] == "drm_detected"
