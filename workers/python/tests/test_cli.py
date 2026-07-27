import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


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


def test_synthesize_segment_audio_uses_mimo_backend_and_model(tmp_path: Path):
    """CLI selects MiMo and forwards the configured voice-design model."""
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
        "modelId": "mimo-v2.5-tts-voicedesign",
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
    backend_class.assert_called_once_with(model_id="mimo-v2.5-tts-voicedesign")
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
    assert speakers == {script["characters"][0]["id"]}


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
