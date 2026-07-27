import base64
import io
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiobook_worker.tts import (
    KokoroTTSBackend,
    MiMoTTSBackend,
    MockTTSBackend,
    _kokoro_voice_for,
    _select_torch_device,
    voice_options,
    voice_registry,
)


def _wav_bytes(duration_seconds: float = 0.1, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * sample_rate))
    return buffer.getvalue()


def test_mimo_backend_sends_voice_design_prompt_and_text_as_assistant(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(api_key="test-key", request_audio=client)

    artifact = backend.synthesize_segment(
        {
            "id": "seg_mimo",
            "text": "夜色渐渐沉了下来。",
            "voiceId": "male_adult_01",
            "emotion": "tense",
            "pace": "slow",
        },
        tmp_path,
    )

    request = client.call_args.args[0]
    assert request["model"] == "mimo-v2.5-tts-voicedesign"
    assert request["messages"][0]["role"] == "user"
    assert "中年男性" in request["messages"][0]["content"]
    assert "语速舒缓，停顿自然" in request["messages"][0]["content"]
    assert request["messages"][1] == {
        "role": "assistant",
        "content": "夜色渐渐沉了下来。",
    }
    assert request["audio"] == {"format": "wav", "optimize_text_preview": False}
    assert artifact.path.read_bytes().startswith(b"RIFF")
    assert artifact.duration_seconds == pytest.approx(0.1)


def test_mimo_backend_uses_independent_character_design_over_fallback_voice(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(api_key="test-key", request_audio=client)
    description = "一位中文成年男性，音色温厚清晰，为该角色建立独立且可辨识的基础音色。"

    backend.synthesize_segment(
        {
            "id": "seg_auto_voice",
            "text": "我知道了。",
            "voiceId": "character_auto_0123456789abcdef",
            "fallbackVoiceId": "male_adult_01",
            "voiceDescription": description,
            "emotion": "neutral",
            "pace": "normal",
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert description in content
    assert "低沉浑厚" not in content


def test_mimo_backend_forces_fixed_pace_for_narration(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(api_key="test-key", request_audio=client)

    backend.synthesize_segment(
        {
            "id": "seg_narration",
            "text": "夜色渐渐沉了下来。",
            "speakerId": "narrator",
            "voiceId": "narrator_default",
            "emotion": "neutral",
            "pace": "fast",
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert "语速适中" in content
    assert "语速偏快" not in content


def test_mimo_backend_describes_teasing_without_turning_it_into_happiness(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(api_key="test-key", request_audio=client)

    backend.synthesize_segment(
        {
            "id": "seg_teasing",
            "text": "哟，还挺乖。",
            "voiceId": "male_adult_02",
            "emotion": "teasing",
            "pace": "normal",
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert "戏谑嘲弄" in content
    assert "轻蔑笑意" in content
    assert "真诚的开心" in content


def test_mimo_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="MIMO_API_KEY"):
        MiMoTTSBackend(api_key=None, key_loader=lambda: None)


def test_select_torch_device_prefers_mps_when_available():
    class FakeTorch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return True

        class cuda:
            @staticmethod
            def is_available():
                return True

    assert _select_torch_device(FakeTorch, "auto") == "mps"


def test_select_torch_device_errors_when_requested_gpu_is_unavailable():
    class FakeTorch:
        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class cuda:
            @staticmethod
            def is_available():
                return False

    try:
        _select_torch_device(FakeTorch, "mps")
    except RuntimeError as error:
        assert "MPS was requested" in str(error)
    else:
        raise AssertionError("expected RuntimeError")


def test_mock_backend_generates_segment_audio_artifact(tmp_path: Path):
    backend = MockTTSBackend()
    segment = {
        "id": "seg_0001",
        "text": "Hello world.",
        "voiceId": "narrator_default",
        "emotion": "neutral",
    }

    artifact = backend.synthesize_segment(segment, tmp_path)

    assert artifact.kind == "segment_audio"
    assert artifact.path.exists()
    assert artifact.path.suffix == ".wav"


def test_voice_registry_declares_language_and_license_metadata():
    voices = voice_registry()

    narrator = voices["narrator_default"]
    assert narrator["languages"] == ["en"]
    assert "licenseNotes" in narrator


def test_mimo_voice_options_are_provider_mapped_and_not_fixed_to_four():
    voices = voice_options("mimo")

    assert len(voices) > 4
    assert {voice["id"] for voice in voices} >= {
        "narrator_default",
        "narrator_female",
        "female_adult_05",
        "male_adult_05",
        "neutral_dialogue_01",
    }
    assert "female_british_01" not in {voice["id"] for voice in voices}


def test_voice_registry_has_kokoro_voices():
    voices = voice_registry()
    for voice_id, entry in voices.items():
        assert "kokoroVoice" in entry, f"{voice_id} missing kokoroVoice"
        assert isinstance(entry["kokoroVoice"], str), f"{voice_id} kokoroVoice not a string"
        assert len(entry["kokoroVoice"]) > 2, f"{voice_id} kokoroVoice too short"
        assert entry["backend"] == "kokoro", f"{voice_id} backend should be kokoro, got {entry['backend']}"


def test_voice_registry_backend_is_kokoro():
    voices = voice_registry()
    for voice_id, entry in voices.items():
        assert entry["backend"] == "kokoro", f"{voice_id} backend should be kokoro"


def test_kokoro_voice_for_maps_known_voices():
    assert _kokoro_voice_for("narrator_default") == "af_heart"
    assert _kokoro_voice_for("female_adult_01") == "af_heart"
    assert _kokoro_voice_for("male_adult_01") == "am_michael"
    assert _kokoro_voice_for("neutral_dialogue_01") == "af_nicole"


def test_kokoro_voice_for_falls_back_on_unknown_id():
    assert _kokoro_voice_for("nonexistent") == "af_heart"  # falls back to narrator_default


def test_voice_assignment_distributes_characters_across_pool():
    """Different characters of the same gender get different voices deterministically."""
    from audiobook_worker.script_builder import _voice_for_gender

    # Two female characters should (likely) get different voices
    voice_a = _voice_for_gender("female", "elizabeth")
    voice_b = _voice_for_gender("female", "jane")
    voice_c = _voice_for_gender("female", "lydia")

    # All should be in the female pool
    assert voice_a.startswith("female_adult_")
    assert voice_b.startswith("female_adult_")
    assert voice_c.startswith("female_adult_")

    # Same character always gets same voice (deterministic)
    assert _voice_for_gender("female", "elizabeth") == voice_a
    assert _voice_for_gender("female", "jane") == voice_b

    # Male voices
    voice_d = _voice_for_gender("male", "darcy")
    voice_e = _voice_for_gender("male", "bingley")
    assert voice_d.startswith("male_adult_")
    assert voice_e.startswith("male_adult_")


def test_parler_backend_synthesize_segment_produces_wav(tmp_path: Path):
    """ParlerTTSBackend.synthesize_segment writes a WAV and returns correct artifact."""
    import numpy as np

    fake_audio = np.zeros(24000, dtype=np.float32)

    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24000
    mock_model.to.return_value = mock_model  # .to(device) returns itself
    mock_model.generate.return_value = MagicMock(
        cpu=lambda: MagicMock(numpy=lambda: fake_audio.reshape(1, -1))
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(input_ids=MagicMock())

    with patch("audiobook_worker.tts.ParlerTTSForConditionalGeneration") as mock_cls, \
         patch("audiobook_worker.tts.AutoTokenizer") as mock_tok_cls:
        mock_cls.from_pretrained.return_value = mock_model
        mock_tok_cls.from_pretrained.return_value = mock_tokenizer

        from audiobook_worker.tts import ParlerTTSBackend
        backend = ParlerTTSBackend()

        segment = {
            "id": "seg_0001",
            "text": "It was a dark and stormy night.",
            "voiceId": "narrator_default",
            "emotion": "neutral",
            "intensity": 0.2,
            "pace": "normal",
        }
        artifact = backend.synthesize_segment(segment, tmp_path)

    assert artifact.kind == "segment_audio"
    assert artifact.path.suffix == ".wav"
    assert artifact.path.exists()
    assert artifact.duration_seconds > 0


def test_parler_backend_builds_description_with_emotion(tmp_path: Path):
    """Emotion modifiers are appended to the base voice description."""
    import numpy as np

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

        from audiobook_worker.tts import ParlerTTSBackend
        backend = ParlerTTSBackend()

        segment = {
            "id": "seg_0002",
            "text": "Get out of my house!",
            "voiceId": "male_adult_01",
            "emotion": "angry",
            "intensity": 0.7,
            "pace": "fast",
        }
        backend.synthesize_segment(segment, tmp_path)

    first_call_args = mock_tokenizer.call_args_list[0][0][0]
    assert "angry" in first_call_args.lower() or "forceful" in first_call_args.lower()


def test_parler_backend_prefers_character_voice_description(tmp_path: Path):
    import numpy as np

    fake_audio = np.zeros(24_000, dtype=np.float32)
    mock_model = MagicMock()
    mock_model.config.sampling_rate = 24_000
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

        from audiobook_worker.tts import ParlerTTSBackend

        backend = ParlerTTSBackend()
        backend.synthesize_segment(
            {
                "id": "seg_auto_parler",
                "text": "我知道了。",
                "voiceId": "character_auto_0123456789abcdef",
                "voiceDescription": "角色专属的稳定中文声线。",
                "emotion": "neutral",
                "pace": "normal",
            },
            tmp_path,
        )

    assert "角色专属的稳定中文声线" in mock_tokenizer.call_args_list[0][0][0]


def test_kokoro_backend_synthesize_segment_produces_wav(tmp_path: Path):
    """KokoroTTSBackend.synthesize_segment writes a WAV and returns correct artifact."""
    import numpy as np
    import torch as _torch

    fake_audio = _torch.tensor(np.zeros(24000, dtype=np.float32))
    mock_result = MagicMock()
    mock_result.audio = fake_audio

    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [mock_result]

    with patch("audiobook_worker.tts.KPipeline") as mock_kp:
        mock_kp.return_value = mock_pipeline

        backend = KokoroTTSBackend()

        segment = {
            "id": "seg_kokoro",
            "text": "It is a truth universally acknowledged.",
            "voiceId": "narrator_default",
            "emotion": "neutral",
        }
        artifact = backend.synthesize_segment(segment, tmp_path)

    assert artifact.kind == "segment_audio"
    assert artifact.path.suffix == ".wav"
    assert artifact.path.exists()
    assert artifact.duration_seconds > 0


def test_kokoro_backend_uses_character_fallback_voice(tmp_path: Path):
    import numpy as np

    mock_result = MagicMock()
    mock_result.audio = np.zeros(24_000, dtype=np.float32)
    mock_pipeline = MagicMock(return_value=[mock_result])

    with patch("audiobook_worker.tts.KPipeline") as mock_kp:
        mock_kp.return_value = mock_pipeline
        backend = KokoroTTSBackend()
        backend.synthesize_segment(
            {
                "id": "seg_auto_kokoro",
                "text": "我知道了。",
                "voiceId": "character_auto_0123456789abcdef",
                "fallbackVoiceId": "male_adult_01",
                "emotion": "neutral",
            },
            tmp_path,
        )

    assert mock_pipeline.call_args.kwargs["voice"] == "am_michael"
