import base64
import io
import json
import threading
import time
import urllib.error
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from audiobook_worker.tts import (
    KokoroTTSBackend,
    MiMoTTSBackend,
    MockTTSBackend,
    MiMoRequestError,
    VOXCPM2_PROMPT_FORMAT_VERSION,
    VoxCPM2TTSBackend,
    _MIMO_VOICE_CLONE_MODEL_ID,
    _MIMO_VOICE_DESIGN_MODEL_ID,
    _mimo_max_attempts,
    _kokoro_voice_for,
    mimo_tts_concurrency,
    mimo_tts_rpm,
    _select_torch_device,
    voxcpm2_profile_control,
    voxcpm2_reference_text,
    voice_options,
    voice_registry,
)
from audiobook_worker import tts as tts_module
from audiobook_worker.voxcpm2_profile_loudness import voxcpm2_profile_loudness


def _wav_bytes(duration_seconds: float = 0.1, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_seconds * sample_rate))
    return buffer.getvalue()


def test_voxcpm2_backend_sends_one_runner_request_for_all_chapter_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = VoxCPM2TTSBackend(
        voice_profile_directory=tmp_path / "voice-profiles" / "voxcpm2",
    )
    monkeypatch.setattr(backend, "_validate_runtime", lambda: None)
    runner_calls: list[dict[str, object]] = []

    def fake_runner(payload: dict[str, object]) -> dict[str, object]:
        runner_calls.append(payload)
        raw_segments = payload["segments"]
        assert isinstance(raw_segments, list)
        results = []
        for item in raw_segments:
            assert isinstance(item, dict)
            output_path = Path(str(item["outputPath"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x08" * 4_800)
            results.append(
                {
                    "id": item["id"],
                    "path": str(output_path),
                    "durationSeconds": 0.1,
                }
            )
        return {"status": "succeeded", "device": "mps", "segments": results}

    monkeypatch.setattr(backend, "_run_runner", fake_runner)
    artifacts = backend.synthesize_segments(
        [
            {
                "id": "seg_0001",
                "text": "第一句。",
                "speakerId": "narrator",
                "voiceId": "narrator_female",
            },
            {
                "id": "seg_0002",
                "text": "第二句。",
                "speakerId": "narrator",
                "voiceId": "narrator_female",
                "emotion": "tense",
            },
        ],
        tmp_path / "segments",
    )

    assert [artifact.path.name for artifact in artifacts] == [
        "seg_0001.wav",
        "seg_0002.wav",
    ]
    assert len(runner_calls) == 1
    assert [item["id"] for item in runner_calls[0]["segments"]] == [
        "seg_0001",
        "seg_0002",
    ]
    assert len(runner_calls[0]["profiles"]) == 1


def test_voxcpm2_payload_separates_stable_profile_from_dynamic_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    backend = VoxCPM2TTSBackend(
        voice_profile_directory=tmp_path / "voice-profiles" / "voxcpm2",
    )
    monkeypatch.setattr(backend, "_validate_runtime", lambda: None)
    runner_calls: list[dict[str, object]] = []

    def fake_runner(payload: dict[str, object]) -> dict[str, object]:
        runner_calls.append(payload)
        results = []
        for item in payload["segments"]:
            output_path = Path(str(item["outputPath"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x08" * 4_800)
            results.append({"id": item["id"], "path": str(output_path), "durationSeconds": 0.1})
        return {"status": "succeeded", "device": "mps", "segments": results}

    monkeypatch.setattr(backend, "_run_runner", fake_runner)
    voice_design = "角色：克制的成年男性，声线低沉浑厚，胸腔共鸣自然，咬字清楚。"
    long_direction = "方向标记" * 100
    backend.synthesize_segments(
        [
            {
                "id": "seg_profile_boundary",
                "text": "现在必须离开。",
                "speakerId": "guard",
                "voiceId": "guard_voice",
                "voiceDesign": voice_design,
                "voiceDirection": long_direction,
                "voiceSceneContext": "SCENE_ONLY_should_not_enter_a_control",
                "emotion": "tense",
                "pace": "fast",
                "language": "zh",
            }
        ],
        tmp_path / "segments",
    )

    payload = runner_calls[0]
    profile = payload["profiles"][0]
    segment = payload["segments"][0]
    assert payload["promptFormatVersion"] == VOXCPM2_PROMPT_FORMAT_VERSION
    assert profile["voiceDesign"] == voice_design
    assert profile["promptFormatVersion"] == VOXCPM2_PROMPT_FORMAT_VERSION
    assert profile["profileLoudness"] == voxcpm2_profile_loudness()
    assert profile["referenceText"] == voxcpm2_reference_text("zh")
    assert "角色：" not in profile["profileControl"]
    assert "低沉" in profile["profileControl"]
    assert "SCENE_ONLY" not in profile["profileControl"]
    assert voice_design not in segment["delivery"]
    assert "谨慎克制" in segment["delivery"]
    assert "语速偏快" in segment["delivery"]
    assert len(segment["delivery"]) < len(long_direction) + 40
    assert "SCENE_ONLY" not in segment["delivery"]


def test_voxcpm2_uses_english_controls_for_non_chinese_segments(tmp_path: Path, monkeypatch):
    backend = VoxCPM2TTSBackend(
        voice_profile_directory=tmp_path / "voice-profiles" / "voxcpm2",
    )
    monkeypatch.setattr(backend, "_validate_runtime", lambda: None)
    calls: list[dict[str, object]] = []

    def fake_runner(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        results = []
        for item in payload["segments"]:
            output_path = Path(str(item["outputPath"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x08" * 4_800)
            results.append({"id": item["id"], "path": str(output_path), "durationSeconds": 0.1})
        return {"status": "succeeded", "segments": results}

    monkeypatch.setattr(backend, "_run_runner", fake_runner)
    backend.synthesize_segments(
        [
            {
                "id": "seg_english",
                "text": "The door opened slowly.",
                "voiceId": "english_guard",
                "voiceDesign": "Role: an adult male voice, low and clear.",
                "emotion": "neutral",
                "pace": "normal",
                "language": "en",
            }
        ],
        tmp_path / "segments",
    )

    profile = calls[0]["profiles"][0]
    segment = calls[0]["segments"][0]
    assert profile["referenceText"] == voxcpm2_reference_text("en")
    assert profile["profileControl"] == "an adult male voice, low and clear."
    assert segment["delivery"] == "natural and restrained, natural conversational pace"


def test_voxcpm2_profile_control_has_english_fallback_without_mutating_design():
    design = "角色：成年男性，声线低沉浑厚，咬字清楚。"

    assert voxcpm2_profile_control(design).startswith("成年男性")
    english = voxcpm2_profile_control(design, "en")
    assert "adult male voice" in english
    assert not any("\u3400" <= character <= "\u9fff" for character in english)
    assert design.startswith("角色：")


def test_voxcpm2_profile_cache_requires_current_loudness_contract(tmp_path: Path):
    profile_path = tmp_path / "profiles" / "guard_zh.wav"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_bytes(_wav_bytes())
    metadata_path = profile_path.with_suffix(".json")
    signature = tts_module._voxcpm2_voice_profile_signature(
        voice_id="guard",
        description="成年男性，低沉而清晰。",
        language="zh",
    )
    metadata = {
        "version": tts_module._VOXCPM2_VOICE_PROFILE_VERSION,
        "backend": "voxcpm2",
        "modelId": "VoxCPM2",
        "promptFormatVersion": VOXCPM2_PROMPT_FORMAT_VERSION,
        "signature": signature,
        "profileLoudness": voxcpm2_profile_loudness(),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert tts_module._voxcpm2_profile_is_usable(
        profile_path,
        metadata_path,
        signature=signature,
    )
    metadata.pop("profileLoudness")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert not tts_module._voxcpm2_profile_is_usable(
        profile_path,
        metadata_path,
        signature=signature,
    )


def test_voxcpm2_profile_signature_includes_loudness_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    before = tts_module._voxcpm2_voice_profile_signature(
        voice_id="guard",
        description="成年男性，低沉而清晰。",
        language="zh",
    )
    monkeypatch.setattr(
        tts_module,
        "voxcpm2_profile_loudness",
        lambda: {**voxcpm2_profile_loudness(), "version": 2},
    )

    assert tts_module._voxcpm2_voice_profile_signature(
        voice_id="guard",
        description="成年男性，低沉而清晰。",
        language="zh",
    ) != before


def test_voxcpm2_narrator_uses_selected_stable_identity(tmp_path: Path, monkeypatch):
    backend = VoxCPM2TTSBackend(
        voice_profile_directory=tmp_path / "voice-profiles" / "voxcpm2",
    )
    monkeypatch.setattr(backend, "_validate_runtime", lambda: None)
    calls: list[dict[str, object]] = []

    def fake_runner(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        results = []
        for item in payload["segments"]:
            output_path = Path(str(item["outputPath"]))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setparams((1, 2, 48_000, 4_800, "NONE", "not compressed"))
                wav_file.writeframes(b"\x00\x08" * 4_800)
            results.append({"id": item["id"], "path": str(output_path), "durationSeconds": 0.1})
        return {"status": "succeeded", "segments": results}

    monkeypatch.setattr(backend, "_run_runner", fake_runner)
    backend.synthesize_segments(
        [
            {
                "id": "seg_narrator",
                "text": "他走进了雨夜。",
                "speakerId": "narrator",
                "voiceId": "narrator_male",
                "voiceDesign": "角色：成年女性，明亮甜美。",
            }
        ],
        tmp_path / "segments",
    )

    profile = calls[0]["profiles"][0]
    assert "成年男性" in profile["voiceDesign"]
    assert "成年女性" not in profile["voiceDesign"]


def test_mimo_backend_sends_voice_design_prompt_and_text_as_assistant(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

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


def test_mimo_voiceclone_creates_one_book_profile_and_reuses_it_for_each_segment(tmp_path: Path):
    encoded = base64.b64encode(_wav_bytes()).decode("ascii")
    client = MagicMock(side_effect=[encoded, encoded, encoded])
    profile_directory = tmp_path / "voice-profiles"
    backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_CLONE_MODEL_ID,
        request_audio=client,
        voice_profile_directory=profile_directory,
    )

    segment = {
        "id": "seg_clone_01",
        "text": "夜色渐渐沉了下来。",
        "speakerId": "narrator",
        "voiceId": "narrator_female",
        "emotion": "neutral",
        "pace": "normal",
    }
    backend.synthesize_segment(segment, tmp_path / "chapter_001")
    backend.synthesize_segment(
        {
            **segment,
            "id": "seg_clone_02",
            "text": "门外忽然传来脚步声。",
            "emotion": "tense",
            "pace": "fast",
        },
        tmp_path / "chapter_001",
    )

    assert client.call_count == 3
    reference_request = client.call_args_list[0].args[0]
    first_segment_request = client.call_args_list[1].args[0]
    second_segment_request = client.call_args_list[2].args[0]
    assert reference_request["model"] == _MIMO_VOICE_DESIGN_MODEL_ID
    assert reference_request["messages"][1]["content"].startswith("这是一段稳定的声音样本")
    assert first_segment_request["model"] == _MIMO_VOICE_CLONE_MODEL_ID
    assert second_segment_request["model"] == _MIMO_VOICE_CLONE_MODEL_ID
    assert first_segment_request["audio"]["voice"].startswith("data:audio/wav;base64,")
    assert first_segment_request["audio"]["voice"] == second_segment_request["audio"]["voice"]
    assert "严格复用参考音频中的同一位说话者" in first_segment_request["messages"][0]["content"]
    assert "语气谨慎克制" in second_segment_request["messages"][0]["content"]
    assert (profile_directory / "narrator_female.wav").is_file()
    assert (profile_directory / "narrator_female.json").is_file()

    # A new chapter/backend instance reads the same book-level profile instead
    # of designing a new voice again.
    next_client = MagicMock(return_value=encoded)
    next_backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_CLONE_MODEL_ID,
        request_audio=next_client,
        voice_profile_directory=profile_directory,
    )
    next_backend.synthesize_segment(
        {**segment, "id": "seg_clone_03", "text": "他停在了门前。"},
        tmp_path / "chapter_002",
    )
    next_client.assert_called_once()
    assert next_client.call_args.args[0]["model"] == _MIMO_VOICE_CLONE_MODEL_ID


def test_mimo_voice_reference_is_not_checked_as_a_speech_segment(tmp_path: Path):
    """Voice references keep their own validation contract and short source text."""
    long_reference = base64.b64encode(_wav_bytes(10.0)).decode("ascii")
    accepted_segment = base64.b64encode(_wav_bytes(0.1)).decode("ascii")
    client = MagicMock(side_effect=[long_reference, accepted_segment])
    profile_directory = tmp_path / "voice-profiles"
    backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_CLONE_MODEL_ID,
        request_audio=client,
        voice_profile_directory=profile_directory,
    )

    backend.synthesize_segment(
        {
            "id": "seg_clone_reference_boundary",
            "text": "嘘。",
            "speakerId": "narrator",
            "voiceId": "narrator_female",
        },
        tmp_path / "chapter_001",
    )

    assert client.call_count == 2
    assert (profile_directory / "narrator_female.wav").is_file()


def test_mimo_backend_uses_independent_character_design_over_fallback_voice(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )
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


def test_mimo_backend_appends_scene_and_fine_grained_direction(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

    backend.synthesize_segment(
        {
            "id": "seg_direction",
            "text": "现在必须离开。",
            "voiceId": "character_auto_0123456789abcdef",
            "voiceDesign": "角色：克制的年轻男性，声线清透但有支撑。",
            "voiceDirection": "语速中等偏快但保持从容，句首短暂停顿，在“必须离开”上加强重音，句尾收紧。",
            "voiceSceneContext": "前文：屋外传来急促脚步；当前：现在必须离开；后文：他抓起外衣。",
            "emotion": "tense",
            "pace": "fast",
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert "角色：克制的年轻男性" in content
    assert "场景：前文：屋外传来急促脚步" in content
    assert "语速中等偏快但保持从容" in content


def test_mimo_backend_uses_dynamic_pace_for_narration(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

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
    assert "一位专业中文有声书旁白" in content
    assert "声音洪亮饱满" in content
    assert "胸腔共鸣自然" in content
    assert "保持稳定统一的旁白声线" in content
    assert "语速偏快" in content
    assert "语速适中" not in content


def test_mimo_backend_locks_narrator_identity_to_selected_male_voice(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

    backend.synthesize_segment(
        {
            "id": "seg_male_narration",
            "text": "他走进了雨夜。",
            "speakerId": "narrator",
            "voiceId": "narrator_male",
            "voiceDesign": "角色：成年女性，明亮甜美。",
            "voiceDirection": "语速很快，情绪紧张。",
            "emotion": "tense",
            "pace": "fast",
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert "固定为同一位成年男性" in content
    assert "成年女性，明亮甜美" not in content
    assert "不要因情绪或语速改变成另一种声音" in content


def test_mimo_backend_ignores_unknown_segment_metadata_for_narration(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

    backend.synthesize_segment(
        {
            "id": "seg_scene_director",
            "text": "门外传来脚步声。",
            "speakerId": "narrator",
            "voiceId": "narrator_default",
            "emotion": "tense",
            "pace": "normal",
            "unusedMetadata": {"note": "ignored"},
        },
        tmp_path,
    )

    content = client.call_args.args[0]["messages"][0]["content"]
    assert "语气谨慎克制" in content


def test_mimo_backend_describes_teasing_without_turning_it_into_happiness(tmp_path: Path):
    client = MagicMock(return_value=base64.b64encode(_wav_bytes()).decode("ascii"))
    backend = MiMoTTSBackend(
        api_key="test-key", model_id=_MIMO_VOICE_DESIGN_MODEL_ID, request_audio=client
    )

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


def test_mimo_runtime_limits_default_and_clamp_unsafe_values(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUDIOBOOK_MIMO_CONCURRENCY", raising=False)
    monkeypatch.delenv("AUDIOBOOK_MIMO_RPM", raising=False)
    monkeypatch.delenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", raising=False)
    assert mimo_tts_concurrency() == 1
    assert mimo_tts_rpm() == 80
    assert _mimo_max_attempts() == 3

    monkeypatch.setenv("AUDIOBOOK_MIMO_CONCURRENCY", "99")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RPM", "99")
    monkeypatch.setenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "0")
    assert mimo_tts_concurrency() == 1
    assert mimo_tts_rpm() == 80
    assert _mimo_max_attempts() == 1

    monkeypatch.setenv("AUDIOBOOK_MIMO_RPM", "12")
    assert mimo_tts_rpm() == 12


def test_mimo_rate_gate_paces_request_starts_without_a_burst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The 80 RPM budget is global and uses a no-burst 0.75 second cadence."""

    rate_state_path = tmp_path / "mimo-rate-state.json"
    clock = {"seconds": 100.0}
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
        return clock["seconds"]

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock["seconds"] += seconds

    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(rate_state_path))
    monkeypatch.setenv("AUDIOBOOK_MIMO_RPM", "80")
    monkeypatch.setattr(tts_module.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(tts_module.time, "sleep", fake_sleep)

    starts: list[float] = []
    for _ in range(81):
        with tts_module._mimo_request_rate_gate() as rate_gate:
            rate_gate.wait_for_turn()
            starts.append(clock["seconds"])

    assert starts[0] == 100.0
    assert all(later - earlier >= 0.75 for earlier, later in zip(starts, starts[1:]))
    # A half-open rolling minute beginning with the first request contains
    # exactly 80 starts; the 81st starts at the next minute boundary.
    assert sum(start < starts[0] + 60.0 for start in starts) == 80
    assert starts[80] - starts[0] == pytest.approx(60.0)
    assert sleep_calls and all(seconds >= 0.75 for seconds in sleep_calls)


def test_mimo_rate_gate_cooldown_blocks_the_next_shared_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rate_state_path = tmp_path / "mimo-rate-state.json"
    clock = {"seconds": 10.0}
    sleep_calls: list[float] = []

    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(rate_state_path))
    monkeypatch.setattr(tts_module.time, "monotonic", lambda: clock["seconds"])

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        clock["seconds"] += seconds

    monkeypatch.setattr(tts_module.time, "sleep", fake_sleep)

    with tts_module._mimo_request_rate_gate() as first_gate:
        first_gate.wait_for_turn()
        first_gate.set_cooldown(5.0)
    with tts_module._mimo_request_rate_gate() as second_gate:
        second_gate.wait_for_turn()

    assert sleep_calls == [5.0]
    state = json.loads(rate_state_path.read_text(encoding="utf-8"))
    assert state["lastStartMonotonic"] == pytest.approx(15.0)


def test_mimo_http_gate_serializes_requests_from_independent_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Only one real HTTP request may cross the shared file-backed gate."""

    rate_state_path = tmp_path / "mimo-rate-state.json"
    first_started = threading.Event()
    release_request = threading.Event()
    active = 0
    maximum_active = 0
    started_at: list[float] = []
    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"audio":{"data":"encoded"}}}]}'

    def fake_urlopen(_request, timeout):
        nonlocal active, maximum_active
        assert timeout == 180
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            started_at.append(time.monotonic())
            first_started.set()
        try:
            release_request.wait(timeout=2)
        finally:
            with lock:
                active -= 1
        return FakeResponse()

    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(rate_state_path))
    monkeypatch.setenv("AUDIOBOOK_MIMO_RPM", "80")
    monkeypatch.setattr("audiobook_worker.tts.urllib.request.urlopen", fake_urlopen)
    backends = [MiMoTTSBackend(api_key="test-key") for _ in range(2)]

    def request(backend: MiMoTTSBackend) -> None:
        try:
            results.append(backend._request_audio_from_api({"model": "test"}))
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=request, args=(backends[0],))
    second = threading.Thread(target=request, args=(backends[1],))
    first.start()
    assert first_started.wait(timeout=1)
    second.start()
    time.sleep(0.05)
    assert maximum_active == 1
    release_request.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert results == ["encoded", "encoded"]
    assert maximum_active == 1
    assert started_at[1] - started_at[0] >= 0.7


def test_mimo_prepares_each_missing_profile_once_before_parallel_synthesis(tmp_path: Path):
    encoded = base64.b64encode(_wav_bytes()).decode("ascii")
    client = MagicMock(side_effect=[encoded, encoded])
    backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_CLONE_MODEL_ID,
        request_audio=client,
        voice_profile_directory=tmp_path / "voice-profiles",
    )
    segments = [
        {
            "id": "seg_0001",
            "text": "第一句。",
            "speakerId": "narrator",
            "voiceId": "narrator_female",
        },
        {
            "id": "seg_0002",
            "text": "第二句。",
            "speakerId": "narrator",
            "voiceId": "narrator_female",
        },
        {
            "id": "seg_0003",
            "text": "第三句。",
            "voiceId": "male_adult_01",
        },
    ]

    backend.prepare_voice_profiles(segments)

    # Two different voices produce two design requests; duplicate narrator
    # segments never trigger another profile build.
    assert client.call_count == 2
    assert (tmp_path / "voice-profiles" / "narrator_female.wav").is_file()
    assert (tmp_path / "voice-profiles" / "male_adult_01.wav").is_file()


def test_mimo_voiceclone_profile_lock_is_shared_by_independent_backends(tmp_path: Path):
    """Concurrent chapter workers must create one shared reference profile.

    Each backend below models a separate chapter-worker process: it has its
    own in-memory cache and lock, but shares the book-level profile directory.
    The profile-specific file lock must prevent two reference-design calls.
    """

    encoded = base64.b64encode(_wav_bytes()).decode("ascii")
    requests: list[dict] = []
    requests_lock = threading.Lock()
    reference_started = threading.Event()
    release_reference = threading.Event()

    def request_audio(payload: dict) -> str:
        with requests_lock:
            requests.append(payload)
        if payload["model"] == _MIMO_VOICE_DESIGN_MODEL_ID:
            reference_started.set()
            release_reference.wait(timeout=2)
        return encoded

    profile_directory = tmp_path / "voice-profiles"
    segment = {
        "id": "seg_shared_profile",
        "text": "夜色渐渐沉了下来。",
        "speakerId": "narrator",
        "voiceId": "narrator_female",
    }
    backends = [
        MiMoTTSBackend(
            api_key="test-key",
            model_id=_MIMO_VOICE_CLONE_MODEL_ID,
            request_audio=request_audio,
            voice_profile_directory=profile_directory,
        )
        for _ in range(2)
    ]
    errors: list[BaseException] = []

    def synthesize(backend: MiMoTTSBackend, output_directory: Path) -> None:
        try:
            backend.synthesize_segment(segment, output_directory)
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=synthesize, args=(backends[0], tmp_path / "chapter_001"))
    second = threading.Thread(target=synthesize, args=(backends[1], tmp_path / "chapter_002"))
    first.start()
    assert reference_started.wait(timeout=1)
    second.start()
    # Give the second backend a chance to reach the same profile lock. It may
    # not make another reference request until the first atomic replacement is
    # complete.
    time.sleep(0.05)
    with requests_lock:
        assert sum(
            request["model"] == _MIMO_VOICE_DESIGN_MODEL_ID for request in requests
        ) == 1
    release_reference.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    with requests_lock:
        assert sum(
            request["model"] == _MIMO_VOICE_DESIGN_MODEL_ID for request in requests
        ) == 1
    assert (profile_directory / "narrator_female.wav").is_file()
    assert (profile_directory / "narrator_female.json").is_file()


def test_mimo_network_request_retries_transient_errors_with_bounded_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[object, int]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"audio":{"data":"encoded"}}}]}'

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        if len(calls) < 3:
            raise urllib.error.URLError("temporary network failure")
        return FakeResponse()

    monkeypatch.setenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(tmp_path / "mimo-rate-state.json"))
    monkeypatch.setattr("audiobook_worker.tts.urllib.request.urlopen", fake_urlopen)
    backend = MiMoTTSBackend(api_key="test-key")

    assert backend._request_audio_from_api({"model": "test"}) == "encoded"
    assert len(calls) == 3
    assert all(timeout == 180 for _, timeout in calls)


def test_mimo_quality_rejection_retries_the_current_segment_before_cache_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    rejected_audio = base64.b64encode(_wav_bytes(10.0)).decode("ascii")
    accepted_audio = base64.b64encode(_wav_bytes(0.1)).decode("ascii")
    encoded_responses = [rejected_audio, accepted_audio]
    calls = 0

    class FakeResponse:
        def __init__(self, encoded_audio: str) -> None:
            self._encoded_audio = encoded_audio

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"audio": {"data": self._encoded_audio}}}]}
            ).encode("utf-8")

    def fake_urlopen(_request, timeout):
        nonlocal calls
        assert timeout == 180
        encoded_audio = encoded_responses[calls]
        calls += 1
        return FakeResponse(encoded_audio)

    monkeypatch.setenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(tmp_path / "mimo-rate-state.json"))
    monkeypatch.setattr("audiobook_worker.tts.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        tts_module._MiMoRequestRateGate,
        "wait_for_turn",
        lambda _self: None,
    )
    backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_DESIGN_MODEL_ID,
    )

    artifact = backend.synthesize_segment(
        {"id": "seg_retry", "text": "嘘。", "pace": "normal"},
        tmp_path,
    )

    assert calls == 2
    assert artifact.path.read_bytes() == base64.b64decode(accepted_audio)
    assert artifact.duration_seconds == pytest.approx(0.1)


def test_mimo_quality_rejection_keeps_an_existing_cache_entry(tmp_path: Path):
    output_path = tmp_path / "seg_existing.wav"
    output_path.write_bytes(b"prior accepted audio")
    rejected_audio = base64.b64encode(_wav_bytes(10.0)).decode("ascii")
    backend = MiMoTTSBackend(
        api_key="test-key",
        model_id=_MIMO_VOICE_DESIGN_MODEL_ID,
        request_audio=lambda _payload: rejected_audio,
    )

    with pytest.raises(MiMoRequestError, match="unusable TTS segment WAV"):
        backend.synthesize_segment(
            {"id": "seg_existing", "text": "嘘。", "pace": "normal"},
            tmp_path,
        )

    assert output_path.read_bytes() == b"prior accepted audio"


def test_mimo_rate_limited_retry_honors_retry_after_and_applies_a_global_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    retry_parameters: list[tuple[int, float | None, bool]] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"audio":{"data":"encoded"}}}]}'

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 180
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "7"},
                io.BytesIO(b"slow down"),
            )
        return FakeResponse()

    def fake_retry_delay(
        attempt: int,
        *,
        retry_after: float | None = None,
        rate_limited: bool = False,
    ) -> float:
        retry_parameters.append((attempt, retry_after, rate_limited))
        return 0.01

    monkeypatch.setenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(tmp_path / "mimo-rate-state.json"))
    monkeypatch.setattr("audiobook_worker.tts.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr(tts_module, "_mimo_retry_delay_seconds", fake_retry_delay)
    backend = MiMoTTSBackend(api_key="test-key")

    assert backend._request_audio_from_api({"model": "test"}) == "encoded"
    assert calls == 2
    assert retry_parameters == [(1, 7.0, True)]


def test_mimo_rate_limited_retry_uses_a_five_second_minimum_without_retry_after(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUDIOBOOK_MIMO_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setattr(tts_module.random, "uniform", lambda _start, _end: 0.0)

    assert tts_module._mimo_retry_delay_seconds(1, rate_limited=True) == 5.0
    assert tts_module._mimo_retry_delay_seconds(
        1,
        retry_after=7.0,
        rate_limited=True,
    ) == 7.0


def test_mimo_network_request_does_not_retry_non_retryable_http_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(b"invalid API key"),
        )

    monkeypatch.setenv("AUDIOBOOK_MIMO_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("AUDIOBOOK_MIMO_RATE_STATE_PATH", str(tmp_path / "mimo-rate-state.json"))
    monkeypatch.setattr("audiobook_worker.tts.urllib.request.urlopen", fake_urlopen)
    backend = MiMoTTSBackend(api_key="test-key")

    with pytest.raises(MiMoRequestError, match="HTTP 401") as error:
        backend._request_audio_from_api({"model": "test"})
    assert calls == 1
    assert error.value.retryable is False


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


def test_kokoro_backend_never_uses_narrator_fallback_voice(tmp_path: Path):
    import numpy as np

    mock_result = MagicMock()
    mock_result.audio = np.zeros(24_000, dtype=np.float32)
    mock_pipeline = MagicMock(return_value=[mock_result])

    with patch("audiobook_worker.tts.KPipeline") as mock_kp:
        mock_kp.return_value = mock_pipeline
        backend = KokoroTTSBackend()
        backend.synthesize_segment(
            {
                "id": "seg_male_narrator_kokoro",
                "text": "这是男性旁白。",
                "speakerId": "narrator",
                "voiceId": "narrator_male",
                "fallbackVoiceId": "af_heart",
            },
            tmp_path,
        )

    assert mock_pipeline.call_args.kwargs["voice"] == "am_michael"
