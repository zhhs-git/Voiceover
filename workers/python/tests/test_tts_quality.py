from __future__ import annotations

import io
import math
import struct
import wave

import pytest

from audiobook_worker.tts_quality import (
    TtsSegmentAudioQualityError,
    analyze_tts_segment_wav,
    maximum_duration_for_segment,
    speech_units_for_text,
    validate_tts_segment_wav,
)


def _tone(
    duration_seconds: float,
    *,
    amplitude: float = 0.12,
    sample_rate: int = 24_000,
) -> list[int]:
    return [
        round(
            math.sin(2 * math.pi * 440 * index / sample_rate) * 32767 * amplitude
        )
        for index in range(round(sample_rate * duration_seconds))
    ]


def _silence(duration_seconds: float, *, sample_rate: int = 24_000) -> list[int]:
    return [0] * round(sample_rate * duration_seconds)


def _wav_bytes(samples: list[int], *, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setparams((1, 2, sample_rate, len(samples), "NONE", "not compressed"))
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
    return buffer.getvalue()


def test_speech_unit_count_handles_chinese_and_word_text():
    assert speech_units_for_text("你好，world 42") == 4
    assert maximum_duration_for_segment("你好", "slow") > maximum_duration_for_segment(
        "你好", "fast"
    )


def test_short_quiet_slow_segment_is_accepted():
    audio = _wav_bytes(
        _tone(0.5, amplitude=0.004)
        + _silence(1.5)
        + _tone(0.5, amplitude=0.004)
    )

    result = analyze_tts_segment_wav(
        audio,
        text="请不要惊动任何人。",
        pace="slow",
    )

    assert result.accepted
    assert result.issues == ()
    assert result.longest_silence_seconds == pytest.approx(1.5, abs=0.25)


def test_duration_exceeding_text_limit_is_rejected_without_pcm_scan():
    audio = _wav_bytes(_tone(0.25) + _silence(10.0))

    result = analyze_tts_segment_wav(audio, text="嘘。", pace="normal")

    assert result.accepted is False
    assert result.issues == ("duration_exceeds_text_limit",)
    assert result.silence_ratio is None


def test_long_trailing_silence_is_rejected_inside_a_plausible_duration():
    text = "这是一段足够长的旁白文字，用来给正常的语速留出宽松余量。"
    audio = _wav_bytes(_tone(1.0) + _silence(9.0))

    result = analyze_tts_segment_wav(audio, text=text, pace="normal")

    assert result.accepted is False
    assert "continuous_silence_exceeds_limit" in result.issues
    assert "silence_ratio_exceeds_limit" in result.issues
    assert result.longest_silence_seconds == pytest.approx(9.0, abs=0.25)


def test_long_interior_silence_is_rejected_inside_a_plausible_duration():
    text = "这是一段足够长的对白文字，用来验证中间空白不会被当作正常停顿。"
    audio = _wav_bytes(_tone(1.0) + _silence(9.0) + _tone(1.0))

    with pytest.raises(TtsSegmentAudioQualityError, match="continuous_silence_exceeds_limit"):
        validate_tts_segment_wav(audio, text=text, pace="normal")
