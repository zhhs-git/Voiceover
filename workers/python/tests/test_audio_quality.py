from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from audiobook_worker.audio_quality import (
    AUDIO_QUALITY_DETECTOR_VERSION,
    _classify_recovery_intervals,
    analyze_audio,
    main,
    repair_short_suspicious_intervals,
)


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is required for audio quality analysis",
)


def _write_wav(path: Path, samples: list[int], sample_rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, sample_rate, len(samples), "NONE", "not compressed"))
        output.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _sine(sample_rate: int, duration: float, frequency: float, amplitude: float) -> list[int]:
    return [
        round(
            math.sin(2 * math.pi * frequency * index / sample_rate)
            * 32767
            * amplitude
        )
        for index in range(round(sample_rate * duration))
    ]


def test_steady_audio_is_normal(tmp_path: Path):
    path = tmp_path / "steady.wav"
    _write_wav(path, _sine(24000, 0.6, 440, 0.25))

    result = analyze_audio(path)

    assert result.status == "normal"
    assert result.suspicious_times == ()
    assert result.issues == ()


def test_short_loud_high_frequency_burst_is_reported(tmp_path: Path):
    sample_rate = 24000
    samples = [0] * round(sample_rate * 0.6)
    burst = _sine(sample_rate, 0.012, 9000, 0.95)
    start = round(sample_rate * 0.24)
    samples[start : start + len(burst)] = burst
    path = tmp_path / "sharp.wav"
    _write_wav(path, samples, sample_rate)

    result = analyze_audio(path)

    assert result.status == "sharp_suspected"
    assert "high_frequency_burst" in result.issues
    assert result.requires_repair
    assert result.actionable_intervals
    assert not result.review_only
    assert any(0.15 <= time_seconds <= 0.35 for time_seconds in result.suspicious_times)


def test_flat_full_scale_burst_is_reported_as_clipping(tmp_path: Path):
    sample_rate = 24000
    samples = [0] * round(sample_rate * 0.6)
    start = round(sample_rate * 0.24)
    samples[start : start + round(sample_rate * 0.012)] = [32767] * round(
        sample_rate * 0.012
    )
    path = tmp_path / "clipped.wav"
    _write_wav(path, samples, sample_rate)

    result = analyze_audio(path)

    assert result.status == "burst_or_clipping_suspected"
    assert "clipping" in result.issues
    assert any(0.15 <= time_seconds <= 0.35 for time_seconds in result.suspicious_times)


def test_sustained_full_scale_signal_is_reported_as_clipping(tmp_path: Path):
    path = tmp_path / "full_scale.wav"
    _write_wav(path, [32767] * round(24000 * 0.2))

    result = analyze_audio(path)

    assert result.status == "burst_or_clipping_suspected"
    assert "clipping" in result.issues


def test_short_abrupt_low_frequency_burst_is_a_spectral_candidate(tmp_path: Path):
    sample_rate = 24000
    samples = _sine(sample_rate, 0.6, 440, 0.05)
    start = round(sample_rate * 0.24)
    loud_window = _sine(sample_rate, 0.012, 440, 0.95)
    samples[start : start + len(loud_window)] = loud_window
    path = tmp_path / "dynamic_maxima.wav"
    _write_wav(path, samples, sample_rate)

    result = analyze_audio(path)

    # This test signal starts and stops abruptly, so its narrow-band transient
    # is a review candidate.  The detector no longer emits the old, misleading
    # ``peak_maximum`` / ``rms_maximum`` classifications.
    assert result.status == "sharp_suspected"
    assert "tonal_spectral_event" in result.issues
    assert any(0.15 <= start_time <= 0.35 for start_time, _ in result.suspicious_intervals)
    assert result.requires_repair


def test_short_interval_repair_preserves_source_and_writes_shorter_wav(tmp_path: Path):
    sample_rate = 24000
    source = tmp_path / "source.wav"
    repaired = tmp_path / "repaired.wav"
    samples = _sine(sample_rate, 0.8, 440, 0.2)
    burst = _sine(sample_rate, 0.04, 9000, 0.95)
    start = round(sample_rate * 0.32)
    samples[start : start + len(burst)] = burst
    _write_wav(source, samples, sample_rate)

    result = repair_short_suspicious_intervals(
        source,
        repaired,
        ((0.32, 0.36),),
    )

    assert result.repaired
    assert result.status == "repaired"
    assert source.is_file()
    assert repaired.is_file()
    with wave.open(str(source), "rb") as original, wave.open(str(repaired), "rb") as output:
        assert output.getnframes() < original.getnframes()
    assert result.to_dict()["crossfadeSeconds"] == pytest.approx(0.03)
    assert AUDIO_QUALITY_DETECTOR_VERSION >= 1


def test_repair_accepts_an_isolated_two_second_defect(tmp_path: Path):
    source = tmp_path / "source.wav"
    repaired = tmp_path / "repaired.wav"
    _write_wav(source, _sine(24000, 3.4, 440, 0.2))

    result = repair_short_suspicious_intervals(
        source,
        repaired,
        ((0.7, 2.6),),
    )

    assert result.repaired
    assert result.intervals[0] == pytest.approx((0.68, 2.62))
    assert repaired.is_file()


def test_scattered_tonal_events_are_review_only_not_repair_targets():
    intervals = (
        (0.1, 0.2),
        (0.6, 0.7),
        (1.1, 1.2),
        (1.6, 1.7),
        (2.1, 2.2),
    )

    actionable, review = _classify_recovery_intervals((), intervals)

    assert actionable == ()
    assert review == intervals


def test_repair_rejects_interval_at_audio_edge(tmp_path: Path):
    source = tmp_path / "source.wav"
    repaired = tmp_path / "repaired.wav"
    _write_wav(source, _sine(24000, 0.6, 440, 0.2))

    result = repair_short_suspicious_intervals(source, repaired, ((0.0, 0.05),))

    assert result.status == "not_eligible"
    assert result.reason == "interval_near_audio_edge"
    assert not repaired.exists()


def test_cli_prints_a_human_readable_result(tmp_path: Path, capsys):
    path = tmp_path / "steady.wav"
    _write_wav(path, _sine(24000, 0.2, 440, 0.2))

    assert main([str(path)]) == 0

    output = capsys.readouterr().out
    assert "文件：" in output
    assert "结果：正常" in output
    assert "风险时间点：无" in output


def test_missing_audio_returns_cli_error(capsys, tmp_path: Path):
    assert main([str(tmp_path / "missing.wav")]) == 2
    assert "检测失败" in capsys.readouterr().out
