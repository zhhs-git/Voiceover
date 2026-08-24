from __future__ import annotations

import io
import math
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_MINIMUM_UNITS_PER_SECOND = {
    "slow": 0.65,
    "normal": 0.9,
    "fast": 1.15,
}
_PAUSE_ALLOWANCE_SECONDS = 8.0
_MINIMUM_MAX_DURATION_SECONDS = 8.0
_SILENCE_WINDOW_SECONDS = 0.25
_SILENCE_THRESHOLD_DBFS = -55.0
_MAX_CONTINUOUS_SILENCE_SECONDS = 8.0
_MAX_SILENCE_RATIO = 0.9
_MIN_DURATION_FOR_SILENCE_RATIO_SECONDS = 6.0


class TtsSegmentAudioQualityError(RuntimeError):
    """A TTS WAV cannot safely be used as a speech segment."""

    def __init__(
        self,
        message: str,
        *,
        result: TtsSegmentAudioQualityResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class TtsSegmentAudioQualityResult:
    speech_units: int
    duration_seconds: float
    maximum_duration_seconds: float
    silence_ratio: float | None
    longest_silence_seconds: float | None
    issues: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.issues

    def describe(self) -> str:
        details = [
            f"duration={self.duration_seconds:.2f}s",
            f"maxDuration={self.maximum_duration_seconds:.2f}s",
        ]
        if self.silence_ratio is not None:
            details.append(f"silenceRatio={self.silence_ratio:.1%}")
        if self.longest_silence_seconds is not None:
            details.append(f"longestSilence={self.longest_silence_seconds:.2f}s")
        return ", ".join(details)

    def to_dict(self) -> dict[str, object]:
        """Return stable camelCase diagnostics for worker/API error payloads."""
        return {
            "speechUnits": self.speech_units,
            "durationSeconds": round(self.duration_seconds, 3),
            "maximumDurationSeconds": round(self.maximum_duration_seconds, 3),
            "silenceRatio": (
                round(self.silence_ratio, 6)
                if self.silence_ratio is not None
                else None
            ),
            "longestSilenceSeconds": (
                round(self.longest_silence_seconds, 3)
                if self.longest_silence_seconds is not None
                else None
            ),
            "issues": list(self.issues),
        }


def speech_units_for_text(text: object) -> int:
    """Count conservative speaking units for mixed Chinese and word-based text."""
    value = str(text or "")
    return len(_CJK_RE.findall(value)) + len(_WORD_RE.findall(value))


def maximum_duration_for_segment(text: object, pace: object = "normal") -> float:
    """Return a deliberately generous spoken-duration ceiling for one segment."""
    pace_id = str(pace or "normal").strip().lower()
    minimum_rate = _MINIMUM_UNITS_PER_SECOND.get(
        pace_id, _MINIMUM_UNITS_PER_SECOND["normal"]
    )
    units = speech_units_for_text(text)
    return max(
        _MINIMUM_MAX_DURATION_SECONDS,
        _PAUSE_ALLOWANCE_SECONDS + units / minimum_rate,
    )


def analyze_tts_segment_wav(
    source: bytes | bytearray | Path | str,
    *,
    text: object,
    pace: object = "normal",
) -> TtsSegmentAudioQualityResult:
    """Inspect a MiMo speech WAV without creating or modifying cache files."""
    maximum_duration_seconds = maximum_duration_for_segment(text, pace)
    speech_units = speech_units_for_text(text)
    try:
        if isinstance(source, (bytes, bytearray)):
            with wave.open(io.BytesIO(bytes(source)), "rb") as wav_file:
                return _analyze_open_wav(
                    wav_file,
                    speech_units=speech_units,
                    maximum_duration_seconds=maximum_duration_seconds,
                )
        with wave.open(str(Path(source)), "rb") as wav_file:
            return _analyze_open_wav(
                wav_file,
                speech_units=speech_units,
                maximum_duration_seconds=maximum_duration_seconds,
            )
    except (OSError, EOFError, wave.Error, ZeroDivisionError) as error:
        raise TtsSegmentAudioQualityError(
            "TTS segment WAV is unreadable or has no usable audio format."
        ) from error


def validate_tts_segment_wav(
    source: bytes | bytearray | Path | str,
    *,
    text: object,
    pace: object = "normal",
) -> TtsSegmentAudioQualityResult:
    """Return an accepted inspection result or raise with the rejection reason."""
    result = analyze_tts_segment_wav(source, text=text, pace=pace)
    if result.accepted:
        return result
    raise TtsSegmentAudioQualityError(
        "TTS segment WAV failed the quality gate "
        f"({', '.join(result.issues)}; {result.describe()}).",
        result=result,
    )


def _analyze_open_wav(
    wav_file: wave.Wave_read,
    *,
    speech_units: int,
    maximum_duration_seconds: float,
) -> TtsSegmentAudioQualityResult:
    frame_count = wav_file.getnframes()
    sample_rate = wav_file.getframerate()
    channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    if frame_count <= 0 or sample_rate <= 0 or channels <= 0:
        raise TtsSegmentAudioQualityError(
            "TTS segment WAV has no audio frames or an invalid sample rate."
        )
    if wav_file.getcomptype() != "NONE" or sample_width not in {1, 2, 3, 4}:
        raise TtsSegmentAudioQualityError(
            "TTS segment WAV must use an uncompressed 8-, 16-, 24-, or 32-bit PCM format."
        )

    duration_seconds = frame_count / sample_rate
    if duration_seconds > maximum_duration_seconds:
        return TtsSegmentAudioQualityResult(
            speech_units=speech_units,
            duration_seconds=duration_seconds,
            maximum_duration_seconds=maximum_duration_seconds,
            silence_ratio=None,
            longest_silence_seconds=None,
            issues=("duration_exceeds_text_limit",),
        )

    silence_ratio, longest_silence_seconds = _silence_stats(
        wav_file,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        duration_seconds=duration_seconds,
    )
    issues: list[str] = []
    if longest_silence_seconds >= _MAX_CONTINUOUS_SILENCE_SECONDS:
        issues.append("continuous_silence_exceeds_limit")
    if (
        duration_seconds >= _MIN_DURATION_FOR_SILENCE_RATIO_SECONDS
        and silence_ratio >= _MAX_SILENCE_RATIO
    ):
        issues.append("silence_ratio_exceeds_limit")
    return TtsSegmentAudioQualityResult(
        speech_units=speech_units,
        duration_seconds=duration_seconds,
        maximum_duration_seconds=maximum_duration_seconds,
        silence_ratio=silence_ratio,
        longest_silence_seconds=longest_silence_seconds,
        issues=tuple(issues),
    )


def _silence_stats(
    wav_file: wave.Wave_read,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
    duration_seconds: float,
) -> tuple[float, float]:
    frames_per_window = max(1, round(sample_rate * _SILENCE_WINDOW_SECONDS))
    silent_seconds = 0.0
    longest_silence_seconds = 0.0
    current_silence_seconds = 0.0
    while True:
        pcm = wav_file.readframes(frames_per_window)
        if not pcm:
            break
        frames_read = len(pcm) // (sample_width * channels)
        if frames_read <= 0:
            break
        window_seconds = frames_read / sample_rate
        if _window_rms_is_silent(pcm, sample_width):
            silent_seconds += window_seconds
            current_silence_seconds += window_seconds
            longest_silence_seconds = max(
                longest_silence_seconds, current_silence_seconds
            )
        else:
            current_silence_seconds = 0.0
    return silent_seconds / duration_seconds, longest_silence_seconds


def _window_rms_is_silent(pcm: bytes, sample_width: int) -> bool:
    sum_squares = 0.0
    sample_count = 0
    max_amplitude = float((1 << (sample_width * 8 - 1)) - 1)
    for sample in _pcm_samples(pcm, sample_width):
        normalized = sample / max_amplitude
        sum_squares += normalized * normalized
        sample_count += 1
    if sample_count == 0:
        return True
    rms = math.sqrt(sum_squares / sample_count)
    threshold = 10 ** (_SILENCE_THRESHOLD_DBFS / 20.0)
    return rms <= threshold


def _pcm_samples(pcm: bytes, sample_width: int):
    if sample_width == 1:
        for sample in pcm:
            yield sample - 128
        return
    if sample_width == 2:
        for (sample,) in struct.iter_unpack("<h", pcm):
            yield sample
        return
    if sample_width == 4:
        for (sample,) in struct.iter_unpack("<i", pcm):
            yield sample
        return
    for offset in range(0, len(pcm) - 2, 3):
        value = pcm[offset] | (pcm[offset + 1] << 8) | (pcm[offset + 2] << 16)
        yield value - (1 << 24) if value & (1 << 23) else value
