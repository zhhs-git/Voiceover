"""Local detection and safe repair helpers for sharp WAV transients.

The detector intentionally remains usable as a standalone command-line tool.
The Stable Audio pipeline consumes its structured result separately and uses
the repair helper only for short, well-bounded defects; this module never
decides whether an asset may be mixed.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import math
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_FRAME_RE = re.compile(
    r"^frame:(?P<index>\d+)\s+pts:\S+\s+pts_time:(?P<time>\S+)"
)
SHARP_PEAK_THRESHOLD_DBFS = -10.0
AUDIO_QUALITY_DETECTOR_VERSION = 1
MAX_REPAIR_INTERVAL_SECONDS = 0.15
MAX_REPAIR_TOTAL_SECONDS = 0.50
REPAIR_PADDING_SECONDS = 0.02
REPAIR_CROSSFADE_SECONDS = 0.03
MIN_REPAIR_RETAINED_PIECE_SECONDS = 0.06
_STAT_RE = re.compile(
    r"^lavfi\.astats\.1\.(?P<name>[A-Za-z0-9_]+)=(?P<value>\S+)"
)
_SPECTRAL_STAT_RE = re.compile(
    r"^lavfi\.aspectralstats\.1\.(?P<name>[A-Za-z0-9_]+)=(?P<value>\S+)"
)
SPECTRAL_CREST_THRESHOLD = 180.0
LOW_FREQUENCY_CREST_THRESHOLD = 150.0
LOW_FREQUENCY_CENTROID_HZ = 1000.0
LOW_FREQUENCY_FLATNESS = 0.04
QUIET_TONAL_RMS_DBFS = -20.0
MIDBAND_TRANSIENT_CREST = 200.0
MIDBAND_TRANSIENT_CENTROID_HZ = 1500.0
MIDBAND_TRANSIENT_FLATNESS = 0.15
MIDBAND_TRANSIENT_FLUX = 0.05
BROADBAND_CENTROID_HZ = 6500.0
BROADBAND_FLATNESS = 0.45
BROADBAND_ROLLOFF_HZ = 12000.0
SPECTRAL_WINDOW_SIZE = 2048


class AudioQualityError(RuntimeError):
    """Raised when an audio file cannot be inspected."""


@dataclass(frozen=True)
class AudioQualityWindow:
    """Short-time statistics used by the detector."""

    index: int
    time_seconds: float
    peak_dbfs: float | None
    rms_dbfs: float | None
    high_rms_dbfs: float | None
    high_peak_dbfs: float | None
    flat_factor: float | None


@dataclass(frozen=True)
class AudioQualityResult:
    """The stable, small result shape for the first validation pass."""

    path: Path
    duration_seconds: float
    status: str
    risk_score: float
    suspicious_times: tuple[float, ...]
    suspicious_intervals: tuple[tuple[float, float], ...]
    issues: tuple[str, ...]
    windows_analyzed: int
    clipped_windows: int
    high_frequency_burst_windows: int

    @property
    def is_suspicious(self) -> bool:
        return self.status != "normal"

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-safe representation for manifests and reports."""

        return {
            "path": str(self.path),
            "durationSeconds": self.duration_seconds,
            "status": self.status,
            "riskScore": self.risk_score,
            "suspiciousTimes": list(self.suspicious_times),
            "suspiciousIntervals": [list(interval) for interval in self.suspicious_intervals],
            "issues": list(self.issues),
            "windowsAnalyzed": self.windows_analyzed,
            "clippedWindows": self.clipped_windows,
            "highFrequencyBurstWindows": self.high_frequency_burst_windows,
        }


@dataclass(frozen=True)
class AudioQualityRepairResult:
    """Outcome of a non-destructive short-interval repair attempt."""

    status: str
    source_path: Path
    output_path: Path | None
    intervals: tuple[tuple[float, float], ...]
    reason: str | None = None

    @property
    def repaired(self) -> bool:
        return self.status == "repaired" and self.output_path is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sourcePath": str(self.source_path),
            "outputPath": str(self.output_path) if self.output_path else None,
            "intervals": [list(interval) for interval in self.intervals],
            "reason": self.reason,
            "crossfadeSeconds": REPAIR_CROSSFADE_SECONDS if self.repaired else None,
        }


def _finite_float(value: str | float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stat_float(value: str) -> float | None:
    """Parse an astats value, retaining +inf for flat clipped samples."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or (math.isinf(number) and number < 0):
        return None
    return number


def _parse_stats(output: str) -> list[dict[str, float]]:
    """Parse ``ametadata=print`` output into one record per frame."""

    records: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        frame_match = _FRAME_RE.match(line)
        if frame_match:
            if current is not None:
                records.append(current)
            time_seconds = _finite_float(frame_match.group("time"))
            if time_seconds is None:
                continue
            current = {
                "index": float(frame_match.group("index")),
                "time_seconds": time_seconds,
            }
            continue

        if current is None:
            continue
        stat_match = _STAT_RE.match(line)
        if not stat_match:
            continue
        value = _stat_float(stat_match.group("value"))
        if value is not None:
            current[stat_match.group("name")] = value

    if current is not None:
        records.append(current)
    return records


def _find_ffmpeg(ffmpeg_path: str | Path | None) -> str:
    executable = str(ffmpeg_path) if ffmpeg_path is not None else shutil.which("ffmpeg")
    if not executable:
        raise AudioQualityError("ffmpeg not found in PATH")
    return executable


def _wav_duration(path: Path) -> tuple[float, int, int]:
    try:
        with wave.open(str(path), "rb") as input_file:
            frame_count = input_file.getnframes()
            sample_rate = input_file.getframerate()
            sample_width = input_file.getsampwidth()
    except (OSError, wave.Error) as error:
        raise AudioQualityError(f"unsupported or unreadable WAV file: {path}") from error

    if frame_count <= 0 or sample_rate <= 0:
        raise AudioQualityError(f"WAV file has no audio frames: {path}")
    return frame_count / sample_rate, sample_rate, sample_width


def _run_stats(
    ffmpeg: str,
    path: Path,
    *,
    filter_expression: str,
    window_seconds: float,
    sample_rate: int,
) -> list[dict[str, float]]:
    # ``astats.reset`` counts audio frames, not seconds.  Split the stream
    # into fixed-size sample frames first, then reset after every one.  Without
    # this, a value such as ``reset=0.05`` is coerced to zero and the reported
    # peak/RMS values accumulate from the beginning of the file.
    window_samples = max(1, round(window_seconds * sample_rate))
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostats",
        "-i",
        str(path),
        "-af",
        (
            f"{filter_expression},"
            "aformat=channel_layouts=mono,"
            f"asetnsamples=n={window_samples}:p=0,"
            "astats=metadata=1:reset=1,"
            "ametadata=print:file=-"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise AudioQualityError(f"unable to start ffmpeg: {error}") from error

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        detail = output.strip().splitlines()[-1] if output.strip() else "unknown ffmpeg error"
        raise AudioQualityError(f"ffmpeg audio analysis failed: {detail}")
    records = _parse_stats(output)
    if not records:
        raise AudioQualityError("ffmpeg returned no audio analysis frames")
    return records


def _parse_spectral_stats(output: str) -> list[dict[str, float]]:
    """Parse per-frame metadata emitted by FFmpeg's ``aspectralstats``."""

    records: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        frame_match = _FRAME_RE.match(line)
        if frame_match:
            if current is not None:
                records.append(current)
            time_seconds = _finite_float(frame_match.group("time"))
            if time_seconds is None:
                continue
            current = {
                "index": float(frame_match.group("index")),
                "time_seconds": time_seconds,
            }
            continue

        if current is None:
            continue
        stat_match = _SPECTRAL_STAT_RE.match(line)
        if not stat_match:
            continue
        value = _stat_float(stat_match.group("value"))
        if value is not None:
            current[stat_match.group("name")] = value

    if current is not None:
        records.append(current)
    return records


def _run_spectral_stats(ffmpeg: str, path: Path) -> list[dict[str, float]]:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostats",
        "-i",
        str(path),
        "-af",
        (
            "aformat=channel_layouts=mono,"
            f"aspectralstats=win_size={SPECTRAL_WINDOW_SIZE}:overlap=0.5,"
            "ametadata=print:file=-"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise AudioQualityError(f"unable to start ffmpeg: {error}") from error

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        detail = output.strip().splitlines()[-1] if output.strip() else "unknown ffmpeg error"
        raise AudioQualityError(f"ffmpeg spectral analysis failed: {detail}")
    records = _parse_spectral_stats(output)
    if not records:
        raise AudioQualityError("ffmpeg returned no spectral analysis frames")
    return records


def _median(values: Iterable[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _merge_windows(
    full_stats: list[dict[str, float]],
    high_stats: list[dict[str, float]],
) -> list[AudioQualityWindow]:
    high_by_index = {int(item["index"]): item for item in high_stats}
    windows: list[AudioQualityWindow] = []
    for item in full_stats:
        index = int(item["index"])
        high = high_by_index.get(index, {})
        windows.append(
            AudioQualityWindow(
                index=index,
                time_seconds=item["time_seconds"],
                peak_dbfs=item.get("Peak_level"),
                rms_dbfs=item.get("RMS_level"),
                high_rms_dbfs=high.get("RMS_level"),
                high_peak_dbfs=high.get("Peak_level"),
                flat_factor=item.get("Flat_factor"),
            )
        )
    return windows


def _group_times(times: Iterable[float], minimum_gap_seconds: float = 0.2) -> tuple[float, ...]:
    grouped: list[float] = []
    for time_seconds in sorted(times):
        if not grouped or time_seconds - grouped[-1] > minimum_gap_seconds:
            grouped.append(round(time_seconds, 2))
    return tuple(grouped)


def _group_intervals(
    times: Iterable[float],
    *,
    frame_duration_seconds: float,
    maximum_gap_seconds: float = 0.2,
) -> tuple[tuple[float, float], ...]:
    """Turn adjacent flagged windows into listenable time ranges."""

    ordered = sorted(times)
    if not ordered:
        return ()
    intervals: list[tuple[float, float]] = []
    start = previous = ordered[0]
    for time_seconds in ordered[1:]:
        if time_seconds - previous <= maximum_gap_seconds:
            previous = time_seconds
            continue
        intervals.append(
            (round(start, 2), round(previous + frame_duration_seconds, 2))
        )
        start = previous = time_seconds
    intervals.append((round(start, 2), round(previous + frame_duration_seconds, 2)))
    return tuple(intervals)


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    maximum_gap_seconds: float = 0.05,
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(intervals)
    if not ordered:
        return ()
    merged: list[list[float]] = [[*ordered[0]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + maximum_gap_seconds:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return tuple((round(start, 2), round(end, 2)) for start, end in merged)


def _safe_repair_intervals(
    intervals: Iterable[tuple[float, float]],
    *,
    duration_seconds: float,
) -> tuple[tuple[tuple[float, float], ...], str | None]:
    """Validate, pad, and merge intervals before a destructive edit.

    The detector reports listening ranges rounded for people.  Repair needs a
    small context margin, but it must never remove a large part of a cue or
    manufacture a crossfade at the edge of a file.
    """

    raw: list[tuple[float, float]] = []
    for interval in intervals:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            return (), "invalid_interval"
        start, end = interval
        if not (
            isinstance(start, (int, float))
            and not isinstance(start, bool)
            and isinstance(end, (int, float))
            and not isinstance(end, bool)
            and math.isfinite(float(start))
            and math.isfinite(float(end))
        ):
            return (), "invalid_interval"
        start_seconds = float(start)
        end_seconds = float(end)
        if not 0 <= start_seconds < end_seconds <= duration_seconds:
            return (), "interval_outside_audio"
        if end_seconds - start_seconds > MAX_REPAIR_INTERVAL_SECONDS + 1e-9:
            return (), "interval_too_long"
        raw.append((start_seconds, end_seconds))

    if not raw:
        return (), "no_intervals"
    if sum(end - start for start, end in raw) > MAX_REPAIR_TOTAL_SECONDS + 1e-9:
        return (), "total_interval_too_long"

    padded = sorted(
        (
            max(0.0, start - REPAIR_PADDING_SECONDS),
            min(duration_seconds, end + REPAIR_PADDING_SECONDS),
        )
        for start, end in raw
    )
    merged: list[list[float]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    if any(
        start < MIN_REPAIR_RETAINED_PIECE_SECONDS
        or duration_seconds - end < MIN_REPAIR_RETAINED_PIECE_SECONDS
        for start, end in merged
    ):
        return (), "interval_near_audio_edge"

    kept_lengths: list[float] = []
    cursor = 0.0
    for start, end in merged:
        kept_lengths.append(start - cursor)
        cursor = end
    kept_lengths.append(duration_seconds - cursor)
    if any(length < MIN_REPAIR_RETAINED_PIECE_SECONDS for length in kept_lengths):
        return (), "insufficient_audio_for_crossfade"

    return tuple((start, end) for start, end in merged), None


def repair_short_suspicious_intervals(
    source_path: str | Path,
    output_path: str | Path,
    intervals: Iterable[tuple[float, float]],
    *,
    ffmpeg_path: str | Path | None = None,
) -> AudioQualityRepairResult:
    """Remove tiny bad spans and crossfade the remaining audio.

    ``source_path`` is never modified.  The caller is responsible for placing
    it in a preserved/rejected location before asking for a repair.  A failed
    attempt leaves no partially-written ``output_path`` behind.
    """

    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        return AudioQualityRepairResult(
            "not_eligible", source, None, (), "source_and_output_must_differ"
        )
    try:
        duration_seconds, _sample_rate, _sample_width = _wav_duration(source)
    except AudioQualityError as error:
        return AudioQualityRepairResult("failed", source, None, (), str(error))

    safe_intervals, reason = _safe_repair_intervals(
        intervals,
        duration_seconds=duration_seconds,
    )
    if reason is not None:
        return AudioQualityRepairResult("not_eligible", source, None, (), reason)

    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in safe_intervals:
        kept.append((cursor, start))
        cursor = end
    kept.append((cursor, duration_seconds))

    labels: list[str] = []
    filters: list[str] = []
    for index, (start, end) in enumerate(kept):
        label = f"kept{index}"
        labels.append(label)
        filters.append(
            f"[0:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[{label}]"
        )
    current = labels[0]
    for index, label in enumerate(labels[1:], start=1):
        joined = f"joined{index}"
        filters.append(
            f"[{current}][{label}]acrossfade=d={REPAIR_CROSSFADE_SECONDS:.6f}:"
            f"c1=tri:c2=tri[{joined}]"
        )
        current = joined

    try:
        executable = _find_ffmpeg(ffmpeg_path)
    except AudioQualityError as error:
        return AudioQualityRepairResult(
            "failed", source, None, safe_intervals, str(error)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.stem}.quality-repair.part{destination.suffix}"
    )
    command = [
        executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{current}]",
        "-c:a",
        "pcm_s16le",
        str(temporary_path),
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return AudioQualityRepairResult(
                "failed", source, None, safe_intervals, "ffmpeg_repair_timeout"
            )
        except OSError as error:
            return AudioQualityRepairResult(
                "failed", source, None, safe_intervals, f"ffmpeg_repair_failed:{error}"
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            return AudioQualityRepairResult(
                "failed",
                source,
                None,
                safe_intervals,
                detail[-1] if detail else "ffmpeg_repair_failed",
            )
        try:
            _wav_duration(temporary_path)
        except AudioQualityError as error:
            return AudioQualityRepairResult(
                "failed", source, None, safe_intervals, str(error)
            )
        temporary_path.replace(destination)
        return AudioQualityRepairResult(
            "repaired", source, destination, safe_intervals
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _spectral_event_times(
    windows: list[AudioQualityWindow],
    *,
    spectral_stats: list[dict[str, float]],
) -> tuple[list[float], list[float]]:
    """Find short tonal and broadband events matching the reviewed examples.

    The accepted bad examples have two shapes: a narrow, sharp spectral crest
    (the events near 0.46s, 14s and 20s) and a brief broadband high-frequency
    burst (the event near 10.12s).  A normal musical or spoken passage can
    have a high spectral crest too, so the tonal rule additionally requires
    either an unusually active mid-band transient or a quiet, narrow spectral
    event.  The latter is a *review candidate*, not proof that the source is
    defective; it is intentionally kept separate from clipping and loud
    broadband bursts.
    """

    if not windows:
        return [], []

    window_times = [window.time_seconds for window in windows]
    tonal_times: list[float] = []
    broadband_times: list[float] = []
    for spectral in spectral_stats:
        time_seconds = spectral["time_seconds"]
        window_index = max(
            0,
            min(len(windows) - 1, bisect_right(window_times, time_seconds) - 1),
        )
        window = windows[window_index]

        crest = spectral.get("crest", 0.0)
        centroid = spectral.get("centroid", 0.0)
        flatness = spectral.get("flatness", 0.0)
        flux = spectral.get("flux", 0.0)
        rolloff = spectral.get("rolloff", 0.0)
        midband_transient = (
            crest >= MIDBAND_TRANSIENT_CREST
            and centroid >= MIDBAND_TRANSIENT_CENTROID_HZ
            and flatness >= MIDBAND_TRANSIENT_FLATNESS
            and flux >= MIDBAND_TRANSIENT_FLUX
        )
        quiet_tonal_transient = (
            window.rms_dbfs is not None
            and window.rms_dbfs <= QUIET_TONAL_RMS_DBFS
            and centroid <= 2000.0
            and flatness <= 0.15
            and (
                crest >= SPECTRAL_CREST_THRESHOLD
                or (
                    crest >= LOW_FREQUENCY_CREST_THRESHOLD
                    and centroid <= LOW_FREQUENCY_CENTROID_HZ
                    and flatness <= LOW_FREQUENCY_FLATNESS
                )
            )
        )
        tonal_event = midband_transient or quiet_tonal_transient
        broadband_event = (
            window.peak_dbfs is not None
            and window.peak_dbfs >= SHARP_PEAK_THRESHOLD_DBFS
            and centroid >= BROADBAND_CENTROID_HZ
            and flatness >= BROADBAND_FLATNESS
            and rolloff >= BROADBAND_ROLLOFF_HZ
        )
        if tonal_event:
            tonal_times.append(time_seconds)
        if broadband_event:
            broadband_times.append(time_seconds)
    return tonal_times, broadband_times


def analyze_audio(
    path: str | Path,
    *,
    ffmpeg_path: str | Path | None = None,
    window_seconds: float = 0.05,
    highpass_hz: float = 6000.0,
) -> AudioQualityResult:
    """Analyze a WAV for clipping and short, perceptually sharp transients.

    The detector deliberately uses conservative rules.  A high-frequency
    sound by itself is not enough; it must also be a short energy burst and
    have a relatively high full-band peak.  Narrow spectral events are kept as
    review candidates because they can be audible artifacts even when their
    peak level is modest.  Flat full-scale samples are treated as clipping
    when ``astats`` reports a flat factor or a full-scale RMS value, which
    avoids flagging an unclipped sine wave that touches 0 dBFS.
    """

    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise AudioQualityError(f"audio file does not exist: {audio_path}")
    duration_seconds, sample_rate, _sample_width = _wav_duration(audio_path)
    if not math.isfinite(window_seconds) or not 0 < window_seconds <= 1:
        raise AudioQualityError("window_seconds must be greater than 0 and at most 1")
    if not math.isfinite(highpass_hz) or highpass_hz <= 0:
        raise AudioQualityError("highpass_hz must be greater than 0")

    ffmpeg = _find_ffmpeg(ffmpeg_path)
    # Keep the filter below Nyquist for short test WAVs as well as Stable
    # Audio's normal 44.1 kHz output.
    cutoff = min(highpass_hz, sample_rate * 0.45)
    full_stats = _run_stats(
        ffmpeg,
        audio_path,
        filter_expression="anull",
        window_seconds=window_seconds,
        sample_rate=sample_rate,
    )
    high_stats = _run_stats(
        ffmpeg,
        audio_path,
        filter_expression=f"highpass=f={cutoff:g}",
        window_seconds=window_seconds,
        sample_rate=sample_rate,
    )
    windows = _merge_windows(full_stats, high_stats)
    spectral_stats = _run_spectral_stats(ffmpeg, audio_path)

    # ``astats`` reports silence as ``-inf``.  Keep that information in the
    # baseline calculation instead of dropping it: a short burst surrounded
    # by silence must not become the apparent average of the whole file.
    high_values = [
        window.high_rms_dbfs if window.high_rms_dbfs is not None else -80.0
        for window in windows
    ]
    high_baseline = _median(high_values)
    clipped_times: list[float] = []
    burst_times: list[float] = []
    for position, window in enumerate(windows):
        peak = window.peak_dbfs
        high_rms = window.high_rms_dbfs
        if (
            peak is not None
            and peak >= -0.1
            and (
                (
                    window.flat_factor is not None
                    and window.flat_factor >= 5.0
                )
                or (
                    window.rms_dbfs is not None
                    and window.rms_dbfs >= -0.1
                )
            )
        ):
            clipped_times.append(window.time_seconds)

        nearby_high_values = (
            high_values[max(0, position - 2) : position]
            + high_values[position + 1 : position + 3]
        )
        local_high_baseline = _median(nearby_high_values)
        has_local_rise = (
            high_rms is not None
            and local_high_baseline is not None
            and high_rms >= local_high_baseline + 6.0
        )
        has_global_rise = (
            high_rms is not None
            and high_baseline is not None
            and high_rms >= high_baseline + 10.0
        )
        if (
            peak is not None
            and peak >= SHARP_PEAK_THRESHOLD_DBFS
            and high_rms is not None
            and high_rms >= -24.0
            and (has_local_rise or has_global_rise)
        ):
            burst_times.append(window.time_seconds)

    tonal_times, broadband_times = _spectral_event_times(
        windows,
        spectral_stats=spectral_stats,
    )

    clipped_windows = len(clipped_times)
    burst_windows = len(burst_times)
    all_suspicious_times = [
        *clipped_times,
        *burst_times,
        *tonal_times,
        *broadband_times,
    ]
    suspicious_times = _group_times(all_suspicious_times, minimum_gap_seconds=0.07)
    frame_times = [window.time_seconds for window in windows]
    frame_differences = [
        later - earlier
        for earlier, later in zip(frame_times, frame_times[1:])
        if later > earlier
    ]
    frame_duration = _median(frame_differences) or window_seconds
    regular_intervals = _group_intervals(
        [*clipped_times, *burst_times],
        frame_duration_seconds=frame_duration,
    )
    # ``aspectralstats`` advances by half a window here, but each emitted
    # feature covers a full 2048-sample analysis window.  Report that full
    # listening span instead of the shorter hop distance.
    spectral_frame_duration = SPECTRAL_WINDOW_SIZE / sample_rate
    spectral_intervals = _group_intervals(
        [*tonal_times, *broadband_times],
        frame_duration_seconds=spectral_frame_duration,
        maximum_gap_seconds=0.07,
    )
    suspicious_intervals = _merge_intervals([
        *regular_intervals,
        *spectral_intervals,
    ])
    issues: list[str] = []
    if clipped_windows:
        issues.append("clipping")
    if burst_windows:
        issues.append("high_frequency_burst")
    if tonal_times:
        issues.append("tonal_spectral_event")
    if broadband_times:
        issues.append("broadband_spectral_event")

    if clipped_windows:
        status = "burst_or_clipping_suspected"
        risk_score = 1.0
    elif burst_windows or tonal_times or broadband_times:
        status = "sharp_suspected"
        suspicious_window_count = (
            burst_windows
            + len(tonal_times)
            + len(broadband_times)
        )
        risk_score = min(
            0.95,
            0.55 + min(0.35, suspicious_window_count / max(len(windows), 1)),
        )
    else:
        status = "normal"
        risk_score = 0.0

    return AudioQualityResult(
        path=audio_path,
        duration_seconds=duration_seconds,
        status=status,
        risk_score=round(risk_score, 3),
        suspicious_times=suspicious_times,
        suspicious_intervals=suspicious_intervals,
        issues=tuple(issues),
        windows_analyzed=len(windows),
        clipped_windows=clipped_windows,
        high_frequency_burst_windows=burst_windows,
    )


def _format_times(times: tuple[float, ...]) -> str:
    if not times:
        return "无"
    return "、".join(f"{time_seconds:.2f}s" for time_seconds in times)


def _format_intervals(intervals: tuple[tuple[float, float], ...]) -> str:
    if not intervals:
        return "无"
    return "、".join(
        f"{start:.2f}–{end:.2f}s"
        for start, end in intervals
    )


def _status_label(status: str) -> str:
    return {
        "normal": "正常",
        "sharp_suspected": "疑似尖锐声",
        "burst_or_clipping_suspected": "疑似爆鸣/削波",
    }.get(status, status)


def print_result(result: AudioQualityResult) -> None:
    print(f"文件：{result.path}")
    print(f"时长：{result.duration_seconds:.2f}s")
    print(f"结果：{_status_label(result.status)}")
    print(f"风险分：{result.risk_score:.3f}")
    print(f"异常时间段：{_format_intervals(result.suspicious_intervals)}")
    print(f"风险时间点：{_format_times(result.suspicious_times)}")
    print(f"原因：{'、'.join(result.issues) if result.issues else '未发现短时高频峰值或削波'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-quality-check",
        description="Check a WAV file for short sharp sounds and clipping.",
    )
    parser.add_argument("audio_path", help="path to a WAV file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze_audio(args.audio_path)
    except AudioQualityError as error:
        print(f"检测失败：{error}")
        return 2
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
