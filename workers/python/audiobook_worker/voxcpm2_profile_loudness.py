"""Loudness contract for cached VoxCPM2 reference-profile WAVs.

The isolated VoxCPM2 runner imports this module directly from its own virtual
environment, so this file deliberately depends on the Python standard library
only.  Segment delivery is intentionally outside this contract: normalizing a
role's reference once retains the model's per-segment emotion and pace.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from pathlib import Path


VOXCPM2_PROFILE_LOUDNESS_VERSION = 1
VOXCPM2_PROFILE_TARGET_LUFS = -20.0
VOXCPM2_PROFILE_TRUE_PEAK_DB = -3.0
VOXCPM2_PROFILE_LOUDNESS_RANGE = 7.0
VOXCPM2_PROFILE_NORMALIZATION_TIMEOUT_SECONDS = 60


class VoxCPM2ProfileLoudnessError(RuntimeError):
    """A reference profile could not be safely normalized."""


def voxcpm2_profile_loudness() -> dict[str, int | float]:
    """Return a fresh, versioned mapping for local profile/cache contracts."""

    return {
        "version": VOXCPM2_PROFILE_LOUDNESS_VERSION,
        "integratedLufs": VOXCPM2_PROFILE_TARGET_LUFS,
        "truePeakDb": VOXCPM2_PROFILE_TRUE_PEAK_DB,
        "loudnessRange": VOXCPM2_PROFILE_LOUDNESS_RANGE,
    }


def profile_loudness_is_current(value: object) -> bool:
    """Return whether persisted/requested loudness data matches this contract."""

    if not isinstance(value, dict):
        return False
    expected = voxcpm2_profile_loudness()
    if set(value) != set(expected) or type(value.get("version")) is not int:
        return False
    if value["version"] != expected["version"]:
        return False
    for key in ("integratedLufs", "truePeakDb", "loudnessRange"):
        actual = value.get(key)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        if float(actual) != expected[key]:
            return False
    return True


def _wav_shape(path: Path) -> tuple[int, int]:
    """Return a readable WAV's sample rate and channel count."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frames = wav_file.getnframes()
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise VoxCPM2ProfileLoudnessError(
            f"VoxCPM2 profile WAV is unreadable: {path.name}"
        ) from error
    if sample_rate <= 0 or channels <= 0 or frames <= 0:
        raise VoxCPM2ProfileLoudnessError(
            f"VoxCPM2 profile WAV is empty or invalid: {path.name}"
        )
    return sample_rate, channels


def _validate_normalized_wav(
    path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    """Confirm ffmpeg produced the worker's portable PCM S16 WAV contract."""

    try:
        with wave.open(str(path), "rb") as wav_file:
            is_expected_shape = (
                wav_file.getframerate() == sample_rate
                and wav_file.getnchannels() == channels
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
                and wav_file.getnframes() > 0
            )
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise VoxCPM2ProfileLoudnessError(
            f"ffmpeg produced an unreadable normalized VoxCPM2 profile: {path.name}"
        ) from error
    if not is_expected_shape:
        raise VoxCPM2ProfileLoudnessError(
            f"ffmpeg changed the expected PCM WAV shape for VoxCPM2 profile: {path.name}"
        )


def _temporary_output_path(path: Path) -> Path:
    suffix = path.suffix or ".wav"
    return path.with_name(
        f".{path.stem}.loudnorm.{os.getpid()}.{time.time_ns()}.part{suffix}"
    )


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()
    return detail.splitlines()[-1] if detail else "no ffmpeg error detail"


def normalize_voxcpm2_profile_wav(
    path: Path,
    *,
    ffmpeg_path: str | None = None,
) -> None:
    """Normalize a generated profile in place only after ffmpeg succeeds.

    ``path`` is a disposable runner candidate, never an accepted profile.  A
    sibling output is validated before atomically replacing that candidate.
    The runner then atomically promotes the candidate to its accepted cache
    path under the existing profile lock.
    """

    sample_rate, channels = _wav_shape(path)
    executable = ffmpeg_path or shutil.which("ffmpeg")
    if not executable:
        raise VoxCPM2ProfileLoudnessError(
            "ffmpeg is required to normalize VoxCPM2 reference profiles."
        )
    temporary_path = _temporary_output_path(path)
    filter_expression = (
        f"loudnorm=I={VOXCPM2_PROFILE_TARGET_LUFS:g}:"
        f"TP={VOXCPM2_PROFILE_TRUE_PEAK_DB:g}:"
        f"LRA={VOXCPM2_PROFILE_LOUDNESS_RANGE:g}"
    )
    command = [
        str(executable),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-af",
        filter_expression,
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(temporary_path),
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=VOXCPM2_PROFILE_NORMALIZATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise VoxCPM2ProfileLoudnessError(
                f"Timed out while normalizing VoxCPM2 profile: {path.name}"
            ) from error
        except OSError as error:
            raise VoxCPM2ProfileLoudnessError(
                f"Unable to run ffmpeg for VoxCPM2 profile {path.name}: {error}"
            ) from error
        if completed.returncode != 0:
            raise VoxCPM2ProfileLoudnessError(
                f"ffmpeg failed to normalize VoxCPM2 profile {path.name}: "
                f"{_failure_detail(completed)}"
            )
        _validate_normalized_wav(
            temporary_path,
            sample_rate=sample_rate,
            channels=channels,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
