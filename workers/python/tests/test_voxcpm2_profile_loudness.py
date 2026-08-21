import subprocess
import wave
from pathlib import Path

import pytest

from audiobook_worker import voxcpm2_profile_loudness as loudness


def _write_wav(
    path: Path,
    *,
    sample_rate: int = 24_000,
    channels: int = 1,
    frame: bytes = b"\x00\x08",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams((channels, 2, sample_rate, 240, "NONE", "not compressed"))
        wav_file.writeframes(frame * 240 * channels)


def test_profile_normalizer_uses_the_fixed_target_and_preserves_pcm_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    profile_path = tmp_path / "profile.wav"
    _write_wav(profile_path, sample_rate=48_000, channels=2, frame=b"\x00\x01")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs):
        commands.append(command)
        _write_wav(
            Path(command[-1]),
            sample_rate=48_000,
            channels=2,
            frame=b"\x00\x02",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(loudness.shutil, "which", lambda _name: "/mock/ffmpeg")
    monkeypatch.setattr(loudness.subprocess, "run", fake_run)

    loudness.normalize_voxcpm2_profile_wav(profile_path)

    command = commands[0]
    assert "loudnorm=I=-20:TP=-3:LRA=7" in command
    assert command[command.index("-c:a") + 1] == "pcm_s16le"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"
    with wave.open(str(profile_path), "rb") as wav_file:
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnchannels() == 2
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
    assert not list(tmp_path.glob(".*.loudnorm.*.part.wav"))


def test_profile_normalizer_keeps_candidate_when_ffmpeg_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    profile_path = tmp_path / "candidate.wav"
    _write_wav(profile_path, frame=b"\x00\x07")
    before = profile_path.read_bytes()
    monkeypatch.setattr(loudness.shutil, "which", lambda _name: "/mock/ffmpeg")
    monkeypatch.setattr(
        loudness.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            1,
            "",
            "synthetic ffmpeg failure",
        ),
    )

    with pytest.raises(loudness.VoxCPM2ProfileLoudnessError, match="synthetic ffmpeg failure"):
        loudness.normalize_voxcpm2_profile_wav(profile_path)

    assert profile_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.loudnorm.*.part.wav"))
