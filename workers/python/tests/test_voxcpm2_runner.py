import json
import wave
from pathlib import Path

import pytest

from audiobook_worker import voxcpm2_runner
from audiobook_worker.voxcpm2_profile_loudness import (
    VoxCPM2ProfileLoudnessError,
    voxcpm2_profile_loudness,
)


def _write_wav(path: Path, sample_rate: int = 24_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setparams((1, 2, sample_rate, 240, "NONE", "not compressed"))
        wav_file.writeframes(b"\x00\x08" * 240)


def test_runner_keeps_profile_and_segment_controls_separate(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[dict[str, object]] = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return [0.0, 0.1, 0.0]

    def fake_atomic_write(path: Path, _audio, sample_rate: int) -> None:
        _write_wav(path, sample_rate)

    monkeypatch.setattr(voxcpm2_runner, "_atomic_write_wav", fake_atomic_write)
    monkeypatch.setattr(
        voxcpm2_runner,
        "normalize_voxcpm2_profile_wav",
        lambda _path: None,
    )
    profile_path = tmp_path / "profiles" / "guard_zh.wav"
    metadata_path = profile_path.with_suffix(".json")
    signature = "profile-signature"
    profile_item = {
        "voiceId": "guard",
        "profilePath": str(profile_path),
        "metadataPath": str(metadata_path),
        "lockPath": str(profile_path.with_suffix(".lock")),
        "signature": signature,
        "voiceDesign": "成年男性，低沉而清晰。",
        "profileControl": "adult male voice, low and clear",
        "referenceText": "The morning light falls softly across the quiet room.",
        "language": "en",
        "promptFormatVersion": 2,
        "profileLoudness": voxcpm2_profile_loudness(),
    }

    first = voxcpm2_runner._ensure_profile(FakeModel(), profile_item, 24_000)
    second = voxcpm2_runner._ensure_profile(FakeModel(), profile_item, 24_000)

    assert first["cacheHit"] is False
    assert second["cacheHit"] is True
    assert calls[0]["text"] == "(adult male voice, low and clear)The morning light falls softly across the quiet room."
    assert calls[0]["cfg_value"] == 2.0
    assert calls[0]["inference_timesteps"] == 10
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["promptFormatVersion"] == 2
    assert metadata["profileLoudness"] == voxcpm2_profile_loudness()
    assert metadata["voiceDesign"] == profile_item["voiceDesign"]
    assert metadata["profileControl"] == profile_item["profileControl"]

    reference_path = tmp_path / "reference.wav"
    _write_wav(reference_path)
    segment_path = tmp_path / "segments" / "seg_0001.wav"
    voxcpm2_runner._synthesize_segment(
        FakeModel(),
        {
            "id": "seg_0001",
            "text": "The door opened.",
            "delivery": "natural and restrained, quick but clear",
            "language": "en",
            "promptFormatVersion": 2,
            "referenceWavPath": str(reference_path),
            "outputPath": str(segment_path),
        },
        24_000,
    )
    assert calls[1]["text"] == "(natural and restrained, quick but clear)The door opened."
    assert calls[1]["reference_wav_path"] == str(reference_path)


def test_runner_keeps_accepted_profile_when_loudness_normalization_fails(
    tmp_path: Path,
    monkeypatch,
):
    class FakeModel:
        def generate(self, **_kwargs):
            return [0.0, 0.1, 0.0]

    def fake_atomic_write(path: Path, _audio, sample_rate: int) -> None:
        _write_wav(path, sample_rate)

    profile_path = tmp_path / "profiles" / "guard_zh.wav"
    metadata_path = profile_path.with_suffix(".json")
    _write_wav(profile_path)
    metadata_path.write_text('{"signature":"old-profile"}', encoding="utf-8")
    accepted_wav = profile_path.read_bytes()
    accepted_metadata = metadata_path.read_bytes()
    monkeypatch.setattr(voxcpm2_runner, "_atomic_write_wav", fake_atomic_write)

    def reject_normalization(_path: Path) -> None:
        raise VoxCPM2ProfileLoudnessError("ffmpeg test failure")

    monkeypatch.setattr(
        voxcpm2_runner,
        "normalize_voxcpm2_profile_wav",
        reject_normalization,
    )
    profile_item = {
        "voiceId": "guard",
        "profilePath": str(profile_path),
        "metadataPath": str(metadata_path),
        "lockPath": str(profile_path.with_suffix(".lock")),
        "signature": "new-profile",
        "voiceDesign": "成年男性，低沉而清晰。",
        "profileControl": "adult male voice, low and clear",
        "referenceText": "The morning light falls softly across the quiet room.",
        "language": "en",
        "promptFormatVersion": 2,
        "profileLoudness": voxcpm2_profile_loudness(),
    }

    with pytest.raises(
        voxcpm2_runner.VoxCPM2RunnerError,
        match="profile loudness normalization failed",
    ):
        voxcpm2_runner._ensure_profile(FakeModel(), profile_item, 24_000)

    assert profile_path.read_bytes() == accepted_wav
    assert metadata_path.read_bytes() == accepted_metadata
    assert not list(profile_path.parent.glob(".*.candidate.*.wav"))


def test_runner_rolls_back_profile_when_sidecar_commit_fails(
    tmp_path: Path,
    monkeypatch,
):
    profile_path = tmp_path / "profiles" / "guard_zh.wav"
    metadata_path = profile_path.with_suffix(".json")
    candidate_path = tmp_path / "profiles" / ".guard_zh.candidate.wav"
    metadata_candidate_path = tmp_path / "profiles" / ".guard_zh.candidate.json"
    _write_wav(profile_path)
    metadata_path.write_bytes(b'{"signature":"old-profile"}')
    _write_wav(candidate_path)
    metadata_candidate_path.write_bytes(b'{"signature":"new-profile"}')
    accepted_wav = profile_path.read_bytes()
    accepted_metadata = metadata_path.read_bytes()
    original_replace = Path.replace

    def fail_metadata_replace(self: Path, target: Path):
        if self == metadata_candidate_path:
            raise OSError("sidecar commit test failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)

    with pytest.raises(OSError, match="sidecar commit test failure"):
        voxcpm2_runner._atomic_replace_profile_and_metadata(
            candidate_path,
            profile_path,
            metadata_candidate_path,
            metadata_path,
        )

    assert profile_path.read_bytes() == accepted_wav
    assert metadata_path.read_bytes() == accepted_metadata
    assert not list(profile_path.parent.glob(".*.backup.*"))
    assert not candidate_path.is_file()
    assert metadata_candidate_path.is_file()
