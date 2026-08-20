import json
import wave
from pathlib import Path

from audiobook_worker import voxcpm2_runner


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
