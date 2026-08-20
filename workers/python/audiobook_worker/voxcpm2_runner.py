"""Run one local VoxCPM2 chapter request inside the model's isolated venv.

The regular audiobook worker intentionally does not import :mod:`voxcpm`: its
runtime and dependency set are separate from the Python 3.11 environment used
for VoxCPM2.  This file is invoked by that isolated interpreter and exchanges
only JSON files with the main worker.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import sys
import time
import wave
from pathlib import Path
from typing import Any


PROMPT_FORMAT_VERSION = 2
PROFILE_VERSION = PROMPT_FORMAT_VERSION


class VoxCPM2RunnerError(RuntimeError):
    """A request cannot be completed by the isolated VoxCPM2 runner."""


def _as_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VoxCPM2RunnerError(f"{name} must be an object.")
    return value


def _as_list(value: object, *, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise VoxCPM2RunnerError(f"{name} must be an array.")
    return [_as_object(item, name=name) for item in value]


def _required_text(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VoxCPM2RunnerError(f"{name} is required.")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _as_object(json.loads(path.read_text(encoding="utf-8")), name="runner request")
    except (OSError, json.JSONDecodeError) as error:
        raise VoxCPM2RunnerError(f"Unable to read runner request: {error}") from error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _is_readable_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except (OSError, wave.Error, ZeroDivisionError):
        return False


def _profile_is_usable(
    profile_path: Path,
    metadata_path: Path,
    *,
    signature: str,
) -> bool:
    if not _is_readable_wav(profile_path):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and metadata.get("version") == PROFILE_VERSION
        and metadata.get("promptFormatVersion") == PROMPT_FORMAT_VERSION
        and metadata.get("signature") == signature
    )


def _seed_from(value: str) -> None:
    """Keep a newly created voice reference reproducible before it is cached."""

    seed = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    random.seed(seed)
    try:
        import numpy as np
        import torch

        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Determinism is a cache-quality improvement, not a reason to reject a
        # valid synthesis when an optional backend does not expose a seed API.
        pass


def _atomic_write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        raise VoxCPM2RunnerError("VoxCPM2 returned an empty waveform.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix or '.wav'}")
    try:
        # The main worker deliberately uses the standard-library ``wave``
        # module for cache and quality checks, so persist plain PCM WAV rather
        # than an implementation-specific floating point subtype.
        sf.write(str(temporary), waveform, sample_rate, subtype="PCM_16")
        if not _is_readable_wav(temporary):
            raise VoxCPM2RunnerError("VoxCPM2 wrote an unreadable WAV file.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _controlled_text(instruction: str, text: str) -> str:
    normalized_instruction = " ".join(instruction.split())
    return f"({normalized_instruction}){text}" if normalized_instruction else text


def _prompt_format_version(value: object, *, name: str) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as error:
        raise VoxCPM2RunnerError(f"{name} must be {PROMPT_FORMAT_VERSION}.") from error
    if version != PROMPT_FORMAT_VERSION:
        raise VoxCPM2RunnerError(
            f"{name} must be {PROMPT_FORMAT_VERSION}, got {version}."
        )
    return version


def _ensure_profile(model: Any, item: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    voice_id = _required_text(item.get("voiceId"), name="profile voiceId")
    profile_path = Path(_required_text(item.get("profilePath"), name="profilePath"))
    metadata_path = Path(_required_text(item.get("metadataPath"), name="metadataPath"))
    lock_path = Path(_required_text(item.get("lockPath"), name="lockPath"))
    signature = _required_text(item.get("signature"), name="profile signature")
    voice_design = _required_text(item.get("voiceDesign"), name="voiceDesign")
    profile_control = _required_text(
        item.get("profileControl"),
        name="profileControl",
    )
    reference_text = _required_text(item.get("referenceText"), name="referenceText")
    language = _required_text(item.get("language"), name="profile language")
    prompt_format_version = _prompt_format_version(
        item.get("promptFormatVersion"),
        name="profile promptFormatVersion",
    )

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if _profile_is_usable(profile_path, metadata_path, signature=signature):
                return {"voiceId": voice_id, "path": str(profile_path), "cacheHit": True}

            _seed_from(signature)
            waveform = model.generate(
                text=_controlled_text(profile_control, reference_text),
                cfg_value=2.0,
                inference_timesteps=10,
                max_len=4096,
            )
            _atomic_write_wav(profile_path, waveform, sample_rate)
            _write_json(
                metadata_path,
                {
                    "version": PROFILE_VERSION,
                    "promptFormatVersion": prompt_format_version,
                    "signature": signature,
                    "voiceId": voice_id,
                    "voiceDesign": voice_design,
                    "profileControl": profile_control,
                    "referenceText": reference_text,
                    "language": language,
                    "backend": "voxcpm2",
                    "modelId": "VoxCPM2",
                },
            )
            return {"voiceId": voice_id, "path": str(profile_path), "cacheHit": False}
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _synthesize_segment(model: Any, item: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    segment_id = _required_text(item.get("id"), name="segment id")
    text = _required_text(item.get("text"), name=f"segment text for {segment_id}")
    output_path = Path(_required_text(item.get("outputPath"), name="segment outputPath"))
    reference_path = Path(_required_text(item.get("referenceWavPath"), name="referenceWavPath"))
    if not _is_readable_wav(reference_path):
        raise VoxCPM2RunnerError(
            f"VoxCPM2 reference WAV is unavailable for segment {segment_id}: {reference_path}"
        )
    delivery = str(item.get("delivery") or "").strip()
    _prompt_format_version(
        item.get("promptFormatVersion"),
        name=f"segment promptFormatVersion for {segment_id}",
    )
    _required_text(item.get("language"), name=f"segment language for {segment_id}")
    _seed_from(f"{segment_id}:{text}:{delivery}:{reference_path}")
    waveform = model.generate(
        text=_controlled_text(delivery, text),
        reference_wav_path=str(reference_path),
        cfg_value=2.0,
        inference_timesteps=10,
        max_len=4096,
    )
    _atomic_write_wav(output_path, waveform, sample_rate)
    with wave.open(str(output_path), "rb") as wav_file:
        duration = wav_file.getnframes() / wav_file.getframerate()
    return {
        "id": segment_id,
        "path": str(output_path),
        "durationSeconds": duration,
        "sampleRate": sample_rate,
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    _prompt_format_version(
        payload.get("promptFormatVersion"),
        name="request promptFormatVersion",
    )
    model_path = Path(_required_text(payload.get("modelPath"), name="modelPath"))
    if not model_path.is_dir():
        raise VoxCPM2RunnerError(f"VoxCPM2 model directory is missing: {model_path}")
    device = str(payload.get("device") or "auto").strip().lower()
    if device not in {"auto", "mps", "cpu"}:
        raise VoxCPM2RunnerError(f"Unsupported VoxCPM2 device: {device}")
    profiles = _as_list(payload.get("profiles"), name="profiles")
    segments = _as_list(payload.get("segments"), name="segments")
    if not profiles and not segments:
        return {"status": "succeeded", "modelLoads": 0, "profiles": [], "segments": []}

    # This import must stay inside the isolated interpreter.  The primary
    # audiobook worker intentionally has no VoxCPM dependency.
    from voxcpm import VoxCPM

    started_at = time.monotonic()
    model = VoxCPM.from_pretrained(
        str(model_path),
        load_denoiser=False,
        optimize=False,
        device=device,
    )
    sample_rate = int(model.tts_model.sample_rate)
    if sample_rate <= 0:
        raise VoxCPM2RunnerError("VoxCPM2 reported an invalid sample rate.")
    profile_results = [_ensure_profile(model, item, sample_rate) for item in profiles]
    segment_results = [_synthesize_segment(model, item, sample_rate) for item in segments]
    return {
        "status": "succeeded",
        "modelLoads": 1,
        "device": str(model.tts_model.device),
        "sampleRate": sample_rate,
        "loadAndGenerationSeconds": time.monotonic() - started_at,
        "profiles": profile_results,
        "segments": segment_results,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    if len(arguments) != 2:
        print("usage: voxcpm2_runner.py INPUT_JSON OUTPUT_JSON", file=sys.stderr)
        return 2
    output_path = Path(arguments[1])
    try:
        result = run(_read_json(Path(arguments[0])))
    except Exception as error:
        _write_json(
            output_path,
            {
                "status": "failed",
                "error": {
                    "code": "voxcpm2_runner_failed",
                    "message": str(error),
                },
            },
        )
        return 1
    _write_json(output_path, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
