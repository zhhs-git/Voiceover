from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiobook_worker.audio_asset_ids import normalize_script_audio_asset_ids


def _default_stable_audio_dir() -> Path:
    """Locate the sibling Stable Audio checkout used by this workspace."""

    module_path = Path(__file__).resolve()
    candidates = (
        # .../audio book/audio book-generator/workers/python/audiobook_worker
        module_path.parents[4] / "stable-audio-3" / "optimized" / "mlx",
        Path.home() / "work" / "audio book" / "stable-audio-3" / "optimized" / "mlx",
        # Keep compatibility with the original checkout location.
        Path.home() / "work" / "stable-audio-3" / "optimized" / "mlx",
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])


DEFAULT_STABLE_AUDIO_DIR = _default_stable_audio_dir()
MANIFEST_VERSION = 2
MAX_AUDIO_DURATION_SECONDS = 120.0
MUSIC_NORMALIZATION_VERSION = 1
MUSIC_TARGET_LUFS = -18.0
MUSIC_TRUE_PEAK_DB = -2.0
MUSIC_LOUDNESS_RANGE = 7.0
MUSIC_NORMALIZATION_TIMEOUT_SECONDS = 300.0


class StableAudioError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        partial_assets: list["AudioAssetResult"] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.partial_assets = partial_assets or []


@dataclass(frozen=True)
class StableAudioConfig:
    root: Path
    executable: Path
    music_model: str = "sm-music"
    sfx_model: str = "sm-sfx"
    decoder: str = "same-s"
    cfg: float = 3.0
    timeout_seconds: float = 3600.0

    @classmethod
    def from_environment(cls) -> "StableAudioConfig":
        root = Path(
            os.environ.get("STABLE_AUDIO_DIR", str(DEFAULT_STABLE_AUDIO_DIR))
        ).expanduser()
        executable = Path(
            os.environ.get("STABLE_AUDIO_BIN", str(root / "sa3"))
        ).expanduser()
        return cls(
            root=root,
            executable=executable,
            music_model=os.environ.get("STABLE_AUDIO_MUSIC_MODEL", "sm-music"),
            sfx_model=os.environ.get("STABLE_AUDIO_SFX_MODEL", "sm-sfx"),
            decoder=os.environ.get("STABLE_AUDIO_DECODER", "same-s"),
            cfg=_float_environment("STABLE_AUDIO_CFG", 3.0),
            timeout_seconds=_float_environment(
                "STABLE_AUDIO_TIMEOUT_SECONDS", 3600.0
            ),
        )


@dataclass(frozen=True)
class AudioAssetSpec:
    asset_id: str
    kind: str
    scene_id: str
    model: str
    prompt: str
    negative_prompt: str
    duration_seconds: float
    plan_signature: str = ""
    seed: int | None = None

    @property
    def manifest_key(self) -> str:
        return f"{self.kind}:{self.asset_id}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "assetId": self.asset_id,
            "kind": self.kind,
            "sceneId": self.scene_id,
            "model": self.model,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "durationSeconds": self.duration_seconds,
            "planSignature": self.plan_signature,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class AudioAssetResult:
    asset_id: str
    kind: str
    scene_id: str
    model: str
    path: Path
    duration_seconds: float
    signature: str
    cache_hit: bool

    def to_artifact(self) -> dict[str, Any]:
        return {
            "kind": f"stable_audio_{self.kind}",
            "path": str(self.path),
            "metadata": {
                "assetId": self.asset_id,
                "sceneId": self.scene_id,
                "model": self.model,
                "durationSeconds": self.duration_seconds,
                "signature": self.signature,
                "cacheHit": self.cache_hit,
            },
        }


@dataclass(frozen=True)
class AudioAssetGenerationResult:
    assets: list[AudioAssetResult]
    warnings: list[str]
    manifest_path: Path


def _float_environment(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StableAudioError(
            "invalid_audio_plan",
            f"audio plan field {field} must be a non-empty string",
        )
    return value.strip()


def _duration(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StableAudioError(
            "invalid_audio_plan",
            f"audio plan field {field} must be a number",
        )
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= MAX_AUDIO_DURATION_SECONDS:
        raise StableAudioError(
            "invalid_audio_plan",
            f"audio plan field {field} must be between 0 and {MAX_AUDIO_DURATION_SECONDS:g}",
        )
    return result


def _safe_asset_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not stem:
        raise StableAudioError("invalid_audio_plan", "audio asset id is empty")
    if stem != value:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        stem = f"{stem[:80]}_{digest}"
    return stem


def _music_palette_anchor(raw_scene: dict[str, Any]) -> str:
    """Return the English theme anchor shared by all variants in a scene."""

    palette = raw_scene.get("musicPalette")
    if not isinstance(palette, dict):
        return ""
    for key in ("promptAnchor", "prompt_anchor", "anchor"):
        value = palette.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _music_prompt_with_anchor(raw_scene: dict[str, Any], prompt: str) -> str:
    """Bind a generated variant to its scene's exact theme anchor.

    Older plans do not have ``promptAnchor`` and pass through unchanged.  New
    plans get one shared phrase prepended at the asset boundary, so a model
    cannot accidentally make low/medium/high unrelated genres by paraphrasing
    the palette in each variant prompt.
    """

    anchor = _music_palette_anchor(raw_scene)
    if not anchor:
        return prompt
    normalized_prompt = " ".join(prompt.casefold().split())
    normalized_anchor = " ".join(anchor.casefold().split())
    if normalized_anchor in normalized_prompt:
        return prompt
    return f"{anchor.rstrip(' .;。，；')}. {prompt}"


def _music_scene_seed(scene_id: str, raw_scene: dict[str, Any]) -> int | None:
    """Derive one stable seed shared by the variants of a scene."""

    if not _music_palette_anchor(raw_scene):
        return None
    seed_source = json.dumps(
        {
            "sceneId": scene_id,
            "promptAnchor": _music_palette_anchor(raw_scene),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    # Stable Audio accepts any integer seed. Keep it positive and inside the
    # range used by the MLX CLI while retaining reproducibility across runs.
    return int(hashlib.sha256(seed_source).hexdigest()[:8], 16) % 2_147_483_646 + 1


def _collect_asset_specs(script: dict[str, Any]) -> list[AudioAssetSpec]:
    raw_plan = script.get("audioPlan") or {}
    if not isinstance(raw_plan, dict):
        raise StableAudioError("invalid_audio_plan", "audioPlan must be an object")
    # Older chapter plans can contain IDs such as ``sfx_1`` in multiple
    # scenes. Normalize before creating manifest keys so those plans remain
    # usable even when they were generated before the planner prompt fix.
    normalize_script_audio_asset_ids(script)
    raw_plan = script.get("audioPlan") or {}
    if not isinstance(raw_plan, dict):
        raise StableAudioError("invalid_audio_plan", "audioPlan must be an object")
    raw_scenes = raw_plan.get("scenes", [])
    if not isinstance(raw_scenes, list):
        raise StableAudioError("invalid_audio_plan", "audioPlan.scenes must be an array")
    plan_signature = hashlib.sha256(
        json.dumps(
            raw_plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    specs: list[AudioAssetSpec] = []
    seen_keys: set[str] = set()
    for scene_index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            raise StableAudioError(
                "invalid_audio_plan",
                f"audioPlan.scenes[{scene_index}] must be an object",
            )
        scene_id = _required_string(
            raw_scene.get("id"), f"audioPlan.scenes[{scene_index}].id"
        )

        raw_variants = raw_scene.get("musicVariants")
        if isinstance(raw_variants, list) and raw_variants:
            for variant_index, raw_variant in enumerate(raw_variants):
                if not isinstance(raw_variant, dict):
                    raise StableAudioError(
                        "invalid_audio_plan",
                        f"audioPlan.scenes[{scene_index}].musicVariants[{variant_index}] must be an object",
                    )
                variant_id = _required_string(
                    raw_variant.get("id"),
                    f"audioPlan.scenes[{scene_index}].musicVariants[{variant_index}].id",
                )
                spec = AudioAssetSpec(
                    asset_id=variant_id,
                    kind="music",
                    scene_id=scene_id,
                    model=_required_string(
                        raw_variant.get("model"),
                        f"audioPlan.scenes[{scene_index}].musicVariants[{variant_index}].model",
                    ),
                    prompt=_music_prompt_with_anchor(
                        raw_scene,
                        _required_string(
                            raw_variant.get("prompt"),
                            f"audioPlan.scenes[{scene_index}].musicVariants[{variant_index}].prompt",
                        ),
                    ),
                    negative_prompt=str(
                        raw_variant.get("negativePrompt") or ""
                    ).strip(),
                    duration_seconds=_duration(
                        raw_variant.get("durationSeconds"),
                        f"audioPlan.scenes[{scene_index}].musicVariants[{variant_index}].durationSeconds",
                    ),
                    plan_signature=plan_signature,
                    seed=_music_scene_seed(scene_id, raw_scene),
                )
                if spec.manifest_key in seen_keys:
                    raise StableAudioError(
                        "invalid_audio_plan",
                        f"duplicate audio asset: {spec.manifest_key}",
                    )
                seen_keys.add(spec.manifest_key)
                specs.append(spec)
        else:
            raw_music = raw_scene.get("music")
            if raw_music is not None:
                if not isinstance(raw_music, dict):
                    raise StableAudioError(
                        "invalid_audio_plan",
                        f"audioPlan.scenes[{scene_index}].music must be an object or null",
                    )
                spec = AudioAssetSpec(
                    asset_id=scene_id,
                    kind="music",
                    scene_id=scene_id,
                    model=_required_string(
                        raw_music.get("model"),
                        f"audioPlan.scenes[{scene_index}].music.model",
                    ),
                    prompt=_music_prompt_with_anchor(
                        raw_scene,
                        _required_string(
                            raw_music.get("prompt"),
                            f"audioPlan.scenes[{scene_index}].music.prompt",
                        ),
                    ),
                    negative_prompt=str(raw_music.get("negativePrompt") or "").strip(),
                    duration_seconds=_duration(
                        raw_music.get("durationSeconds"),
                        f"audioPlan.scenes[{scene_index}].music.durationSeconds",
                    ),
                    plan_signature=plan_signature,
                    seed=_music_scene_seed(scene_id, raw_scene),
                )
                if spec.manifest_key in seen_keys:
                    raise StableAudioError(
                        "invalid_audio_plan",
                        f"duplicate audio asset: {spec.manifest_key}",
                    )
                seen_keys.add(spec.manifest_key)
                specs.append(spec)

        raw_sfx = raw_scene.get("sfx", [])
        if not isinstance(raw_sfx, list):
            raise StableAudioError(
                "invalid_audio_plan",
                f"audioPlan.scenes[{scene_index}].sfx must be an array",
            )
        for sfx_index, raw_effect in enumerate(raw_sfx):
            if not isinstance(raw_effect, dict):
                raise StableAudioError(
                    "invalid_audio_plan",
                    f"audioPlan.scenes[{scene_index}].sfx[{sfx_index}] must be an object",
                )
            effect_id = _required_string(
                raw_effect.get("id"),
                f"audioPlan.scenes[{scene_index}].sfx[{sfx_index}].id",
            )
            spec = AudioAssetSpec(
                asset_id=effect_id,
                kind="sfx",
                scene_id=scene_id,
                model=_required_string(
                    raw_effect.get("model"),
                    f"audioPlan.scenes[{scene_index}].sfx[{sfx_index}].model",
                ),
                prompt=_required_string(
                    raw_effect.get("prompt"),
                    f"audioPlan.scenes[{scene_index}].sfx[{sfx_index}].prompt",
                ),
                negative_prompt=str(raw_effect.get("negativePrompt") or "").strip(),
                duration_seconds=_duration(
                    raw_effect.get("durationSeconds"),
                    f"audioPlan.scenes[{scene_index}].sfx[{sfx_index}].durationSeconds",
                ),
                plan_signature=plan_signature,
            )
            if spec.manifest_key in seen_keys:
                raise StableAudioError(
                    "invalid_audio_plan",
                    f"duplicate audio asset: {spec.manifest_key}",
                )
            seen_keys.add(spec.manifest_key)
            specs.append(spec)

    return specs


def _asset_signature(spec: AudioAssetSpec, config: StableAudioConfig) -> str:
    payload = {
        "manifestVersion": MANIFEST_VERSION,
        "assetId": spec.asset_id,
        "kind": spec.kind,
        "sceneId": spec.scene_id,
        "model": spec.model,
        "prompt": spec.prompt,
        "negativePrompt": spec.negative_prompt,
        "durationSeconds": spec.duration_seconds,
        "planSignature": spec.plan_signature,
        "seed": spec.seed,
        "decoder": config.decoder,
        "cfg": config.cfg if spec.negative_prompt else None,
    }
    if spec.kind == "music":
        payload.update(
            {
                "musicNormalizationVersion": MUSIC_NORMALIZATION_VERSION,
                "musicTargetLufs": MUSIC_TARGET_LUFS,
                "musicTruePeakDb": MUSIC_TRUE_PEAK_DB,
            }
        )
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_path(output_directory: Path) -> Path:
    return output_directory / "manifest.json"


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": MANIFEST_VERSION, "assets": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": MANIFEST_VERSION, "assets": {}}
    if not isinstance(value, dict) or not isinstance(value.get("assets", {}), dict):
        return {"version": MANIFEST_VERSION, "assets": {}}
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prune_audio_asset_manifest(
    output_directory: Path,
    specs: list[AudioAssetSpec],
) -> Path:
    """Remove manifest entries that are no longer in the current chapter plan.

    Keep old WAV files on disk for recoverability, but stop advertising stale
    asset IDs (for example ``scene_001`` after a plan changes to ``scene_1``).
    """
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    manifest["version"] = MANIFEST_VERSION
    manifest_assets = manifest.setdefault("assets", {})
    allowed_keys = {spec.manifest_key for spec in specs}
    stale_keys = [key for key in manifest_assets if key not in allowed_keys]
    for key in stale_keys:
        del manifest_assets[key]
    if stale_keys or not manifest_path.is_file():
        _write_manifest(manifest_path, manifest)
    return manifest_path


def _load_audio_script(script_path: Path) -> dict[str, Any]:
    """Read a chapter script and persist any legacy asset-ID repair."""

    try:
        script = json.loads(script_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StableAudioError("script_not_found", str(error)) from error
    except (OSError, json.JSONDecodeError) as error:
        raise StableAudioError("invalid_script", str(error)) from error
    if not isinstance(script, dict):
        raise StableAudioError("invalid_script", "chapter script must be an object")

    changes = normalize_script_audio_asset_ids(script)
    if changes:
        try:
            script_path.write_text(
                json.dumps(script, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise StableAudioError(
                "script_write_failed",
                f"Unable to persist normalized audio asset IDs: {error}",
            ) from error
    return script


def collect_audio_asset_specs(
    script_path: Path,
    *,
    asset_id: str | None = None,
    asset_kind: str | None = None,
) -> list[AudioAssetSpec]:
    """Read and validate the Stable Audio assets declared by a chapter script.

    The Gradio handoff uses the same normalized plan as the CLI generator.  Keeping
    this in one place prevents the browser workflow and the legacy command-line
    workflow from disagreeing about asset IDs, models, or output filenames.
    """
    script = _load_audio_script(script_path)

    all_specs = _collect_asset_specs(script)
    specs = all_specs
    if asset_id is not None or asset_kind is not None:
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise StableAudioError(
                "invalid_audio_asset_selection",
                "assetId is required when selecting one audio asset",
            )
        if asset_kind not in {"music", "sfx"}:
            raise StableAudioError(
                "invalid_audio_asset_selection",
                "assetKind must be music or sfx when selecting one audio asset",
            )
        specs = [
            spec
            for spec in specs
            if spec.asset_id == asset_id.strip() and spec.kind == asset_kind
        ]
        if not specs:
            raise StableAudioError(
                "audio_asset_not_found",
                f"Audio asset not found: {asset_kind}:{asset_id}",
                details={"assetId": asset_id, "assetKind": asset_kind},
            )
    return specs


def asset_output_path(output_directory: Path, spec: AudioAssetSpec) -> Path:
    """Return the stable destination used by both CLI and Gradio generation."""
    subdirectory = output_directory / ("music" if spec.kind == "music" else "sfx")
    return subdirectory / f"{_safe_asset_stem(spec.asset_id)}.wav"


def asset_signature(
    spec: AudioAssetSpec,
    config: StableAudioConfig | None = None,
) -> str:
    return _asset_signature(spec, config or StableAudioConfig.from_environment())


def cached_audio_asset(
    output_directory: Path,
    spec: AudioAssetSpec,
    *,
    config: StableAudioConfig | None = None,
) -> AudioAssetResult | None:
    """Return a valid manifest-backed asset, or None when it must be generated."""
    config = config or StableAudioConfig.from_environment()
    output_path = asset_output_path(output_directory, spec)
    signature = _asset_signature(spec, config)
    manifest = _read_manifest(_manifest_path(output_directory))
    previous = manifest.get("assets", {}).get(spec.manifest_key)
    if not (
        isinstance(previous, dict)
        and previous.get("signature") == signature
        and _is_readable_wav(output_path)
    ):
        return None
    return AudioAssetResult(
        asset_id=spec.asset_id,
        kind=spec.kind,
        scene_id=spec.scene_id,
        model=spec.model,
        path=output_path,
        duration_seconds=_wav_duration(output_path),
        signature=signature,
        cache_hit=True,
    )


def import_generated_audio_asset(
    output_directory: Path,
    spec: AudioAssetSpec,
    source_path: Path,
    *,
    config: StableAudioConfig | None = None,
) -> AudioAssetResult:
    """Import a WAV generated by the Gradio UI and update the asset manifest."""
    if not _is_readable_wav(source_path):
        raise StableAudioError(
            "invalid_stable_audio_output",
            f"Stable Audio did not create a readable WAV for {spec.asset_id}",
            details={"assetId": spec.asset_id, "path": str(source_path)},
        )
    config = config or StableAudioConfig.from_environment()
    output_path = asset_output_path(output_directory, spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    if spec.kind == "music":
        try:
            _normalize_music_asset(output_path)
        except StableAudioError:
            output_path.unlink(missing_ok=True)
            raise

    duration_seconds = _wav_duration(output_path)
    signature = _asset_signature(spec, config)
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    manifest["version"] = MANIFEST_VERSION
    manifest_assets = manifest.setdefault("assets", {})

    manifest_assets[spec.manifest_key] = {
        "assetId": spec.asset_id,
        "kind": spec.kind,
        "sceneId": spec.scene_id,
        "model": spec.model,
        "path": str(output_path),
        "durationSeconds": duration_seconds,
        "signature": signature,
    }
    _write_manifest(manifest_path, manifest)
    return AudioAssetResult(
        asset_id=spec.asset_id,
        kind=spec.kind,
        scene_id=spec.scene_id,
        model=spec.model,
        path=output_path,
        duration_seconds=duration_seconds,
        signature=signature,
        cache_hit=False,
    )


def quarantine_audio_asset(
    output_directory: Path,
    spec: AudioAssetSpec,
) -> Path | None:
    """Keep a rejected generated WAV out of the manifest and final mix.

    The file is moved into a chapter-local ``rejected`` directory rather than
    deleted, so an unexpected quality-detector result remains recoverable for
    diagnosis.  Since the manifest entry is removed first, the mixer will
    never pick up a quarantined asset.
    """

    output_path = asset_output_path(output_directory, spec)
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    manifest["version"] = MANIFEST_VERSION
    manifest_assets = manifest.setdefault("assets", {})
    manifest_assets.pop(spec.manifest_key, None)
    _write_manifest(manifest_path, manifest)

    if not output_path.is_file():
        return None
    rejected_directory = output_directory / "rejected" / spec.kind
    rejected_directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = rejected_directory / (
        f"{_safe_asset_stem(spec.asset_id)}-{stamp}{output_path.suffix}"
    )
    suffix = 1
    while destination.exists():
        destination = rejected_directory / (
            f"{_safe_asset_stem(spec.asset_id)}-{stamp}-{suffix}{output_path.suffix}"
        )
        suffix += 1
    output_path.replace(destination)
    return destination


def _is_readable_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except (OSError, wave.Error, ZeroDivisionError):
        return False


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def _normalize_music_asset(path: Path) -> None:
    """Normalize one generated music asset before it enters the mix bus.

    Stable Audio outputs can have materially different integrated loudness even
    when their prompts look similar.  Normalize the finite source WAV once,
    before it is looped or cross-faded, so the mixer's artistic fader remains an
    artistic control rather than compensating for source-level variation.
    """

    executable = shutil.which("ffmpeg")
    if not executable:
        raise StableAudioError(
            "audio_normalization_unavailable",
            "ffmpeg is required to normalize Stable Audio music assets",
        )

    temporary_path = path.with_name(f".{path.stem}.normalized.part{path.suffix}")
    filter_expression = (
        f"loudnorm=I={MUSIC_TARGET_LUFS:g}:TP={MUSIC_TRUE_PEAK_DB:g}:"
        f"LRA={MUSIC_LOUDNESS_RANGE:g}"
    )
    command = [
        executable,
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
        "44100",
        "-ac",
        "2",
        str(temporary_path),
    ]
    try:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=MUSIC_NORMALIZATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise StableAudioError(
                "audio_normalization_timeout",
                f"Timed out while normalizing music asset: {path.name}",
            ) from error
        except OSError as error:
            raise StableAudioError(
                "audio_normalization_failed",
                f"Unable to run ffmpeg for music asset: {error}",
            ) from error

        if completed.returncode != 0 or not _is_readable_wav(temporary_path):
            detail = _failure_detail(completed)
            raise StableAudioError(
                "audio_normalization_failed",
                f"Failed to normalize music asset {path.name}: {detail}",
                details={"path": str(path), "returnCode": completed.returncode},
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _format_seconds(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _command_for(
    spec: AudioAssetSpec,
    output_path: Path,
    config: StableAudioConfig,
) -> list[str]:
    expected_model = config.music_model if spec.kind == "music" else config.sfx_model
    if spec.model != expected_model:
        raise StableAudioError(
            "invalid_audio_plan",
            f"{spec.kind} asset {spec.asset_id} must use {expected_model}, got {spec.model}",
        )
    command = [
        str(config.executable),
        "--prompt",
        spec.prompt,
        "--dit",
        spec.model,
        "--decoder",
        config.decoder,
        "--seconds",
        _format_seconds(spec.duration_seconds),
        "--out",
        str(output_path.resolve()),
    ]
    if spec.negative_prompt:
        command.extend(["--cfg", _format_seconds(config.cfg), "--negative-prompt", spec.negative_prompt])
    if spec.seed is not None:
        command.extend(["--seed", str(spec.seed)])
    return command


def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return f"Stable Audio exited with code {completed.returncode}"
    return output.splitlines()[-1]


def generate_audio_assets(
    script_path: Path,
    output_directory: Path,
    *,
    force: bool = False,
    asset_id: str | None = None,
    asset_kind: str | None = None,
    config: StableAudioConfig | None = None,
) -> AudioAssetGenerationResult:
    config = config or StableAudioConfig.from_environment()
    script = _load_audio_script(script_path)

    all_specs = _collect_asset_specs(script)
    specs = all_specs
    if asset_id is not None or asset_kind is not None:
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise StableAudioError(
                "invalid_audio_asset_selection",
                "assetId is required when selecting one audio asset",
            )
        if asset_kind not in {"music", "sfx"}:
            raise StableAudioError(
                "invalid_audio_asset_selection",
                "assetKind must be music or sfx when selecting one audio asset",
            )
        specs = [
            spec
            for spec in specs
            if spec.asset_id == asset_id.strip() and spec.kind == asset_kind
        ]
        if not specs:
            raise StableAudioError(
                "audio_asset_not_found",
                f"Audio asset not found: {asset_kind}:{asset_id}",
                details={"assetId": asset_id, "assetKind": asset_kind},
            )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    manifest["version"] = MANIFEST_VERSION
    manifest_assets = manifest.setdefault("assets", {})

    # A selected-asset regeneration must preserve other assets from the
    # current plan, while removing entries left by an older plan.
    allowed_keys = {spec.manifest_key for spec in all_specs}
    for key in list(manifest_assets):
        if key not in allowed_keys:
            del manifest_assets[key]

    if not specs:
        _write_manifest(manifest_path, manifest)
        return AudioAssetGenerationResult(
            assets=[],
            warnings=["no_audio_assets"],
            manifest_path=manifest_path,
        )

    if not config.executable.is_file():
        raise StableAudioError(
            "stable_audio_unavailable",
            f"Stable Audio executable not found: {config.executable}",
            details={"executable": str(config.executable)},
        )

    results: list[AudioAssetResult] = []
    for spec in specs:
        signature = _asset_signature(spec, config)
        subdirectory = output_directory / ("music" if spec.kind == "music" else "sfx")
        output_path = subdirectory / f"{_safe_asset_stem(spec.asset_id)}.wav"
        previous = manifest_assets.get(spec.manifest_key)
        cache_hit = (
            not force
            and isinstance(previous, dict)
            and previous.get("signature") == signature
            and _is_readable_wav(output_path)
        )

        if not cache_hit:
            subdirectory.mkdir(parents=True, exist_ok=True)
            output_path.unlink(missing_ok=True)
            command = _command_for(spec, output_path, config)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(config.root),
                    capture_output=True,
                    text=True,
                    timeout=config.timeout_seconds,
                    check=False,
                )
            except FileNotFoundError as error:
                raise StableAudioError(
                    "stable_audio_unavailable",
                    f"Unable to execute Stable Audio: {error}",
                    partial_assets=results,
                ) from error
            except OSError as error:
                raise StableAudioError(
                    "stable_audio_unavailable",
                    f"Unable to execute Stable Audio: {error}",
                    partial_assets=results,
                ) from error
            except subprocess.TimeoutExpired as error:
                raise StableAudioError(
                    "stable_audio_timeout",
                    f"Stable Audio timed out while generating {spec.asset_id}",
                    details={"assetId": spec.asset_id},
                    partial_assets=results,
                ) from error
            if completed.returncode != 0:
                raise StableAudioError(
                    "stable_audio_generation_failed",
                    f"Failed to generate {spec.asset_id}: {_failure_detail(completed)}",
                    details={
                        "assetId": spec.asset_id,
                        "returnCode": completed.returncode,
                    },
                    partial_assets=results,
                )
            if not _is_readable_wav(output_path):
                raise StableAudioError(
                    "invalid_stable_audio_output",
                    f"Stable Audio did not create a readable WAV for {spec.asset_id}",
                    details={"assetId": spec.asset_id, "path": str(output_path)},
                    partial_assets=results,
                )
            if spec.kind == "music":
                try:
                    _normalize_music_asset(output_path)
                except StableAudioError as error:
                    output_path.unlink(missing_ok=True)
                    raise StableAudioError(
                        error.code,
                        str(error),
                        details=error.details,
                        partial_assets=results,
                    ) from error

        actual_duration = _wav_duration(output_path)
        result = AudioAssetResult(
            asset_id=spec.asset_id,
            kind=spec.kind,
            scene_id=spec.scene_id,
            model=spec.model,
            path=output_path,
            duration_seconds=actual_duration,
            signature=signature,
            cache_hit=cache_hit,
        )
        results.append(result)
        manifest_assets[spec.manifest_key] = {
            "assetId": spec.asset_id,
            "kind": spec.kind,
            "sceneId": spec.scene_id,
            "model": spec.model,
            "path": str(output_path),
            "durationSeconds": actual_duration,
            "signature": signature,
        }
        _write_manifest(manifest_path, manifest)

    return AudioAssetGenerationResult(
        assets=results,
        warnings=[],
        manifest_path=manifest_path,
    )
