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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from audiobook_worker.audio_asset_ids import normalize_script_audio_asset_ids
from audiobook_worker.audio_quality import (
    AUDIO_QUALITY_DETECTOR_VERSION,
    AudioQualityResult,
    analyze_audio,
    repair_short_suspicious_intervals,
)
from audiobook_worker.llm import normalize_serialized_audio_plan_music_coverage


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
MANIFEST_VERSION = 3
MAX_AUDIO_DURATION_SECONDS = 120.0
MUSIC_NORMALIZATION_VERSION = 1
MUSIC_TARGET_LUFS = -18.0
MUSIC_TRUE_PEAK_DB = -2.0
MUSIC_LOUDNESS_RANGE = 7.0
MUSIC_NORMALIZATION_TIMEOUT_SECONDS = 300.0
QUALITY_REPORT_VERSION = 2
QUALITY_REGENERATION_ATTEMPTS = 2
FALLBACK_MUSIC_ASSET_ID = "chapter_fallback_music"


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
    quality_enabled: bool = True

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
            quality_enabled=_bool_environment(
                "AUDIOBOOK_AUDIO_ASSET_QUALITY_ENABLED", True
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
    quality: dict[str, Any] | None = None

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
                "quality": self.quality,
            },
        }


@dataclass(frozen=True)
class AudioAssetGenerationResult:
    assets: list[AudioAssetResult]
    warnings: list[str]
    manifest_path: Path


@dataclass(frozen=True)
class _AssetQualityOutcome:
    accepted: bool
    status: str
    report_path: Path
    quality: dict[str, Any] | None
    reason: str | None = None
    rejected_paths: tuple[Path, ...] = ()


def _float_environment(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _bool_environment(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


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
    temporary_path = path.with_name(f".{path.name}.part")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _quality_report_path(output_directory: Path, spec: AudioAssetSpec) -> Path:
    return (
        output_directory
        / "quality"
        / spec.kind
        / f"{_safe_asset_stem(spec.asset_id)}.json"
    )


def _write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.part")
    try:
        temporary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _new_quality_report(spec: AudioAssetSpec, signature: str) -> dict[str, Any]:
    return {
        "version": QUALITY_REPORT_VERSION,
        "detectorVersion": AUDIO_QUALITY_DETECTOR_VERSION,
        "assetKey": spec.manifest_key,
        "assetKind": spec.kind,
        "assetId": spec.asset_id,
        "signature": signature,
        "accepted": False,
        "attempts": [],
    }


def _quality_metadata_is_current(value: Any) -> bool:
    if not (
        isinstance(value, dict)
        and value.get("detectorVersion") == AUDIO_QUALITY_DETECTOR_VERSION
        and value.get("status") in {"passed", "repaired", "review_only"}
        and isinstance(value.get("reportPath"), str)
        and value.get("reportPath")
    ):
        return False
    report_path = Path(value["reportPath"])
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(report, dict)
        and report.get("detectorVersion") == AUDIO_QUALITY_DETECTOR_VERSION
        and report.get("accepted") is True
    )


def _audio_asset_entry(
    spec: AudioAssetSpec,
    *,
    path: Path,
    duration_seconds: float,
    signature: str,
    quality: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "assetId": spec.asset_id,
        "kind": spec.kind,
        "sceneId": spec.scene_id,
        "model": spec.model,
        "path": str(path),
        "durationSeconds": duration_seconds,
        "signature": signature,
        "quality": quality,
    }


def _archive_audio_candidate(
    output_directory: Path,
    spec: AudioAssetSpec,
    path: Path,
    *,
    reason: str,
) -> Path | None:
    """Move one rejected candidate aside without reusing a filename."""

    if not path.is_file():
        return None
    rejected_directory = output_directory / "rejected" / spec.kind
    rejected_directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_reason = _safe_asset_stem(reason)[:40]
    destination = rejected_directory / (
        f"{_safe_asset_stem(spec.asset_id)}-{stamp}-{safe_reason}{path.suffix}"
    )
    suffix = 1
    while destination.exists():
        destination = rejected_directory / (
            f"{_safe_asset_stem(spec.asset_id)}-{stamp}-{safe_reason}-{suffix}{path.suffix}"
        )
        suffix += 1
    path.replace(destination)
    return destination


def _record_rejected_asset(
    manifest: dict[str, Any],
    spec: AudioAssetSpec,
    *,
    reason: str,
    report_path: Path,
    rejected_paths: list[Path],
) -> None:
    manifest.setdefault("assets", {}).pop(spec.manifest_key, None)
    rejected_assets = manifest.setdefault("rejectedAssets", {})
    rejected_assets[spec.manifest_key] = {
        "assetId": spec.asset_id,
        "kind": spec.kind,
        "sceneId": spec.scene_id,
        "reason": reason,
        "reportPath": str(report_path),
        "paths": [str(path) for path in rejected_paths],
        "detectorVersion": AUDIO_QUALITY_DETECTOR_VERSION,
    }


def _record_accepted_asset(
    manifest: dict[str, Any],
    spec: AudioAssetSpec,
    *,
    path: Path,
    duration_seconds: float,
    signature: str,
    quality: dict[str, Any] | None,
) -> None:
    manifest.setdefault("assets", {})[spec.manifest_key] = _audio_asset_entry(
        spec,
        path=path,
        duration_seconds=duration_seconds,
        signature=signature,
        quality=quality,
    )
    rejected_assets = manifest.get("rejectedAssets")
    if isinstance(rejected_assets, dict):
        rejected_assets.pop(spec.manifest_key, None)


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
    rejected_assets = manifest.get("rejectedAssets")
    stale_rejected_keys: list[str] = []
    if isinstance(rejected_assets, dict):
        stale_rejected_keys = [key for key in rejected_assets if key not in allowed_keys]
        for key in stale_rejected_keys:
            del rejected_assets[key]
    quality_fallbacks = manifest.get("qualityFallbacks")
    stale_fallback_keys: list[str] = []
    if isinstance(quality_fallbacks, dict):
        stale_fallback_keys = [key for key in quality_fallbacks if key not in allowed_keys]
        for key in stale_fallback_keys:
            del quality_fallbacks[key]
    if stale_keys or stale_rejected_keys or stale_fallback_keys or not manifest_path.is_file():
        _write_manifest(manifest_path, manifest)
    return manifest_path


def _validate_declared_audio_models(
    script: dict[str, Any],
    config: StableAudioConfig,
) -> None:
    """Reject an explicit incompatible model before coverage can fill a gap."""

    raw_plan = script.get("audioPlan")
    raw_scenes = raw_plan.get("scenes") if isinstance(raw_plan, dict) else None
    if not isinstance(raw_scenes, list):
        return

    def validate(value: object, expected: str, field: str) -> None:
        if not isinstance(value, dict) or "model" not in value:
            return
        model = value.get("model")
        if not isinstance(model, str) or model.strip() != expected:
            raise StableAudioError(
                "invalid_audio_plan",
                f"{field} must use {expected}, got {model}",
            )

    for scene_index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            continue
        prefix = f"audioPlan.scenes[{scene_index}]"
        validate(raw_scene.get("music"), config.music_model, f"{prefix}.music")
        raw_variants = raw_scene.get("musicVariants")
        if isinstance(raw_variants, list):
            for variant_index, raw_variant in enumerate(raw_variants):
                validate(
                    raw_variant,
                    config.music_model,
                    f"{prefix}.musicVariants[{variant_index}]",
                )
        raw_sfx = raw_scene.get("sfx")
        if isinstance(raw_sfx, list):
            for sfx_index, raw_sfx_item in enumerate(raw_sfx):
                validate(
                    raw_sfx_item,
                    config.sfx_model,
                    f"{prefix}.sfx[{sfx_index}]",
                )


def _load_audio_script(
    script_path: Path,
    *,
    config: StableAudioConfig | None = None,
) -> dict[str, Any]:
    """Read a chapter script and normalize legacy IDs and music coverage."""

    try:
        script = json.loads(script_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StableAudioError("script_not_found", str(error)) from error
    except (OSError, json.JSONDecodeError) as error:
        raise StableAudioError("invalid_script", str(error)) from error
    if not isinstance(script, dict):
        raise StableAudioError("invalid_script", "chapter script must be an object")

    if config is not None:
        _validate_declared_audio_models(script, config)

    script_changed = bool(normalize_script_audio_asset_ids(script))
    raw_segments = script.get("segments")
    if isinstance(raw_segments, list) and raw_segments:
        normalized_plan = normalize_serialized_audio_plan_music_coverage(
            script.get("audioPlan"),
            segment_count=len(raw_segments),
            language=str(script.get("language") or "zh"),
            segments=[item if isinstance(item, dict) else {} for item in raw_segments],
        )
        if script.get("audioPlan") != normalized_plan:
            script["audioPlan"] = normalized_plan
            script_changed = True

    if script_changed:
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

    Keeping this in one place prevents callers and the backend CLI generator
    from disagreeing about asset IDs, models, or output filenames.
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
    """Return the stable destination used by all backend asset workflows."""
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
        and (
            not config.quality_enabled
            or _quality_metadata_is_current(previous.get("quality"))
        )
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
        quality=(previous.get("quality") if isinstance(previous, dict) else None),
    )


def import_generated_audio_asset(
    output_directory: Path,
    spec: AudioAssetSpec,
    source_path: Path,
    *,
    config: StableAudioConfig | None = None,
) -> AudioAssetResult:
    """Import one externally generated WAV through the normal quality gate."""
    if not _is_readable_wav(source_path):
        raise StableAudioError(
            "invalid_stable_audio_output",
            f"Stable Audio did not create a readable WAV for {spec.asset_id}",
            details={"assetId": spec.asset_id, "path": str(source_path)},
        )
    config = config or StableAudioConfig.from_environment()
    output_path = asset_output_path(output_directory, spec)
    _copy_regenerated_candidate(source_path, output_path, spec)
    signature = _asset_signature(spec, config)
    manifest_path = _manifest_path(output_directory)
    manifest = _read_manifest(manifest_path)
    manifest["version"] = MANIFEST_VERSION
    outcome = _run_audio_asset_quality_gate(
        output_directory,
        spec,
        output_path,
        config=config,
        signature=signature,
        source="external_import",
    )
    if not outcome.accepted:
        _record_rejected_asset(
            manifest,
            spec,
            reason=outcome.reason or "quality_rejected",
            report_path=outcome.report_path,
            rejected_paths=list(outcome.rejected_paths),
        )
        _write_manifest(manifest_path, manifest)
        raise StableAudioError(
            "audio_quality_rejected",
            f"Stable Audio asset failed quality checks: {spec.manifest_key}",
            details={
                "assetId": spec.asset_id,
                "assetKind": spec.kind,
                "reason": outcome.reason,
                "reportPath": str(outcome.report_path),
            },
        )

    duration_seconds = _wav_duration(output_path)
    _record_accepted_asset(
        manifest,
        spec,
        path=output_path,
        duration_seconds=duration_seconds,
        signature=signature,
        quality=outcome.quality,
    )
    quality_fallbacks = manifest.get("qualityFallbacks")
    if isinstance(quality_fallbacks, dict):
        quality_fallbacks.pop(spec.manifest_key, None)
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
        quality=outcome.quality,
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
    destination = _archive_audio_candidate(
        output_directory,
        spec,
        output_path,
        reason="manual-quarantine",
    )
    _record_rejected_asset(
        manifest,
        spec,
        reason="manual_quarantine",
        report_path=_quality_report_path(output_directory, spec),
        rejected_paths=[destination] if destination is not None else [],
    )
    _write_manifest(manifest_path, manifest)
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


def _quality_metadata(
    report_path: Path,
    *,
    status: str,
    source: str,
    generation_attempts: int,
    repair_intervals: tuple[tuple[float, float], ...] = (),
    review_only: bool = False,
) -> dict[str, Any]:
    return {
        "detectorVersion": AUDIO_QUALITY_DETECTOR_VERSION,
        "reportPath": str(report_path),
        "status": status,
        "source": source,
        "generationAttempts": generation_attempts,
        "repairIntervals": [list(interval) for interval in repair_intervals],
        "reviewOnly": review_only,
    }


def _quality_attempt(
    *,
    source: str,
    generation_attempt: int,
    generation_seed: int | None = None,
    quality: AudioQualityResult | None = None,
    error: str | None = None,
    repair: dict[str, Any] | None = None,
    result: str,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "source": source,
        "generationAttempt": generation_attempt,
        "result": result,
    }
    if generation_seed is not None:
        item["generationSeed"] = generation_seed
    if quality is not None:
        item["quality"] = quality.to_dict()
    if error:
        item["quality"] = {"status": "analysis_failed", "error": error}
    if repair is not None:
        item["repair"] = repair
    return item


def _inspect_audio_asset_candidate(
    output_directory: Path,
    spec: AudioAssetSpec,
    candidate_path: Path,
    *,
    report: dict[str, Any],
    report_path: Path,
    source: str,
    generation_attempt: int,
    generation_seed: int | None = None,
) -> _AssetQualityOutcome:
    """Inspect one candidate and optionally repair a short actionable span.

    Any source that requires repair is moved out of its normal asset location
    *before* a repair is attempted. That keeps a concurrently-started mix from
    ever finding the bad WAV by a fallback filesystem path. Tonal-only review
    observations remain in place and are accepted with their diagnostics.
    """

    rejected_paths: list[Path] = []
    try:
        quality_result = analyze_audio(candidate_path)
    except Exception as error:
        archived = _archive_audio_candidate(
            output_directory, spec, candidate_path, reason=f"{source}-analysis-failed"
        )
        if archived is not None:
            rejected_paths.append(archived)
        report["attempts"].append(
            _quality_attempt(
                source=source,
                generation_attempt=generation_attempt,
                generation_seed=generation_seed,
                error=str(error),
                result="rejected",
            )
        )
        return _AssetQualityOutcome(
            False,
            "analysis_failed",
            report_path,
            None,
            "quality_detection_failed",
            tuple(rejected_paths),
        )

    if not quality_result.requires_repair:
        quality_status = "review_only" if quality_result.review_only else "passed"
        report["attempts"].append(
            _quality_attempt(
                source=source,
                generation_attempt=generation_attempt,
                generation_seed=generation_seed,
                quality=quality_result,
                result=quality_status,
            )
        )
        return _AssetQualityOutcome(
            True,
            quality_status,
            report_path,
            _quality_metadata(
                report_path,
                status=quality_status,
                source=source,
                generation_attempts=generation_attempt,
                review_only=quality_result.review_only,
            ),
        )

    archived_source = _archive_audio_candidate(
        output_directory, spec, candidate_path, reason=f"{source}-suspicious"
    )
    if archived_source is not None:
        rejected_paths.append(archived_source)
    if archived_source is None:
        report["attempts"].append(
            _quality_attempt(
                source=source,
                generation_attempt=generation_attempt,
                generation_seed=generation_seed,
                quality=quality_result,
                result="rejected",
            )
        )
        return _AssetQualityOutcome(
            False,
            "rejected",
            report_path,
            None,
            "quality_candidate_archive_failed",
            tuple(rejected_paths),
        )

    repair_result = repair_short_suspicious_intervals(
        archived_source,
        candidate_path,
        quality_result.actionable_intervals or quality_result.suspicious_intervals,
    )
    report["attempts"].append(
        _quality_attempt(
            source=source,
            generation_attempt=generation_attempt,
            generation_seed=generation_seed,
            quality=quality_result,
            repair=repair_result.to_dict(),
            result="repair_attempted" if repair_result.repaired else "rejected",
        )
    )
    if not repair_result.repaired or repair_result.output_path is None:
        failed_repair = _archive_audio_candidate(
            output_directory, spec, candidate_path, reason=f"{source}-repair-failed"
        )
        if failed_repair is not None:
            rejected_paths.append(failed_repair)
        return _AssetQualityOutcome(
            False,
            "rejected",
            report_path,
            None,
            repair_result.reason or "quality_repair_not_eligible",
            tuple(rejected_paths),
        )

    try:
        repaired_quality = analyze_audio(repair_result.output_path)
    except Exception as error:
        repaired_quality = None
        repair_error = str(error)
    else:
        repair_error = None
    if repaired_quality is not None and not repaired_quality.requires_repair:
        report["attempts"].append(
            _quality_attempt(
                source="repair",
                generation_attempt=generation_attempt,
                generation_seed=generation_seed,
                quality=repaired_quality,
                result="repaired",
            )
        )
        return _AssetQualityOutcome(
            True,
            "repaired",
            report_path,
            _quality_metadata(
                report_path,
                status="repaired",
                source=source,
                generation_attempts=generation_attempt,
                repair_intervals=repair_result.intervals,
                review_only=repaired_quality.review_only,
            ),
            rejected_paths=tuple(rejected_paths),
        )

    failed_repair = _archive_audio_candidate(
        output_directory, spec, candidate_path, reason=f"{source}-repair-rejected"
    )
    if failed_repair is not None:
        rejected_paths.append(failed_repair)
    report["attempts"].append(
        _quality_attempt(
            source="repair",
            generation_attempt=generation_attempt,
            generation_seed=generation_seed,
            quality=repaired_quality,
            error=repair_error,
            result="rejected",
        )
    )
    return _AssetQualityOutcome(
        False,
        "rejected",
        report_path,
        None,
        "quality_repair_verification_failed",
        tuple(rejected_paths),
    )


def _regeneration_seed(signature: str, attempt: int) -> int:
    digest = hashlib.sha256(
        f"{signature}:audio-quality-regeneration:{attempt}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % 2_147_483_646 + 1


def _generate_cli_audio_candidate(
    spec: AudioAssetSpec,
    output_path: Path,
    config: StableAudioConfig,
) -> None:
    """Generate one asset through the configured Stable Audio CLI."""

    if not config.executable.is_file():
        raise StableAudioError(
            "stable_audio_unavailable",
            f"Stable Audio executable not found: {config.executable}",
            details={"executable": str(config.executable)},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        ) from error
    except OSError as error:
        raise StableAudioError(
            "stable_audio_unavailable",
            f"Unable to execute Stable Audio: {error}",
        ) from error
    except subprocess.TimeoutExpired as error:
        raise StableAudioError(
            "stable_audio_timeout",
            f"Stable Audio timed out while generating {spec.asset_id}",
            details={"assetId": spec.asset_id},
        ) from error
    if completed.returncode != 0:
        raise StableAudioError(
            "stable_audio_generation_failed",
            f"Failed to generate {spec.asset_id}: {_failure_detail(completed)}",
            details={
                "assetId": spec.asset_id,
                "returnCode": completed.returncode,
            },
        )
    if not _is_readable_wav(output_path):
        raise StableAudioError(
            "invalid_stable_audio_output",
            f"Stable Audio did not create a readable WAV for {spec.asset_id}",
            details={"assetId": spec.asset_id, "path": str(output_path)},
        )
    if spec.kind == "music":
        try:
            _normalize_music_asset(output_path)
        except StableAudioError:
            output_path.unlink(missing_ok=True)
            raise


def _regenerate_audio_asset_cli(
    spec: AudioAssetSpec,
    output_path: Path,
    config: StableAudioConfig,
    *,
    signature: str,
    attempt: int,
) -> AudioAssetSpec:
    """Regenerate one rejected asset with an auditable deterministic seed."""

    retry_spec = replace(spec, seed=_regeneration_seed(signature, attempt))
    _generate_cli_audio_candidate(retry_spec, output_path, config)
    return retry_spec


def _copy_regenerated_candidate(
    source_path: Path,
    output_path: Path,
    spec: AudioAssetSpec,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    try:
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(output_path)
        if spec.kind == "music":
            _normalize_music_asset(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run_audio_asset_quality_gate(
    output_directory: Path,
    spec: AudioAssetSpec,
    output_path: Path,
    *,
    config: StableAudioConfig,
    signature: str,
    source: str,
) -> _AssetQualityOutcome:
    """Accept one quality-safe asset or reject it without stopping the chapter."""

    report_path = _quality_report_path(output_directory, spec)
    if not config.quality_enabled:
        return _AssetQualityOutcome(
            True,
            "skipped",
            report_path,
            {
                "enabled": False,
                "status": "skipped",
                "detectorVersion": AUDIO_QUALITY_DETECTOR_VERSION,
            },
        )

    report = _new_quality_report(spec, signature)
    rejected_paths: list[Path] = []
    first = _inspect_audio_asset_candidate(
        output_directory,
        spec,
        output_path,
        report=report,
        report_path=report_path,
        source=source,
        generation_attempt=0,
        generation_seed=spec.seed,
    )
    rejected_paths.extend(first.rejected_paths)
    if first.accepted:
        report["accepted"] = True
        report["status"] = first.status
        _write_quality_report(report_path, report)
        return first

    latest_reason = first.reason or "quality_rejected"
    for attempt in range(1, QUALITY_REGENERATION_ATTEMPTS + 1):
        retry_seed = _regeneration_seed(signature, attempt)
        try:
            retry_spec = _regenerate_audio_asset_cli(
                spec,
                output_path,
                config,
                signature=signature,
                attempt=attempt,
            )
        except StableAudioError as error:
            latest_reason = error.code
            failed_candidate = _archive_audio_candidate(
                output_directory,
                spec,
                output_path,
                reason="cli-regeneration-failed",
            )
            if failed_candidate is not None:
                rejected_paths.append(failed_candidate)
            report["attempts"].append(
                {
                    "source": "stable_audio_cli_regeneration",
                    "generationAttempt": attempt,
                    "generationSeed": retry_seed,
                    "result": "generation_failed",
                    "error": {"code": error.code, "message": str(error)},
                }
            )
            continue

        regenerated = _inspect_audio_asset_candidate(
            output_directory,
            spec,
            output_path,
            report=report,
            report_path=report_path,
            source="stable_audio_cli_regeneration",
            generation_attempt=attempt,
            generation_seed=retry_spec.seed,
        )
        rejected_paths.extend(regenerated.rejected_paths)
        if regenerated.accepted:
            report["accepted"] = True
            report["status"] = regenerated.status
            _write_quality_report(report_path, report)
            return regenerated
        latest_reason = regenerated.reason or latest_reason

    report["accepted"] = False
    report["status"] = "rejected"
    report["reason"] = latest_reason
    _write_quality_report(report_path, report)
    return _AssetQualityOutcome(
        False,
        "rejected",
        report_path,
        None,
        latest_reason,
        tuple(rejected_paths),
    )


def _record_audio_generation_failure(
    output_directory: Path,
    manifest: dict[str, Any],
    spec: AudioAssetSpec,
    output_path: Path,
    *,
    signature: str,
    source: str,
    error: StableAudioError,
) -> None:
    """Quarantine a failed CLI candidate without discarding the chapter state."""

    report_path = _quality_report_path(output_directory, spec)
    rejected_candidate = _archive_audio_candidate(
        output_directory,
        spec,
        output_path,
        reason=f"{source}-generation-failed",
    )
    report = _new_quality_report(spec, signature)
    report["status"] = "generation_failed"
    report["reason"] = error.code
    report["attempts"].append(
        {
            "source": source,
            "generationAttempt": 0,
            "generationSeed": spec.seed,
            "result": "generation_failed",
            "error": {"code": error.code, "message": str(error)},
        }
    )
    _write_quality_report(report_path, report)
    _record_rejected_asset(
        manifest,
        spec,
        reason=error.code,
        report_path=report_path,
        rejected_paths=[rejected_candidate] if rejected_candidate is not None else [],
    )


def _refresh_quality_music_fallbacks(
    manifest: dict[str, Any],
    specs: list[AudioAssetSpec],
) -> None:
    """Map rejected music to an approved same/nearby-scene substitute.

    The mixer already knows how to loop a selected music asset over a cue.  A
    manifest-level mapping lets it keep that behavior without ever looking at
    an on-disk WAV that was removed from ``assets`` by the quality gate.
    """

    assets = manifest.get("assets")
    rejected_assets = manifest.get("rejectedAssets")
    if not isinstance(assets, dict) or not isinstance(rejected_assets, dict):
        manifest.pop("qualityFallbacks", None)
        return

    music_specs = [spec for spec in specs if spec.kind == "music"]
    accepted = [spec for spec in music_specs if spec.manifest_key in assets]
    accepted_by_scene: dict[str, list[AudioAssetSpec]] = {}
    for spec in accepted:
        accepted_by_scene.setdefault(spec.scene_id, []).append(spec)

    scene_positions: dict[str, int] = {}
    for position, spec in enumerate(music_specs):
        scene_positions.setdefault(spec.scene_id, position)

    fallbacks: dict[str, dict[str, str]] = {}
    for spec in music_specs:
        if spec.manifest_key in assets or spec.manifest_key not in rejected_assets:
            continue
        same_scene = accepted_by_scene.get(spec.scene_id, [])
        if same_scene:
            target = same_scene[0]
            reason = "quality_same_scene_variant"
        else:
            source_position = scene_positions.get(spec.scene_id, 0)
            candidates = sorted(
                accepted,
                key=lambda candidate: (
                    abs(scene_positions.get(candidate.scene_id, source_position) - source_position),
                    scene_positions.get(candidate.scene_id, source_position),
                ),
            )
            if not candidates:
                continue
            target = candidates[0]
            reason = "quality_nearby_scene"
        fallbacks[spec.manifest_key] = {
            "assetKey": target.manifest_key,
            "reason": reason,
        }

    if fallbacks:
        manifest["qualityFallbacks"] = fallbacks
    else:
        manifest.pop("qualityFallbacks", None)


def _chapter_fallback_music_spec(
    specs: list[AudioAssetSpec],
) -> AudioAssetSpec | None:
    """Build one conservative music bed for a chapter with no survivor.

    The fallback is deliberately an explicit manifest asset. The mixer can
    then map rejected scene music to it without ever searching for a rejected
    WAV by filename.
    """

    base = next((spec for spec in specs if spec.kind == "music"), None)
    if base is None:
        return None
    used_ids = {spec.asset_id for spec in specs if spec.kind == "music"}
    asset_id = FALLBACK_MUSIC_ASSET_ID
    suffix = 2
    while asset_id in used_ids:
        asset_id = f"{FALLBACK_MUSIC_ASSET_ID}_{suffix}"
        suffix += 1
    seed_source = (
        f"{base.plan_signature}:chapter-fallback:{asset_id}".encode("utf-8")
    )
    seed = int(hashlib.sha256(seed_source).hexdigest()[:8], 16) % 2_147_483_646 + 1
    prompt = (
        f"{base.prompt.rstrip(' ,.;，。；')}, continuous understated instrumental "
        "background bed, gentle texture, no sudden accents"
    )
    return replace(
        base,
        asset_id=asset_id,
        scene_id="chapter_fallback",
        prompt=prompt,
        seed=seed,
    )


def _has_approved_music(
    manifest: dict[str, Any],
    specs: list[AudioAssetSpec],
) -> bool:
    assets = manifest.get("assets")
    return isinstance(assets, dict) and any(
        spec.kind == "music" and spec.manifest_key in assets
        for spec in specs
    )


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
    script = _load_audio_script(script_path, config=config)

    planned_specs = _collect_asset_specs(script)
    fallback_spec = _chapter_fallback_music_spec(planned_specs)
    all_specs = [*planned_specs, *([fallback_spec] if fallback_spec else [])]
    specs = planned_specs
    selected_asset = asset_id is not None or asset_kind is not None
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
    rejected_assets = manifest.get("rejectedAssets")
    if isinstance(rejected_assets, dict):
        for key in list(rejected_assets):
            if key not in allowed_keys:
                del rejected_assets[key]
    quality_fallbacks = manifest.get("qualityFallbacks")
    if isinstance(quality_fallbacks, dict):
        for key in list(quality_fallbacks):
            if key not in allowed_keys:
                del quality_fallbacks[key]

    if not specs:
        _write_manifest(manifest_path, manifest)
        return AudioAssetGenerationResult(
            assets=[],
            warnings=["no_audio_assets"],
            manifest_path=manifest_path,
        )

    results: list[AudioAssetResult] = []
    warnings: list[str] = []

    def process_spec(
        spec: AudioAssetSpec,
        *,
        continue_after_generation_failure: bool = False,
    ) -> AudioAssetResult | None:
        signature = _asset_signature(spec, config)
        output_path = asset_output_path(output_directory, spec)
        previous = manifest_assets.get(spec.manifest_key)
        matching_cached_candidate = (
            not force
            and isinstance(previous, dict)
            and previous.get("signature") == signature
            and _is_readable_wav(output_path)
        )
        cache_hit = matching_cached_candidate and (
            not config.quality_enabled
            or _quality_metadata_is_current(
                previous.get("quality") if isinstance(previous, dict) else None
            )
        )

        if cache_hit:
            return AudioAssetResult(
                asset_id=spec.asset_id,
                kind=spec.kind,
                scene_id=spec.scene_id,
                model=spec.model,
                path=output_path,
                duration_seconds=_wav_duration(output_path),
                signature=signature,
                cache_hit=True,
                quality=(previous.get("quality") if isinstance(previous, dict) else None),
            )

        if matching_cached_candidate:
            source = "cache_validation"
        else:
            manifest_assets.pop(spec.manifest_key, None)
            try:
                _generate_cli_audio_candidate(spec, output_path, config)
            except StableAudioError as error:
                if continue_after_generation_failure:
                    _record_audio_generation_failure(
                        output_directory,
                        manifest,
                        spec,
                        output_path,
                        signature=signature,
                        source="stable_audio_cli_fallback",
                        error=error,
                    )
                    warnings.append(
                        f"audio_generation_failed:{spec.manifest_key}:{error.code}"
                    )
                    _write_manifest(manifest_path, manifest)
                    return None
                raise StableAudioError(
                    error.code,
                    str(error),
                    details=error.details,
                    partial_assets=results,
                ) from error
            source = "stable_audio_cli"

        outcome = _run_audio_asset_quality_gate(
            output_directory,
            spec,
            output_path,
            config=config,
            signature=signature,
            source=source,
        )
        if not outcome.accepted:
            _record_rejected_asset(
                manifest,
                spec,
                reason=outcome.reason or "quality_rejected",
                report_path=outcome.report_path,
                rejected_paths=list(outcome.rejected_paths),
            )
            warnings.append(
                f"audio_quality_rejected:{spec.manifest_key}:{outcome.reason or 'unknown'}"
            )
            _write_manifest(manifest_path, manifest)
            return None

        actual_duration = _wav_duration(output_path)
        result = AudioAssetResult(
            asset_id=spec.asset_id,
            kind=spec.kind,
            scene_id=spec.scene_id,
            model=spec.model,
            path=output_path,
            duration_seconds=actual_duration,
            signature=signature,
            cache_hit=(
                matching_cached_candidate
                and outcome.status in {"passed", "review_only"}
            ),
            quality=outcome.quality,
        )
        _record_accepted_asset(
            manifest,
            spec,
            path=output_path,
            duration_seconds=actual_duration,
            signature=signature,
            quality=outcome.quality,
        )
        _write_manifest(manifest_path, manifest)
        return result

    for spec in specs:
        result = process_spec(spec)
        if result is not None:
            results.append(result)

    if not selected_asset and fallback_spec is not None:
        if _has_approved_music(manifest, planned_specs):
            # A normal planned music asset recovered. Keep the old fallback
            # WAV on disk for diagnosis, but stop advertising it to the mixer.
            manifest_assets.pop(fallback_spec.manifest_key, None)
            rejected_assets = manifest.get("rejectedAssets")
            if isinstance(rejected_assets, dict):
                rejected_assets.pop(fallback_spec.manifest_key, None)
        elif not _has_approved_music(manifest, all_specs):
            fallback_result = process_spec(
                fallback_spec,
                continue_after_generation_failure=True,
            )
            if fallback_result is not None:
                results.append(fallback_result)
                warnings.append("quality_music_fallback_generated")
            else:
                warnings.append("quality_music_fallback_unavailable")

    _refresh_quality_music_fallbacks(manifest, all_specs)
    if specs and not results:
        warnings.append("no_quality_approved_audio_assets")
    _write_manifest(manifest_path, manifest)

    return AudioAssetGenerationResult(
        assets=results,
        warnings=warnings,
        manifest_path=manifest_path,
    )
