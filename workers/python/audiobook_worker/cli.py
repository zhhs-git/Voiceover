from __future__ import annotations

import argparse
import hashlib
import json
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

from audiobook_worker.audio import (
    DEFAULT_MUSIC_GAIN,
    DEFAULT_SFX_GAIN,
    DEFAULT_VOICE_GAIN,
    assemble_chapter_audio,
    mix_chapter_audio,
)
from audiobook_worker.dialogue import resolve_text_language
from audiobook_worker.llm import (
    AudioPlanningRequest,
    MockLLMAnalyzer,
    audio_plan_to_dict,
    default_analyzer,
    ensure_audio_music_coverage,
    select_active_audio_characters,
)
from audiobook_worker.model_settings import DEFAULT_TTS_MODEL_ID, VOXCPM2_MODEL_ID
from audiobook_worker.segment_merge import (
    DEFAULT_MAX_TTS_CHARACTERS,
    DEFAULT_MAX_TTS_WORDS,
    merge_tts_segments,
    split_tts_segments,
)
from audiobook_worker.stable_audio import StableAudioError, generate_audio_assets
from audiobook_worker.script_builder import (
    apply_narrator_voice,
    build_chapter_script,
    build_chapter_script_with_corrections,
    refresh_script_voice_assignments,
)
from audiobook_worker.tts import (
    KokoroTTSBackend,
    MiMoTTSBackend,
    MockTTSBackend,
    ParlerTTSBackend,
    VOXCPM2_BATCH_ADAPTER_VERSION,
    VOXCPM2_PROMPT_FORMAT_VERSION,
    VoxCPM2TTSBackend,
    voxcpm2_language_for_segment,
    voice_options,
)
from audiobook_worker.tts_quality import (
    TtsSegmentAudioQualityError,
    analyze_tts_segment_wav,
)
from audiobook_worker.voxcpm2_profile_loudness import voxcpm2_profile_loudness
from audiobook_worker.transcription import (
    DEFAULT_WHISPER_MODEL,
    TranscriptionError,
    transcribe_audio,
)
from audiobook_worker.workflow import (
    ANALYSIS_WORKFLOW_STEPS,
    GENERATION_WORKFLOW_STEPS,
    start_workflow,
    update_workflow,
)


# Bump this whenever the chunking/validation contract changes.  It prevents
# an old, potentially truncated WAV from being accepted as a cache hit after
# the safer splitting logic is deployed.
_TTS_SEGMENTATION_VERSION = 7
_TTS_TIMELINE_MANIFEST_VERSION = 1
_TTS_TIMELINE_MANIFEST_NAME = "timeline.json"
# Bump this when analysis prompts or stage contracts change.  A speaker-stage
# prompt change must not allow a failed run to resume with an older attribution
# artifact that was produced by the previous prompt.
_ANALYSIS_CACHE_VERSION = 4
_ANALYSIS_STAGES = (
    "characters",
    "voice_design",
    "speakers",
    "delivery",
    "voice_direction",
)
_AUDIO_PLANNING_CACHE_VERSION = 1
_AUDIO_PLANNING_STAGES = ("scene_structure", "music", "sfx")
_AUDIO_PLANNING_STAGE_LABELS = {
    "scene_structure": "场景结构分析",
    "music": "背景音乐设计",
    "sfx": "音效证据分析",
}


def _workflow_state_for_script(script_path: Path, chapter_id: str) -> tuple[Path, dict[str, Any]]:
    analysis_directory = _analysis_directory(script_path.parent, chapter_id)
    state_path = analysis_directory / "state.json"
    return state_path, _read_json_object(state_path)


def _start_generation_workflow(
    state: dict[str, Any],
    *,
    first_step: str,
    preserve_completed: bool = False,
) -> None:
    previous = state.get("workflow", {}).get("generation") if preserve_completed else None
    start_workflow(
        state,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        first_step=first_step,
    )
    if not isinstance(previous, dict):
        return
    previous_steps = previous.get("steps")
    if not isinstance(previous_steps, dict):
        return
    generation = state["workflow"]["generation"]
    for step_id, value in previous_steps.items():
        if step_id == first_step or step_id not in generation["steps"]:
            continue
        if isinstance(value, dict) and value.get("status") in {"succeeded", "skipped"}:
            generation["steps"][step_id] = dict(value)


def _response(
    status: str,
    *,
    warnings: list[str] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "warnings": warnings or [],
        "artifacts": artifacts or [],
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _invalidate_chapter_audio_artifacts(script_directory: Path, chapter_id: str) -> None:
    """Invalidate derived audio when a chapter script is replaced.

    Keep individual generated WAVs recoverable, but remove the assembled voice,
    final mix, and Stable Audio manifest so the UI cannot present an old result
    as belonging to the new scene plan.  TTS sidecars are additionally guarded
    by the cache signature above.
    """
    work_directory = script_directory.parent
    (work_directory / "audio" / f"{chapter_id}.wav").unlink(missing_ok=True)
    (work_directory / "audio" / f"{chapter_id}_mixed.wav").unlink(missing_ok=True)
    (work_directory / "audio-assets" / chapter_id / "manifest.json").unlink(
        missing_ok=True
    )
    (work_directory / "segments" / chapter_id / _TTS_TIMELINE_MANIFEST_NAME).unlink(
        missing_ok=True
    )


def _invalidate_chapter_mix_artifacts(script_directory: Path, chapter_id: str) -> None:
    """Invalidate only derived music/SFX and the final mix.

    Post-TTS audio planning must not remove the already generated voice track.
    """
    work_directory = script_directory.parent
    (work_directory / "audio" / f"{chapter_id}_mixed.wav").unlink(missing_ok=True)
    (work_directory / "audio-assets" / chapter_id / "manifest.json").unlink(
        missing_ok=True
    )


def _analysis_directory(script_directory: Path, chapter_id: str) -> Path:
    directory = script_directory.parent / "analysis" / chapter_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _voice_direction_map(script_path: Path, chapter_id: str) -> dict[int, str]:
    """Read internal dynamic directions without adding them to the script IR."""
    analysis_path = _analysis_directory(script_path.parent, chapter_id) / "voice_direction.json"
    payload = _read_json_object(analysis_path)
    raw_directions = payload.get("directions", [])
    if not isinstance(raw_directions, list):
        raw_directions = payload.get("segmentAnnotations", [])
    result: dict[int, str] = {}
    for item in raw_directions:
        if not isinstance(item, dict) or "segmentIndex" not in item:
            continue
        direction = str(
            item.get("direction")
            or item.get("voiceDirection")
            or item.get("performanceDirection")
            or ""
        ).strip()
        if direction:
            result[int(item["segmentIndex"])] = direction
    return result


def _clip_voice_context(value: Any, limit: int = 90) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[:limit]}…"


def _decorate_segments_for_voice(
    script_path: Path,
    script: dict[str, Any],
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chapter_id = str(script.get("chapterId") or script_path.stem)
    directions = _voice_direction_map(script_path, chapter_id)
    original_segments = script.get("segments", [])
    if not isinstance(original_segments, list):
        original_segments = []
    script_text = " ".join(
        str(item.get("text") or "")
        for item in original_segments
        if isinstance(item, dict)
    )
    script_language = resolve_text_language(
        script_text,
        str(script.get("language") or "").strip() or None,
    )
    index_by_id = {
        str(item.get("id")): index
        for index, item in enumerate(original_segments)
        if isinstance(item, dict) and item.get("id") is not None
    }
    decorated: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        item["language"] = script_language
        source_ids = item.get("sourceSegmentIds") or [item.get("id")]
        source_index = next(
            (index_by_id[str(source_id)] for source_id in source_ids if str(source_id) in index_by_id),
            None,
        )
        if source_index is not None:
            if not item.get("voiceDirection"):
                item["voiceDirection"] = directions.get(source_index, "")
            context_parts: list[str] = []
            if source_index > 0:
                context_parts.append(
                    f"前文：{_clip_voice_context(original_segments[source_index - 1].get('text'))}"
                )
            context_parts.append(
                f"当前：{_clip_voice_context(original_segments[source_index].get('text'))}"
            )
            if source_index + 1 < len(original_segments):
                context_parts.append(
                    f"后文：{_clip_voice_context(original_segments[source_index + 1].get('text'))}"
                )
            item["voiceSceneContext"] = "；".join(context_parts)
        decorated.append(item)
    return decorated


def _analysis_input_signature(
    *,
    book_id: str,
    chapter_id: str,
    text: str,
    language: str,
    known_characters: Any,
) -> str:
    """Build a stable key for reusable analysis-stage artifacts."""

    roster: list[dict[str, Any]] = []
    if isinstance(known_characters, list):
        for item in known_characters:
            if not isinstance(item, dict):
                continue
            roster.append(
                {
                    "id": item.get("id"),
                    "canonicalName": item.get("canonicalName"),
                    "aliases": item.get("aliases", []),
                    "gender": item.get("gender", "unknown"),
                    "ageClass": item.get("ageClass", "unknown"),
                    "voiceDesign": item.get("voiceDesign", ""),
                }
            )
    payload = {
        "version": _ANALYSIS_CACHE_VERSION,
        "bookId": book_id,
        "chapterId": chapter_id,
        "text": text,
        "language": language,
        "knownCharacters": roster,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_status(state: dict[str, Any], key: str) -> str | None:
    value = state.get(key)
    return value.get("status") if isinstance(value, dict) else None


def _analysis_resume_stage(
    state: dict[str, Any],
    *,
    input_signature: str,
    explicit_stage: str | None,
) -> str | None:
    metadata = state.get("analysis")
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != _ANALYSIS_CACHE_VERSION
        or metadata.get("inputSignature") != input_signature
    ):
        return None
    if explicit_stage:
        return explicit_stage
    if _state_status(state, "script") == "succeeded":
        return None
    for stage in _ANALYSIS_STAGES:
        if _state_status(state, stage) != "succeeded":
            return stage
    return "script"


def _load_cached_analysis_stages(
    analysis_directory: Path,
    state: dict[str, Any],
    *,
    input_signature: str,
    resume_from_stage: str | None,
) -> dict[str, dict[str, Any]]:
    if not resume_from_stage:
        return {}
    metadata = state.get("analysis")
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != _ANALYSIS_CACHE_VERSION
        or metadata.get("inputSignature") != input_signature
    ):
        return {}
    if resume_from_stage == "script":
        reusable = _ANALYSIS_STAGES
    elif resume_from_stage in _ANALYSIS_STAGES:
        reusable = _ANALYSIS_STAGES[: _ANALYSIS_STAGES.index(resume_from_stage)]
    else:
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for stage in reusable:
        if _state_status(state, stage) != "succeeded":
            continue
        payload = _read_json_object(analysis_directory / f"{stage}.json")
        if payload:
            cached[stage] = payload
    return cached


def _audio_planning_input_signature(request: AudioPlanningRequest) -> str:
    """Build a cache key for post-TTS audio planning inputs."""

    payload = {
        "version": _AUDIO_PLANNING_CACHE_VERSION,
        "bookId": request.book_id,
        "chapterId": request.chapter_id,
        "language": request.language,
        "text": request.text,
        "segments": request.segments,
        "transcript": request.transcript,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audio_planning_resume_stage(
    state: dict[str, Any],
    *,
    input_signature: str,
    explicit_stage: str | None,
) -> str | None:
    if explicit_stage is not None and explicit_stage not in {
        *_AUDIO_PLANNING_STAGES,
        "complete",
    }:
        raise ValueError(
            "audioPlanningFromStage must be scene_structure, music, sfx, or complete"
        )
    metadata = state.get("audioPlanning")
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != _AUDIO_PLANNING_CACHE_VERSION
        or metadata.get("inputSignature") != input_signature
    ):
        return explicit_stage
    if explicit_stage:
        return explicit_stage
    if _state_status(state, "audioPlan") == "succeeded":
        return None
    stages = metadata.get("stages")
    if not isinstance(stages, dict):
        return "scene_structure"
    for stage in _AUDIO_PLANNING_STAGES:
        entry = stages.get(stage)
        if not isinstance(entry, dict) or entry.get("status") != "succeeded":
            return stage
    return "complete"


def _load_cached_audio_planning_stages(
    analysis_directory: Path,
    state: dict[str, Any],
    *,
    input_signature: str,
    resume_from_stage: str | None,
) -> dict[str, dict[str, Any]]:
    if not resume_from_stage:
        return {}
    metadata = state.get("audioPlanning")
    if (
        not isinstance(metadata, dict)
        or metadata.get("version") != _AUDIO_PLANNING_CACHE_VERSION
        or metadata.get("inputSignature") != input_signature
    ):
        return {}
    if resume_from_stage == "complete":
        reusable = _AUDIO_PLANNING_STAGES
    elif resume_from_stage in _AUDIO_PLANNING_STAGES:
        reusable = _AUDIO_PLANNING_STAGES[: _AUDIO_PLANNING_STAGES.index(resume_from_stage)]
    else:
        return {}
    cached: dict[str, dict[str, Any]] = {}
    stage_metadata = metadata.get("stages")
    if not isinstance(stage_metadata, dict):
        stage_metadata = {}
    for stage in reusable:
        entry = stage_metadata.get(stage)
        if not isinstance(entry, dict) or entry.get("status") != "succeeded":
            continue
        payload = _read_json_object(
            analysis_directory / f"audio_plan_{stage}.json"
        )
        if payload:
            cached[stage] = payload
    return cached


def _clear_analysis_stage_state(state: dict[str, Any]) -> None:
    for key in (*_ANALYSIS_STAGES, "script", "failedStage", "lastCompletedStage"):
        state.pop(key, None)


def _voice_identity_strategy(backend_name: str) -> str:
    """Name the stable-voice mechanism used in a segment cache signature."""

    backend = str(backend_name or "").strip().casefold()
    if backend == "mimo":
        return "mimo-reference-audio-voiceclone-v1"
    if backend == "voxcpm2":
        return "voxcpm2-local-reference-wav-v1"
    return f"{backend or 'default'}-native-voice-v1"


def _backend_requires_segment_quality(backend_name: str) -> bool:
    """Keep cloud and local clone WAVs behind the same speech quality gate."""

    return str(backend_name or "").strip().casefold() in {"mimo", "voxcpm2"}


def _tts_model_id_for_request(request: dict[str, Any]) -> str | None:
    """Resolve a stable model id so all stages sign the same backend cache."""

    configured = str(request.get("modelId") or "").strip()
    if configured:
        return configured
    backend = str(request.get("backend") or "mimo").strip().casefold()
    if backend == "mimo":
        return DEFAULT_TTS_MODEL_ID
    if backend == "voxcpm2":
        return VOXCPM2_MODEL_ID
    return None


def _segment_cache_signature(
    segment: dict[str, Any],
    backend_name: str,
    model_id: str | None,
    *,
    max_words: int = DEFAULT_MAX_TTS_WORDS,
    max_characters: int = DEFAULT_MAX_TTS_CHARACTERS,
) -> str:
    payload = {
        "ttsSegmentationVersion": _TTS_SEGMENTATION_VERSION,
        "voiceIdentityStrategy": _voice_identity_strategy(backend_name),
        "backend": backend_name,
        "modelId": model_id or "default",
        "maxWords": max_words,
        "maxCharacters": max_characters,
        "text": segment.get("text", ""),
        "voiceId": segment.get("voiceId", "narrator_default"),
        "fallbackVoiceId": segment.get("fallbackVoiceId"),
        "voiceDesign": segment.get("voiceDesign"),
        "voiceDescription": segment.get("voiceDescription"),
        "voiceDirection": segment.get("voiceDirection"),
        "voiceSceneContext": segment.get("voiceSceneContext"),
        "emotion": segment.get("emotion", "neutral"),
        "intensity": segment.get("intensity"),
        "pace": segment.get("pace", "normal"),
        "sourceSegmentIds": segment.get("sourceSegmentIds", [segment["id"]]),
    }
    if str(backend_name or "").strip().casefold() == "voxcpm2":
        payload["voxcpm2PromptFormatVersion"] = VOXCPM2_PROMPT_FORMAT_VERSION
        payload["voxcpm2BatchAdapterVersion"] = VOXCPM2_BATCH_ADAPTER_VERSION
        payload["voxcpm2Language"] = voxcpm2_language_for_segment(segment)
        payload["voxcpm2ProfileLoudness"] = voxcpm2_profile_loudness()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_cache_metadata_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(audio_path.suffix + ".json")


def _read_cached_segment_artifact(
    segment: dict[str, Any],
    output_directory: Path,
    expected_signature: str,
    *,
    backend_name: str,
) -> dict[str, Any] | None:
    audio_path = output_directory / f"{segment['id']}.wav"
    metadata_path = _segment_cache_metadata_path(audio_path)
    if not audio_path.exists() or not metadata_path.exists():
        return None
    if not _is_readable_wav(audio_path):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("signature") != expected_signature:
        return None
    if not _segment_audio_passes_quality(segment, audio_path, backend_name):
        _invalidate_segment_cache(audio_path)
        return None
    return {
        "kind": "segment_audio",
        "path": str(audio_path),
        "metadata": {
            "durationSeconds": metadata.get("durationSeconds", 0),
            "device": metadata.get("device"),
            "sourceSegmentIds": metadata.get("sourceSegmentIds", segment.get("sourceSegmentIds", [segment["id"]])),
            "segmentId": segment["id"],
            "cacheHit": True,
        },
    }


def _invalidate_segment_cache(audio_path: Path) -> None:
    """Remove one unusable cached segment and its sidecar as one logical item."""
    audio_path.unlink(missing_ok=True)
    _segment_cache_metadata_path(audio_path).unlink(missing_ok=True)


def _segment_audio_passes_quality(
    segment: dict[str, Any],
    audio_path: Path,
    backend_name: str,
) -> bool:
    if not _is_readable_wav(audio_path):
        return False
    if not _backend_requires_segment_quality(backend_name):
        return True
    try:
        return analyze_tts_segment_wav(
            audio_path,
            text=segment.get("text", ""),
            pace=segment.get("pace", "normal"),
        ).accepted
    except TtsSegmentAudioQualityError:
        return False


def _write_segment_cache_metadata(
    audio_path: Path,
    *,
    signature: str,
    duration_seconds: float,
    backend_name: str,
    model_id: str | None,
    device: str | None,
    source_segment_ids: list[str],
) -> None:
    _write_json(
        _segment_cache_metadata_path(audio_path),
        {
            "signature": signature,
            "backend": backend_name,
            "modelId": model_id or "default",
            "durationSeconds": duration_seconds,
            "device": device,
            "sourceSegmentIds": source_segment_ids,
            "segmentId": audio_path.stem,
        },
    )


def _tts_timeline_manifest_path(segment_directory: Path) -> Path:
    return segment_directory / _TTS_TIMELINE_MANIFEST_NAME


def _timeline_signature(records: list[dict[str, Any]], gap_seconds: float) -> str:
    payload = {
        "version": _TTS_TIMELINE_MANIFEST_VERSION,
        "gapSeconds": gap_seconds,
        "segments": [
            {
                "id": record.get("id"),
                "sourceSegmentIds": record.get("sourceSegmentIds", []),
                "signature": record.get("signature"),
            }
            for record in records
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timeline_records(
    segments: list[dict[str, Any]],
    segment_directory: Path,
    *,
    backend_name: str,
    model_id: str | None,
    max_words: int,
    max_characters: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for segment in segments:
        path = segment_directory / f"{segment['id']}.wav"
        records.append(
            {
                "id": str(segment["id"]),
                "sourceSegmentIds": _segment_source_ids(segment),
                "signature": _segment_cache_signature(
                    segment,
                    backend_name,
                    model_id,
                    max_words=max_words,
                    max_characters=max_characters,
                ),
                "durationSeconds": round(_wav_duration_seconds(path), 6),
            }
        )
    return records


def _write_tts_timeline_manifest(
    segment_directory: Path,
    segments: list[dict[str, Any]],
    *,
    backend_name: str,
    model_id: str | None,
    max_words: int,
    max_characters: int,
    gap_seconds: float,
    assembled_audio_path: Path | None = None,
    assembled_duration_seconds: float | None = None,
) -> dict[str, Any]:
    records = _timeline_records(
        segments,
        segment_directory,
        backend_name=backend_name,
        model_id=model_id,
        max_words=max_words,
        max_characters=max_characters,
    )
    manifest: dict[str, Any] = {
        "version": _TTS_TIMELINE_MANIFEST_VERSION,
        "ttsSegmentationVersion": _TTS_SEGMENTATION_VERSION,
        "backend": backend_name,
        "modelId": model_id or "default",
        "maxWords": max_words,
        "maxCharacters": max_characters,
        "gapSeconds": gap_seconds,
        "segments": records,
        "timelineSignature": _timeline_signature(records, gap_seconds),
    }
    if assembled_audio_path is not None:
        manifest["assembledAudioPath"] = str(assembled_audio_path)
    if assembled_duration_seconds is not None:
        manifest["assembledDurationSeconds"] = round(
            assembled_duration_seconds, 6
        )
    _write_json(_tts_timeline_manifest_path(segment_directory), manifest)
    return manifest


def _read_tts_timeline_manifest(segment_directory: Path) -> dict[str, Any] | None:
    path = _tts_timeline_manifest_path(segment_directory)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _expected_timeline_signature(
    segments: list[dict[str, Any]],
    *,
    backend_name: str,
    model_id: str | None,
    max_words: int,
    max_characters: int,
    gap_seconds: float,
) -> str:
    records = [
        {
            "id": str(segment["id"]),
            "sourceSegmentIds": _segment_source_ids(segment),
            "signature": _segment_cache_signature(
                segment,
                backend_name,
                model_id,
                max_words=max_words,
                max_characters=max_characters,
            ),
        }
        for segment in segments
    ]
    return _timeline_signature(records, gap_seconds)


def _manifest_matches_expected_timeline(
    manifest: dict[str, Any] | None,
    segments: list[dict[str, Any]],
    *,
    backend_name: str,
    model_id: str | None,
    max_words: int,
    max_characters: int,
    gap_seconds: float,
) -> bool:
    if not manifest or manifest.get("version") != _TTS_TIMELINE_MANIFEST_VERSION:
        return False
    return manifest.get("timelineSignature") == _expected_timeline_signature(
        segments,
        backend_name=backend_name,
        model_id=model_id,
        max_words=max_words,
        max_characters=max_characters,
        gap_seconds=gap_seconds,
    )


def _cached_segment_files_match_expected(
    expected_segments: list[dict[str, Any]],
    segment_paths: list[Path],
    *,
    backend_name: str,
    model_id: str | None,
    max_words: int,
    max_characters: int,
) -> bool:
    if len(expected_segments) != len(segment_paths):
        return False
    for segment, path in zip(expected_segments, segment_paths, strict=True):
        metadata = _read_segment_cache_metadata(path)
        if metadata is None or metadata.get("signature") != _segment_cache_signature(
            segment,
            backend_name,
            model_id,
            max_words=max_words,
            max_characters=max_characters,
        ):
            return False
        if not _segment_audio_passes_quality(segment, path, backend_name):
            _invalidate_segment_cache(path)
            return False
    return True


def _is_readable_wav(path: Path) -> bool:
    """Return whether path is a non-empty, readable WAV file."""
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except (OSError, wave.Error, ZeroDivisionError):
        return False


def _read_segment_cache_metadata(audio_path: Path) -> dict[str, Any] | None:
    metadata_path = _segment_cache_metadata_path(audio_path)
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _segment_source_ids(segment: dict[str, Any]) -> list[str]:
    raw_source_ids = segment.get("sourceSegmentIds")
    if isinstance(raw_source_ids, list):
        source_ids = [str(value) for value in raw_source_ids if value is not None]
        if source_ids:
            return source_ids
    return [str(segment["id"])]


def _mix_segment_descriptor(
    path: Path,
    fallback_segment: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    descriptor = dict(fallback_segment)
    if metadata is None:
        descriptor["sourceSegmentIds"] = _segment_source_ids(fallback_segment)
        return descriptor

    segment_id = metadata.get("segmentId")
    if isinstance(segment_id, str) and segment_id.strip():
        descriptor["id"] = segment_id.strip()
    source_ids = metadata.get("sourceSegmentIds")
    if isinstance(source_ids, list):
        normalized_source_ids = [
            str(value) for value in source_ids if value is not None
        ]
        if normalized_source_ids:
            descriptor["sourceSegmentIds"] = normalized_source_ids
    descriptor.setdefault("id", path.stem)
    descriptor.setdefault("sourceSegmentIds", [descriptor["id"]])
    return descriptor


def _resolve_mix_segment_audio(
    expected_segments: list[dict[str, Any]],
    segment_directory: Path,
    *,
    backend_name: str,
) -> tuple[list[Path], list[dict[str, Any]], list[str], list[str]]:
    """Resolve the actual cached WAVs used to build the mix timeline.

    Newer synthesis runs write a sidecar containing ``sourceSegmentIds``.
    Older runs may have no sidecars and may have skipped a failed segment
    while still producing a valid assembled chapter WAV.  In that legacy
    case, use the readable files that are present, in expected script order;
    the caller separately verifies that their durations match the voice track.
    """

    exact_paths: list[Path] = []
    missing_ids: list[str] = []
    for segment in expected_segments:
        path = segment_directory / f"{segment['id']}.wav"
        if _segment_audio_passes_quality(segment, path, backend_name):
            exact_paths.append(path)
        else:
            if _backend_requires_segment_quality(backend_name) and path.is_file():
                _invalidate_segment_cache(path)
            missing_ids.append(str(segment["id"]))

    if not missing_ids:
        descriptors = [
            _mix_segment_descriptor(
                path,
                segment,
                _read_segment_cache_metadata(path),
            )
            for path, segment in zip(exact_paths, expected_segments, strict=True)
        ]
        return exact_paths, descriptors, [], []

    readable_paths = sorted(
        (
            path
            for path in segment_directory.glob("*.wav")
            if _is_readable_wav(path)
        ),
        key=lambda path: path.name,
    )
    if not readable_paths:
        return [], [], missing_ids, []

    metadata_by_path = {
        path: _read_segment_cache_metadata(path) for path in readable_paths
    }
    has_sidecar = any(metadata is not None for metadata in metadata_by_path.values())
    expected_by_id = {
        str(segment["id"]): (index, segment)
        for index, segment in enumerate(expected_segments)
    }

    if not has_sidecar:
        # This is the format used by the user's already assembled chapter:
        # exact files exist for most synthesized items, but one old TTS item
        # is absent and there is no metadata to describe a merge.
        resolved_paths: list[Path] = []
        resolved_segments: list[dict[str, Any]] = []
        for segment in expected_segments:
            path = segment_directory / f"{segment['id']}.wav"
            if _segment_audio_passes_quality(segment, path, backend_name):
                resolved_paths.append(path)
                resolved_segments.append(_mix_segment_descriptor(path, segment, None))
        if not resolved_paths:
            return [], [], missing_ids, []
        warning = (
            "legacy_segment_cache_recovered_missing:"
            + ",".join(missing_ids)
        )
        return resolved_paths, resolved_segments, missing_ids, [warning]

    required_source_ids = {
        source_id
        for segment in expected_segments
        for source_id in _segment_source_ids(segment)
    }
    selected: dict[Path, tuple[int, dict[str, Any], set[str]]] = {}
    for path in readable_paths:
        metadata = metadata_by_path[path]
        segment_id = path.stem
        source_ids: set[str] = set()
        if metadata is not None:
            raw_segment_id = metadata.get("segmentId")
            if isinstance(raw_segment_id, str) and raw_segment_id.strip():
                segment_id = raw_segment_id.strip()
            raw_source_ids = metadata.get("sourceSegmentIds")
            if isinstance(raw_source_ids, list):
                source_ids = {
                    str(value) for value in raw_source_ids if value is not None
                }

        expected_match = expected_by_id.get(segment_id)
        if expected_match is not None:
            index, fallback_segment = expected_match
            if not _segment_audio_passes_quality(
                fallback_segment,
                path,
                backend_name,
            ):
                if _backend_requires_segment_quality(backend_name):
                    _invalidate_segment_cache(path)
                continue
            descriptor = _mix_segment_descriptor(path, fallback_segment, metadata)
            selected[path] = (index, descriptor, set(_segment_source_ids(descriptor)))
            continue

        if not source_ids:
            continue
        candidate_indices = [
            index
            for index, segment in enumerate(expected_segments)
            if source_ids.intersection(_segment_source_ids(segment))
        ]
        if not candidate_indices:
            continue
        fallback_segment = expected_segments[min(candidate_indices)]
        if not _segment_audio_passes_quality(
            fallback_segment,
            path,
            backend_name,
        ):
            if _backend_requires_segment_quality(backend_name):
                _invalidate_segment_cache(path)
            continue
        descriptor = _mix_segment_descriptor(path, fallback_segment, metadata)
        selected[path] = (
            min(candidate_indices),
            descriptor,
            set(_segment_source_ids(descriptor)),
        )

    covered_source_ids = {
        source_id
        for _, _, source_ids in selected.values()
        for source_id in source_ids
    }
    unresolved_source_ids = sorted(required_source_ids - covered_source_ids)
    if unresolved_source_ids:
        return [], [], missing_ids, []

    ordered = sorted(selected.values(), key=lambda item: item[0])
    return (
        [
            path
            for path, _ in sorted(
                selected.items(), key=lambda item: (item[1][0], item[0].name)
            )
        ],
        [item[1] for item in ordered],
        missing_ids,
        ["segment_cache_recovered_from_metadata"],
    )


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() <= 0:
            raise ValueError(f"WAV has an invalid sample rate: {path}")
        return wav_file.getnframes() / wav_file.getframerate()


def _protected_audio_source_segment_ids(script: dict[str, Any]) -> set[str]:
    original_segments = script.get("segments") or []
    if not isinstance(original_segments, list):
        return set()

    raw_plan = script.get("audioPlan") or {}
    raw_scenes = raw_plan.get("scenes", []) if isinstance(raw_plan, dict) else []
    if not isinstance(raw_scenes, list):
        return set()

    protected_indices: set[int] = set()
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, dict):
            continue
        for key in ("startSegmentIndex", "endSegmentIndex"):
            value = raw_scene.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                protected_indices.add(value)
        raw_sfx = raw_scene.get("sfx", [])
        if not isinstance(raw_sfx, list):
            continue
        for raw_effect in raw_sfx:
            if not isinstance(raw_effect, dict):
                continue
            value = raw_effect.get("anchorSegmentIndex")
            if isinstance(value, int) and not isinstance(value, bool):
                protected_indices.add(value)

    return {
        str(original_segments[index]["id"])
        for index in protected_indices
        if 0 <= index < len(original_segments)
        and isinstance(original_segments[index], dict)
        and original_segments[index].get("id") is not None
    }


def _tts_segments_for_request(
    original_segments: list[dict[str, Any]],
    request: dict[str, Any],
    protected_source_segment_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    max_words = int(request.get("maxMergedSegmentWords", DEFAULT_MAX_TTS_WORDS))
    max_characters = int(
        request.get("maxMergedSegmentCharacters", DEFAULT_MAX_TTS_CHARACTERS)
    )
    if request.get("mergeSegments", True):
        return merge_tts_segments(
            original_segments,
            max_words=max_words,
            max_characters=max_characters,
            protected_source_segment_ids=protected_source_segment_ids,
        )
    return split_tts_segments(
        original_segments,
        max_words=max_words,
        max_characters=max_characters,
    )


def _segment_artifact_payload(
    segment: dict[str, Any],
    artifact: Any,
    *,
    output_directory: Path,
    signature: str,
    backend_name: str,
    model_id: str | None,
    cache_segments: bool,
    device: str | None,
) -> dict[str, Any]:
    """Validate and serialize one TTS result without changing timeline order."""
    expected_audio_path = output_directory / f"{segment['id']}.wav"
    try:
        artifact_path = Path(artifact.path)
    except (TypeError, ValueError) as error:
        expected_audio_path.unlink(missing_ok=True)
        raise ValueError(
            f"TTS returned an invalid audio path for segment {segment['id']}: {error}"
        ) from error
    if artifact_path.resolve() != expected_audio_path.resolve() or not _is_readable_wav(
        expected_audio_path
    ):
        _invalidate_segment_cache(expected_audio_path)
        raise ValueError(
            f"TTS did not create a readable WAV for segment {segment['id']}."
        )
    if not _segment_audio_passes_quality(segment, expected_audio_path, backend_name):
        _invalidate_segment_cache(expected_audio_path)
        raise ValueError(
            f"TTS generated an unusable WAV for segment {segment['id']}."
        )

    source_segment_ids = segment.get("sourceSegmentIds", [segment["id"]])
    if cache_segments:
        _write_segment_cache_metadata(
            artifact.path,
            signature=signature,
            duration_seconds=artifact.duration_seconds,
            backend_name=backend_name,
            model_id=model_id,
            device=device,
            source_segment_ids=source_segment_ids,
        )
    return {
        "kind": artifact.kind,
        "path": str(artifact.path),
        "metadata": {
            "segmentId": segment["id"],
            "durationSeconds": artifact.duration_seconds,
            "device": device,
            "sourceSegmentIds": source_segment_ids,
            "cacheHit": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobook-worker",
        description="Run local audiobook processing worker commands.",
    )
    parser.add_argument("command", nargs="?", help="worker command to run")
    parser.add_argument("input_path", nargs="?", help="path to JSON worker input")
    parser.add_argument("output_path", nargs="?", help="path to write JSON worker output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.input_path is None or args.output_path is None:
        parser.error("command requires input_path and output_path")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    request = json.loads(input_path.read_text(encoding="utf-8"))
    try:
        payload = _dispatch(args.command, request)
        _write_json(output_path, payload)
        if payload.get("error", {}).get("code") == "unknown_command":
            return 2
        return 0 if payload.get("status") != "failed" else 1
    except KeyError as error:
        payload = _response(
            "failed",
            error={
                "code": "invalid_worker_input",
                "message": f"Missing required worker input field: {error.args[0]}",
            },
        )
        _write_json(output_path, payload)
        return 1



def _dispatch(command: str, request: dict[str, Any]) -> dict[str, Any]:
    handler = _dispatch_table.get(command)
    if handler is not None:
        return handler(request)
    return _response(
        "failed",
        error={
            "code": "unknown_command",
            "message": f"Unknown worker command: {command}",
        },
    )


def _extract_book(request: dict[str, Any]) -> dict[str, Any]:
    from audiobook_worker.chapters import detect_chapters
    from audiobook_worker.extract import extract_book_text
    from ebooklib import epub as epublib

    book_path = Path(request["bookPath"])
    title = book_path.stem
    if book_path.suffix.lower() == ".epub":
        try:
            book = epublib.read_epub(str(book_path))
            title_meta = book.get_metadata("DC", "title")
            if title_meta:
                title = title_meta[0][0]
        except Exception:
            pass

    result = extract_book_text(book_path)
    chapters = detect_chapters(result.text)

    output_dir = Path(request["outputDirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        (output_dir / f"{chapter.id}.txt").write_text(chapter.text, encoding="utf-8")

    return _response(
        "succeeded",
        warnings=result.warnings,
        artifacts=[
            {
                "kind": "book_extraction",
                "path": str(output_dir),
                "metadata": {
                    "title": title,
                    "chapterCount": len(chapters),
                    "chapters": [
                        {
                            "id": c.id,
                            "title": c.title,
                            "textLength": len(c.text),
                            "textPath": str(output_dir / f"{c.id}.txt"),
                        }
                        for c in chapters
                    ],
                    "requiresOcr": result.requires_ocr,
                },
            }
        ],
    )


def _analyze_chapter(request: dict[str, Any]) -> dict[str, Any]:
    chapter_text = Path(request["chapterTextPath"]).read_text(encoding="utf-8")
    chapter_id = str(request["chapterId"])
    language = resolve_text_language(chapter_text, request.get("language"))
    output_directory = Path(request["outputDirectory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    analysis_directory = _analysis_directory(output_directory, chapter_id)
    state_path = analysis_directory / "state.json"
    state = _read_json_object(state_path)
    input_signature = _analysis_input_signature(
        book_id=str(request["bookId"]),
        chapter_id=chapter_id,
        text=chapter_text,
        language=language,
        known_characters=request.get("knownCharacters"),
    )
    explicit_resume = request.get("resumeFromStage") or request.get("fromStage")
    if explicit_resume is not None and explicit_resume not in {
        *_ANALYSIS_STAGES,
        "script",
    }:
        return _response(
            "failed",
            error={
                "code": "invalid_analysis_stage",
                "message": "resumeFromStage must be characters, voice_design, speakers, delivery, voice_direction, or script",
            },
        )
    resume_from_stage = _analysis_resume_stage(
        state,
        input_signature=input_signature,
        explicit_stage=explicit_resume,
    )
    cached_stages = _load_cached_analysis_stages(
        analysis_directory,
        state,
        input_signature=input_signature,
        resume_from_stage=resume_from_stage,
    )

    if resume_from_stage is None:
        _clear_analysis_stage_state(state)
        for stage in _ANALYSIS_STAGES:
            (analysis_directory / f"{stage}.json").unlink(missing_ok=True)
        # These artifacts belong to a previous script/audio timeline and must
        # not be mistaken for the new chapter analysis.
        (analysis_directory / "transcript.json").unlink(missing_ok=True)
        (analysis_directory / "audio_plan.json").unlink(missing_ok=True)
        state.setdefault("workflow", {}).pop("generation", None)
    state["analysis"] = {
        "status": "running",
        "version": _ANALYSIS_CACHE_VERSION,
        "inputSignature": input_signature,
    }
    start_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        first_step=resume_from_stage or "characters",
    )
    for cached_stage in cached_stages:
        update_workflow(
            state,
            "analysis",
            ANALYSIS_WORKFLOW_STEPS,
            step=cached_stage,
            step_status="succeeded",
        )
    update_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        current_step=resume_from_stage or "characters",
        status="running",
    )
    _write_json(state_path, state)

    def publish_stage(stage: str, payload: dict[str, Any]) -> None:
        _write_json(analysis_directory / f"{stage}.json", payload)
        state[stage] = {
            "status": "succeeded",
            "artifact": str(analysis_directory / f"{stage}.json"),
        }
        state["lastCompletedStage"] = stage
        update_workflow(
            state,
            "analysis",
            ANALYSIS_WORKFLOW_STEPS,
            step=stage,
            step_status="succeeded",
        )
        stage_index = ANALYSIS_WORKFLOW_STEPS.index(stage)
        if stage_index + 1 < len(ANALYSIS_WORKFLOW_STEPS):
            update_workflow(
                state,
                "analysis",
                ANALYSIS_WORKFLOW_STEPS,
                step=ANALYSIS_WORKFLOW_STEPS[stage_index + 1],
                step_status="running",
            )
        _write_json(state_path, state)

    try:
        script = build_chapter_script(
            book_id=request["bookId"],
            chapter_id=chapter_id,
            title=request.get("title", chapter_id),
            text=chapter_text,
            language=language,
            analyzer=MockLLMAnalyzer()
            if request.get("mockLlm")
            else default_analyzer(str(request.get("llmModelId") or "") or None),
            known_characters=request.get("knownCharacters"),
            analysis_stage_callback=publish_stage,
            analysis_cached_stages=cached_stages,
            analysis_resume_from_stage=resume_from_stage,
            narrator_voice_id=request.get("narratorVoiceId"),
        )
        script_path = output_directory / f"{chapter_id}.json"
        _write_json(script_path, script)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        state = _read_json_object(state_path)
        state["analysis"] = {
            "status": "failed",
            "version": _ANALYSIS_CACHE_VERSION,
            "inputSignature": input_signature,
        }
        failed_stage = next(
            (
                stage
                for stage in _ANALYSIS_STAGES
                if _state_status(state, stage) != "succeeded"
            ),
            "script",
        )
        state["failedStage"] = failed_stage
        state["error"] = {"code": "chapter_analysis_failed", "message": str(error)}
        update_workflow(
            state,
            "analysis",
            ANALYSIS_WORKFLOW_STEPS,
            step=failed_stage,
            step_status="failed",
            error=str(error),
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={"code": "chapter_analysis_failed", "message": str(error)},
        )

    state = _read_json_object(state_path)
    state["analysis"] = {
        "status": "succeeded",
        "version": _ANALYSIS_CACHE_VERSION,
        "inputSignature": input_signature,
    }
    state["script"] = {"status": "succeeded", "artifact": str(script_path)}
    update_workflow(
        state,
        "analysis",
        ANALYSIS_WORKFLOW_STEPS,
        step="script",
        step_status="succeeded",
        status="succeeded",
        current_step=None,
    )
    state.pop("failedStage", None)
    state.pop("error", None)
    _write_json(state_path, state)
    _invalidate_chapter_audio_artifacts(output_directory, chapter_id)
    return _response(
        "succeeded",
        artifacts=[{"kind": "chapter_script", "path": str(script_path)}],
    )


def _transcribe_chapter_audio(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script = json.loads(script_path.read_text(encoding="utf-8"))
    chapter_id = str(script.get("chapterId") or script_path.stem)
    voice_path = Path(request["voiceAudioPath"])
    analysis_directory = Path(
        request.get("analysisDirectory")
        or _analysis_directory(script_path.parent, chapter_id)
    )
    analysis_directory.mkdir(parents=True, exist_ok=True)
    output_path = analysis_directory / "transcript.json"
    state_path = analysis_directory / "state.json"
    state = _read_json_object(state_path)
    if not isinstance(state.get("workflow", {}).get("generation"), dict):
        _start_generation_workflow(state, first_step="transcript")
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="voice",
            step_status="succeeded",
        )
    else:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="transcript",
            step_status="running",
        )
    _write_json(state_path, state)
    try:
        transcript = transcribe_audio(
            voice_path,
            model=str(request.get("whisperModel") or DEFAULT_WHISPER_MODEL),
            language=str(script.get("language") or ""),
            python_executable=(
                str(request["whisperPython"])
                if request.get("whisperPython")
                else None
            ),
        )
        _write_json(output_path, transcript)
    except (OSError, json.JSONDecodeError, KeyError, TranscriptionError) as error:
        state = _read_json_object(state_path)
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="transcript",
            step_status="failed",
            error=str(error),
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={"code": "transcription_failed", "message": str(error)},
        )
    state = _read_json_object(state_path)
    state["transcript"] = {"status": "succeeded", "artifact": str(output_path)}
    update_workflow(
        state,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        step="transcript",
        step_status="succeeded",
    )
    _write_json(state_path, state)
    return _response(
        "succeeded",
        artifacts=[
            {
                "kind": "chapter_transcript",
                "path": str(output_path),
                "metadata": {
                    "segmentCount": len(transcript.get("segments", [])),
                    "durationSeconds": transcript.get("durationSeconds"),
                    "model": transcript.get("model"),
                },
            }
        ],
    )


def _plan_chapter_audio(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script = json.loads(script_path.read_text(encoding="utf-8"))
    chapter_id = str(script.get("chapterId") or script_path.stem)
    transcript_path = Path(request.get("transcriptPath") or "")
    analysis_directory = Path(
        request.get("analysisDirectory")
        or _analysis_directory(script_path.parent, chapter_id)
    )
    analysis_directory.mkdir(parents=True, exist_ok=True)
    state_path = analysis_directory / "state.json"
    state = _read_json_object(state_path)
    if not isinstance(state.get("workflow", {}).get("generation"), dict):
        _start_generation_workflow(state, first_step="audio_plan")
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="voice",
            step_status="succeeded",
        )
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="transcript",
            step_status="succeeded",
        )
    else:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="running",
        )
    _write_json(state_path, state)
    if not transcript_path.is_file():
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="failed",
            error="请先完成原章节配音并转录 Whisper 文本。",
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={
                "code": "missing_transcript",
                "message": "请先完成原章节配音并转录 Whisper 文本。",
            },
        )
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_transcript_segments = transcript.get("segments", [])
    if not isinstance(raw_transcript_segments, list) or not raw_transcript_segments:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="failed",
            error="Whisper 转录结果为空。",
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={"code": "invalid_transcript", "message": "Whisper 转录结果为空。"},
        )

    analyzer = (
        MockLLMAnalyzer()
        if request.get("mockLlm")
        else default_analyzer(str(request.get("llmModelId") or "") or None)
    )
    language = str(script.get("language") or "")
    planner_segments = [
        {
            "segmentIndex": index,
            "id": segment.get("id"),
            "type": segment.get("type"),
            "text": segment.get("text", ""),
            "speakerId": segment.get("speakerId", "narrator"),
            "emotion": segment.get("emotion", "neutral"),
            "pace": segment.get("pace", "normal"),
        }
        for index, segment in enumerate(script.get("segments", []))
        if isinstance(segment, dict)
    ]
    planner_request = AudioPlanningRequest(
        book_id=str(script.get("bookId") or request.get("bookId") or ""),
        chapter_id=str(script.get("chapterId") or request.get("chapterId") or ""),
        text=str(request.get("chapterText") or ""),
        language=language,
        segments=planner_segments,
        transcript=[item for item in raw_transcript_segments if isinstance(item, dict)],
        characters=select_active_audio_characters(
            [item for item in script.get("characters", []) if isinstance(item, dict)],
            planner_segments,
        ),
    )
    if not planner_request.text:
        text_path = request.get("chapterTextPath")
        if isinstance(text_path, str) and text_path:
            planner_request = replace(
                planner_request,
                text=Path(text_path).read_text(encoding="utf-8"),
            )
        else:
            planner_request = replace(
                planner_request,
                text="\n".join(str(item.get("text", "")) for item in planner_request.segments),
            )
    planner_request = replace(
        planner_request,
        language=resolve_text_language(planner_request.text, planner_request.language),
    )

    explicit_audio_stage = request.get("audioPlanningFromStage") or request.get(
        "planningFromStage"
    )
    if explicit_audio_stage is not None:
        explicit_audio_stage = str(explicit_audio_stage)
    try:
        audio_input_signature = _audio_planning_input_signature(planner_request)
        audio_resume_stage = _audio_planning_resume_stage(
            state,
            input_signature=audio_input_signature,
            explicit_stage=explicit_audio_stage,
        )
        cached_audio_stages = _load_cached_audio_planning_stages(
            analysis_directory,
            state,
            input_signature=audio_input_signature,
            resume_from_stage=audio_resume_stage,
        )
    except ValueError as error:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="failed",
            error=str(error),
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={"code": "invalid_audio_planning_stage", "message": str(error)},
        )

    planning_metadata: dict[str, Any] = {
        "status": "running",
        "version": _AUDIO_PLANNING_CACHE_VERSION,
        "inputSignature": audio_input_signature,
        "stages": {},
        "currentStage": audio_resume_stage or "scene_structure",
    }
    for stage in cached_audio_stages:
        planning_metadata["stages"][stage] = {
            "status": "succeeded",
            "artifact": str(analysis_directory / f"audio_plan_{stage}.json"),
        }
    state["audioPlanning"] = planning_metadata
    state.pop("audioPlan", None)
    update_workflow(
        state,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        step="audio_plan",
        step_status="running",
        detail=(
            f"正在进行{_AUDIO_PLANNING_STAGE_LABELS.get(planning_metadata['currentStage'], '音频规划')}。"
        ),
    )
    _write_json(state_path, state)

    def publish_audio_stage(stage: str, payload: dict[str, Any]) -> None:
        artifact_path = analysis_directory / f"audio_plan_{stage}.json"
        _write_json(artifact_path, payload)
        current_state = _read_json_object(state_path)
        metadata = current_state.setdefault("audioPlanning", {})
        metadata.setdefault("stages", {})[stage] = {
            "status": "succeeded",
            "artifact": str(artifact_path),
        }
        metadata["lastCompletedStage"] = stage
        next_stage_index = _AUDIO_PLANNING_STAGES.index(stage) + 1
        next_stage = (
            _AUDIO_PLANNING_STAGES[next_stage_index]
            if next_stage_index < len(_AUDIO_PLANNING_STAGES)
            else None
        )
        metadata["currentStage"] = next_stage
        update_workflow(
            current_state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="running" if next_stage else "running",
            detail=(
                f"正在进行{_AUDIO_PLANNING_STAGE_LABELS[next_stage]}。"
                if next_stage
                else "音频规划阶段已完成，正在保存计划。"
            ),
        )
        _write_json(state_path, current_state)

    planner_request = replace(
        planner_request,
        stage_callback=publish_audio_stage,
        cached_stages=cached_audio_stages,
        resume_from_stage=audio_resume_stage,
    )

    try:
        plan = analyzer.plan_audio(planner_request)
        # Enforce the product invariant at the CLI boundary as well as inside
        # the real LLM stage.  This also protects mock/custom analyzers and
        # older callers that return a partial plan with silent dialogue gaps.
        plan = ensure_audio_music_coverage(
            plan,
            segment_count=len(planner_request.segments),
            language=planner_request.language,
            segments=planner_request.segments,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        state = _read_json_object(state_path)
        planning_metadata = state.setdefault("audioPlanning", {})
        planning_metadata["status"] = "failed"
        planning_metadata["error"] = str(error)
        stage_metadata = planning_metadata.get("stages")
        if not isinstance(stage_metadata, dict):
            stage_metadata = {}
            planning_metadata["stages"] = stage_metadata
        planning_metadata["failedStage"] = next(
            (
                stage
                for stage in _AUDIO_PLANNING_STAGES
                if not isinstance(stage_metadata.get(stage), dict)
                or stage_metadata[stage].get("status") != "succeeded"
            ),
            "finalize",
        )
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="audio_plan",
            step_status="failed",
            error=str(error),
        )
        _write_json(state_path, state)
        return _response(
            "failed",
            error={"code": "audio_planning_failed", "message": str(error)},
        )

    script["audioPlan"] = audio_plan_to_dict(plan)
    _write_json(script_path, script)
    plan_path = analysis_directory / "audio_plan.json"
    _write_json(plan_path, script["audioPlan"])
    _invalidate_chapter_mix_artifacts(script_path.parent, str(script.get("chapterId") or script_path.stem))
    state = _read_json_object(state_path)
    planning_metadata = state.setdefault("audioPlanning", {})
    planning_metadata["status"] = "succeeded"
    planning_metadata["currentStage"] = None
    planning_metadata.pop("failedStage", None)
    planning_metadata.pop("error", None)
    state["audioPlan"] = {"status": "succeeded", "artifact": str(plan_path)}
    update_workflow(
        state,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        step="audio_plan",
        step_status="succeeded",
    )
    _write_json(state_path, state)
    return _response(
        "succeeded",
        artifacts=[
            {"kind": "chapter_script", "path": str(script_path)},
            {"kind": "audio_plan", "path": str(plan_path)},
        ],
        warnings=["no_audio_assets"] if not plan.scenes else [],
    )


def _tts_backend_for_request(request: dict[str, Any]):
    """Build the requested TTS backend for direct and chapter synthesis.

    Keeping this in one place ensures direct voice previews, single-segment
    regeneration, and whole-chapter synthesis use exactly the same backend
    and reference-profile configuration.
    """

    backend_name = str(request.get("backend") or "mimo").strip().casefold()
    model_id = _tts_model_id_for_request(request)
    voice_profile_directory = request.get("voiceProfileDirectory")
    if backend_name == "parler":
        return ParlerTTSBackend(model_id) if model_id else ParlerTTSBackend()
    if backend_name == "mimo":
        mimo_kwargs: dict[str, Any] = {
            "model_id": model_id or DEFAULT_TTS_MODEL_ID
        }
        if isinstance(voice_profile_directory, str) and voice_profile_directory:
            mimo_kwargs["voice_profile_directory"] = voice_profile_directory
        return MiMoTTSBackend(**mimo_kwargs)
    if backend_name == "voxcpm2":
        voxcpm2_kwargs: dict[str, Any] = {
            "model_id": model_id or VOXCPM2_MODEL_ID
        }
        if isinstance(voice_profile_directory, str) and voice_profile_directory:
            voxcpm2_kwargs["voice_profile_directory"] = voice_profile_directory
        voxcpm2_kwargs["service_context"] = {
            "bookId": str(request.get("bookId") or ""),
            "chapterId": str(request.get("chapterId") or ""),
            "backend": backend_name,
            "modelId": model_id or VOXCPM2_MODEL_ID,
        }
        return VoxCPM2TTSBackend(**voxcpm2_kwargs)
    if backend_name == "kokoro":
        return KokoroTTSBackend()
    return MockTTSBackend()


def _synthesize_segment_audio(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script = apply_narrator_voice(
        json.loads(script_path.read_text(encoding="utf-8")),
        request.get("narratorVoiceId"),
    )
    segment = next(
        item for item in script["segments"] if item["id"] == request["segmentId"]
    )
    segment = _decorate_segments_for_voice(script_path, script, [segment])[0]
    backend_name = str(request.get("backend") or "mimo").strip().casefold()
    backend = _tts_backend_for_request(request)
    artifact = backend.synthesize_segment(segment, Path(request["outputDirectory"]))
    if not _segment_audio_passes_quality(segment, artifact.path, backend_name):
        _invalidate_segment_cache(artifact.path)
        return _response(
            "failed",
            error={
                "code": "invalid_segment_audio",
                "message": f"TTS generated an unusable WAV for segment {segment['id']}.",
                "details": {"segmentId": segment["id"]},
            },
        )
    return _response(
        "succeeded",
        artifacts=[
            {
                "kind": artifact.kind,
                "path": str(artifact.path),
                "metadata": {"durationSeconds": artifact.duration_seconds},
            }
        ],
    )


def _generate_audio_assets(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script = _read_json_object(script_path)
    chapter_id = str(script.get("chapterId") or request.get("chapterId") or script_path.stem)
    state_path, state = _workflow_state_for_script(script_path, chapter_id)
    if not isinstance(state.get("workflow", {}).get("generation"), dict):
        _start_generation_workflow(state, first_step="stable_audio")
    else:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="stable_audio",
            step_status="running",
        )
    _write_json(state_path, state)
    mixed_output_path = request.get("mixedOutputPath")
    if isinstance(mixed_output_path, str) and mixed_output_path:
        # Changing or regenerating an asset invalidates the previous final mix.
        Path(mixed_output_path).unlink(missing_ok=True)
    try:
        result = generate_audio_assets(
            Path(request["scriptPath"]),
            Path(request["outputDirectory"]),
            force=request.get("force") is True,
            asset_id=request.get("assetId"),
            asset_kind=request.get("assetKind"),
        )
    except StableAudioError as error:
        state = _read_json_object(state_path)
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="stable_audio",
            step_status="failed",
            error=str(error),
        )
        _write_json(state_path, state)
        partial_artifacts = [asset.to_artifact() for asset in error.partial_assets]
        manifest_path = Path(request["outputDirectory"]) / "manifest.json"
        if manifest_path.is_file():
            partial_artifacts.append(
                {
                    "kind": "stable_audio_manifest",
                    "path": str(manifest_path),
                    "metadata": {"partial": bool(partial_artifacts)},
                }
            )
        return _response(
            "failed",
            artifacts=partial_artifacts,
            error={
                "code": error.code,
                "message": str(error),
                "details": error.details,
            },
        )

    artifacts = [asset.to_artifact() for asset in result.assets]
    manifest_metadata: dict[str, Any] = {
        "assetCount": len(result.assets),
        "warningCount": len(result.warnings),
    }
    try:
        manifest_payload = _read_json_object(result.manifest_path)
    except (OSError, ValueError):
        manifest_payload = {}
    rejected_assets = manifest_payload.get("rejectedAssets")
    if isinstance(rejected_assets, dict):
        manifest_metadata["rejectedAssets"] = [
            {"assetKey": key, **value}
            for key, value in rejected_assets.items()
            if isinstance(key, str) and isinstance(value, dict)
        ]
    quality_fallbacks = manifest_payload.get("qualityFallbacks")
    if isinstance(quality_fallbacks, dict):
        manifest_metadata["qualityFallbacks"] = quality_fallbacks
    artifacts.append(
        {
            "kind": "stable_audio_manifest",
            "path": str(result.manifest_path),
            "metadata": manifest_metadata,
        }
    )
    state = _read_json_object(state_path)
    update_workflow(
        state,
        "generation",
        GENERATION_WORKFLOW_STEPS,
        step="stable_audio",
        step_status="succeeded",
    )
    _write_json(state_path, state)
    return _response(
        "succeeded",
        warnings=result.warnings,
        artifacts=artifacts,
    )


def _synthesize_chapter_audio(request: dict[str, Any]) -> dict[str, Any]:
    script = apply_narrator_voice(
        json.loads(Path(request["scriptPath"]).read_text(encoding="utf-8")),
        request.get("narratorVoiceId"),
    )
    script_path = Path(request["scriptPath"])
    chapter_id = str(script.get("chapterId") or request.get("chapterId") or script_path.stem)
    state_path, state = _workflow_state_for_script(script_path, chapter_id)
    _start_generation_workflow(state, first_step="voice")
    _write_json(state_path, state)

    def finish_voice(status: str, error: str | None = None) -> None:
        current = _read_json_object(state_path)
        update_workflow(
            current,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="voice",
            step_status=status,
            error=error,
        )
        _write_json(state_path, current)

    mixed_output_path = request.get("mixedOutputPath")
    if isinstance(mixed_output_path, str) and mixed_output_path:
        # A regenerated voice track invalidates the previous final mix. The
        # user must explicitly run mix_chapter_audio again.
        Path(mixed_output_path).unlink(missing_ok=True)
    original_segments = _decorate_segments_for_voice(
        script_path,
        script,
        script["segments"],
    )
    segments = _tts_segments_for_request(
        original_segments,
        request,
        _protected_audio_source_segment_ids(script),
    )
    output_directory = Path(request["outputDirectory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    # Do not leave a manifest from a previous segmentation run available if
    # this retry fails before the new cache is complete.
    _tts_timeline_manifest_path(output_directory).unlink(missing_ok=True)
    backend_name = str(request.get("backend") or "mimo").strip().casefold()
    model_id = _tts_model_id_for_request(request)
    cache_segments = request.get("cacheSegments", True)
    max_words = int(request.get("maxMergedSegmentWords", DEFAULT_MAX_TTS_WORDS))
    max_characters = int(
        request.get("maxMergedSegmentCharacters", DEFAULT_MAX_TTS_CHARACTERS)
    )
    backend = None

    def get_backend():
        nonlocal backend
        if backend is None:
            backend = _tts_backend_for_request(request)
        return backend

    artifacts_by_index: dict[int, dict[str, Any]] = {}
    pending_segments: list[tuple[int, dict[str, Any], str]] = []
    expected_audio_paths = {output_directory / f"{segment['id']}.wav" for segment in segments}
    for index, segment in enumerate(segments):
        expected_audio_path = output_directory / f"{segment['id']}.wav"
        signature = _segment_cache_signature(
            segment,
            backend_name,
            model_id,
            max_words=max_words,
            max_characters=max_characters,
        )
        if cache_segments:
            cached_artifact = _read_cached_segment_artifact(
                segment,
                output_directory,
                signature,
                backend_name=backend_name,
            )
            if cached_artifact is not None:
                artifacts_by_index[index] = cached_artifact
                continue

        # A failed retry must not leave an older artifact at the same path for
        # a later assembly step to mistake for the newly requested audio.
        _invalidate_segment_cache(expected_audio_path)
        pending_segments.append((index, segment, signature))

    active_backend = None
    if pending_segments:
        active_backend = get_backend()
        prepare_voice_profiles = getattr(active_backend, "prepare_voice_profiles", None)
        if backend_name == "mimo" and callable(prepare_voice_profiles):
            profile_directory = request.get("voiceProfileDirectory")
            try:
                prepare_voice_profiles(
                    [segment for _, segment, _ in pending_segments],
                    (
                        profile_directory
                        if isinstance(profile_directory, str)
                        else output_directory / ".voice-profiles"
                    ),
                )
            except Exception as error:
                failed_segment = pending_segments[0][1]
                finish_voice("failed", str(error))
                return _response(
                    "failed",
                    error={
                        "code": "tts_synthesis_failed",
                        "message": str(error),
                        "details": {"segmentId": failed_segment["id"]},
                    },
                )

        results: dict[int, Any] = {}
        failures: dict[int, Exception] = {}
        if backend_name == "voxcpm2":
            # VoxCPM2 submits the whole missing subset in one request. The
            # web-owned service owns one model and may batch independent items
            # across admitted chapters; cache hits never reach that service.
            try:
                synthesized = active_backend.synthesize_segments(
                    [
                        {
                            **segment,
                            "sourcePosition": index,
                            "cacheSignature": signature,
                        }
                        for index, segment, signature in pending_segments
                    ],
                    output_directory,
                )
            except Exception as error:
                failures[pending_segments[0][0]] = error
            else:
                if len(synthesized) != len(pending_segments):
                    failures[pending_segments[0][0]] = RuntimeError(
                        "VoxCPM2 did not return every requested segment."
                    )
                else:
                    results = {
                        index: artifact
                        for (index, _, _), artifact in zip(
                            pending_segments,
                            synthesized,
                            strict=True,
                        )
                    }
        else:
            # MiMo applies an account-wide rate/concurrency limit. Process every
            # segment in source order, even when the worker is invoked directly;
            # the backend additionally serializes the actual HTTP boundary across
            # all child processes.
            for index, segment, _ in pending_segments:
                try:
                    results[index] = active_backend.synthesize_segment(
                        segment,
                        output_directory,
                    )
                except Exception as error:
                    failures[index] = error
                    break

        if failures:
            failed_index = min(failures)
            failed_segment = segments[failed_index]
            expected_audio_path = output_directory / f"{failed_segment['id']}.wav"
            _invalidate_segment_cache(expected_audio_path)
            finish_voice("failed", str(failures[failed_index]))
            return _response(
                "failed",
                error={
                    "code": "tts_synthesis_failed",
                    "message": str(failures[failed_index]),
                    "details": {"segmentId": failed_segment["id"]},
                },
            )

        device = getattr(active_backend, "_device", None)
        for index, segment, signature in pending_segments:
            try:
                artifacts_by_index[index] = _segment_artifact_payload(
                    segment,
                    results[index],
                    output_directory=output_directory,
                    signature=signature,
                    backend_name=backend_name,
                    model_id=model_id,
                    cache_segments=cache_segments,
                    device=device,
                )
            except ValueError as error:
                finish_voice("failed", str(error))
                return _response(
                    "failed",
                    error={
                        "code": "invalid_segment_audio",
                        "message": str(error),
                        "details": {"segmentId": segment["id"]},
                    },
                )

    artifacts = [artifacts_by_index[index] for index in range(len(segments)) if index in artifacts_by_index]

    missing_audio = [
        segment["id"]
        for segment in segments
        if not _is_readable_wav(output_directory / f"{segment['id']}.wav")
    ]
    if missing_audio or len(artifacts) != len(segments):
        finish_voice("failed", "TTS did not produce every expected segment audio file.")
        return _response(
            "failed",
            error={
                "code": "incomplete_segment_audio",
                "message": "TTS did not produce every expected segment audio file.",
                "details": {
                    "missingSegmentIds": missing_audio,
                    "expectedCount": len(segments),
                    "actualCount": len(artifacts),
                },
            },
        )

    for stale_audio in output_directory.glob("*.wav"):
        if stale_audio not in expected_audio_paths:
            stale_audio.unlink()
            stale_metadata = _segment_cache_metadata_path(stale_audio)
            if stale_metadata.exists():
                stale_metadata.unlink()
    gap_seconds = float(request.get("gapSeconds", 0.5))
    _write_tts_timeline_manifest(
        output_directory,
        segments,
        backend_name=backend_name,
        model_id=model_id,
        max_words=max_words,
        max_characters=max_characters,
        gap_seconds=gap_seconds,
    )
    metadata: dict[str, Any] = {
        "originalSegmentCount": len(original_segments),
        "synthesizedSegmentCount": len(segments),
        "cachedSegmentCount": sum(1 for artifact in artifacts if artifact["metadata"].get("cacheHit")),
        "segmentIds": [segment["id"] for segment in segments],
    }
    if backend_name == "voxcpm2" and active_backend is not None:
        raw_service_metrics = getattr(active_backend, "service_metrics", None)
        if isinstance(raw_service_metrics, dict):
            metadata["voxcpm2Service"] = {
                name: value
                for name, value in raw_service_metrics.items()
                if isinstance(name, str)
                and isinstance(value, int)
                and not isinstance(value, bool)
            }
    payload = _response("succeeded", artifacts=artifacts)
    payload["metadata"] = metadata
    # The public "原章节配音" stage also includes the following assembly
    # command.  Keep it running until the complete chapter WAV exists.
    finish_voice("running", "片段配音已完成，正在组装原章节音频。")
    return payload


def _assemble_chapter_audio(request: dict[str, Any]) -> dict[str, Any]:
    segment_directory = Path(request["segmentAudioDirectory"])
    script_path = request.get("scriptPath")
    state_path: Path | None = None
    if isinstance(script_path, str) and script_path:
        script_file = Path(script_path)
        script = _read_json_object(script_file)
        chapter_id = str(script.get("chapterId") or request.get("chapterId") or script_file.stem)
        state_path, state = _workflow_state_for_script(script_file, chapter_id)
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="voice",
            step_status="running",
            detail="正在组装原章节音频。",
        )
        _write_json(state_path, state)

    def finish_assembly(status: str, error: str | None = None) -> None:
        if state_path is None:
            return
        state = _read_json_object(state_path)
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="voice",
            step_status=status,
            error=error,
        )
        _write_json(state_path, state)

    if script_path:
        try:
            script = apply_narrator_voice(
                json.loads(Path(script_path).read_text(encoding="utf-8")),
                request.get("narratorVoiceId"),
            )
            decorated_segments = _decorate_segments_for_voice(
                Path(script_path),
                script,
                script["segments"],
            )
            expected_segments = _tts_segments_for_request(
                decorated_segments,
                request,
                _protected_audio_source_segment_ids(script),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            finish_assembly("failed", str(error))
            return _response(
                "failed",
                error={
                    "code": "invalid_assembly_input",
                    "message": str(error),
                },
            )
        backend_name = str(request.get("backend") or "mimo").strip().casefold()
        segment_paths = [
            segment_directory / f"{segment['id']}.wav" for segment in expected_segments
        ]
        invalid_segments = [
            (segment, path)
            for segment, path in zip(expected_segments, segment_paths)
            if not _segment_audio_passes_quality(
                segment,
                path,
                backend_name,
            )
        ]
        if _backend_requires_segment_quality(backend_name):
            for _, path in invalid_segments:
                if path.is_file():
                    _invalidate_segment_cache(path)
        missing_audio = [
            segment["id"]
            for segment, _ in invalid_segments
        ]
        if missing_audio:
            finish_assembly("failed", "Cannot assemble the chapter because segment audio is missing or unreadable.")
            return _response(
                "failed",
                error={
                    "code": "incomplete_segment_audio",
                    "message": "Cannot assemble the chapter because segment audio is missing or unreadable.",
                    "details": {"missingSegmentIds": missing_audio},
                },
            )
    else:
        # Preserve the standalone assembly command used by older workflows.
        segment_paths = sorted(segment_directory.glob("*.wav"))

    gap_seconds = float(request.get("gapSeconds", 0.5))
    try:
        artifact = assemble_chapter_audio(
            segment_paths,
            Path(request["outputPath"]),
            gap_seconds=gap_seconds,
        )
    except (OSError, ValueError, wave.Error) as error:
        finish_assembly("failed", str(error))
        return _response(
            "failed",
            error={
                "code": "chapter_assembly_failed",
                "message": str(error),
                },
            )
    if script_path and "expected_segments" in locals():
        _write_tts_timeline_manifest(
            segment_directory,
            expected_segments,
            backend_name=str(request.get("backend") or "mimo"),
            model_id=_tts_model_id_for_request(request),
            max_words=int(
                request.get("maxMergedSegmentWords", DEFAULT_MAX_TTS_WORDS)
            ),
            max_characters=int(
                request.get(
                    "maxMergedSegmentCharacters", DEFAULT_MAX_TTS_CHARACTERS
                )
            ),
            gap_seconds=gap_seconds,
            assembled_audio_path=Path(request["outputPath"]),
            assembled_duration_seconds=artifact.duration_seconds,
        )
    finish_assembly("succeeded")
    return _response(
        "succeeded",
        artifacts=[
            {
                "kind": artifact.kind,
                "path": str(artifact.path),
                "metadata": {"durationSeconds": artifact.duration_seconds},
            }
        ],
    )


def _mix_chapter_audio(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script_snapshot = _read_json_object(script_path)
    chapter_id = str(script_snapshot.get("chapterId") or request.get("chapterId") or script_path.stem)
    state_path, state = _workflow_state_for_script(script_path, chapter_id)
    if not isinstance(state.get("workflow", {}).get("generation"), dict):
        _start_generation_workflow(state, first_step="mix")
    else:
        update_workflow(
            state,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="mix",
            step_status="running",
        )
    _write_json(state_path, state)

    def finish_mix(status: str, error: str | None = None) -> None:
        current = _read_json_object(state_path)
        update_workflow(
            current,
            "generation",
            GENERATION_WORKFLOW_STEPS,
            step="mix",
            step_status=status,
            error=error,
        )
        _write_json(state_path, current)

    try:
        script = apply_narrator_voice(
            json.loads(script_path.read_text(encoding="utf-8")),
            request.get("narratorVoiceId"),
        )
        decorated_segments = _decorate_segments_for_voice(
            script_path,
            script,
            script["segments"],
        )
        expected_segments = _tts_segments_for_request(
            decorated_segments,
            request,
            _protected_audio_source_segment_ids(script),
        )
        backend_name = str(request.get("backend") or "mimo").strip().casefold()
        model_id = _tts_model_id_for_request(request)
        max_words = int(
            request.get("maxMergedSegmentWords", DEFAULT_MAX_TTS_WORDS)
        )
        max_characters = int(
            request.get(
                "maxMergedSegmentCharacters", DEFAULT_MAX_TTS_CHARACTERS
            )
        )
        segment_directory = Path(request["segmentAudioDirectory"])
        segment_paths, tts_segments, missing_audio, recovery_warnings = (
            _resolve_mix_segment_audio(
                expected_segments,
                segment_directory,
                backend_name=backend_name,
            )
        )
        if not segment_paths:
            finish_mix("failed", "Cannot mix the chapter because segment audio is missing or unreadable.")
            return _response(
                "failed",
                error={
                    "code": "incomplete_segment_audio",
                    "message": "Cannot mix the chapter because segment audio is missing or unreadable.",
                    "details": {"missingSegmentIds": missing_audio},
                },
            )

        gap_seconds = float(request.get("gapSeconds", 0.5))
        voice_duration = _wav_duration_seconds(Path(request["voiceAudioPath"]))
        timeline_duration = sum(
            _wav_duration_seconds(path) for path in segment_paths
        ) + gap_seconds * max(0, len(segment_paths) - 1)
        if abs(voice_duration - timeline_duration) > 0.25:
            manifest = _read_tts_timeline_manifest(segment_directory)
            manifest_is_usable = (
                len(segment_paths) == len(expected_segments)
                and _manifest_matches_expected_timeline(
                    manifest,
                    expected_segments,
                    backend_name=backend_name,
                    model_id=model_id,
                    max_words=max_words,
                    max_characters=max_characters,
                    gap_seconds=gap_seconds,
                )
                and all(_is_readable_wav(path) for path in segment_paths)
            )
            cache_is_usable = _cached_segment_files_match_expected(
                expected_segments,
                segment_paths,
                backend_name=backend_name,
                model_id=model_id,
                max_words=max_words,
                max_characters=max_characters,
            )
            # Very old runs have neither sidecars nor a manifest. If every
            # current segment file is present, the only safe recovery still
            # available is to rebuild the assembled container from those
            # files; this avoids forcing a fresh TTS request for legacy books.
            legacy_cache_is_usable = (
                len(segment_paths) == len(expected_segments)
                and all(_is_readable_wav(path) for path in segment_paths)
                and not any(
                    _read_segment_cache_metadata(path) is not None
                    for path in segment_paths
                )
            )
            if manifest_is_usable or cache_is_usable or legacy_cache_is_usable:
                # The TTS WAVs are valid and belong to the current script. The
                # old assembly step may nevertheless have used a different
                # merge decision; rebuild only the chapter container and keep
                # every cached TTS file.
                repaired = assemble_chapter_audio(
                    segment_paths,
                    Path(request["voiceAudioPath"]),
                    gap_seconds=gap_seconds,
                )
                voice_duration = repaired.duration_seconds
                timeline_duration = sum(
                    _wav_duration_seconds(path) for path in segment_paths
                ) + gap_seconds * max(0, len(segment_paths) - 1)
                _write_tts_timeline_manifest(
                    segment_directory,
                    expected_segments,
                    backend_name=backend_name,
                    model_id=model_id,
                    max_words=max_words,
                    max_characters=max_characters,
                    gap_seconds=gap_seconds,
                    assembled_audio_path=Path(request["voiceAudioPath"]),
                    assembled_duration_seconds=voice_duration,
                )
                recovery_warnings.append(
                    "voice_timeline_reassembled_from_cached_segments"
                )
        if abs(voice_duration - timeline_duration) > 0.25:
            finish_mix(
                "failed",
                "The cached segment timeline does not match the original chapter audio. Please regenerate the original chapter audio first.",
            )
            return _response(
                "failed",
                error={
                    "code": "segment_timeline_mismatch",
                    "message": "The cached segment timeline does not match the original chapter audio. Please regenerate the original chapter audio first.",
                    "details": {
                        "voiceDurationSeconds": round(voice_duration, 3),
                        "segmentTimelineDurationSeconds": round(timeline_duration, 3),
                        "missingSegmentIds": missing_audio,
                    },
                },
            )

        artifact = mix_chapter_audio(
            request["voiceAudioPath"],
            segment_paths,
            tts_segments,
            script,
            request["audioAssetsDirectory"],
            request["outputPath"],
            gap_seconds=gap_seconds,
            voice_gain=float(request.get("voiceGain", DEFAULT_VOICE_GAIN)),
            music_gain=float(request.get("musicGain", DEFAULT_MUSIC_GAIN)),
            sfx_gain=float(request.get("sfxGain", DEFAULT_SFX_GAIN)),
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        wave.Error,
    ) as error:
        finish_mix("failed", str(error))
        return _response(
            "failed",
            error={
                "code": "chapter_mixing_failed",
                "message": str(error),
            },
        )
    finish_mix("succeeded")
    return _response(
        "succeeded",
        warnings=[
            *recovery_warnings,
            *artifact.metadata.get("warnings", []),
        ],
        artifacts=[
            {
                "kind": artifact.kind,
                "path": str(artifact.path),
                "metadata": {
                    "durationSeconds": artifact.duration_seconds,
                    **artifact.metadata,
                },
            }
        ],
    )


def _apply_corrections(request: dict[str, Any]) -> dict[str, Any]:
    output_directory = Path(request["outputDirectory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    corrections = request.get("corrections", {})
    analyzer = (
        MockLLMAnalyzer()
        if request.get("mockLlm")
        else default_analyzer(str(request.get("llmModelId") or "") or None)
    )

    artifacts = []
    for chapter in request["chapters"]:
        chapter_text = Path(chapter["textPath"]).read_text(encoding="utf-8")
        language = resolve_text_language(chapter_text, request.get("language"))
        script = build_chapter_script_with_corrections(
            book_id=request["bookId"],
            chapter_id=chapter["chapterId"],
            title=chapter.get("title", chapter["chapterId"]),
            text=chapter_text,
            language=language,
            corrections=corrections,
            analyzer=analyzer,
            known_characters=request.get("knownCharacters"),
            narrator_voice_id=request.get("narratorVoiceId"),
        )
        script_path = output_directory / f"{chapter['chapterId']}.json"
        _write_json(script_path, script)
        _invalidate_chapter_audio_artifacts(output_directory, chapter["chapterId"])
        artifacts.append({
            "kind": "chapter_script",
            "path": str(script_path),
            "metadata": {
                "chapterId": chapter["chapterId"],
                "characterCount": len(script.get("characters", [])),
                "segmentCount": len(script.get("segments", [])),
            },
        })

    return _response("succeeded", artifacts=artifacts)


def _refresh_voice_assignments(request: dict[str, Any]) -> dict[str, Any]:
    script_path = Path(request["scriptPath"])
    script = json.loads(script_path.read_text(encoding="utf-8"))
    refreshed = refresh_script_voice_assignments(
        script,
        force_legacy_auto=bool(request.get("forceLegacyAuto", False)),
        narrator_voice_id=request.get("narratorVoiceId"),
    )
    output_path = Path(request.get("outputPath") or script_path)
    _write_json(output_path, refreshed)
    _invalidate_chapter_audio_artifacts(output_path.parent, str(refreshed.get("chapterId") or output_path.stem))
    return _response(
        "succeeded",
        artifacts=[
            {
                "kind": "chapter_script",
                "path": str(output_path),
                "metadata": {
                    "chapterId": refreshed.get("chapterId"),
                    "characterCount": len(refreshed.get("characters", [])),
                    "segmentCount": len(refreshed.get("segments", [])),
                },
            }
        ],
    )


def _read_file(request: dict[str, Any]) -> dict[str, Any]:
    content = Path(request["path"]).read_text(encoding="utf-8")
    return json.loads(content)


def _write_file(request: dict[str, Any]) -> dict[str, Any]:
    path = Path(request["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(request["content"], encoding="utf-8")
    return _response("succeeded")


def _check_rights(request: dict[str, Any]) -> dict[str, Any]:
    from audiobook_worker.rights import classify_rights

    result = classify_rights(
        input_path=Path(request["bookPath"]),
        metadata=request.get("metadata", {}),
    )
    payload = _response("succeeded")
    payload["classification"] = result.classification
    payload["reason"] = result.reason
    payload["requiresAttestation"] = result.requires_attestation
    payload["evidence"] = result.evidence
    return payload


def _list_voices(request: dict[str, Any]) -> dict[str, Any]:
    payload = _response("succeeded")
    payload["voices"] = voice_options(request.get("backend", "mimo"))
    return payload


def _convert_to_mp3(request: dict[str, Any]) -> dict[str, Any]:
    """Convert a WAV file to MP3 using ffmpeg and write to the given output path."""
    import shutil
    import subprocess
    wav_path = Path(request["wavPath"])
    out_path = Path(request["outputPath"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return _response("failed", error={"message": "ffmpeg not found in PATH"})
    bitrate_kbps = request.get("bitrateKbps")
    codec_options = ["-q:a", "2"]
    if bitrate_kbps is not None:
        if isinstance(bitrate_kbps, bool) or not isinstance(bitrate_kbps, int):
            return _response("failed", error={"message": "invalid MP3 bitrate"})
        normalized_bitrate = bitrate_kbps
        if normalized_bitrate not in {128, 192, 256, 320}:
            return _response("failed", error={"message": "unsupported MP3 bitrate"})
        # Batch exports intentionally use CBR so the selected bitrate is exact.
        codec_options = ["-b:a", f"{normalized_bitrate}k"]
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", *codec_options, str(out_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        msg = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1]
        return _response("failed", error={"message": msg})
    return _response("succeeded", artifacts=[{"kind": "mp3", "path": str(out_path)}])


_dispatch_table: dict[str, Any] = {
    "extract_book": _extract_book,
    "analyze_chapter": _analyze_chapter,
    "transcribe_chapter_audio": _transcribe_chapter_audio,
    "plan_chapter_audio": _plan_chapter_audio,
    "synthesize_segment_audio": _synthesize_segment_audio,
    "generate_audio_assets": _generate_audio_assets,
    "synthesize_chapter_audio": _synthesize_chapter_audio,
    "assemble_chapter_audio": _assemble_chapter_audio,
    "mix_chapter_audio": _mix_chapter_audio,
    "apply_corrections": _apply_corrections,
    "refresh_voice_assignments": _refresh_voice_assignments,
    "_read_file": _read_file,
    "_write_file": _write_file,
    "check_rights": _check_rights,
    "list_voices": _list_voices,
    "convert_to_mp3": _convert_to_mp3,
}


if __name__ == "__main__":
    raise SystemExit(main())
