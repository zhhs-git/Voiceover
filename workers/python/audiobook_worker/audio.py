from __future__ import annotations

import json
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audiobook_worker.audio_asset_ids import normalize_script_audio_asset_ids


DEFAULT_VOICE_GAIN = 1.412538  # +3 dB relative to the assembled voice track
DEFAULT_MUSIC_GAIN = 0.40  # about +2 dB from the normalized 0.32x baseline
DEFAULT_SFX_GAIN = 0.35
DEFAULT_MUSIC_CROSSFADE_SECONDS = 0.75


@dataclass(frozen=True)
class ChapterAudioArtifact:
    kind: str
    path: Path
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioTimelineSegment:
    """One synthesized WAV's position in the chapter timeline."""

    segment_id: str
    source_segment_ids: tuple[str, ...]
    path: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class AudioMixTrack:
    """A Stable Audio asset scheduled on the chapter timeline."""

    kind: str
    asset_id: str
    scene_id: str
    path: Path
    start_seconds: float
    duration_seconds: float
    loop: bool
    gain: float
    fade_in_seconds: float
    fade_out_seconds: float


@dataclass(frozen=True)
class AudioMixPlan:
    timeline: tuple[AudioTimelineSegment, ...]
    tracks: tuple[AudioMixTrack, ...]
    duration_seconds: float
    warnings: tuple[str, ...]


def assemble_chapter_audio(
    segment_paths: list[Path],
    output_path: Path | str,
    *,
    gap_seconds: float = 0.5,
) -> ChapterAudioArtifact:
    if not segment_paths:
        raise ValueError("at least one segment audio path is required")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = None
    total_frames = 0
    with wave.open(str(output), "wb") as out_wav:
        for index, segment_path in enumerate(segment_paths):
            with wave.open(str(segment_path), "rb") as in_wav:
                if params is None:
                    params = in_wav.getparams()
                    out_wav.setparams(params)
                elif in_wav.getparams()[:3] != params[:3]:
                    raise ValueError("all segment WAV files must share channels, sample width, and frame rate")

                frames = in_wav.readframes(in_wav.getnframes())
                out_wav.writeframes(frames)
                total_frames += in_wav.getnframes()

            if index < len(segment_paths) - 1 and params is not None:
                gap_frames = int(params.framerate * gap_seconds)
                out_wav.writeframes(b"\x00" * params.sampwidth * params.nchannels * gap_frames)
                total_frames += gap_frames

    assert params is not None
    return ChapterAudioArtifact(
        kind="chapter_audio",
        path=output,
        duration_seconds=total_frames / params.framerate,
    )


def build_audio_mix_plan(
    segment_paths: list[Path],
    tts_segments: list[dict[str, Any]],
    script: dict[str, Any],
    asset_directory: Path | str,
    *,
    gap_seconds: float = 0.5,
    music_gain: float = DEFAULT_MUSIC_GAIN,
    sfx_gain: float = DEFAULT_SFX_GAIN,
    music_crossfade_seconds: float = DEFAULT_MUSIC_CROSSFADE_SECONDS,
) -> AudioMixPlan:
    """Convert the script's segment-index cues into real audio timestamps.

    The LLM writes indexes into the original script. TTS may merge or split
    those segments, so this function first builds timing from the actual WAV
    durations and then uses each synthesized item's ``sourceSegmentIds`` to
    map the indexes back to the timeline.
    """

    if len(segment_paths) != len(tts_segments):
        raise ValueError(
            "segment audio count does not match the synthesized TTS segment count"
        )
    if gap_seconds < 0 or not math.isfinite(gap_seconds):
        raise ValueError("gap_seconds must be a finite non-negative number")
    if music_gain < 0 or sfx_gain < 0:
        raise ValueError("audio gains must be non-negative")
    if music_crossfade_seconds < 0 or not math.isfinite(music_crossfade_seconds):
        raise ValueError(
            "music_crossfade_seconds must be a finite non-negative number"
        )

    # Keep direct mix callers compatible with older plans that reused an SFX
    # ID in multiple scenes, even if the plan was not regenerated first.
    normalize_script_audio_asset_ids(script)

    timeline: list[AudioTimelineSegment] = []
    cursor = 0.0
    for index, (segment_path, tts_segment) in enumerate(
        zip(segment_paths, tts_segments, strict=True)
    ):
        duration = _wav_duration(segment_path)
        segment_id = str(tts_segment.get("id") or f"tts_{index:04d}")
        source_ids = tuple(
            str(source_id)
            for source_id in (
                tts_segment.get("sourceSegmentIds") or [tts_segment.get("id")]
            )
            if source_id is not None
        )
        if not source_ids:
            source_ids = (segment_id,)
        start = cursor
        end = start + duration
        timeline.append(
            AudioTimelineSegment(
                segment_id=segment_id,
                source_segment_ids=source_ids,
                path=Path(segment_path),
                start_seconds=start,
                end_seconds=end,
            )
        )
        cursor = end + (gap_seconds if index < len(segment_paths) - 1 else 0.0)

    source_ranges: dict[str, tuple[float, float]] = {}
    for item in timeline:
        for source_id in item.source_segment_ids:
            previous = source_ranges.get(source_id)
            if previous is None:
                source_ranges[source_id] = (item.start_seconds, item.end_seconds)
            else:
                source_ranges[source_id] = (
                    min(previous[0], item.start_seconds),
                    max(previous[1], item.end_seconds),
                )

    original_segments = script.get("segments") or []
    if not isinstance(original_segments, list):
        raise ValueError("script.segments must be an array")
    original_ids = [str(item.get("id")) for item in original_segments if isinstance(item, dict)]
    original_texts = [
        str(item.get("text") or "") if isinstance(item, dict) else ""
        for item in original_segments
    ]
    asset_root = Path(asset_directory)
    manifest = _read_audio_asset_manifest(asset_root)
    warnings: list[str] = []
    tracks: list[AudioMixTrack] = []

    def interval_for_index(raw_index: Any, label: str) -> tuple[float, float] | None:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            warnings.append(f"invalid_{label}_index")
            return None
        if raw_index < 0 or raw_index >= len(original_ids):
            warnings.append(f"out_of_range_{label}_index:{raw_index}")
            return None
        source_id = original_ids[raw_index]
        interval = source_ranges.get(source_id)
        if interval is None:
            warnings.append(f"missing_tts_timeline_segment:{source_id}")
        return interval

    def event_interval_for_effect(
        raw_index: Any,
        anchor_text: Any,
        label: str,
    ) -> tuple[float, float] | None:
        """Refine an SFX anchor to the cue text inside its TTS segment.

        The segment index remains the authoritative mapping.  When the planner
        also supplies an exact ``anchorText`` (for example ``吱呀``), use its
        character position to avoid placing a cue at the beginning of a long
        narration segment.  This is deliberately a proportional estimate: TTS
        timing is not available until after synthesis, but it is much more
        accurate than treating every event as the segment's first millisecond.
        """
        interval = interval_for_index(raw_index, label)
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            return None
        if raw_index < 0 or raw_index >= len(original_texts):
            return None
        cue = str(anchor_text or "").strip()
        if not cue:
            return interval
        match = _find_audio_anchor_match(cue, original_texts, raw_index)
        if match is None:
            warnings.append(
                f"unmatched_sfx_anchor_text:{label.removeprefix('sfx_anchor:')}"
            )
            return interval
        refined_interval = _interval_for_audio_anchor_match(
            match,
            original_ids,
            source_ranges,
        )
        if refined_interval is None:
            warnings.append(
                f"unmapped_sfx_anchor_text:{label.removeprefix('sfx_anchor:')}"
            )
            return interval
        if match[0] != raw_index:
            warnings.append(
                f"repaired_sfx_anchor_index:{label.removeprefix('sfx_anchor:')}"
                f":{raw_index}->{match[0]}"
            )
        if match[0] != match[1]:
            warnings.append(
                f"sfx_anchor_spans_segments:{label.removeprefix('sfx_anchor:')}"
            )
        return refined_interval

    raw_plan = script.get("audioPlan") or {}
    raw_scenes = raw_plan.get("scenes", []) if isinstance(raw_plan, dict) else []
    if not isinstance(raw_scenes, list):
        warnings.append("invalid_audio_plan_scenes")
        raw_scenes = []

    # Resolve scene boundaries before scheduling music. A scene boundary is
    # expressed in source-segment indexes, while the mixer works in the actual
    # synthesized timeline.
    resolved_scenes: list[
        tuple[int, dict[str, Any], str, float, float, Path | None]
    ] = []
    for scene_index, raw_scene in enumerate(raw_scenes):
        if not isinstance(raw_scene, dict):
            warnings.append(f"invalid_scene:{scene_index}")
            continue
        scene_id = str(raw_scene.get("id") or f"scene_{scene_index + 1:03d}")
        scene_start = interval_for_index(
            raw_scene.get("startSegmentIndex"), f"scene_start:{scene_id}"
        )
        scene_end = interval_for_index(
            raw_scene.get("endSegmentIndex"), f"scene_end:{scene_id}"
        )
        if scene_start is None or scene_end is None:
            continue
        scene_start_seconds = scene_start[0]
        scene_end_seconds = max(scene_start_seconds, scene_end[1])

        raw_music = raw_scene.get("music")
        asset_path: Path | None = None
        if (
            isinstance(raw_music, dict)
            and not (
                isinstance(raw_scene.get("musicVariants"), list)
                and raw_scene.get("musicVariants")
            )
            and scene_end_seconds > scene_start_seconds
        ):
            asset_id = str(scene_id)
            asset_path = _resolve_audio_asset_path(
                asset_root, manifest, "music", asset_id
            )
            if asset_path is None:
                warnings.append(f"missing_music_asset:{asset_id}")

        resolved_scenes.append(
            (
                scene_index,
                raw_scene,
                scene_id,
                scene_start_seconds,
                scene_end_seconds,
                asset_path,
            )
            )

    chapter_duration = cursor
    scheduled_music: list[dict[str, Any]] = []

    def break_intervals(
        raw_scene: dict[str, Any],
        scene_start_seconds: float,
        scene_end_seconds: float,
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        scene_start_index = raw_scene.get("startSegmentIndex")
        scene_end_index = raw_scene.get("endSegmentIndex")
        raw_breaks = raw_scene.get("musicBreaks", [])
        if not isinstance(raw_breaks, list):
            return intervals
        for break_index, raw_break in enumerate(raw_breaks):
            if not isinstance(raw_break, dict):
                continue
            after_index = raw_break.get("afterSegmentIndex")
            duration = raw_break.get("durationSeconds")
            if (
                isinstance(after_index, bool)
                or not isinstance(after_index, int)
                or not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
            ):
                warnings.append(f"invalid_music_break:{break_index}")
                continue
            if not 2.0 <= float(duration) <= 6.0:
                warnings.append(f"invalid_music_break_duration:{after_index}")
                continue
            if (
                isinstance(scene_start_index, bool)
                or not isinstance(scene_start_index, int)
                or isinstance(scene_end_index, bool)
                or not isinstance(scene_end_index, int)
                or not scene_start_index <= after_index < scene_end_index
            ):
                warnings.append(f"music_break_outside_scene:{after_index}")
                continue
            anchor = interval_for_index(
                after_index, f"music_break:{after_index}"
            )
            if anchor is None:
                continue
            start = max(scene_start_seconds, anchor[1])
            end = min(scene_end_seconds, start + float(duration))
            if end > start:
                intervals.append((start, end))
        return sorted(intervals)

    def subtract_breaks(
        start_seconds: float,
        end_seconds: float,
        breaks: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        pieces: list[tuple[float, float]] = []
        cursor_seconds = start_seconds
        for break_start, break_end in breaks:
            if break_end <= cursor_seconds or break_start >= end_seconds:
                continue
            if break_start > cursor_seconds:
                pieces.append((cursor_seconds, min(break_start, end_seconds)))
            cursor_seconds = max(cursor_seconds, break_end)
            if cursor_seconds >= end_seconds:
                break
        if cursor_seconds < end_seconds:
            pieces.append((cursor_seconds, end_seconds))
        return [
            (start, end)
            for start, end in pieces
            if end - start > 0.01
        ]

    for (
        _scene_index,
        raw_scene,
        scene_id,
        scene_start_seconds,
        scene_end_seconds,
        legacy_asset_path,
    ) in resolved_scenes:
        breaks = break_intervals(
            raw_scene,
            scene_start_seconds,
            scene_end_seconds,
        )
        raw_variants = raw_scene.get("musicVariants")
        variant_paths: dict[str, Path] = {}
        if isinstance(raw_variants, list) and raw_variants:
            for raw_variant in raw_variants:
                if not isinstance(raw_variant, dict):
                    continue
                variant_id = str(raw_variant.get("id") or "").strip()
                if not variant_id:
                    continue
                path = _resolve_audio_asset_path(
                    asset_root,
                    manifest,
                    "music",
                    variant_id,
                )
                if path is None:
                    warnings.append(f"missing_music_asset:{variant_id}")
                    continue
                variant_paths[variant_id] = path

            raw_cues = raw_scene.get("musicCues", [])
            cues = raw_cues if isinstance(raw_cues, list) else []
            scene_start_index = raw_scene.get("startSegmentIndex")
            scene_end_index = raw_scene.get("endSegmentIndex")
            preferred_id = next(
                (
                    str(raw_variant.get("id"))
                    for raw_variant in raw_variants
                    if isinstance(raw_variant, dict)
                    and str(raw_variant.get("level") or "").lower() == "low"
                    and str(raw_variant.get("id") or "").strip() in variant_paths
                ),
                next(iter(variant_paths), None),
            )

            # A malformed cue must not create an accidental hole or overlap in
            # the final mix.  Source-segment cue ranges are normalized here;
            # intentional silence is represented only by musicBreaks, which
            # are subtracted from these ranges below.
            valid_cues: list[dict[str, Any]] = []
            cue_candidates: list[tuple[int, int, str, int]] = []
            for cue_index, cue in enumerate(cues):
                if not isinstance(cue, dict):
                    warnings.append(f"invalid_music_cue:{scene_id}:{cue_index}")
                    continue
                start_index = cue.get("startSegmentIndex")
                end_index = cue.get("endSegmentIndex")
                variant_id = str(cue.get("variantId") or "").strip()
                if (
                    isinstance(start_index, bool)
                    or not isinstance(start_index, int)
                    or isinstance(end_index, bool)
                    or not isinstance(end_index, int)
                    or not isinstance(scene_start_index, int)
                    or not isinstance(scene_end_index, int)
                    or start_index < scene_start_index
                    or end_index > scene_end_index
                    or end_index < start_index
                    or variant_id not in variant_paths
                ):
                    warnings.append(f"invalid_music_cue:{scene_id}:{cue_index}")
                    continue
                cue_candidates.append((start_index, end_index, variant_id, cue_index))

            cue_candidates.sort(key=lambda item: (item[0], item[1], item[3]))
            cue_cursor = scene_start_index if isinstance(scene_start_index, int) else 0

            def append_fallback_cue(start_index: int, end_index: int) -> None:
                if preferred_id is None or start_index > end_index:
                    return
                valid_cues.append({
                    "startSegmentIndex": start_index,
                    "endSegmentIndex": end_index,
                    "variantId": preferred_id,
                })
                warnings.append(
                    f"filled_music_cue_gap:{scene_id}:{start_index}-{end_index}"
                )

            for start_index, end_index, variant_id, cue_index in cue_candidates:
                if start_index < cue_cursor:
                    warnings.append(f"overlapping_music_cue:{scene_id}:{cue_index}")
                    if end_index < cue_cursor:
                        continue
                    start_index = cue_cursor
                if start_index > cue_cursor:
                    append_fallback_cue(cue_cursor, start_index - 1)
                if end_index < start_index:
                    continue
                valid_cues.append({
                    "startSegmentIndex": start_index,
                    "endSegmentIndex": end_index,
                    "variantId": variant_id,
                })
                cue_cursor = end_index + 1

            if (
                isinstance(scene_end_index, int)
                and cue_cursor <= scene_end_index
            ):
                append_fallback_cue(cue_cursor, scene_end_index)

            for cue in valid_cues:
                start_interval = interval_for_index(
                    cue.get("startSegmentIndex"),
                    f"music_cue_start:{scene_id}",
                )
                end_interval = interval_for_index(
                    cue.get("endSegmentIndex"),
                    f"music_cue_end:{scene_id}",
                )
                if start_interval is None or end_interval is None:
                    continue
                cue_start = max(scene_start_seconds, start_interval[0])
                cue_end = min(scene_end_seconds, end_interval[1])
                if cue_end <= cue_start:
                    continue
                asset_id = str(cue.get("variantId") or "").strip()
                for piece_start, piece_end in subtract_breaks(
                    cue_start,
                    cue_end,
                    breaks,
                ):
                    scheduled_music.append(
                        {
                            "asset_id": asset_id,
                            "scene_id": scene_id,
                            "path": variant_paths[asset_id],
                            "start": piece_start,
                            "end": piece_end,
                            "legacy_bridge": False,
                        }
                    )
            continue

        if legacy_asset_path is None:
            continue
        for piece_start, piece_end in subtract_breaks(
            scene_start_seconds,
            scene_end_seconds,
            breaks,
        ):
            scheduled_music.append(
                {
                    "asset_id": scene_id,
                    "scene_id": scene_id,
                    "path": legacy_asset_path,
                    "start": piece_start,
                    "end": piece_end,
                    "legacy_bridge": not breaks,
                }
            )

    scheduled_music.sort(key=lambda item: (item["start"], item["end"]))
    for index, item in enumerate(scheduled_music):
        start_seconds = float(item["start"])
        end_seconds = float(item["end"])
        next_start = (
            float(scheduled_music[index + 1]["start"])
            if index + 1 < len(scheduled_music)
            else chapter_duration
        )
        if item["legacy_bridge"]:
            if next_start > start_seconds:
                end_seconds = min(
                    chapter_duration,
                    next_start + music_crossfade_seconds,
                )
            else:
                warnings.append(
                    f"non_monotonic_music_scene:{item['scene_id']}"
                )
                end_seconds = chapter_duration
        elif (
            index + 1 < len(scheduled_music)
            and next_start <= end_seconds + 0.01
        ):
            end_seconds = min(
                chapter_duration,
                next_start + music_crossfade_seconds,
            )
        duration_seconds = max(0.0, end_seconds - start_seconds)
        if duration_seconds <= 0:
            warnings.append(
                f"zero_duration_music_asset:{item['asset_id']}"
            )
            continue
        has_next = index + 1 < len(scheduled_music)
        fade_out = (
            min(music_crossfade_seconds, duration_seconds / 2)
            if has_next or end_seconds < chapter_duration
            else min(1.5, duration_seconds / 3)
        )
        tracks.append(
            AudioMixTrack(
                kind="music",
                asset_id=str(item["asset_id"]),
                scene_id=str(item["scene_id"]),
                path=Path(item["path"]),
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                loop=True,
                gain=music_gain,
                fade_in_seconds=min(1.0, duration_seconds / 3),
                fade_out_seconds=fade_out,
            )
        )

    for (
        _scene_index,
        raw_scene,
        scene_id,
        _scene_start_seconds,
        _scene_end_seconds,
        _asset_path,
    ) in resolved_scenes:
        raw_sfx = raw_scene.get("sfx", [])
        if not isinstance(raw_sfx, list):
            warnings.append(f"invalid_sfx_list:{scene_id}")
            continue
        for effect_index, raw_effect in enumerate(raw_sfx):
            if not isinstance(raw_effect, dict):
                warnings.append(f"invalid_sfx:{scene_id}:{effect_index}")
                continue
            asset_id = str(raw_effect.get("id") or "").strip()
            if not asset_id:
                warnings.append(f"missing_sfx_id:{scene_id}:{effect_index}")
                continue
            anchor = event_interval_for_effect(
                raw_effect.get("anchorSegmentIndex"),
                raw_effect.get("anchorText"),
                f"sfx_anchor:{asset_id}",
            )
            if anchor is None:
                continue
            asset_path = _resolve_audio_asset_path(
                asset_root, manifest, "sfx", asset_id
            )
            if asset_path is None:
                warnings.append(f"missing_sfx_asset:{asset_id}")
                continue
            effect_duration = _wav_duration(asset_path)
            timing = str(raw_effect.get("timing") or "during")
            if timing == "before":
                start_seconds = anchor[0] - effect_duration
            elif timing == "after":
                start_seconds = anchor[1]
            else:
                start_seconds = anchor[0]
            if start_seconds < 0:
                # A ``before`` cue at the beginning of a chapter has no
                # negative timeline to occupy. Shift the complete effect to
                # zero instead of trimming it down to zero and silently
                # dropping it from the final mix.
                warnings.append(f"sfx_shifted_to_chapter_start:{asset_id}")
                clipped_start = 0.0
                clipped_duration = effect_duration
            else:
                clipped_start = start_seconds
                clipped_duration = effect_duration
            tracks.append(
                AudioMixTrack(
                    kind="sfx",
                    asset_id=asset_id,
                    scene_id=scene_id,
                    path=asset_path,
                    start_seconds=clipped_start,
                    duration_seconds=clipped_duration,
                    loop=False,
                    gain=sfx_gain,
                    fade_in_seconds=min(0.08, clipped_duration / 4),
                    fade_out_seconds=min(0.35, clipped_duration / 3),
                )
            )

    if not tracks:
        warnings.append("no_mix_tracks")
    duration = max(
        [cursor, *[track.start_seconds + track.duration_seconds for track in tracks]]
    ) if timeline else 0.0
    return AudioMixPlan(
        timeline=tuple(timeline),
        tracks=tuple(tracks),
        duration_seconds=duration,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def mix_chapter_audio(
    voice_audio_path: Path | str,
    segment_paths: list[Path],
    tts_segments: list[dict[str, Any]],
    script: dict[str, Any],
    asset_directory: Path | str,
    output_path: Path | str,
    *,
    gap_seconds: float = 0.5,
    voice_gain: float = DEFAULT_VOICE_GAIN,
    music_gain: float = DEFAULT_MUSIC_GAIN,
    sfx_gain: float = DEFAULT_SFX_GAIN,
    music_crossfade_seconds: float = DEFAULT_MUSIC_CROSSFADE_SECONDS,
    ffmpeg_path: str | None = None,
) -> ChapterAudioArtifact:
    """Mix the assembled human voice track with scheduled Stable Audio tracks."""

    voice_path = Path(voice_audio_path)
    output = Path(output_path)
    if not _is_readable_wav(voice_path):
        raise ValueError(f"voice audio is missing or unreadable: {voice_path}")
    if voice_path.resolve() == output.resolve():
        raise ValueError("mixed output path must be different from voice audio path")
    if voice_gain <= 0 or not math.isfinite(voice_gain):
        raise ValueError("voice_gain must be a finite positive number")

    plan = build_audio_mix_plan(
        segment_paths,
        tts_segments,
        script,
        asset_directory,
        gap_seconds=gap_seconds,
        music_gain=music_gain,
        sfx_gain=sfx_gain,
        music_crossfade_seconds=music_crossfade_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not plan.tracks and math.isclose(voice_gain, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        shutil.copyfile(voice_path, output)
    else:
        executable = ffmpeg_path or shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError(
                "ffmpeg is required to mix voice, background music, and sound effects"
            )
        command = _build_ffmpeg_mix_command(
            executable,
            voice_path,
            plan.tracks,
            plan.duration_seconds,
            output,
            voice_gain=voice_gain,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not _is_readable_wav(output):
            output.unlink(missing_ok=True)
            detail = (completed.stderr or completed.stdout or "").strip()
            detail = detail.splitlines()[-1] if detail else "ffmpeg exited with an error"
            raise RuntimeError(f"chapter audio mixing failed: {detail}")

    duration = _wav_duration(output)
    return ChapterAudioArtifact(
        kind="mixed_chapter_audio",
        path=output,
        duration_seconds=duration,
        metadata={
            "musicTrackCount": sum(track.kind == "music" for track in plan.tracks),
            "sfxTrackCount": sum(track.kind == "sfx" for track in plan.tracks),
            "warnings": list(plan.warnings),
        },
    )


def _wav_duration(path: Path | str) -> float:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() <= 0:
            raise ValueError(f"WAV has an invalid sample rate: {path}")
        return wav_file.getnframes() / wav_file.getframerate()


def _is_readable_wav(path: Path | str) -> bool:
    try:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except (OSError, wave.Error, ZeroDivisionError):
        return False


def _normalize_audio_anchor_text(value: Any) -> str:
    """Normalize cue text so punctuation and segment breaks do not matter."""

    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum()
    )


def _find_audio_anchor_match(
    anchor_text: Any,
    segment_texts: list[str],
    preferred_index: int,
) -> tuple[int, int, int, int, list[tuple[int, int]]] | None:
    """Find a cue in the concatenated normalized chapter text.

    The returned tuple contains the first and last source segment, the
    normalized character start/end, and the normalized character offsets for
    every source segment.  This allows a cue such as ``门“吱呀”一声被推开`` to
    be mapped even when the segmenter produced ``门`` and ``吱呀`` separately.
    """

    needle = _normalize_audio_anchor_text(anchor_text)
    if not needle or not segment_texts:
        return None

    normalized_segments = [
        _normalize_audio_anchor_text(text) for text in segment_texts
    ]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for text in normalized_segments:
        end = cursor + len(text)
        offsets.append((cursor, end))
        cursor = end
    haystack = "".join(normalized_segments)
    if not haystack:
        return None

    def segment_at(position: int) -> int | None:
        return next(
            (
                index
                for index, (start, end) in enumerate(offsets)
                if start <= position < end
            ),
            None,
        )

    candidates: list[tuple[int, int, int, int]] = []
    search_from = 0
    while True:
        position = haystack.find(needle, search_from)
        if position < 0:
            break
        end_position = position + len(needle)
        start_index = segment_at(position)
        end_index = segment_at(end_position - 1)
        if start_index is not None and end_index is not None:
            candidates.append((abs(start_index - preferred_index), position, start_index, end_index))
        search_from = position + 1

    if not candidates:
        return None
    _, start_position, start_index, end_index = min(candidates)
    return start_index, end_index, start_position, start_position + len(needle), offsets


def _interval_for_audio_anchor_match(
    match: tuple[int, int, int, int, list[tuple[int, int]]],
    original_ids: list[str],
    source_ranges: dict[str, tuple[float, float]],
) -> tuple[float, float] | None:
    """Project a normalized cue span onto the available synthesized timeline."""

    start_index, end_index, match_start, match_end, offsets = match
    covered_ranges: list[tuple[float, float]] = []
    for index in range(start_index, end_index + 1):
        if index >= len(original_ids) or index >= len(offsets):
            continue
        source_range = source_ranges.get(original_ids[index])
        segment_start, segment_end = offsets[index]
        normalized_length = segment_end - segment_start
        if source_range is None or normalized_length <= 0:
            continue
        covered_start = max(match_start, segment_start)
        covered_end = min(match_end, segment_end)
        if covered_start >= covered_end:
            continue
        timeline_start, timeline_end = source_range
        timeline_duration = timeline_end - timeline_start
        ratio_start = (covered_start - segment_start) / normalized_length
        ratio_end = (covered_end - segment_start) / normalized_length
        covered_ranges.append(
            (
                timeline_start + timeline_duration * ratio_start,
                timeline_start + timeline_duration * ratio_end,
            )
        )

    if not covered_ranges:
        return None
    return min(item[0] for item in covered_ranges), max(item[1] for item in covered_ranges)


def _read_audio_asset_manifest(asset_directory: Path) -> dict[str, Any]:
    path = asset_directory / "manifest.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _resolve_audio_asset_path(
    asset_directory: Path,
    manifest: dict[str, Any],
    kind: str,
    asset_id: str,
) -> Path | None:
    assets = manifest.get("assets", {})
    if isinstance(assets, dict):
        value = assets.get(f"{kind}:{asset_id}")
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            path = Path(value["path"])
            if path.is_file() and _is_readable_wav(path):
                return path
    safe_id = "".join(character if character.isalnum() or character in "-_." else "_" for character in asset_id).strip("._")
    if not safe_id:
        return None
    fallback = asset_directory / ("music" if kind == "music" else "sfx") / f"{safe_id}.wav"
    return fallback if _is_readable_wav(fallback) else None


def _build_ffmpeg_mix_command(
    executable: str,
    voice_path: Path,
    tracks: tuple[AudioMixTrack, ...],
    duration_seconds: float,
    output_path: Path,
    *,
    voice_gain: float = DEFAULT_VOICE_GAIN,
) -> list[str]:
    """Build an ffmpeg graph with voice side-chain ducking for music."""

    command: list[str] = [executable, "-y", "-i", str(voice_path)]
    for track in tracks:
        if track.loop:
            command.extend(["-stream_loop", "-1"])
        command.extend(["-i", str(track.path)])

    filters = [
        "[0:a]aresample=async=1:first_pts=0,aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"volume={voice_gain:.6f}[voice]"
    ]
    music_labels: list[str] = []
    sfx_labels: list[str] = []
    for index, track in enumerate(tracks, start=1):
        label = f"track{index}"
        delay_ms = max(0, round(track.start_seconds * 1000))
        fade_in = max(0.0, min(track.fade_in_seconds, track.duration_seconds / 2))
        fade_out = max(0.0, min(track.fade_out_seconds, track.duration_seconds / 2))
        fade_out_start = max(0.0, track.duration_seconds - fade_out)
        chain = [
            "aresample=async=1:first_pts=0",
            f"volume={track.gain:.6f}",
            f"atrim=duration={track.duration_seconds:.6f}",
        ]
        if fade_in > 0:
            chain.append(f"afade=t=in:st=0:d={fade_in:.6f}")
        if fade_out > 0:
            chain.append(f"afade=t=out:st={fade_out_start:.6f}:d={fade_out:.6f}")
        chain.extend([
            f"adelay={delay_ms}:all=1",
            f"atrim=duration={duration_seconds:.6f}",
        ])
        filters.append(f"[{index}:a]{','.join(chain)}[{label}]")
        (music_labels if track.kind == "music" else sfx_labels).append(f"[{label}]")

    output_inputs = ["[voice]"]
    if music_labels:
        if len(music_labels) == 1:
            music_label = music_labels[0]
        else:
            music_label = "[musicbed]"
            filters.append(
                "".join(music_labels)
                + f"amix=inputs={len(music_labels)}:duration=longest:normalize=0{music_label}"
            )
        filters.append(
            f"{music_label}[voice]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=300:makeup=1[duckedmusic]"
        )
        output_inputs.append("[duckedmusic]")
    output_inputs.extend(sfx_labels)
    filters.append(
        "".join(output_inputs)
        + f"amix=inputs={len(output_inputs)}:duration=longest:normalize=0,alimiter=limit=0.95[out]"
    )
    command.extend([
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-t",
        f"{duration_seconds:.6f}",
        "-c:a",
        "pcm_s16le",
        "-ac",
        "2",
        str(output_path),
    ])
    return command
