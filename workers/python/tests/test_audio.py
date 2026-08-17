import json
import shutil
import wave
from pathlib import Path

import pytest

from audiobook_worker.audio import (
    DEFAULT_MUSIC_CROSSFADE_SECONDS,
    AudioMixTrack,
    DEFAULT_MUSIC_GAIN,
    DEFAULT_SFX_GAIN,
    DEFAULT_VOICE_GAIN,
    _build_ffmpeg_mix_command,
    _wav_duration,
    assemble_chapter_audio,
    build_audio_mix_plan,
    mix_chapter_audio,
)
from audiobook_worker.tts import MockTTSBackend


def test_assembles_chapter_audio_from_segments(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "Hello."}, segment_dir)
    second = backend.synthesize_segment({"id": "seg_0002", "text": "World."}, segment_dir)
    output_path = tmp_path / "chapter_001.wav"

    artifact = assemble_chapter_audio([first.path, second.path], output_path)

    assert artifact.kind == "chapter_audio"
    assert artifact.path == output_path
    assert output_path.exists()
    with wave.open(str(output_path), "rb") as wav:
        assert wav.getnframes() > 0
        assert wav.getnchannels() == 1


def test_builds_mix_timestamps_from_actual_tts_segment_durations(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "Hello."}, segment_dir)
    second = backend.synthesize_segment({"id": "seg_0002", "text": "World."}, segment_dir)
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "music").mkdir(parents=True)
    (asset_dir / "sfx").mkdir(parents=True)
    backend.synthesize_segment({"id": "scene_001", "text": "music"}, asset_dir / "music")
    backend.synthesize_segment({"id": "door", "text": "effect"}, asset_dir / "sfx")

    script = {
        "segments": [
            {"id": "seg_0001"},
            {"id": "seg_0002"},
        ],
        "audioPlan": {
            "scenes": [
                {
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 1,
                    "music": {"model": "sm-music"},
                    "sfx": [
                        {
                            "id": "door",
                            "anchorSegmentIndex": 1,
                            "timing": "after",
                        }
                    ],
                }
            ]
        },
    }
    plan = build_audio_mix_plan(
        [first.path, second.path],
        [
            {"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]},
            {"id": "seg_0002", "sourceSegmentIds": ["seg_0002"]},
        ],
        script,
        asset_dir,
    )

    assert [round(item.start_seconds, 2) for item in plan.timeline] == [0.0, 0.75]
    assert plan.tracks[0].kind == "music"
    assert round(plan.tracks[0].duration_seconds, 2) == 1.0
    assert plan.tracks[1].kind == "sfx"
    assert plan.tracks[1].gain == pytest.approx(DEFAULT_SFX_GAIN)
    assert plan.tracks[1].fade_in_seconds == pytest.approx(0.0625)
    assert plan.tracks[1].fade_out_seconds == pytest.approx(0.25 / 3)
    assert round(plan.tracks[1].start_seconds, 2) == 1.0


def test_music_scenes_overlap_and_cover_tts_gaps(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    segment_paths = [
        backend.synthesize_segment(
            {"id": f"seg_{index:04d}", "text": f"Line {index}."},
            segment_dir,
        ).path
        for index in range(4)
    ]
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "music").mkdir(parents=True)
    for scene_id in ("scene_001", "scene_002"):
        backend.synthesize_segment(
            {"id": scene_id, "text": f"music {scene_id}"},
            asset_dir / "music",
        )

    plan = build_audio_mix_plan(
        segment_paths,
        [
            {
                "id": f"seg_{index:04d}",
                "sourceSegmentIds": [f"seg_{index:04d}"],
            }
            for index in range(4)
        ],
        {
            "segments": [
                {"id": f"seg_{index:04d}"}
                for index in range(4)
            ],
            "audioPlan": {
                "scenes": [
                    {
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 0,
                        "music": {"model": "sm-music"},
                        "sfx": [],
                    },
                    {
                        "id": "scene_002",
                        "startSegmentIndex": 1,
                        "endSegmentIndex": 3,
                        "music": {"model": "sm-music"},
                        "sfx": [],
                    },
                ]
            },
        },
        asset_dir,
    )

    music_tracks = [track for track in plan.tracks if track.kind == "music"]
    assert len(music_tracks) == 2
    first, second = music_tracks
    assert second.start_seconds == pytest.approx(0.75)
    assert first.start_seconds + first.duration_seconds == pytest.approx(
        second.start_seconds + DEFAULT_MUSIC_CROSSFADE_SECONDS
    )
    assert first.fade_out_seconds == pytest.approx(
        DEFAULT_MUSIC_CROSSFADE_SECONDS
    )
    assert second.start_seconds < first.start_seconds + first.duration_seconds
    assert second.start_seconds + second.duration_seconds == pytest.approx(
        plan.duration_seconds
    )


def test_music_variants_switch_and_leave_intentional_breaks(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    long_text = " ".join(["word"] * 20)
    segment_paths = [
        backend.synthesize_segment(
            {"id": f"seg_{index:04d}", "text": long_text},
            segment_dir,
        ).path
        for index in range(3)
    ]
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "music").mkdir(parents=True)
    for asset_id in ("scene_001_low", "scene_001_medium"):
        backend.synthesize_segment(
            {"id": asset_id, "text": "music"},
            asset_dir / "music",
        )

    plan = build_audio_mix_plan(
        segment_paths,
        [
            {
                "id": f"seg_{index:04d}",
                "sourceSegmentIds": [f"seg_{index:04d}"],
            }
            for index in range(3)
        ],
        {
            "segments": [
                {"id": f"seg_{index:04d}"}
                for index in range(3)
            ],
            "audioPlan": {
                "version": 2,
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 2,
                    "musicVariants": [
                        {"id": "scene_001_low", "level": "low"},
                        {"id": "scene_001_medium", "level": "medium"},
                    ],
                    "musicCues": [
                        {
                            "id": "cue_001",
                            "startSegmentIndex": 0,
                            "endSegmentIndex": 1,
                            "variantId": "scene_001_low",
                        },
                        {
                            "id": "cue_002",
                            "startSegmentIndex": 2,
                            "endSegmentIndex": 2,
                            "variantId": "scene_001_medium",
                        },
                    ],
                    "musicBreaks": [{
                        "afterSegmentIndex": 1,
                        "durationSeconds": 2,
                    }],
                    "sfx": [],
                }],
            },
        },
        asset_dir,
    )

    music_tracks = [track for track in plan.tracks if track.kind == "music"]
    assert [track.asset_id for track in music_tracks] == [
        "scene_001_low",
        "scene_001_medium",
    ]
    assert music_tracks[0].start_seconds == pytest.approx(0.0)
    assert music_tracks[0].duration_seconds < music_tracks[1].start_seconds
    assert music_tracks[1].start_seconds - (
        music_tracks[0].start_seconds + music_tracks[0].duration_seconds
    ) >= 1.9


def test_v2_music_fills_cue_gaps_and_keeps_breaks_inside_their_scene(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    long_text = " ".join(["word"] * 20)
    segment_paths = [
        backend.synthesize_segment(
            {"id": f"seg_{index:04d}", "text": long_text},
            segment_dir,
        ).path
        for index in range(4)
    ]
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "music").mkdir(parents=True)
    for asset_id in ("scene_001_low", "scene_001_high", "scene_002_low"):
        backend.synthesize_segment(
            {"id": asset_id, "text": "music"},
            asset_dir / "music",
        )

    plan = build_audio_mix_plan(
        segment_paths,
        [
            {
                "id": f"seg_{index:04d}",
                "sourceSegmentIds": [f"seg_{index:04d}"],
            }
            for index in range(4)
        ],
        {
            "segments": [
                {"id": f"seg_{index:04d}"}
                for index in range(4)
            ],
            "audioPlan": {
                "version": 2,
                "scenes": [
                    {
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 1,
                        "musicVariants": [
                            {"id": "scene_001_low", "level": "low"},
                            {"id": "scene_001_high", "level": "high"},
                        ],
                        "musicCues": [{
                            "id": "cue_001",
                            "startSegmentIndex": 1,
                            "endSegmentIndex": 1,
                            "variantId": "scene_001_high",
                        }],
                        "musicBreaks": [{
                            "afterSegmentIndex": 0,
                            "durationSeconds": 2,
                        }],
                        "sfx": [],
                    },
                    {
                        "id": "scene_002",
                        "startSegmentIndex": 2,
                        "endSegmentIndex": 3,
                        "musicVariants": [{
                            "id": "scene_002_low",
                            "level": "low",
                        }],
                        "musicCues": [{
                            "id": "cue_002",
                            "startSegmentIndex": 2,
                            "endSegmentIndex": 3,
                            "variantId": "scene_002_low",
                        }],
                        # This break belongs to scene_001 and must not mute
                        # scene_002 when a plan is malformed.
                        "musicBreaks": [{
                            "afterSegmentIndex": 0,
                            "durationSeconds": 2,
                        }],
                        "sfx": [],
                    },
                ],
            },
        },
        asset_dir,
    )

    music_tracks = [track for track in plan.tracks if track.kind == "music"]
    assert [track.asset_id for track in music_tracks] == [
        "scene_001_low",
        "scene_001_high",
        "scene_002_low",
    ]
    assert music_tracks[1].start_seconds - (
        music_tracks[0].start_seconds + music_tracks[0].duration_seconds
    ) >= 1.9
    assert music_tracks[2].start_seconds < plan.duration_seconds
    assert "filled_music_cue_gap:scene_001:0-0" in plan.warnings
    assert "music_break_outside_scene:0" in plan.warnings


def test_quality_rejected_assets_never_use_filesystem_fallback(tmp_path: Path):
    backend = MockTTSBackend()
    segment_directory = tmp_path / "segments"
    segment_paths = [
        backend.synthesize_segment(
            {"id": f"seg_{index:04d}", "text": "line"},
            segment_directory,
        ).path
        for index in range(2)
    ]
    asset_directory = tmp_path / "audio-assets"
    valid_music = backend.synthesize_segment(
        {"id": "scene_002", "text": "approved music"},
        asset_directory / "music",
    ).path
    # A stale conventional filename exists for both rejected assets. The mix
    # plan must not pick either one merely because it is readable on disk.
    backend.synthesize_segment(
        {"id": "scene_001", "text": "rejected music"},
        asset_directory / "music",
    )
    backend.synthesize_segment(
        {"id": "door", "text": "rejected sfx"},
        asset_directory / "sfx",
    )
    (asset_directory / "manifest.json").write_text(
        json.dumps(
            {
                "version": 3,
                "assets": {
                    "music:scene_002": {"path": str(valid_music)},
                },
                "rejectedAssets": {
                    "music:scene_001": {"reason": "quality_retries_exhausted"},
                    "sfx:door": {"reason": "quality_retries_exhausted"},
                },
                "qualityFallbacks": {
                    "music:scene_001": {
                        "assetKey": "music:scene_002",
                        "reason": "quality_nearby_scene",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_audio_mix_plan(
        segment_paths,
        [
            {"id": "seg_0000", "sourceSegmentIds": ["seg_0000"]},
            {"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]},
        ],
        {
            "segments": [{"id": "seg_0000"}, {"id": "seg_0001"}],
            "audioPlan": {
                "scenes": [
                    {
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 0,
                        "music": {"model": "sm-music"},
                        "sfx": [{
                            "id": "door",
                            "anchorSegmentIndex": 0,
                            "timing": "during",
                        }],
                    },
                    {
                        "id": "scene_002",
                        "startSegmentIndex": 1,
                        "endSegmentIndex": 1,
                        "music": {"model": "sm-music"},
                        "sfx": [],
                    },
                ]
            },
        },
        asset_directory,
    )

    music_tracks = [track for track in plan.tracks if track.kind == "music"]
    assert len(music_tracks) == 2
    assert music_tracks[0].path == valid_music
    assert not [track for track in plan.tracks if track.kind == "sfx"]
    assert "quality_rejected_sfx:door" in plan.warnings
    assert any(warning.startswith("quality_music_fallback:scene_001:") for warning in plan.warnings)


def test_sfx_anchor_spanning_segments_is_kept_at_chapter_start(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "门"}, segment_dir)
    second = backend.synthesize_segment({"id": "seg_0002", "text": "吱呀"}, segment_dir)
    third = backend.synthesize_segment(
        {"id": "seg_0003", "text": "一声被推开"},
        segment_dir,
    )
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "sfx").mkdir(parents=True)
    effect = backend.synthesize_segment(
        {"id": "door", "text": "door creak"},
        asset_dir / "sfx",
    )

    plan = build_audio_mix_plan(
        [first.path, second.path, third.path],
        [
            {"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]},
            {"id": "seg_0002", "sourceSegmentIds": ["seg_0002"]},
            {"id": "seg_0003", "sourceSegmentIds": ["seg_0003"]},
        ],
        {
            "segments": [
                {"id": "seg_0001", "text": "门"},
                {"id": "seg_0002", "text": "吱呀"},
                {"id": "seg_0003", "text": "一声被推开"},
            ],
            "audioPlan": {
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 2,
                    "music": None,
                    "sfx": [{
                        "id": "door",
                        "anchorSegmentIndex": 0,
                        "anchorText": "门“吱呀”一声被推开",
                        "timing": "before",
                    }],
                }],
            },
        },
        asset_dir,
    )

    sfx_tracks = [track for track in plan.tracks if track.kind == "sfx"]
    assert len(sfx_tracks) == 1
    assert sfx_tracks[0].start_seconds == 0.0
    assert sfx_tracks[0].duration_seconds == pytest.approx(
        _wav_duration(effect.path)
    )
    assert "sfx_anchor_spans_segments:door" in plan.warnings
    assert "sfx_shifted_to_chapter_start:door" in plan.warnings


def test_unmatched_before_sfx_is_not_silently_dropped(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "门"}, segment_dir)
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "sfx").mkdir(parents=True)
    backend.synthesize_segment({"id": "door", "text": "door creak"}, asset_dir / "sfx")

    plan = build_audio_mix_plan(
        [first.path],
        [{"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]}],
        {
            "segments": [{"id": "seg_0001", "text": "门"}],
            "audioPlan": {
                "scenes": [{
                    "id": "scene_001",
                    "startSegmentIndex": 0,
                    "endSegmentIndex": 0,
                    "music": None,
                    "sfx": [{
                        "id": "door",
                        "anchorSegmentIndex": 0,
                        "anchorText": "不存在的原文",
                        "timing": "before",
                    }],
                }],
            },
        },
        asset_dir,
    )

    assert [track.kind for track in plan.tracks] == ["sfx"]
    assert "unmatched_sfx_anchor_text:door" in plan.warnings
    assert "sfx_shifted_to_chapter_start:door" in plan.warnings


def test_mixing_without_assets_keeps_a_voice_only_copy(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "Hello."}, segment_dir)
    voice_path = tmp_path / "voice.wav"
    assemble_chapter_audio([first.path], voice_path)
    output_path = tmp_path / "mixed.wav"

    artifact = mix_chapter_audio(
        voice_path,
        [first.path],
        [{"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]}],
        {"segments": [{"id": "seg_0001"}], "audioPlan": {"scenes": []}},
        tmp_path / "missing-assets",
        output_path,
        voice_gain=1.0,
    )

    assert artifact.kind == "mixed_chapter_audio"
    assert output_path.read_bytes() == voice_path.read_bytes()
    assert "no_mix_tracks" in artifact.metadata["warnings"]


def test_ffmpeg_mix_applies_default_voice_gain_before_ducking():
    command = _build_ffmpeg_mix_command(
        "ffmpeg",
        Path("voice.wav"),
        (),
        2.0,
        Path("mixed.wav"),
    )

    filter_complex = command[command.index("-filter_complex") + 1]
    assert f"volume={DEFAULT_VOICE_GAIN:.6f}" in filter_complex


def test_default_music_gain_is_raised_after_loudness_normalization():
    assert DEFAULT_MUSIC_GAIN == pytest.approx(0.40)

    command = _build_ffmpeg_mix_command(
        "ffmpeg",
        Path("voice.wav"),
        (
            # The command builder only needs track metadata for this assertion.
            # Its input path is not opened while constructing the command.
            AudioMixTrack(
                kind="music",
                asset_id="scene_001",
                scene_id="scene_001",
                path=Path("music.wav"),
                start_seconds=0.0,
                duration_seconds=2.0,
                loop=False,
                gain=DEFAULT_MUSIC_GAIN,
                fade_in_seconds=0.0,
                fade_out_seconds=0.0,
            ),
        ),
        2.0,
        Path("mixed.wav"),
    )
    filter_complex = command[command.index("-filter_complex") + 1]
    assert f"volume={DEFAULT_MUSIC_GAIN:.6f}" in filter_complex


def test_default_sfx_gain_and_fades_are_applied():
    assert DEFAULT_SFX_GAIN == pytest.approx(0.35)

    command = _build_ffmpeg_mix_command(
        "ffmpeg",
        Path("voice.wav"),
        (
            AudioMixTrack(
                kind="sfx",
                asset_id="door",
                scene_id="scene_001",
                path=Path("door.wav"),
                start_seconds=0.0,
                duration_seconds=2.0,
                loop=False,
                gain=DEFAULT_SFX_GAIN,
                fade_in_seconds=0.08,
                fade_out_seconds=0.35,
            ),
        ),
        2.0,
        Path("mixed.wav"),
    )
    filter_complex = command[command.index("-filter_complex") + 1]
    assert f"volume={DEFAULT_SFX_GAIN:.6f}" in filter_complex
    assert "afade=t=in:st=0:d=0.080000" in filter_complex
    assert "afade=t=out:st=1.650000:d=0.350000" in filter_complex


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the integration mix")
def test_mixes_music_and_sfx_into_a_chapter_track(tmp_path: Path):
    backend = MockTTSBackend()
    segment_dir = tmp_path / "segments"
    first = backend.synthesize_segment({"id": "seg_0001", "text": "Hello."}, segment_dir)
    voice_path = tmp_path / "voice.wav"
    assemble_chapter_audio([first.path], voice_path)
    asset_dir = tmp_path / "audio-assets"
    (asset_dir / "music").mkdir(parents=True)
    (asset_dir / "sfx").mkdir(parents=True)
    backend.synthesize_segment({"id": "scene_001", "text": "music"}, asset_dir / "music")
    backend.synthesize_segment({"id": "door", "text": "effect"}, asset_dir / "sfx")
    output_path = tmp_path / "mixed.wav"

    artifact = mix_chapter_audio(
        voice_path,
        [first.path],
        [{"id": "seg_0001", "sourceSegmentIds": ["seg_0001"]}],
        {
            "segments": [{"id": "seg_0001"}],
            "audioPlan": {
                "scenes": [
                    {
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 0,
                        "music": {"model": "sm-music"},
                        "sfx": [
                            {
                                "id": "door",
                                "anchorSegmentIndex": 0,
                                "timing": "during",
                            }
                        ],
                    }
                ]
            },
        },
        asset_dir,
        output_path,
    )

    assert artifact.kind == "mixed_chapter_audio"
    assert output_path.exists()
    with wave.open(str(output_path), "rb") as wav:
        assert wav.getnframes() > 0
        assert wav.getnchannels() == 2
