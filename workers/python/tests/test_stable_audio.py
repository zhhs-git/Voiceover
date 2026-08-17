import json
import subprocess
import wave
from pathlib import Path

import pytest

from audiobook_worker import stable_audio
from audiobook_worker.audio_quality import AudioQualityRepairResult, AudioQualityResult
from audiobook_worker.stable_audio import (
    StableAudioConfig,
    StableAudioError,
    generate_audio_assets,
)


def _write_test_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 44100, 4410, "NONE", "not compressed"))
        output.writeframes(b"\x00\x00" * 4410)


def _command_output_path(command: list[str]) -> Path:
    if "--out" in command:
        return Path(command[command.index("--out") + 1])
    return Path(command[-1])


def _quality_result(path: Path, *, suspicious: bool) -> AudioQualityResult:
    return AudioQualityResult(
        path=path,
        duration_seconds=0.1,
        status="sharp_suspected" if suspicious else "normal",
        risk_score=0.8 if suspicious else 0.0,
        suspicious_times=(0.04,) if suspicious else (),
        suspicious_intervals=((0.02, 0.08),) if suspicious else (),
        issues=("high_frequency_burst",) if suspicious else (),
        windows_analyzed=2,
        clipped_windows=0,
        high_frequency_burst_windows=1 if suspicious else 0,
    )


def _script_payload() -> dict:
    return {
        "bookId": "book_123",
        "chapterId": "chapter_001",
        "segments": [{"id": "seg_0001", "text": "雨落在街上。"}],
        "audioPlan": {
            "scenes": [{
                "id": "scene_001",
                "startSegmentIndex": 0,
                "endSegmentIndex": 0,
                "music": {
                    "model": "sm-music",
                    "durationSeconds": 30,
                    "prompt": "TrackType: Music, VocalType: Instrumental, dark historical suspense bed",
                    "negativePrompt": "vocals, speech",
                },
                "sfx": [{
                    "id": "sfx_001",
                    "model": "sm-sfx",
                    "anchorSegmentIndex": 0,
                    "timing": "during",
                    "durationSeconds": 5,
                    "prompt": "TrackType: SFX, heavy rain on wet stone pavement",
                    "negativePrompt": "music, speech",
                }],
            }]
        },
    }


def test_generates_music_and_sfx_with_expected_commands_and_cache(tmp_path, monkeypatch):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(
        root=tmp_path,
        executable=executable,
        timeout_seconds=10,
        quality_enabled=False,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        output_path = _command_output_path(command)
        _write_test_wav(output_path)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)

    first = generate_audio_assets(
        script_path,
        tmp_path / "audio-assets",
        config=config,
    )

    assert len(first.assets) == 2
    assert all(not asset.cache_hit for asset in first.assets)
    assert len(calls) == 3
    generation_calls = [call for call in calls if "--out" in call[0]]
    assert len(generation_calls) == 2
    normalization_calls = [call for call in calls if "--out" not in call[0]]
    assert len(normalization_calls) == 1
    assert "loudnorm=I=-18:TP=-2:LRA=7" in normalization_calls[0][0]
    music_command = generation_calls[0][0]
    sfx_command = generation_calls[1][0]
    assert "sm-music" in music_command
    assert "sm-sfx" in sfx_command
    assert "--cfg" in music_command
    assert "--negative-prompt" in music_command
    assert str(
        (tmp_path / "audio-assets" / "music" / "scene_001.wav").resolve()
    ) in music_command
    assert (tmp_path / "audio-assets" / "manifest.json").is_file()

    second = generate_audio_assets(
        script_path,
        tmp_path / "audio-assets",
        config=config,
    )

    assert len(calls) == 3
    assert all(asset.cache_hit for asset in second.assets)


def test_normalizes_duplicate_sfx_ids_across_scenes_and_persists_plan(tmp_path):
    payload = _script_payload()
    payload["segments"] = [
        {"id": "seg_0001", "text": "雨落在街上。"},
        {"id": "seg_0002", "text": "远处又传来雨声。"},
    ]
    first_scene = payload["audioPlan"]["scenes"][0]
    payload["audioPlan"]["scenes"].append({
        **first_scene,
        "id": "scene_002",
        "startSegmentIndex": 1,
        "endSegmentIndex": 1,
        "music": {**first_scene["music"]},
        "sfx": [{**first_scene["sfx"][0]}],
    })
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    specs = stable_audio.collect_audio_asset_specs(script_path)

    assert [spec.asset_id for spec in specs if spec.kind == "sfx"] == [
        "sfx_001",
        "scene_002_sfx_001",
    ]
    normalized = json.loads(script_path.read_text(encoding="utf-8"))
    assert normalized["audioPlan"]["scenes"][1]["sfx"][0]["id"] == (
        "scene_002_sfx_001"
    )


def test_empty_audio_plan_does_not_require_stable_audio_executable(tmp_path):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps({"audioPlan": {"scenes": []}}), encoding="utf-8")
    result = generate_audio_assets(
        script_path,
        tmp_path / "audio-assets",
        config=StableAudioConfig(
            root=tmp_path,
            executable=tmp_path / "missing-sa3",
        ),
    )

    assert result.assets == []
    assert result.warnings == ["no_audio_assets"]
    assert result.manifest_path.is_file()


def test_generates_three_same_theme_music_variants_and_reuses_cache(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(
        json.dumps(
            {
                "audioPlan": {
                    "version": 2,
                    "scenes": [{
                        "id": "scene_001",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 2,
                        "musicVariants": [
                            {
                                "id": "scene_001_low",
                                "level": "low",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, VocalType: Instrumental, low",
                            },
                            {
                                "id": "scene_001_medium",
                                "level": "medium",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, VocalType: Instrumental, medium",
                            },
                            {
                                "id": "scene_001_high",
                                "level": "high",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, VocalType: Instrumental, high",
                            },
                        ],
                        "musicCues": [{
                            "id": "cue_001",
                            "startSegmentIndex": 0,
                            "endSegmentIndex": 2,
                            "variantId": "scene_001_low",
                        }],
                        "sfx": [],
                    }],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    output_directory = tmp_path / "audio-assets"

    first = generate_audio_assets(script_path, output_directory, config=config)
    assert {asset.asset_id for asset in first.assets} == {
        "scene_001_low",
        "scene_001_medium",
        "scene_001_high",
    }
    assert len(calls) == 6
    assert all(
        (output_directory / "music" / f"scene_001_{level}.wav").is_file()
        for level in ("low", "medium", "high")
    )

    second = generate_audio_assets(script_path, output_directory, config=config)
    assert len(calls) == 6
    assert all(asset.cache_hit for asset in second.assets)


def test_music_variants_share_palette_anchor_and_deterministic_seed(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "chapter.json"
    anchor = (
        "Ancient Chinese frontier camp at night, xiao, low strings and muted wood, "
        "Dorian colour, low-mid register, warm firelit texture, restrained pulse"
    )
    script_path.write_text(
        json.dumps(
            {
                "audioPlan": {
                    "version": 2,
                    "scenes": [{
                        "id": "scene_anchor",
                        "startSegmentIndex": 0,
                        "endSegmentIndex": 2,
                        "musicPalette": {"promptAnchor": anchor},
                        "musicVariants": [
                            {
                                "id": "scene_anchor_low",
                                "level": "low",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, low arrangement",
                            },
                            {
                                "id": "scene_anchor_medium",
                                "level": "medium",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, medium arrangement",
                            },
                            {
                                "id": "scene_anchor_high",
                                "level": "high",
                                "model": "sm-music",
                                "durationSeconds": 30,
                                "prompt": "TrackType: Music, high arrangement",
                            },
                        ],
                        "musicCues": [{
                            "id": "cue_anchor",
                            "startSegmentIndex": 0,
                            "endSegmentIndex": 2,
                            "variantId": "scene_anchor_low",
                        }],
                        "sfx": [],
                    }],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    result = generate_audio_assets(script_path, tmp_path / "audio-assets", config=config)

    assert len(result.assets) == 3
    generation_calls = [call for call in calls if "--out" in call]
    assert len(generation_calls) == 3
    assert all(anchor in call[call.index("--prompt") + 1] for call in generation_calls)
    seeds = [call[call.index("--seed") + 1] for call in generation_calls]
    assert len(set(seeds)) == 1


def test_generation_prunes_manifest_entries_from_an_older_audio_plan(tmp_path, monkeypatch):
    old_script = _script_payload()
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(old_script, ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)

    def fake_run(command, **kwargs):
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    output_directory = tmp_path / "audio-assets"
    generate_audio_assets(script_path, output_directory, config=config)

    current_scene = old_script["audioPlan"]["scenes"][0]
    current_script = {
        **old_script,
        "audioPlan": {
            "scenes": [{
                **current_scene,
                "id": "scene_1",
                "music": {**current_scene["music"]},
                "sfx": [{**current_scene["sfx"][0], "id": "sfx_1"}],
            }],
        },
    }
    script_path.write_text(json.dumps(current_script, ensure_ascii=False), encoding="utf-8")
    generate_audio_assets(script_path, output_directory, config=config)

    manifest = json.loads(
        (output_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert set(manifest["assets"]) == {"music:scene_1", "sfx:sfx_1"}


def test_audio_plan_change_invalidates_stable_audio_cache(tmp_path, monkeypatch):
    payload = _script_payload()
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    output_directory = tmp_path / "audio-assets"
    generate_audio_assets(script_path, output_directory, config=config)

    payload["audioPlan"]["scenes"][0]["music"]["prompt"] += ", restrained tension"
    script_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    second = generate_audio_assets(script_path, output_directory, config=config)

    assert len([call for call in calls if "--out" in call]) == 4
    assert all(not asset.cache_hit for asset in second.assets)


def test_can_force_regenerate_only_one_selected_asset(tmp_path, monkeypatch):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    output_directory = tmp_path / "audio-assets"
    generate_audio_assets(script_path, output_directory, config=config)

    selected = generate_audio_assets(
        script_path,
        output_directory,
        force=True,
        asset_id="sfx_001",
        asset_kind="sfx",
        config=config,
    )

    assert [asset.asset_id for asset in selected.assets] == ["sfx_001"]
    assert len([call for call, _kwargs in calls if "--out" in call]) == 3
    assert calls[-1][0][calls[-1][0].index("--dit") + 1] == "sm-sfx"
    assert str((output_directory / "sfx" / "sfx_001.wav").resolve()) in calls[-1][0]
    assert (output_directory / "music" / "scene_001.wav").is_file()


def test_selected_asset_must_exist_in_the_audio_plan(tmp_path):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(StableAudioError) as raised:
        generate_audio_assets(
            script_path,
            tmp_path / "audio-assets",
            asset_id="missing",
            asset_kind="music",
            config=StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False),
        )

    assert raised.value.code == "audio_asset_not_found"


def test_rejects_audio_plan_model_that_does_not_match_config(tmp_path):
    payload = _script_payload()
    payload["audioPlan"]["scenes"][0]["music"]["model"] = "medium"
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(payload), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(StableAudioError) as raised:
        generate_audio_assets(
            script_path,
            tmp_path / "audio-assets",
            config=StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False),
        )

    assert raised.value.code == "invalid_audio_plan"


def test_preserves_completed_assets_when_a_later_asset_fails(tmp_path, monkeypatch):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=False)
    generation_count = 0

    def fake_run(command, **kwargs):
        nonlocal generation_count
        if "--out" not in command:
            _write_test_wav(_command_output_path(command))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        generation_count += 1
        if generation_count == 1:
            _write_test_wav(_command_output_path(command))
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="generation failed")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)

    with pytest.raises(StableAudioError) as raised:
        generate_audio_assets(
            script_path,
            tmp_path / "audio-assets",
            config=config,
        )

    assert raised.value.code == "stable_audio_generation_failed"
    assert [asset.asset_id for asset in raised.value.partial_assets] == ["scene_001"]


def test_cached_asset_without_quality_record_is_checked_without_regenerating(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    output_directory = tmp_path / "audio-assets"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    disabled_config = StableAudioConfig(
        root=tmp_path,
        executable=executable,
        quality_enabled=False,
    )
    generate_audio_assets(script_path, output_directory, config=disabled_config)
    call_count_before_validation = len(calls)
    analyzed: list[Path] = []

    def normal_analysis(path):
        analyzed.append(Path(path))
        return _quality_result(Path(path), suspicious=False)

    monkeypatch.setattr(stable_audio, "analyze_audio", normal_analysis)
    enabled_config = StableAudioConfig(
        root=tmp_path,
        executable=executable,
        quality_enabled=True,
    )
    result = generate_audio_assets(script_path, output_directory, config=enabled_config)

    assert len(calls) == call_count_before_validation
    assert len(analyzed) == 2
    assert all(asset.cache_hit for asset in result.assets)
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["assets"]["music:scene_001"]["quality"]["status"] == "passed"
    assert (output_directory / "quality" / "music" / "scene_001.json").is_file()


def test_short_sfx_quality_failure_regenerates_only_that_asset(tmp_path, monkeypatch):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=True)
    output_directory = tmp_path / "audio-assets"

    def fake_run(command, **kwargs):
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    quality_calls = 0

    def fake_analyze(path):
        nonlocal quality_calls
        quality_calls += 1
        return _quality_result(Path(path), suspicious=quality_calls == 1)

    regenerated: list[int] = []

    def fake_regenerate(spec, _config, *, signature, attempt):
        regenerated.append(attempt)
        remote = tmp_path / f"remote-{attempt}.wav"
        _write_test_wav(remote)
        return remote

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    monkeypatch.setattr(stable_audio, "analyze_audio", fake_analyze)
    monkeypatch.setattr(
        stable_audio,
        "repair_short_suspicious_intervals",
        lambda source, output, intervals: AudioQualityRepairResult(
            "not_eligible", Path(source), None, (), "short_sfx"
        ),
    )
    monkeypatch.setattr(stable_audio, "_request_gradio_regeneration", fake_regenerate)

    result = generate_audio_assets(
        script_path,
        output_directory,
        asset_id="sfx_001",
        asset_kind="sfx",
        config=config,
    )

    assert [asset.asset_id for asset in result.assets] == ["sfx_001"]
    assert regenerated == [1]
    assert result.assets[0].quality["source"] == "gradio_regeneration"
    assert (output_directory / "rejected" / "sfx").is_dir()
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert "sfx:sfx_001" in manifest["assets"]
    assert "sfx:sfx_001" not in manifest.get("rejectedAssets", {})


def test_quality_retries_exhausted_quarantines_asset_without_failing_stage(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=True)
    output_directory = tmp_path / "audio-assets"

    def fake_run(command, **kwargs):
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    regenerated: list[int] = []

    def fake_regenerate(spec, _config, *, signature, attempt):
        regenerated.append(attempt)
        remote = tmp_path / f"remote-{attempt}.wav"
        _write_test_wav(remote)
        return remote

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    monkeypatch.setattr(
        stable_audio,
        "analyze_audio",
        lambda path: _quality_result(Path(path), suspicious=True),
    )
    monkeypatch.setattr(
        stable_audio,
        "repair_short_suspicious_intervals",
        lambda source, output, intervals: AudioQualityRepairResult(
            "not_eligible", Path(source), None, (), "short_sfx"
        ),
    )
    monkeypatch.setattr(stable_audio, "_request_gradio_regeneration", fake_regenerate)

    result = generate_audio_assets(
        script_path,
        output_directory,
        asset_id="sfx_001",
        asset_kind="sfx",
        config=config,
    )

    assert result.assets == []
    assert regenerated == [1, 2]
    assert any(warning.startswith("audio_quality_rejected:sfx:sfx_001") for warning in result.warnings)
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert "sfx:sfx_001" not in manifest["assets"]
    assert manifest["rejectedAssets"]["sfx:sfx_001"]["reason"] == "short_sfx_direct_regeneration"
    assert (output_directory / "quality" / "sfx" / "sfx_001.json").is_file()


def test_short_music_defect_is_repaired_and_rechecked_before_manifest_write(
    tmp_path,
    monkeypatch,
):
    script_path = tmp_path / "chapter.json"
    script_path.write_text(json.dumps(_script_payload(), ensure_ascii=False), encoding="utf-8")
    executable = tmp_path / "sa3"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    config = StableAudioConfig(root=tmp_path, executable=executable, quality_enabled=True)
    output_directory = tmp_path / "audio-assets"

    def fake_run(command, **kwargs):
        _write_test_wav(_command_output_path(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    analysis_count = 0

    def fake_analyze(path):
        nonlocal analysis_count
        analysis_count += 1
        return _quality_result(Path(path), suspicious=analysis_count == 1)

    def fake_repair(source, output, intervals):
        _write_test_wav(Path(output))
        return AudioQualityRepairResult(
            "repaired",
            Path(source),
            Path(output),
            ((0.02, 0.08),),
        )

    monkeypatch.setattr(stable_audio.subprocess, "run", fake_run)
    monkeypatch.setattr(stable_audio, "analyze_audio", fake_analyze)
    monkeypatch.setattr(stable_audio, "repair_short_suspicious_intervals", fake_repair)
    monkeypatch.setattr(
        stable_audio,
        "_request_gradio_regeneration",
        lambda *args, **kwargs: pytest.fail("repaired asset must not be regenerated"),
    )

    result = generate_audio_assets(
        script_path,
        output_directory,
        asset_id="scene_001",
        asset_kind="music",
        config=config,
    )

    assert [asset.quality["status"] for asset in result.assets] == ["repaired"]
    assert analysis_count == 2
    assert list((output_directory / "rejected" / "music").glob("*.wav"))


def test_rejected_music_uses_same_scene_then_nearby_approved_fallback():
    first = stable_audio.AudioAssetSpec(
        asset_id="scene_001_low",
        kind="music",
        scene_id="scene_001",
        model="sm-music",
        prompt="low",
        negative_prompt="",
        duration_seconds=30,
    )
    same_scene = stable_audio.AudioAssetSpec(
        asset_id="scene_001_medium",
        kind="music",
        scene_id="scene_001",
        model="sm-music",
        prompt="medium",
        negative_prompt="",
        duration_seconds=30,
    )
    nearby = stable_audio.AudioAssetSpec(
        asset_id="scene_002_low",
        kind="music",
        scene_id="scene_002",
        model="sm-music",
        prompt="nearby",
        negative_prompt="",
        duration_seconds=30,
    )
    manifest = {
        "assets": {same_scene.manifest_key: {}, nearby.manifest_key: {}},
        "rejectedAssets": {first.manifest_key: {}},
    }

    stable_audio._refresh_quality_music_fallbacks(manifest, [first, same_scene, nearby])

    assert manifest["qualityFallbacks"][first.manifest_key] == {
        "assetKey": same_scene.manifest_key,
        "reason": "quality_same_scene_variant",
    }
    manifest["assets"].pop(same_scene.manifest_key)
    stable_audio._refresh_quality_music_fallbacks(manifest, [first, same_scene, nearby])
    assert manifest["qualityFallbacks"][first.manifest_key] == {
        "assetKey": nearby.manifest_key,
        "reason": "quality_nearby_scene",
    }
