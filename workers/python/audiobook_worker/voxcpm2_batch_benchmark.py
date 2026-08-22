"""Benchmark and gate resident VoxCPM2 batch sizes without touching book caches.

Run this program only while MiMo work is idle.  It is executed with the
isolated VoxCPM2 interpreter, loads one model instance, synthesizes the same
representative four-or-more segment fixture at batch sizes 1, 2, and 4, and
checks the temporary WAVs through the existing assembly and no-track final-mix
path.  ``auto`` remains batch size one until this program writes a valid
selection file for the current adapter version.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

if __package__:
    from .audio import assemble_chapter_audio, build_audio_mix_plan, mix_chapter_audio
    from .model_settings import voxcpm2_paths
    from .voxcpm2_batch_adapter import BATCH_ADAPTER_VERSION, VoxCPM2BatchAdapter
    from .voxcpm2_runner import _atomic_write_wav
else:  # pragma: no cover - exercised by the isolated direct-script runtime.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from audiobook_worker.audio import (  # type: ignore[no-redef]
        assemble_chapter_audio,
        build_audio_mix_plan,
        mix_chapter_audio,
    )
    from audiobook_worker.model_settings import voxcpm2_paths  # type: ignore[no-redef]
    from voxcpm2_batch_adapter import BATCH_ADAPTER_VERSION, VoxCPM2BatchAdapter  # type: ignore[no-redef]
    from voxcpm2_runner import _atomic_write_wav  # type: ignore[no-redef]


BENCHMARK_FORMAT_VERSION = 1
_REQUIRED_BATCH_SIZES = (1, 2, 4)
_MINIMUM_MULTI_BATCH_THROUGHPUT_IMPROVEMENT = 0.03


class VoxCPM2BatchBenchmarkError(RuntimeError):
    """The benchmark fixture, runtime, or temporary output is invalid."""


def _required_text(value: object, *, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VoxCPM2BatchBenchmarkError(f"{name} is required.")
    return text


def _fixture_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Load a repeatable, cache-free four-or-more segment benchmark fixture."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VoxCPM2BatchBenchmarkError(f"Cannot read benchmark fixture: {error}") from error
    if not isinstance(raw, dict):
        raise VoxCPM2BatchBenchmarkError("benchmark fixture must be an object.")
    raw_items = raw.get("items")
    if not isinstance(raw_items, list):
        raise VoxCPM2BatchBenchmarkError("benchmark fixture items must be an array.")
    if len(raw_items) < max(_REQUIRED_BATCH_SIZES) or len(raw_items) % max(_REQUIRED_BATCH_SIZES):
        raise VoxCPM2BatchBenchmarkError(
            "benchmark fixture must contain a multiple of four representative items."
        )

    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(raw_items):
        if not isinstance(candidate, dict):
            raise VoxCPM2BatchBenchmarkError(f"benchmark item {index} must be an object.")
        item_id = _required_text(candidate.get("id"), name=f"benchmark item {index} id")
        if item_id in seen_ids:
            raise VoxCPM2BatchBenchmarkError(f"duplicate benchmark item id: {item_id}")
        reference_path = Path(
            _required_text(
                candidate.get("referenceWavPath"),
                name=f"benchmark item {item_id} referenceWavPath",
            )
        ).expanduser()
        if not reference_path.is_file():
            raise VoxCPM2BatchBenchmarkError(
                f"benchmark reference WAV is unavailable for {item_id}: {reference_path}"
            )
        text = _required_text(candidate.get("text"), name=f"benchmark item {item_id} text")
        delivery = " ".join(str(candidate.get("delivery") or "").split())
        signature = str(candidate.get("cacheSignature") or "").strip()
        if not signature:
            signature = hashlib.sha256(
                "\u0000".join((item_id, text, delivery, str(reference_path.resolve()))).encode("utf-8")
            ).hexdigest()
        items.append(
            {
                "id": item_id,
                "text": text,
                "delivery": delivery,
                "referenceWavPath": str(reference_path.resolve()),
                "cacheSignature": signature,
            }
        )
        seen_ids.add(item_id)
    return raw, items


def select_batch_size(runs: list[dict[str, Any]]) -> int:
    """Pick the fastest fully valid run, otherwise retain the B=1 fallback."""

    valid: list[dict[str, Any]] = []
    for run in runs:
        try:
            batch_size = int(run.get("batchSize"))
            throughput = float(run.get("audioSecondsPerWallSecond"))
        except (TypeError, ValueError):
            continue
        validation = run.get("validation")
        if (
            batch_size not in _REQUIRED_BATCH_SIZES
            or not isinstance(validation, dict)
            or run.get("status") != "succeeded"
            or not validation.get("allWavsReadable")
            or not validation.get("assemblyValid")
            or not validation.get("timelineValid")
            or not validation.get("finalMixValid")
            or run.get("memoryPressure") is True
            or throughput <= 0
        ):
            continue
        valid.append(run)
    baseline = next((run for run in valid if int(run["batchSize"]) == 1), None)
    if baseline is None:
        return 1
    # Preserve the lower-risk batch when measurements tie within floating-point
    # noise; a larger batch is only selected when it materially wins.
    best = max(
        valid,
        key=lambda run: (
            float(run["audioSecondsPerWallSecond"]),
            -int(run["batchSize"]),
        ),
    )
    if int(best["batchSize"]) == 1:
        return 1
    baseline_throughput = float(baseline["audioSecondsPerWallSecond"])
    if float(best["audioSecondsPerWallSecond"]) < baseline_throughput * (
        1 + _MINIMUM_MULTI_BATCH_THROUGHPUT_IMPROVEMENT
    ):
        return 1
    return int(best["batchSize"])


def _readable_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as error:
        raise VoxCPM2BatchBenchmarkError(f"benchmark wrote an unreadable WAV: {path}") from error
    if frames <= 0 or sample_rate <= 0:
        raise VoxCPM2BatchBenchmarkError(f"benchmark wrote an empty WAV: {path}")
    return frames / sample_rate


def _darwin_memory_counters() -> dict[str, int]:
    if sys.platform != "darwin":
        return {}
    counters: dict[str, int] = {}
    try:
        completed = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return counters
    if completed.returncode != 0:
        return counters
    for line in completed.stdout.splitlines():
        match = re.match(r"^Pages (pageouts|swapouts):\s+(\d+)", line.strip())
        if match:
            counters[match.group(1)] = int(match.group(2))
    try:
        swap = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return counters
    used = re.search(r"used\s*=\s*([0-9.]+)([KMG])", swap.stdout)
    if used:
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[used.group(2)]
        counters["swapUsedBytes"] = int(float(used.group(1)) * multiplier)
    return counters


def _memory_snapshot() -> dict[str, int]:
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.  The app's production target is
    # macOS, but retaining the platform distinction makes reports portable.
    rss_bytes = max_rss if sys.platform == "darwin" else max_rss * 1024
    snapshot = {"peakRssBytes": rss_bytes, **_darwin_memory_counters()}
    try:
        import torch

        if torch.backends.mps.is_available():
            snapshot["mpsAllocatedBytes"] = int(torch.mps.current_allocated_memory())
            snapshot["mpsDriverAllocatedBytes"] = int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return snapshot


def _memory_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in set(before) | set(after)
        if key not in {"peakRssBytes", "mpsAllocatedBytes", "mpsDriverAllocatedBytes"}
    }


def _validate_temporary_chapter(
    segment_paths: list[Path],
    items: list[dict[str, str]],
    directory: Path,
) -> dict[str, Any]:
    """Exercise source-order assembly, timeline mapping, and final-mix output."""

    tts_segments = [
        {
            "id": item["id"],
            "text": item["text"],
            "sourceSegmentIds": [f"source_{index:04d}"],
        }
        for index, item in enumerate(items)
    ]
    script = {
        "segments": [
            {"id": f"source_{index:04d}", "text": item["text"]}
            for index, item in enumerate(items)
        ],
        "audioPlan": {},
    }
    voice_path = directory / "chapter-voice.wav"
    mixed_path = directory / "chapter-mix.wav"
    assembled = assemble_chapter_audio(segment_paths, voice_path)
    plan = build_audio_mix_plan(segment_paths, tts_segments, script, directory / "assets")
    mixed = mix_chapter_audio(
        voice_path,
        segment_paths,
        tts_segments,
        script,
        directory / "assets",
        mixed_path,
        voice_gain=1.0,
    )
    expected_source_ids = [f"source_{index:04d}" for index in range(len(items))]
    mapped_source_ids = [
        source_id
        for timeline_item in plan.timeline
        for source_id in timeline_item.source_segment_ids
    ]
    return {
        "assemblyValid": _readable_duration(assembled.path) > 0,
        "timelineValid": mapped_source_ids == expected_source_ids,
        "finalMixValid": _readable_duration(mixed.path) > 0,
        "assembledDurationSeconds": assembled.duration_seconds,
        "mixedDurationSeconds": mixed.duration_seconds,
    }


def _run_batch_size(
    adapter: VoxCPM2BatchAdapter,
    items: list[dict[str, str]],
    *,
    batch_size: int,
    sample_rate: int,
    temporary_root: Path,
) -> dict[str, Any]:
    started_memory = _memory_snapshot()
    started_at = time.perf_counter()
    scenario_directory = temporary_root / f"batch-{batch_size}"
    segment_directory = scenario_directory / "segments"
    segment_paths: list[Path] = []
    audio_seconds = 0.0
    batch_count = 0
    try:
        for offset in range(0, len(items), batch_size):
            request_items = items[offset : offset + batch_size]
            results = adapter.generate_batch(request_items)
            if len(results) != len(request_items):
                raise VoxCPM2BatchBenchmarkError("batch adapter omitted a benchmark result")
            for item, result in zip(request_items, results, strict=True):
                if result.segment_id != item["id"]:
                    raise VoxCPM2BatchBenchmarkError(
                        f"batch adapter mapped {result.segment_id} to {item['id']}"
                    )
                output_path = segment_directory / f"{item['id']}.wav"
                _atomic_write_wav(output_path, result.waveform, sample_rate)
                audio_seconds += _readable_duration(output_path)
                segment_paths.append(output_path)
            batch_count += 1
        validation = {
            "allWavsReadable": len(segment_paths) == len(items)
            and all(_readable_duration(path) > 0 for path in segment_paths),
            **_validate_temporary_chapter(segment_paths, items, scenario_directory),
        }
        status = "succeeded"
        error: str | None = None
    except Exception as caught:
        validation = {
            "allWavsReadable": False,
            "assemblyValid": False,
            "timelineValid": False,
            "finalMixValid": False,
        }
        status = "failed"
        error = f"{type(caught).__name__}: {caught}"
    wall_seconds = time.perf_counter() - started_at
    finished_memory = _memory_snapshot()
    memory_delta = _memory_delta(started_memory, finished_memory)
    # Any newly consumed swap is a hard no for an automatic larger batch.
    memory_pressure = (
        memory_delta.get("swapUsedBytes", 0) > 0
        or memory_delta.get("swapouts", 0) > 0
        or memory_delta.get("pageouts", 0) > 0
    )
    return {
        "batchSize": batch_size,
        "status": status,
        "wallSeconds": round(wall_seconds, 6),
        "audioSeconds": round(audio_seconds, 6),
        "audioSecondsPerWallSecond": round(audio_seconds / wall_seconds, 6)
        if wall_seconds > 0
        else 0.0,
        "batchCount": batch_count,
        "validWavCount": len(segment_paths) if status == "succeeded" else 0,
        "validation": validation,
        "memoryBefore": started_memory,
        "memoryAfter": finished_memory,
        "memoryDelta": memory_delta,
        "memoryPressure": memory_pressure,
        "error": error,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_benchmark(
    fixture_path: Path,
    report_path: Path,
    selection_path: Path,
    *,
    model_path: Path,
    device: str,
) -> dict[str, Any]:
    fixture, items = load_fixture(fixture_path)
    del fixture
    if not model_path.is_dir():
        raise VoxCPM2BatchBenchmarkError(f"VoxCPM2 model directory is missing: {model_path}")
    if device not in {"auto", "mps", "cpu"}:
        raise VoxCPM2BatchBenchmarkError(f"unsupported VoxCPM2 device: {device}")

    from voxcpm import VoxCPM

    load_started_at = time.perf_counter()
    pipeline = VoxCPM.from_pretrained(
        str(model_path),
        load_denoiser=False,
        optimize=False,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started_at
    adapter = VoxCPM2BatchAdapter(pipeline)
    sample_rate = int(pipeline.tts_model.sample_rate)
    if sample_rate <= 0:
        raise VoxCPM2BatchBenchmarkError("VoxCPM2 reported an invalid sample rate")

    warm_started_at = time.perf_counter()
    for reference_path in sorted({item["referenceWavPath"] for item in items}):
        adapter._reference_prompt_cache(Path(reference_path))
    warm_seconds = time.perf_counter() - warm_started_at
    with tempfile.TemporaryDirectory(prefix="voxcpm2-batch-benchmark-") as temporary:
        temporary_root = Path(temporary)
        runs = [
            _run_batch_size(
                adapter,
                items,
                batch_size=batch_size,
                sample_rate=sample_rate,
                temporary_root=temporary_root,
            )
            for batch_size in _REQUIRED_BATCH_SIZES
        ]

    selected_batch_size = select_batch_size(runs)
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    report = {
        "version": BENCHMARK_FORMAT_VERSION,
        "adapterVersion": BATCH_ADAPTER_VERSION,
        "generatedAt": completed_at,
        "fixtureSha256": _fixture_digest(fixture_path),
        "fixtureItemCount": len(items),
        "modelPath": str(model_path.resolve()),
        "requestedDevice": device,
        "actualDevice": str(getattr(pipeline.tts_model, "device", "")),
        "sampleRate": sample_rate,
        "modelLoads": 1,
        "modelLoadSeconds": round(load_seconds, 6),
        "referenceWarmupSeconds": round(warm_seconds, 6),
        "runs": runs,
        "selectedBatchSize": selected_batch_size,
    }
    _write_json(report_path, report)
    _write_json(
        selection_path,
        {
            "version": BENCHMARK_FORMAT_VERSION,
            "adapterVersion": BATCH_ADAPTER_VERSION,
            "selectedBatchSize": selected_batch_size,
            "benchmarkReport": str(report_path.resolve()),
            "fixtureSha256": report["fixtureSha256"],
            "generatedAt": completed_at,
        },
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark VoxCPM2 tensor batch sizes 1, 2, and 4")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    arguments = parser.parse_args(argv)
    model_path = arguments.model_path or voxcpm2_paths()["model"]
    try:
        report = run_benchmark(
            arguments.fixture,
            arguments.report,
            arguments.selection,
            model_path=model_path,
            device=arguments.device,
        )
    except Exception as error:
        print(f"VoxCPM2 batch benchmark failed: {error}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "report": str(arguments.report),
                "selection": str(arguments.selection),
                "selectedBatchSize": report["selectedBatchSize"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
