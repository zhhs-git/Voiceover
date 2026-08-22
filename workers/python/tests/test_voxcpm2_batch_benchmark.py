import json
from pathlib import Path

import pytest

from audiobook_worker.voxcpm2_batch_benchmark import (
    VoxCPM2BatchBenchmarkError,
    load_fixture,
    select_batch_size,
)


def _successful_run(batch_size: int, throughput: float, *, memory_pressure: bool = False):
    return {
        "batchSize": batch_size,
        "status": "succeeded",
        "audioSecondsPerWallSecond": throughput,
        "memoryPressure": memory_pressure,
        "validation": {
            "allWavsReadable": True,
            "assemblyValid": True,
            "timelineValid": True,
            "finalMixValid": True,
        },
    }


def test_load_fixture_normalizes_items_and_requires_four_reference_wavs(tmp_path: Path):
    references = []
    for index in range(4):
        reference = tmp_path / f"profile-{index}.wav"
        reference.write_bytes(b"reference")
        references.append(reference)
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": f"item-{index}",
                        "text": f"第 {index} 条测试文本。",
                        "delivery": "  自然  克制  ",
                        "referenceWavPath": str(reference),
                    }
                    for index, reference in enumerate(references)
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _fixture, items = load_fixture(fixture_path)

    assert len(items) == 4
    assert items[0]["delivery"] == "自然 克制"
    assert items[0]["cacheSignature"]
    assert items[0]["referenceWavPath"] == str(references[0].resolve())


def test_load_fixture_rejects_a_non_multiple_of_four(tmp_path: Path):
    reference = tmp_path / "profile.wav"
    reference.write_bytes(b"reference")
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": f"item-{index}",
                        "text": "测试文本。",
                        "referenceWavPath": str(reference),
                    }
                    for index in range(3)
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(VoxCPM2BatchBenchmarkError, match="multiple of four"):
        load_fixture(fixture_path)


def test_selection_uses_fastest_valid_run_and_rejects_memory_pressure():
    runs = [
        _successful_run(1, 1.0),
        _successful_run(2, 1.3),
        _successful_run(4, 1.7, memory_pressure=True),
    ]

    assert select_batch_size(runs) == 2
    runs[1]["validation"]["timelineValid"] = False
    assert select_batch_size(runs) == 1
