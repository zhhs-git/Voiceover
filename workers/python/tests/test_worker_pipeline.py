import json
import os
import subprocess
import sys
from pathlib import Path


def run_worker(command: str, input_payload: dict, tmp_path: Path) -> dict:
    input_path = tmp_path / f"{command}.input.json"
    output_path = tmp_path / f"{command}.output.json"
    input_path.write_text(json.dumps(input_payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "audiobook_worker.cli", command, str(input_path), str(output_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "AUDIOBOOK_LLM_MODEL": "mock"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_generates_script_segment_audio_and_chapter_audio(tmp_path: Path):
    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text('She waited. "Come in," Elizabeth said.', encoding="utf-8")
    script_dir = tmp_path / "scripts"

    analyze = run_worker(
        "analyze_chapter",
        {
            "bookId": "book_123",
            "chapterId": "chapter_001",
            "title": "Chapter 1",
            "language": "en",
            "chapterTextPath": str(chapter_path),
            "outputDirectory": str(script_dir),
        },
        tmp_path,
    )

    assert analyze["status"] == "succeeded"
    script_path = Path(analyze["artifacts"][0]["path"])
    script = json.loads(script_path.read_text(encoding="utf-8"))
    assert script["segments"][1]["speakerId"] == script["characters"][0]["id"]

    segment_dir = tmp_path / "segments"
    for segment in script["segments"]:
        synthesize = run_worker(
            "synthesize_segment_audio",
            {
                "bookId": "book_123",
                "chapterId": "chapter_001",
                "segmentId": segment["id"],
                "scriptPath": str(script_path),
                "outputDirectory": str(segment_dir),
                "backend": "mock",
            },
            tmp_path,
        )
        assert synthesize["status"] == "succeeded"

    assemble = run_worker(
        "assemble_chapter_audio",
        {
            "bookId": "book_123",
            "chapterId": "chapter_001",
            "segmentAudioDirectory": str(segment_dir),
            "outputPath": str(tmp_path / "chapter_001.wav"),
        },
        tmp_path,
    )

    assert assemble["status"] == "succeeded"
    assert Path(assemble["artifacts"][0]["path"]).exists()


def test_analyzes_chinese_chapter_without_explicit_language(tmp_path: Path):
    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text("院子里很安静。张三说道：“走吧。”", encoding="utf-8")

    result = run_worker(
        "analyze_chapter",
        {
            "bookId": "book_zh",
            "chapterId": "chapter_001",
            "title": "第一章",
            "chapterTextPath": str(chapter_path),
            "outputDirectory": str(tmp_path / "scripts"),
        },
        tmp_path,
    )

    script = json.loads(Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8"))
    dialogue = next(segment for segment in script["segments"] if segment["type"] == "dialogue")
    assert script["language"] == "zh"
    assert dialogue["text"] == "走吧。"
    assert dialogue["speakerId"] == script["characters"][0]["id"]


def test_corrections_flow_applies_alias_merge_and_regenerates(tmp_path: Path):
    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text('"Over here," Lizzy called. "Coming," Elizabeth replied.', encoding="utf-8")
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()

    # Step 1: Run initial analysis
    analyze = run_worker(
        "analyze_chapter",
        {
            "bookId": "book_123",
            "chapterId": "chapter_001",
            "title": "Chapter 1",
            "language": "en",
            "chapterTextPath": str(chapter_path),
            "outputDirectory": str(script_dir),
        },
        tmp_path,
    )
    assert analyze["status"] == "succeeded"

    # Step 2: Apply alias merge correction
    corrections = run_worker(
        "apply_corrections",
        {
            "bookId": "book_123",
            "chapters": [
                {"chapterId": "chapter_001", "textPath": str(chapter_path), "title": "Chapter 1"}
            ],
            "corrections": {
                "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
                "genderOverrides": [],
                "voiceOverrides": [],
            },
            "outputDirectory": str(script_dir),
            "language": "en",
        },
        tmp_path,
    )
    assert corrections["status"] == "succeeded"
    corrected_path = Path(corrections["artifacts"][0]["path"])
    corrected_script = json.loads(corrected_path.read_text())

    # Speakers should be unified to elizabeth only
    speakers = {seg["speakerId"] for seg in corrected_script["segments"] if seg["type"] == "dialogue"}
    assert len(speakers) == 1
    assert speakers == {corrected_script["characters"][0]["id"]}

    # Characters should have only one elizabeth entry
    character_ids = {c["id"] for c in corrected_script["characters"]}
    assert len(character_ids) == 1
    assert next(iter(character_ids)) == corrected_script["characters"][0]["id"]


def test_corrections_auto_detects_chinese_when_stale_language_is_sent(tmp_path: Path):
    chapter_path = tmp_path / "chapter_001.txt"
    chapter_path.write_text(
        "“这满街的白幡是做什么?“\n"
        "”嗬，官老爷都系白腰带?’\n"
        "“你是几日没出门了，连这都不知道?护国长公主薨了啊!“",
        encoding="utf-8",
    )
    script_dir = tmp_path / "scripts"

    result = run_worker(
        "apply_corrections",
        {
            "bookId": "book_zh",
            "chapters": [
                {
                    "chapterId": "chapter_001",
                    "textPath": str(chapter_path),
                    "title": "第一章",
                }
            ],
            "corrections": {},
            # Simulates the stale value that previously came from the UI.
            "language": "en",
            "outputDirectory": str(script_dir),
        },
        tmp_path,
    )

    script = json.loads(Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8"))
    assert script["language"] == "zh"
    assert [segment["text"] for segment in script["segments"][:3]] == [
        "这满街的白幡是做什么?",
        "嗬，官老爷都系白腰带?",
        "你是几日没出门了，连这都不知道?护国长公主薨了啊!",
    ]
