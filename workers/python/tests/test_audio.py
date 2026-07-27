import wave
from pathlib import Path

from audiobook_worker.audio import assemble_chapter_audio
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
