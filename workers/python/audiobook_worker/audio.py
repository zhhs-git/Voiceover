from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChapterAudioArtifact:
    kind: str
    path: Path
    duration_seconds: float


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
