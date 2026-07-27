from __future__ import annotations

import re
from dataclasses import dataclass


CHAPTER_HEADING_RE = re.compile(
    r"^(?:(?:chapter|part|book|section)\s+(?:\d+|[ivxlcdm]+)\b.*|"
    r"第[零〇一二三四五六七八九十百千万两\d]+[章节卷部回篇].*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Chapter:
    id: str
    title: str
    text: str
    start_offset: int
    end_offset: int
    confidence: float


def detect_chapters(text: str) -> list[Chapter]:
    matches = list(CHAPTER_HEADING_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return [
            Chapter(
                id="chapter_001",
                title="Chapter 1",
                text=stripped,
                start_offset=0,
                end_offset=len(text),
                confidence=0.35,
            )
        ]

    chapters: list[Chapter] = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body_start = match.end()
        body = text[body_start:next_start].strip()
        chapters.append(
            Chapter(
                id=f"chapter_{index + 1:03d}",
                title=match.group(0).strip(),
                text=body,
                start_offset=match.start(),
                end_offset=next_start,
                confidence=0.82,
            )
        )
    return chapters
