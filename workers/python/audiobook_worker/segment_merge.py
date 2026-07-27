from __future__ import annotations

from copy import deepcopy
import re


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# MiMo can return a syntactically valid WAV even when a long Chinese request
# is cut short.  Keep requests deliberately small and split at punctuation
# whenever possible.  The value is intentionally conservative because the
# backend does not return a transcript that we could use for completeness
# verification.
DEFAULT_MAX_TTS_WORDS = 200
DEFAULT_MAX_TTS_CHARACTERS = 40
_CJK_BOUNDARIES = frozenset("。！？!?；;，,、：:\n\r…")


def merge_tts_segments(
    segments: list[dict],
    *,
    max_words: int = DEFAULT_MAX_TTS_WORDS,
    max_characters: int = DEFAULT_MAX_TTS_CHARACTERS,
) -> list[dict]:
    expanded = split_tts_segments(
        segments,
        max_words=max_words,
        max_characters=max_characters,
    )
    merged: list[dict] = []

    for segment in expanded:
        current = deepcopy(segment)
        current["sourceSegmentIds"] = list(
            segment.get("sourceSegmentIds") or [segment["id"]]
        )
        current_word_count = len(current.get("text", "").split())

        if not merged:
            current["_wordCount"] = current_word_count
            merged.append(current)
            continue

        previous = merged[-1]
        combined_text = _join_text(previous.get("text", ""), current.get("text", ""))
        if _can_merge(previous, current) and _within_tts_limits(
            combined_text,
            max_words=max_words,
            max_characters=max_characters,
        ):
            previous["text"] = combined_text
            previous["sourceSegmentIds"].extend(current["sourceSegmentIds"])
            previous["_wordCount"] = len(combined_text.split())
        else:
            current["_wordCount"] = current_word_count
            merged.append(current)

    for segment in merged:
        segment.pop("_wordCount", None)
    return merged


def split_tts_segments(
    segments: list[dict],
    *,
    max_words: int = DEFAULT_MAX_TTS_WORDS,
    max_characters: int = DEFAULT_MAX_TTS_CHARACTERS,
) -> list[dict]:
    """Split individual script segments before they reach a TTS backend.

    Chinese text has no whitespace, so word-count based limits are ineffective
    for it.  Every returned item keeps the original text verbatim and records
    the source segment id, allowing the caller to verify complete coverage.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")

    result: list[dict] = []
    for segment in segments:
        source_id = str(segment["id"])
        source_ids = list(segment.get("sourceSegmentIds") or [source_id])
        text = str(segment.get("text", ""))
        chunks = _split_text_for_tts(
            text,
            max_words=max_words,
            max_characters=max_characters,
        )
        if len(chunks) == 1:
            item = deepcopy(segment)
            item["sourceSegmentIds"] = source_ids
            result.append(item)
            continue

        for part_index, chunk in enumerate(chunks, start=1):
            item = deepcopy(segment)
            item["id"] = f"{source_id}_part_{part_index:04d}"
            item["text"] = chunk
            item["sourceSegmentIds"] = source_ids.copy()
            item["splitPart"] = part_index
            item["splitPartCount"] = len(chunks)
            result.append(item)
    return result


def _split_text_for_tts(
    text: str,
    *,
    max_words: int,
    max_characters: int,
) -> list[str]:
    if not _within_tts_limits(
        text,
        max_words=max_words,
        max_characters=max_characters,
    ):
        return _split_cjk_text(text, max_characters) if _contains_cjk(text) else _split_word_text(text, max_words)
    return [text]


def _split_cjk_text(text: str, max_characters: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_characters:
            chunks.append(text[start:])
            break

        hard_end = start + max_characters
        boundary = _last_safe_boundary(text, start, hard_end, max_characters)
        chunks.append(text[start:boundary])
        start = boundary
    return chunks or [""]


def _last_safe_boundary(text: str, start: int, hard_end: int, max_characters: int) -> int:
    candidates = [
        position
        for position in range(start + 1, hard_end + 1)
        if text[position - 1] in _CJK_BOUNDARIES
    ]
    # Do not create a one-character punctuation chunk merely because the
    # source starts with a stray full stop or an OCR line break.
    minimum_preferred_length = max(8, max_characters // 2)
    preferred = [
        position
        for position in candidates
        if position - start >= minimum_preferred_length
    ]
    return max(preferred or [], default=hard_end)


def _split_word_text(text: str, max_words: int) -> list[str]:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start_word = 0
    while start_word < len(words):
        end_word = min(start_word + max_words, len(words))
        start = 0 if start_word == 0 else words[start_word].start()
        end = len(text) if end_word == len(words) else words[end_word].start()
        chunks.append(text[start:end])
        start_word = end_word
    return chunks


def _within_tts_limits(text: str, *, max_words: int, max_characters: int) -> bool:
    if _contains_cjk(text):
        return len(text) <= max_characters
    return len(text.split()) <= max_words


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def _join_text(left: str, right: str) -> str:
    left = str(left)
    right = str(right)
    if not left:
        return right
    if not right:
        return left
    if _contains_cjk(left + right) or left[-1].isspace() or right[0].isspace():
        return left + right
    return f"{left} {right}"


def _can_merge(left: dict, right: dict) -> bool:
    return (
        left.get("voiceId") == right.get("voiceId")
        and left.get("fallbackVoiceId") == right.get("fallbackVoiceId")
        and left.get("voiceDescription") == right.get("voiceDescription")
        and left.get("emotion", "neutral") == right.get("emotion", "neutral")
        and left.get("pace", "normal") == right.get("pace", "normal")
    )
