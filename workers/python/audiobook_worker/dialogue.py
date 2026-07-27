from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


QUOTE_RE = re.compile(
    r'“(?P<curly>[^”]+)”|‘(?P<single>[^’]+)’|'
    r'「(?P<corner>[^」]+)」|『(?P<double_corner>[^』]+)』|'
    r'"(?P<ascii>[^"\n]+)"'
)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
SPEECH_VERBS = (
    r"said|asked|replied|whispered|shouted|answered|called|cried|exclaimed|"
    r"observed|remarked|interrupted|declared|protested|murmured|muttered|"
    r"continued|returned|added|laughed|sighed|urged|insisted|repeated|"
    r"began|rejoined|interposed|responded|demanded|inquired"
)
# Pattern 1: "Speaker said" or "Speaker cried"
TRAILING_SPEECH_TAG_RE = re.compile(
    rf"^\s*,?\s*(?P<speaker>(?:Mr\.|Mrs\.|Miss\.|Sir\s+)?[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?)\s+(?:{SPEECH_VERBS})\b"
)
# Pattern 2a: "said Speaker" — proper name (possibly with title)
TRAILING_SPEECH_TAG_INVERTED_RE = re.compile(
    rf"^\s*,?\s*(?:{SPEECH_VERBS})\s+(?P<speaker>(?:Mr\.|Mrs\.|Miss\.|Sir\s+)?[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?)"
)
# Pattern 2b: "cried his wife" / "said her sister" — relational noun (lowercase)
TRAILING_SPEECH_TAG_RELATIONAL_RE = re.compile(
    rf"^\s*,?\s*(?:{SPEECH_VERBS})\s+(?:his|her|the)\s+(?P<speaker>[a-z][A-Za-z.'-]*)"
)
CHINESE_SPEECH_VERBS = (
    "低声说道|轻声说道|高声说道|低声说|轻声说|高声说|冷笑道|喃喃道|"
    "嘟囔道|反问道|问道|答道|回答|喊道|叫道|嚷道|笑道|喜道|哭道|叹道|说道|"
    "说|问|喊|叫|嚷"
)
CHINESE_SPEAKER = r"[\u3400-\u4dbf\u4e00-\u9fff·]{1,8}"
CHINESE_LEADING_TAG_RE = re.compile(
    rf"(?:^|[。！？!?\n])\s*(?P<speaker>{CHINESE_SPEAKER})\s*"
    rf"(?:{CHINESE_SPEECH_VERBS})\s*[：:]?\s*$"
)
CHINESE_TRAILING_TAG_RE = re.compile(
    rf"^\s*[，,。.!！?？]?\s*(?P<speaker>{CHINESE_SPEAKER})\s*"
    rf"(?:{CHINESE_SPEECH_VERBS})(?=$|[，,。.!！?？\s])"
)
ENGLISH_LEADING_TAG_RE = re.compile(
    rf"^\s*(?P<speaker>(?:Mr\.|Mrs\.|Miss\.|Ms\.|Sir\s+)?"
    rf"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?)\s+"
    rf"(?:{SPEECH_VERBS})\s*[,：:]?\s*$"
)

_CHINESE_PRONOUN_SPEAKERS = {
    "我",
    "你",
    "他",
    "她",
    "它",
    "我俩",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "自己",
}

# These expressions describe text that is being mentioned, named, written,
# quoted, or used as a sound effect.  They are deliberately context based:
# a short quote at the beginning of a paragraph is still allowed to be a
# genuine line of dialogue when there is no stronger contrary evidence.
CHINESE_QUOTED_MATERIAL_CONTEXT_RE = re.compile(
    r"(?:被(?:人)?称(?:为|作|过|呼)?|称(?:为|作|过|呼)?|"
    r"叫作|叫做|唤作|唤做|所谓|名为|因为|由于|封为|册封为|"
    r"尊称|称呼|"
    r"这(?:个|种)?称呼|这(?:两|两三个|几)?个字|这句话|这句|"
    r"写着|写道|题为|标着|刻着|刻有|贴着|挂着|上面写|"
    r"纸上|信中|书中|诗中|歌词中|公告中|碑上|牌上|匾上|"
    r"梦中|回忆中|想起(?:了)?|提到|提及|记载|记录)"
)
CHINESE_QUOTED_MATERIAL_AFTER_RE = re.compile(
    r"^[，,、；;：:。.!！?？\s]*(?:这(?:个|两个)?(?:字|词|称呼|名字|说法|句话)|"
    r"一声|一阵|的(?:称呼|名字|字样)|(?:被|给)?(?:写|刻|标)着|"
    r"后面|之后|被|而|死|成|一事|一案|一词|一字)"
)
CHINESE_SOUND_EFFECT_CONTENT_RE = re.compile(
    r"^(?:吱呀|咯吱|咔嚓|砰|啪|嘭|扑通|哗啦|咣当|轰隆|"
    r"呼啦|呜咽|呜呜|嘶|嗖|嘎吱|喀嚓|喵|汪|喔|咳咳)[！!。．…，,：:？?]*$"
)
ENGLISH_QUOTED_MATERIAL_CONTEXT_RE = re.compile(
    r"\b(?:called|known\s+as|named|titled|labelled|labeled|"
    r"referred\s+to\s+as|the\s+word|the\s+term|the\s+phrase|"
    r"written|reads|listed\s+as|according\s+to|book|letter|poem|song)\b",
    re.IGNORECASE,
)
ENGLISH_SPEECH_PREFIX_RE = re.compile(
    rf"(?:{SPEECH_VERBS})\s*[,：:]?\s*$", re.IGNORECASE
)

SegmentType = Literal["narration", "dialogue"]


@dataclass(frozen=True)
class DialogueSegment:
    type: SegmentType
    text: str
    start_offset: int
    end_offset: int
    speaker_hint: str | None = None
    warnings: list[str] = field(default_factory=list)


def detect_text_language(text: str) -> str:
    han_count = len(CJK_RE.findall(text))
    ascii_count = len(ASCII_LETTER_RE.findall(text))
    if han_count >= 1 and ascii_count == 0:
        return "zh"
    if han_count >= 2 and han_count >= ascii_count * 0.25:
        return "zh"
    return "en"


def resolve_text_language(text: str, requested: str | None = None) -> str:
    """Resolve the analysis language from the chapter text.

    The source text is the authority for role analysis. In particular, do not
    let a stale UI/default value such as "en" send an obviously Chinese
    chapter through the English quote repair and LLM prompt path.
    """
    detected = detect_text_language(text)
    if not requested:
        return detected

    normalized = requested.strip().lower().split("-", 1)[0]
    if normalized not in {"zh", "en"}:
        return detected
    return detected if normalized != detected else normalized


def segment_dialogue(text: str, language: str | None = None) -> list[DialogueSegment]:
    language = resolve_text_language(text, language)
    scan_text = _normalize_malformed_chinese_quote_lines(text) if language.startswith("zh") else text
    segments: list[DialogueSegment] = []
    cursor = 0
    for match in QUOTE_RE.finditer(scan_text):
        if match.start() > cursor:
            narration = text[cursor : match.start()].strip()
            if narration:
                segments.append(
                    DialogueSegment(
                        type="narration",
                        text=narration,
                        start_offset=cursor,
                        end_offset=match.start(),
                    )
                )

        dialogue_text = _quoted_text(match).strip()
        speaker_hint = _infer_speaker(
            text_before_quote=text[: match.start()],
            text_after_quote=text[match.end() :],
            language=language,
        )
        quoted_material = not speaker_hint and _is_quoted_material(
            dialogue_text,
            text_before_quote=text[: match.start()],
            text_after_quote=text[match.end() :],
            language=language,
        )
        segment_type: SegmentType = "narration" if quoted_material else "dialogue"
        warnings = ["quoted_material"] if quoted_material else (
            [] if speaker_hint else ["speaker_unknown"]
        )
        segments.append(
            DialogueSegment(
                type=segment_type,
                text=dialogue_text,
                start_offset=match.start(),
                end_offset=match.end(),
                speaker_hint=None if quoted_material else speaker_hint,
                warnings=warnings,
            )
        )
        cursor = match.end()

    if cursor < len(text):
        narration = text[cursor:].strip()
        if narration:
            segments.append(
                DialogueSegment(
                    type="narration",
                    text=narration,
                    start_offset=cursor,
                    end_offset=len(text),
                )
            )

    return segments or [
        DialogueSegment(
            type="narration",
            text=text.strip(),
            start_offset=0,
            end_offset=len(text),
        )
    ]


def _normalize_malformed_chinese_quote_lines(text: str) -> str:
    """Repair line-delimited OCR quote direction without changing source offsets."""
    normalized: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        opener = body.find("“")
        if opener < 0 and body.startswith("”"):
            body = "“" + body[1:]
            opener = 0
        if opener >= 0 and "”" not in body[opener + 1 :]:
            repaired = False
            for closer in ("“", "’", "'", '"'):
                close_at = body.rfind(closer)
                if close_at > opener:
                    body = body[:close_at] + "”" + body[close_at + 1 :]
                    repaired = True
                    break
            if not repaired and ending:
                body += "”"
                ending = ending[1:]
        normalized.append(body + ending)
    return "".join(normalized)


def _quoted_text(match: re.Match[str]) -> str:
    return next(group for group in match.groups() if group is not None)


def _infer_speaker(
    *, text_before_quote: str, text_after_quote: str, language: str
) -> str | None:
    if language.startswith("zh"):
        leading = CHINESE_LEADING_TAG_RE.search(text_before_quote)
        if leading:
            speaker = leading.group("speaker")
            return None if speaker in _CHINESE_PRONOUN_SPEAKERS else speaker
        trailing = CHINESE_TRAILING_TAG_RE.match(text_after_quote)
        if trailing:
            speaker = trailing.group("speaker")
            return None if speaker in _CHINESE_PRONOUN_SPEAKERS else speaker
    leading = ENGLISH_LEADING_TAG_RE.match(text_before_quote)
    if leading:
        return leading.group("speaker")
    return _infer_english_trailing_speaker(text_after_quote)


def _infer_english_trailing_speaker(text_after_quote: str) -> str | None:
    for pattern in (
        TRAILING_SPEECH_TAG_RE,
        TRAILING_SPEECH_TAG_INVERTED_RE,
        TRAILING_SPEECH_TAG_RELATIONAL_RE,
    ):
        match = pattern.match(text_after_quote)
        if match:
            return match.group("speaker")
    return None


def _is_quoted_material(
    content: str,
    *,
    text_before_quote: str,
    text_after_quote: str,
    language: str,
) -> bool:
    """Identify quoted text that should be read as narration, not a character.

    Quotation marks are not sufficient evidence of direct speech.  Novels
    commonly quote titles, names, labels, remembered phrases, and sound
    effects inside an otherwise narrative sentence.  The classifier only
    takes the conservative, high-signal cases here; ordinary untagged quotes
    remain dialogue candidates for the LLM to resolve from context.
    """
    before = _local_quote_context(text_before_quote)
    after = text_after_quote[:100]
    compact_content = re.sub(r"\s+", "", content)

    if not compact_content:
        return False

    # Explicit speech syntax wins over all material heuristics.  This also
    # covers pronoun-led Chinese speech such as “他说‘走吧’”.
    if language.startswith("zh"):
        if re.search(rf"(?:{CHINESE_SPEECH_VERBS})\s*[：:]?\s*$", before):
            return False
    elif ENGLISH_SPEECH_PREFIX_RE.search(before):
        return False

    if language.startswith("zh"):
        if CHINESE_QUOTED_MATERIAL_CONTEXT_RE.search(before):
            return True
        if CHINESE_QUOTED_MATERIAL_AFTER_RE.match(after):
            return True
        if CHINESE_SOUND_EFFECT_CONTENT_RE.fullmatch(compact_content):
            return True
        # A book/poem/notice followed by a colon and a quote is a citation,
        # even when the cited material contains sentence punctuation.
        if re.search(r"《[^》]{1,80}》\s*[：:]\s*$", before):
            return True
        return False

    return bool(ENGLISH_QUOTED_MATERIAL_CONTEXT_RE.search(before))


def _local_quote_context(text_before_quote: str) -> str:
    """Return context belonging to this quote, not an earlier quoted line."""
    # OCR often uses an opening curly mark where the closing mark should be;
    # treating either curly direction as a boundary prevents an old malformed
    # quote from contaminating the next line's classification.
    closing_marks = "“”‘’「」『』\""
    previous_quote = max(text_before_quote.rfind(mark) for mark in closing_marks)
    if previous_quote >= 0:
        text_before_quote = text_before_quote[previous_quote + 1 :]
    return text_before_quote[-100:]
