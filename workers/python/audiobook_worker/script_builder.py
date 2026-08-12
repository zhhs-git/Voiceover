from __future__ import annotations

import hashlib
import re

from audiobook_worker.dialogue import resolve_text_language, segment_dialogue
from audiobook_worker.llm import (
    CharacterContext,
    ChapterAnalysisRequest,
    MockLLMAnalyzer,
)


NARRATOR_FEMALE_VOICE_ID = "narrator_female"
NARRATOR_MALE_VOICE_ID = "narrator_male"
# Kept for scripts produced before the book-scoped narrator setting existed.
# It is the same stable female voice as narrator_female.
LEGACY_NARRATOR_VOICE_ID = "narrator_default"
DEFAULT_NARRATOR_VOICE_ID = LEGACY_NARRATOR_VOICE_ID


def normalize_narrator_voice_id(value: object = None) -> str:
    """Return one of the two supported stable narrator identities.

    Unknown, missing, and legacy values intentionally resolve to the existing
    female narrator so old books remain compatible and never become random.
    """
    normalized = str(value or "").strip().casefold()
    if normalized == NARRATOR_MALE_VOICE_ID:
        return NARRATOR_MALE_VOICE_ID
    if normalized == NARRATOR_FEMALE_VOICE_ID:
        return NARRATOR_FEMALE_VOICE_ID
    return DEFAULT_NARRATOR_VOICE_ID


def is_narrator_voice_id(value: object) -> bool:
    return str(value or "").strip().casefold() in {
        LEGACY_NARRATOR_VOICE_ID,
        NARRATOR_FEMALE_VOICE_ID,
        NARRATOR_MALE_VOICE_ID,
    }


VOICE_REGISTRY = {
    # ── Narrators ───────────────────────────────────────────────────────
    "narrator_default": {
        "id": "narrator_default",
        "displayName": "Narrator (Warm Female)",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_heart",
    },
    "narrator_female": {
        "id": "narrator_female",
        "displayName": "Narrator (Female)",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_heart",
    },
    "narrator_male": {
        "id": "narrator_male",
        "displayName": "Narrator (Stable Male)",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_michael",
    },

    # ── Female voices (5 distinct) ──────────────────────────────────────
    "female_adult_01": {
        "id": "female_adult_01",
        "displayName": "Female — Warm & Expressive",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "happy", "sad", "angry", "excited", "afraid"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_heart",
    },
    "female_adult_02": {
        "id": "female_adult_02",
        "displayName": "Female — Bright & Clear",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "happy", "excited", "angry"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_bella",
    },
    "female_adult_03": {
        "id": "female_adult_03",
        "displayName": "Female — Gentle & Soft",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "sad", "afraid", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_nicole",
    },
    "female_adult_04": {
        "id": "female_adult_04",
        "displayName": "Female — Energetic & Lively",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "excited", "happy", "angry"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_sky",
    },
    "female_adult_05": {
        "id": "female_adult_05",
        "displayName": "Female — Measured & Elegant",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_sarah",
    },

    # ── Male voices (5 distinct) ────────────────────────────────────────
    "male_adult_01": {
        "id": "male_adult_01",
        "displayName": "Male — Deep & Resonant",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "angry", "tense", "excited"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_michael",
    },
    "male_adult_02": {
        "id": "male_adult_02",
        "displayName": "Male — Crisp & Articulate",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "happy", "angry"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_liam",
    },
    "male_adult_03": {
        "id": "male_adult_03",
        "displayName": "Male — Warm & Friendly",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "happy", "excited", "sad"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_onyx",
    },
    "male_adult_04": {
        "id": "male_adult_04",
        "displayName": "Male — Strong & Authoritative",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "angry", "tense", "excited"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_eric",
    },
    "male_adult_05": {
        "id": "male_adult_05",
        "displayName": "Male — Measured & Calm",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "sad", "tense", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "am_puck",
    },

    # ── British voices (for period works like Austen) ────────────────────
    "female_british_01": {
        "id": "female_british_01",
        "displayName": "Female — British (Bright)",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "happy", "excited", "sad"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "bf_isabella",
    },
    "female_british_02": {
        "id": "female_british_02",
        "displayName": "Female — British (Elegant)",
        "genderPresentation": "female",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "tense", "sad", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "bf_lily",
    },
    "male_british_01": {
        "id": "male_british_01",
        "displayName": "Male — British (Refined)",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "angry", "tense", "happy"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "bm_george",
    },
    "male_british_02": {
        "id": "male_british_02",
        "displayName": "Male — British (Warm)",
        "genderPresentation": "male",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral", "happy", "sad", "excited"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "bm_lewis",
    },

    # ── Neutral / fallback ──────────────────────────────────────────────
    "neutral_dialogue_01": {
        "id": "neutral_dialogue_01",
        "displayName": "Neutral Dialogue",
        "genderPresentation": "neutral",
        "ageClass": "adult",
        "languages": ["en"],
        "styles": ["neutral"],
        "backend": "kokoro",
        "licenseNotes": "Kokoro-82M Apache 2.0",
        "kokoroVoice": "af_nicole",
    },
}

# The fixed catalog remains available for explicit reader choices and for
# local-backend fallback.  MiMo automatic character voices are designed from
# an identity-specific prompt instead of being constrained to these entries.
_FEMALE_VOICE_POOL = [
    "female_adult_01",
    "female_adult_02",
    "female_adult_03",
    "female_adult_04",
    "female_adult_05",
]

_MALE_VOICE_POOL = [
    "male_adult_01",
    "male_adult_02",
    "male_adult_03",
    "male_adult_04",
    "male_adult_05",
]


# Bump this when the automatic routing rules change.  Version 3 moves
# automatic characters from the fixed catalog to independent identity voices.
VOICE_ASSIGNMENT_VERSION = 3
_AUTO_CHARACTER_VOICE_PREFIX = "character_auto_"

# Character IDs are book-scoped registry IDs.  The language model may return a
# temporary candidate ID, but that value must never become the durable identity
# used by scripts, the database, or voice routing.
_SYSTEM_CHARACTER_ID_PREFIX = "char_"
_IDENTITY_CONFIDENCE_THRESHOLD = 0.6
_IDENTITY_STATUSES = frozenset({"provisional", "confirmed", "merged"})

_AGE_CLASSES = frozenset({"child", "young", "adult", "older", "unknown"})
_VOICE_SOURCES = frozenset({"auto", "manual"})

# These preferences are only used when a local backend needs a finite fallback
# voice.  They do not constrain MiMo's generated character voice design.
_VOICE_PREFERENCES: dict[str, dict[str, list[str]]] = {
    "female": {
        "child": ["female_adult_02", "female_adult_04", "female_adult_03"],
        "young": ["female_adult_02", "female_adult_04", "female_adult_03"],
        "adult": ["female_adult_01", "female_adult_03", "female_adult_05"],
        "older": ["female_adult_05", "female_adult_01", "female_adult_03"],
        "unknown": _FEMALE_VOICE_POOL,
    },
    "male": {
        "child": ["male_adult_02", "male_adult_03"],
        "young": ["male_adult_02", "male_adult_03"],
        "adult": ["male_adult_02", "male_adult_03", "male_adult_05"],
        "older": ["male_adult_01", "male_adult_04", "male_adult_05"],
        "unknown": _MALE_VOICE_POOL,
    },
}

_PARENT_VOICE_PREFERENCES: dict[str, list[str]] = {
    "female": ["female_adult_05", "female_adult_01", "female_adult_03"],
    "male": ["male_adult_01", "male_adult_04", "male_adult_05"],
}

_IDENTITY_TERMS = {
    "elder": (
        "祖父", "祖母", "爷爷", "奶奶", "外公", "外婆", "老爷子", "老太太",
        "老夫人", "老人", "老者", "老汉", "老太", "grandfather", "grandmother",
        "grandpa", "grandma", "elderly", "old man", "old woman",
    ),
    "parent": (
        "父亲", "母亲", "爸爸", "妈妈", "爹", "娘", "爹爹", "娘亲", "父王", "母后",
        "father", "mother", "dad", "mom", "mum",
    ),
    "child": (
        "小孩", "孩子", "孩童", "幼童", "宝宝", "娃娃", "童子", "小童", "小姑娘",
        "小男孩", "小女孩", "child", "kid", "baby", "toddler", "boy", "girl",
    ),
    "young": (
        "儿子", "女儿", "少爷", "小姐", "少年", "少女", "青年", "公子", "姑娘", "丫头",
        "小子", "young man", "young woman", "son", "daughter", "teen", "teenager",
    ),
}


def build_chapter_script(
    *,
    book_id: str,
    chapter_id: str,
    title: str,
    text: str,
    language: str,
    analyzer=None,
    known_characters: list[dict] | None = None,
    analysis_stage_callback=None,
    analysis_cached_stages: dict[str, dict] | None = None,
    analysis_resume_from_stage: str | None = None,
    narrator_voice_id: str | None = None,
) -> dict:
    narrator_voice_id = normalize_narrator_voice_id(narrator_voice_id)
    language = resolve_text_language(text, language)
    raw_segments = segment_dialogue(text, language=language)
    analyzer = analyzer or MockLLMAnalyzer()

    # Convert known_characters dicts (from frontend/DB) to the script shape.
    # The canonical name is the identity/voice key; a model-generated id is
    # only an internal reference and is allowed to change between responses.
    known_script_characters, known_id_replacements = _merge_script_characters_with_replacements(
        [_known_character_to_script(c) for c in (known_characters or [])],
        [],
        book_id=book_id,
        assign_system_ids=False,
    )
    ctx = [
        CharacterContext(
            id=character["id"],
            canonical_name=character["canonicalName"],
            aliases=character["aliases"],
            gender=character["gender"],
            age_class=character["ageClass"],
            voice_design=(
                str(character.get("voiceDesign") or "")
                if str(character.get("voiceDesignSource") or "").strip().lower() != "fallback"
                else ""
            ),
        )
        for character in known_script_characters
    ]

    analysis = analyzer.analyze_chapter(
        ChapterAnalysisRequest(
            book_id=book_id,
            chapter_id=chapter_id,
            text=text,
            language=language,
            known_characters=ctx,
            stage_callback=analysis_stage_callback,
            cached_stages=analysis_cached_stages or {},
            resume_from_stage=analysis_resume_from_stage,
        )
    )
    annotations = {
        annotation.segment_index: annotation for annotation in analysis.segment_annotations
    }
    characters, discovered_id_replacements = _merge_script_characters_with_replacements(
        known_script_characters,
        [_character_to_script(character) for character in analysis.characters],
        book_id=book_id,
        assign_system_ids=True,
    )
    id_replacements = _resolve_id_replacements(
        {**known_id_replacements, **discovered_id_replacements}
    )

    segments = []
    voice_ids = {narrator_voice_id}
    for index, raw_segment in enumerate(raw_segments):
        annotation = annotations.get(index)
        speaker_id = "narrator"
        emotion = "neutral"
        pace = "normal"
        confidence = 0.9
        warnings = list(raw_segment.warnings)

        segment_type = raw_segment.type
        if annotation:
            warnings = sorted(set(warnings + annotation.warnings))
            pace = _normalize_pace(annotation.pace)

        # A quoted-material marker is a high-confidence semantic decision from
        # the pre-segmentation pass (or from the model).  Never let a bad model
        # annotation turn a title, label, citation, or sound effect into a
        # character line.
        is_quoted_material = "quoted_material" in warnings
        if is_quoted_material:
            speaker_id = "narrator"
            emotion = "neutral"
            confidence = annotation.confidence if annotation else 0.9
            segment_type = "narration"
        elif annotation:
            speaker_id = _resolve_speaker_id(
                annotation.speaker_id,
                characters,
                id_replacements,
            )
            emotion = _normalize_emotion(annotation.emotion)
            if emotion in {"neutral", "happy", "excited"} and _has_teasing_cue(
                raw_segment.text
            ):
                # Preserve an explicit taunt even when the model reduces it to
                # cheerful speech because of a light interjection or smile.
                emotion = "teasing"
            confidence = annotation.confidence
            segment_type = "narration" if speaker_id == "narrator" else "dialogue"
        elif raw_segment.type == "dialogue":
            speaker_id = "unknown"
            confidence = 0.35

        voice_id = _assign_voice(speaker_id, characters, narrator_voice_id)
        voice_description = _voice_description_for_speaker(speaker_id, characters)
        fallback_voice_id = _fallback_voice_for_speaker(speaker_id, characters)
        voice_ids.add(voice_id)
        segment = {
            "id": f"seg_{index + 1:04d}",
            "type": segment_type,
            "text": raw_segment.text,
            "speakerId": speaker_id,
            "voiceId": voice_id,
            "emotion": emotion,
            "intensity": _emotion_intensity(emotion),
            "pace": pace,
            "confidence": confidence,
            "source": {
                "startOffset": raw_segment.start_offset,
                "endOffset": raw_segment.end_offset,
            },
            "warnings": warnings,
        }
        if voice_description:
            segment["voiceDesign"] = voice_description
            segment["voiceDescription"] = voice_description
        if fallback_voice_id:
            segment["fallbackVoiceId"] = fallback_voice_id
        segments.append(segment)

    return {
        "bookId": book_id,
        "chapterId": chapter_id,
        "title": title,
        "language": language,
        "narratorVoiceId": narrator_voice_id,
        "characters": characters,
        "voices": _voice_metadata_for_ids(voice_ids, characters),
        "segments": segments,
        "audioPlan": {"scenes": []},
    }


def build_chapter_script_with_corrections(
    *,
    book_id: str,
    chapter_id: str,
    title: str,
    text: str,
    language: str,
    corrections: dict,
    analyzer=None,
    known_characters: list[dict] | None = None,
    narrator_voice_id: str | None = None,
) -> dict:
    if corrections is None:
        raise ValueError("corrections must be a dict, got None")
    if not isinstance(corrections, dict):
        raise ValueError(
            f"corrections must be a dict, got {type(corrections).__name__}"
        )

    for item in corrections.get("aliasMerges", []):
        if "from" not in item:
            raise KeyError("aliasMerges item missing required key 'from'")
        if "to" not in item:
            raise KeyError("aliasMerges item missing required key 'to'")

    for item in corrections.get("genderOverrides", []):
        if "characterId" not in item:
            raise KeyError("genderOverrides item missing required key 'characterId'")
        if "gender" not in item:
            raise KeyError("genderOverrides item missing required key 'gender'")

    for item in corrections.get("voiceOverrides", []):
        if "characterId" not in item:
            raise KeyError("voiceOverrides item missing required key 'characterId'")
        if "voiceId" not in item:
            raise KeyError("voiceOverrides item missing required key 'voiceId'")

    alias_map: dict[str, str] = {}
    for merge in corrections.get("aliasMerges", []):
        source = str(merge["from"]).strip()
        target = str(merge["to"]).strip()
        if not source or not target:
            raise ValueError("aliasMerges entries must contain non-empty from/to")
        alias_map[source.casefold()] = target

    for source in list(alias_map):
        seen: set[str] = set()
        target = alias_map[source]
        while target.casefold() in alias_map:
            key = target.casefold()
            if key in seen or key == source:
                raise ValueError("aliasMerges contains a cycle")
            seen.add(key)
            target = alias_map[key]
        alias_map[source] = target

    if alias_map:
        for alias, canonical in alias_map.items():
            pattern = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(alias) + r"(?![A-Za-z0-9_])",
                re.IGNORECASE,
            )
            text = pattern.sub(canonical, text)

    gender_overrides: dict[str, str] = {}
    for override in corrections.get("genderOverrides", []):
        gender_overrides[override["characterId"]] = override["gender"]

    voice_overrides: dict[str, str] = {}
    for override in corrections.get("voiceOverrides", []):
        voice_overrides[override["characterId"]] = override["voiceId"]

    voice_design_overrides: dict[str, str] = {}
    for override in corrections.get("voiceDesignOverrides", []):
        if "characterId" not in override:
            raise KeyError("voiceDesignOverrides item missing required key 'characterId'")
        if "voiceDesign" not in override:
            raise KeyError("voiceDesignOverrides item missing required key 'voiceDesign'")
        voice_design_overrides[override["characterId"]] = str(
            override["voiceDesign"]
        ).strip()

    script = build_chapter_script(
        book_id=book_id,
        chapter_id=chapter_id,
        title=title,
        text=text,
        language=language,
        analyzer=analyzer,
        known_characters=known_characters,
        narrator_voice_id=narrator_voice_id,
    )

    character_ids = {character["id"] for character in script["characters"]}

    def resolve_override_ids(overrides: dict[str, str]) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for character_id, value in overrides.items():
            candidate = _resolve_speaker_id(character_id, script["characters"], {})
            resolved[candidate if candidate in character_ids else character_id] = value
        return resolved

    gender_overrides = resolve_override_ids(gender_overrides)
    voice_overrides = resolve_override_ids(voice_overrides)
    voice_design_overrides = resolve_override_ids(voice_design_overrides)

    for character in script["characters"]:
        char_id = character["id"]
        if char_id in gender_overrides:
            character["gender"] = str(gender_overrides[char_id]).strip().lower()
            # A corrected gender changes automatic routing, but never silently
            # replaces a voice the reader selected by hand.
            if character.get("voiceSource") != "manual":
                character["voiceSource"] = "auto"
                character.pop("voiceAssignmentVersion", None)
                character.pop("voiceProfile", None)
        if char_id in voice_overrides:
            character["voiceId"] = voice_overrides[char_id]
            character["voiceSource"] = "manual"
            character.pop("voiceAssignmentVersion", None)
            character.pop("voiceProfile", None)
            character.pop("voiceDescription", None)
            character.pop("fallbackVoiceId", None)

        if char_id in voice_design_overrides and voice_design_overrides[char_id]:
            character["voiceDesign"] = voice_design_overrides[char_id]
            character["voiceDesignSource"] = "manual"
            character["voiceDescription"] = voice_design_overrides[char_id]

    _ensure_unique_character_voices(script["characters"])

    character_voices = {
        character["id"]: character["voiceId"]
        for character in script["characters"]
    }
    character_descriptions = {
        character["id"]: character.get("voiceDesign") or character.get("voiceDescription")
        for character in script["characters"]
    }
    character_fallbacks = {
        character["id"]: character.get("fallbackVoiceId")
        for character in script["characters"]
    }
    for segment in script["segments"]:
        speaker_id = segment["speakerId"]
        if speaker_id in voice_overrides:
            segment["voiceId"] = voice_overrides[speaker_id]
        elif speaker_id in character_voices:
            segment["voiceId"] = character_voices[speaker_id]
        description = character_descriptions.get(speaker_id)
        if description and speaker_id not in voice_overrides:
            segment["voiceDesign"] = description
            segment["voiceDescription"] = description
        else:
            segment.pop("voiceDesign", None)
            segment.pop("voiceDescription", None)
        fallback_voice_id = character_fallbacks.get(speaker_id)
        if fallback_voice_id and speaker_id not in voice_overrides:
            segment["fallbackVoiceId"] = fallback_voice_id
        else:
            segment.pop("fallbackVoiceId", None)

    narrator_voice_id = normalize_narrator_voice_id(
        narrator_voice_id or script.get("narratorVoiceId")
    )
    for segment in script["segments"]:
        if segment.get("speakerId") == "narrator":
            segment["voiceId"] = narrator_voice_id
            segment.pop("voiceDesign", None)
            segment.pop("voiceDescription", None)
            segment.pop("fallbackVoiceId", None)
    script["narratorVoiceId"] = narrator_voice_id
    voice_ids = {s["voiceId"] for s in script["segments"]}
    voice_ids.add(narrator_voice_id)
    script["voices"] = _voice_metadata_for_ids(voice_ids, script["characters"])

    return script


def refresh_script_voice_assignments(
    script: dict,
    *,
    force_legacy_auto: bool = False,
    narrator_voice_id: str | None = None,
) -> dict:
    """Upgrade the voice routing of an already analysed chapter script.

    This is intentionally model-free: segmentation, text, speaker attribution,
    and emotion remain untouched.  It is used for a one-time migration of old
    scripts whose saved voice IDs predate identity-aware assignment.

    ``force_legacy_auto`` is only appropriate for an explicit repair action.
    It treats unversioned records as old automatic assignments, including
    former collision resolutions that cannot otherwise be distinguished from a
    legacy manual choice.
    """
    if not isinstance(script, dict):
        raise ValueError("script must be a dict")
    raw_characters = script.get("characters", [])
    if not isinstance(raw_characters, list):
        raise ValueError("script characters must be a list")

    source_characters: list[dict] = []
    for raw_character in raw_characters:
        if not isinstance(raw_character, dict):
            continue
        character = dict(raw_character)
        if (
            force_legacy_auto
            and not character.get("voiceSource")
            and not character.get("voiceAssignmentVersion")
        ):
            character["voiceSource"] = "auto"
            character.pop("voiceAssignmentVersion", None)
            character.pop("voiceProfile", None)
            character.pop("voiceDescription", None)
            character.pop("fallbackVoiceId", None)
        source_characters.append(character)

    characters, replacements = _merge_script_characters_with_replacements(
        [], source_characters
    )
    result = dict(script)
    result["characters"] = characters
    segments: list[dict] = []
    narrator_voice_id = normalize_narrator_voice_id(
        narrator_voice_id or script.get("narratorVoiceId")
    )
    voice_ids = {narrator_voice_id}
    for raw_segment in script.get("segments", []):
        if not isinstance(raw_segment, dict):
            continue
        segment = dict(raw_segment)
        speaker_id = _resolve_speaker_id(
            str(segment.get("speakerId") or "unknown"), characters, replacements
        )
        segment["speakerId"] = speaker_id
        if speaker_id == "narrator":
            segment["voiceId"] = narrator_voice_id
            segment.pop("voiceDesign", None)
            segment.pop("voiceDescription", None)
            segment.pop("fallbackVoiceId", None)
        elif speaker_id == "unknown":
            segment["voiceId"] = "neutral_dialogue_01"
            segment.pop("voiceDescription", None)
            segment.pop("fallbackVoiceId", None)
        else:
            voice_id = _assign_voice(speaker_id, characters)
            segment["voiceId"] = voice_id
            description = _voice_description_for_speaker(speaker_id, characters)
            if description:
                segment["voiceDescription"] = description
            else:
                segment.pop("voiceDescription", None)
            fallback_voice_id = _fallback_voice_for_speaker(speaker_id, characters)
            if fallback_voice_id:
                segment["fallbackVoiceId"] = fallback_voice_id
            else:
                segment.pop("fallbackVoiceId", None)
        voice_ids.add(segment["voiceId"])
        segments.append(segment)

    result["segments"] = segments
    result["narratorVoiceId"] = narrator_voice_id
    result["voices"] = _voice_metadata_for_ids(voice_ids, characters)
    return result


def apply_narrator_voice(script: dict, narrator_voice_id: str | None = None) -> dict:
    """Normalize narrator routing in an existing script without re-analysis."""
    if not isinstance(script, dict):
        raise ValueError("script must be a dict")
    inferred_voice_id = next(
        (
            segment.get("voiceId")
            for segment in script.get("segments", [])
            if isinstance(segment, dict)
            and segment.get("speakerId") == "narrator"
            and segment.get("voiceId")
        ),
        None,
    )
    selected = normalize_narrator_voice_id(
        narrator_voice_id or script.get("narratorVoiceId") or inferred_voice_id
    )
    result = dict(script)
    result["narratorVoiceId"] = selected
    segments = []
    voice_ids: set[str] = {selected}
    for raw_segment in script.get("segments", []):
        if not isinstance(raw_segment, dict):
            continue
        segment = dict(raw_segment)
        if segment.get("speakerId") == "narrator":
            segment["voiceId"] = selected
            segment.pop("voiceDesign", None)
            segment.pop("voiceDescription", None)
            segment.pop("fallbackVoiceId", None)
        voice_ids.add(str(segment.get("voiceId") or "neutral_dialogue_01"))
        segments.append(segment)
    result["segments"] = segments
    result["voices"] = _voice_metadata_for_ids(voice_ids, result.get("characters", []))
    return result


def _voice_metadata_for_ids(voice_ids: set[str], characters: list[dict]) -> list[dict]:
    """Build script voice metadata for fixed and generated voice identities."""
    characters_by_voice = {
        str(character.get("voiceId") or ""): character
        for character in characters
        if character.get("voiceSource") == "auto"
    }
    styles = [
        "neutral",
        "happy",
        "sad",
        "angry",
        "afraid",
        "tense",
        "teasing",
        "whispering",
        "excited",
        "tired",
    ]
    metadata: list[dict] = []
    for voice_id in sorted(voice_ids):
        if voice_id in VOICE_REGISTRY:
            metadata.append(dict(VOICE_REGISTRY[voice_id]))
            continue
        character = characters_by_voice.get(voice_id)
        if not character or not voice_id.startswith(_AUTO_CHARACTER_VOICE_PREFIX):
            continue
        gender = str(character.get("gender") or "unknown").strip().lower()
        metadata.append(
            {
                "id": voice_id,
                "displayName": "角色自动音色（身份生成）",
                "genderPresentation": (
                    gender if gender in {"female", "male", "neutral", "unknown"} else "unknown"
                ),
                "ageClass": _normalize_age_class(character.get("ageClass")),
                "languages": ["zh"],
                "styles": styles,
                "backend": "mimo",
                "licenseNotes": "Generated per-character identity voice design.",
            }
        )
    return metadata


def _merge_script_characters_with_replacements(
    known: list[dict],
    discovered: list[dict],
    *,
    book_id: str | None = None,
    assign_system_ids: bool = False,
) -> tuple[list[dict], dict[str, str]]:
    """Merge character records while preserving the book-level identity.

    The model can return ``li_huaiyu``, ``李怀玉`` or a temporary id such as
    ``character_7`` for the same person.  Matching canonical names and
    meaningful aliases lets us keep one id and one voice, then the returned
    replacement map rewrites segment speaker references to that stable id.
    """
    merged: list[dict] = []
    replacements: dict[str, str] = {}

    for incoming, is_discovered in [
        *[(item, False) for item in known],
        *[(item, True) for item in discovered],
    ]:
        character = _normalize_script_character(incoming)
        character_id = character["id"]
        if not character_id:
            continue

        existing_index = _find_matching_character(character, merged)
        if existing_index is None:
            if is_discovered and assign_system_ids and book_id:
                candidate_id = character_id
                character_id = _system_character_id(
                    book_id,
                    character.get("canonicalName", ""),
                    character.get("aliases", []),
                )
                if any(existing.get("id") == character_id for existing in merged):
                    collision_digest = hashlib.sha256(
                        f"{character_id}:{candidate_id}:{len(merged)}".encode("utf-8")
                    ).hexdigest()[:8]
                    character_id = f"{character_id}_{collision_digest}"
                character["id"] = character_id
                if candidate_id != character_id:
                    replacements[candidate_id] = character_id
            merged.append(character)
            continue

        existing = merged[existing_index]
        if character_id != existing["id"]:
            replacements[character_id] = existing["id"]
        _merge_character_record(existing, character)

    _ensure_unique_character_voices(merged)
    return merged, _resolve_id_replacements(replacements)


def _merge_script_characters(known: list[dict], discovered: list[dict]) -> list[dict]:
    """Backward-compatible character merge helper without the replacement map."""
    merged, _ = _merge_script_characters_with_replacements(known, discovered)
    return merged


def _known_character_to_script(character: dict) -> dict:
    # Keep every voice-routing field supplied by a saved script or the desktop
    # roster.  ``_normalize_script_character`` will upgrade legacy values and
    # fill the automatic fields when they are absent.
    return _normalize_script_character(character)


def _normalize_script_character(character: dict) -> dict:
    character_id = str(character.get("id", "")).strip()
    canonical_name = str(
        character.get("canonicalName", character.get("canonical_name", ""))
    ).strip()
    canonical_name = canonical_name or character_id
    aliases = _normalize_aliases(character.get("aliases", []))
    gender = _effective_gender(
        character.get("gender", "unknown"), canonical_name, aliases
    )
    age_class = _effective_age_class(
        character.get("ageClass", character.get("age_class", "unknown")),
        canonical_name,
        aliases,
    )
    voice_id = str(character.get("voiceId") or "").strip()
    confidence = _coerce_confidence(character.get("confidence", 0.0))
    requested_identity_status = str(character.get("identityStatus") or "").strip().lower()
    identity_status = (
        requested_identity_status
        if requested_identity_status in _IDENTITY_STATUSES
        else (
            "confirmed"
            if confidence >= _IDENTITY_CONFIDENCE_THRESHOLD
            else "provisional"
        )
    )
    source = _voice_source_for_character(
        character,
        gender=gender,
        canonical_name=canonical_name or character_id,
        existing_voice_id=voice_id,
    )
    profile = _voice_profile_key(gender, age_class, canonical_name, aliases)

    normalized = {
        "id": character_id,
        "canonicalName": canonical_name,
        "aliases": aliases,
        "gender": gender,
        "ageClass": age_class,
        "voiceId": voice_id,
        "voiceSource": source,
        "identityStatus": identity_status,
        "confidence": confidence,
    }
    voice_design = str(character.get("voiceDesign") or "").strip()
    if voice_design:
        normalized["voiceDesign"] = voice_design
        design_source = str(character.get("voiceDesignSource") or "").strip().lower()
        normalized["voiceDesignSource"] = design_source or (
            "manual" if source == "manual" else "llm"
        )
    if source == "manual":
        if character.get("voiceDescription"):
            normalized["voiceDescription"] = str(character["voiceDescription"])
        return normalized

    # Keep an already-current automatic selection locked in place.  New,
    # legacy, or profile-changed records deliberately remain unlocked until
    # the complete chapter roster is available for collision resolution.
    if _is_current_auto_assignment(character, profile):
        normalized["voiceAssignmentVersion"] = VOICE_ASSIGNMENT_VERSION
        normalized["voiceProfile"] = profile
        if voice_id:
            normalized["voiceId"] = voice_id
        if character.get("voiceDescription"):
            normalized["voiceDescription"] = str(character["voiceDescription"])
        fallback_voice_id = str(character.get("fallbackVoiceId") or "").strip()
        if fallback_voice_id:
            normalized["fallbackVoiceId"] = fallback_voice_id
    return normalized


def _normalize_aliases(value) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    aliases: list[str] = []
    seen: set[str] = set()
    for alias in value:
        normalized = str(alias).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            aliases.append(normalized)
            seen.add(key)
    return aliases


def _coerce_confidence(value) -> float:
    try:
        return min(1.0, max(0.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _find_matching_character(character: dict, merged: list[dict]) -> int | None:
    character_id = _identity_key(character.get("id"))
    canonical = _identity_key(character.get("canonicalName"))

    # Exact ids remain the strongest match, including ids supplied by an
    # external correction or a previously saved script.
    for index, existing in enumerate(merged):
        if character_id and character_id == _identity_key(existing.get("id")):
            return index

    # A canonical name is authoritative when it is not merely a generic
    # title.  This is what merges model ids such as ``li_huaiyu`` and
    # ``character_7`` without merging every character called “小姐”.
    if canonical and not _is_generic_character_label(canonical):
        for index, existing in enumerate(merged):
            if canonical == _identity_key(existing.get("canonicalName")):
                return index

    # Family roles such as “父亲/儿子” are generic in the abstract, but are
    # often the only stable identity the text provides.  If there is exactly
    # one compatible role already known in this book, merge it when a model
    # changes its temporary id.  Ambiguous honorifics like “小姐” deliberately
    # remain non-mergeable.
    if canonical and _is_mergeable_family_label(canonical):
        candidates = [
            index
            for index, existing in enumerate(merged)
            if canonical == _identity_key(existing.get("canonicalName"))
            and _compatible_character_identity(existing, character)
        ]
        if len(candidates) == 1:
            return candidates[0]

    incoming_keys = {
        _identity_key(value)
        for value in [character.get("canonicalName"), *character.get("aliases", [])]
        if _identity_key(value) and not _is_generic_character_label(_identity_key(value))
    }
    if not incoming_keys:
        return None
    for index, existing in enumerate(merged):
        existing_keys = {
            _identity_key(value)
            for value in [
                existing.get("canonicalName"),
                *existing.get("aliases", []),
            ]
            if _identity_key(value)
            and not _is_generic_character_label(_identity_key(value))
        }
        if incoming_keys & existing_keys:
            return index
    return None


def _merge_character_record(existing: dict, incoming: dict) -> None:
    previous_profile = _voice_profile_key(
        existing.get("gender", "unknown"),
        existing.get("ageClass", "unknown"),
        existing.get("canonicalName", ""),
        existing.get("aliases", []),
    )
    existing_name = str(existing.get("canonicalName", "")).strip()
    incoming_name = str(incoming.get("canonicalName", "")).strip()
    aliases = [
        *existing.get("aliases", []),
        *incoming.get("aliases", []),
    ]
    if incoming_name and _identity_key(incoming_name) != _identity_key(existing_name):
        aliases.append(incoming_name)
    existing["aliases"] = _normalize_aliases(
        [alias for alias in aliases if _identity_key(alias) != _identity_key(existing_name)]
    )

    if not existing_name and incoming_name:
        existing["canonicalName"] = incoming_name
    if existing.get("gender") in (None, "", "unknown") and incoming.get("gender") not in (
        None,
        "",
        "unknown",
    ):
        existing["gender"] = incoming["gender"]
    if existing.get("ageClass") in (None, "", "unknown") and incoming.get("ageClass") not in (
        None,
        "",
        "unknown",
    ):
        existing["ageClass"] = incoming["ageClass"]

    incoming_voice_design = str(incoming.get("voiceDesign") or "").strip()
    existing_design_source = str(existing.get("voiceDesignSource") or "").strip().lower()
    incoming_design_source = str(incoming.get("voiceDesignSource") or "llm").strip().lower()
    if incoming_voice_design and existing.get("voiceSource") != "manual" and (
        not str(existing.get("voiceDesign") or "").strip()
        or existing_design_source == "fallback"
    ):
        existing["voiceDesign"] = incoming_voice_design
        existing["voiceDesignSource"] = incoming_design_source

    if incoming.get("identityStatus") == "confirmed":
        existing["identityStatus"] = "confirmed"
    elif existing.get("identityStatus") not in _IDENTITY_STATUSES:
        existing["identityStatus"] = incoming.get("identityStatus", "provisional")

    # A manual choice belongs to the reader, not to the model.  For automatic
    # choices, the final identity profile below recomputes a suitable voice
    # after age/gender information from both records has been merged.
    if existing.get("voiceSource") != "manual" and incoming.get("voiceSource") == "manual":
        existing["voiceId"] = incoming.get("voiceId", "")
        existing["voiceSource"] = "manual"
        if incoming.get("voiceDescription"):
            existing["voiceDescription"] = incoming["voiceDescription"]
        else:
            existing.pop("voiceDescription", None)
        if incoming_voice_design:
            existing["voiceDesign"] = incoming_voice_design
            existing["voiceDesignSource"] = incoming_design_source
        existing.pop("fallbackVoiceId", None)
        existing.pop("voiceAssignmentVersion", None)
        existing.pop("voiceProfile", None)
    elif existing.get("voiceSource") != "manual":
        existing["voiceSource"] = "auto"
        updated_profile = _voice_profile_key(
            existing.get("gender", "unknown"),
            existing.get("ageClass", "unknown"),
            existing.get("canonicalName", ""),
            existing.get("aliases", []),
        )
        if updated_profile != previous_profile:
            existing.pop("voiceAssignmentVersion", None)
            existing.pop("voiceProfile", None)

    existing["confidence"] = max(
        float(existing.get("confidence", 0.0) or 0.0),
        float(incoming.get("confidence", 0.0) or 0.0),
    )


def _identity_key(value) -> str:
    return "".join(
        character
        for character in str(value or "").strip().casefold()
        if character.isalnum()
    )


def _system_character_id(
    book_id: str,
    canonical_name: str,
    aliases: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return a stable opaque ID for a new book-scoped character.

    This is deliberately derived by the worker, not copied from the LLM.  The
    registry keeps the generated value in the script/database; the book-scoped
    identity key also makes a repeated analysis deterministic before the roster
    has been restored from SQLite.
    """
    identity = _identity_key(canonical_name)
    if not identity:
        identity = next(
            (
                _identity_key(alias)
                for alias in (aliases or [])
                if _identity_key(alias)
            ),
            "character",
        )
    book_key = _identity_key(book_id) or "book"
    digest = hashlib.sha256(
        f"audiobook-character-v1:{book_key}:{identity}".encode("utf-8")
    ).hexdigest()[:20]
    return f"{_SYSTEM_CHARACTER_ID_PREFIX}{digest}"


def _is_generic_character_label(value: str) -> bool:
    return value in {
        "小姐",
        "少爷",
        "姑娘",
        "公子",
        "夫人",
        "太太",
        "老爷",
        "殿下",
        "陛下",
        "皇上",
        "皇后",
        "公主",
        "王爷",
        "世子",
        "大人",
        "先生",
        "女士",
        "母亲",
        "父亲",
        "娘",
        "爹",
        "妈妈",
        "爸爸",
        "mother",
        "father",
        "wife",
        "husband",
        "miss",
        "mrs",
        "ms",
        "mr",
        "sir",
        "madam",
        "lady",
        "lord",
        "girl",
        "boy",
        "woman",
        "man",
    }


def _is_mergeable_family_label(value: str) -> bool:
    return value in {
        "父亲",
        "母亲",
        "儿子",
        "女儿",
        "父王",
        "母后",
        "father",
        "mother",
        "son",
        "daughter",
    }


def _compatible_character_identity(existing: dict, incoming: dict) -> bool:
    existing_gender = str(existing.get("gender", "unknown")).strip().lower()
    incoming_gender = str(incoming.get("gender", "unknown")).strip().lower()
    if (
        existing_gender not in {"", "unknown"}
        and incoming_gender not in {"", "unknown"}
        and existing_gender != incoming_gender
    ):
        return False
    existing_age = str(existing.get("ageClass", "unknown")).strip().lower()
    incoming_age = str(incoming.get("ageClass", "unknown")).strip().lower()
    return (
        existing_age in {"", "unknown"}
        or incoming_age in {"", "unknown"}
        or existing_age == incoming_age
    )


def _resolve_id_replacements(replacements: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for source, target in replacements.items():
        current = target
        seen: set[str] = set()
        while current in replacements and current not in seen:
            seen.add(current)
            current = replacements[current]
        if source != current:
            resolved[source] = current
    return resolved


def _resolve_speaker_id(
    speaker_id: str,
    characters: list[dict],
    replacements: dict[str, str],
) -> str:
    speaker = str(speaker_id or "unknown").strip()
    if speaker in {"narrator", "unknown"}:
        return speaker
    speaker = _resolve_id_replacements({speaker: replacements.get(speaker, speaker)}).get(
        speaker, speaker
    )
    if any(character.get("id") == speaker for character in characters):
        return speaker

    key = _identity_key(speaker)
    if key:
        for character in characters:
            values = [character.get("canonicalName"), *character.get("aliases", [])]
            if any(key == _identity_key(value) for value in values):
                return character["id"]
    return speaker


def _character_to_script(character) -> dict:
    return _normalize_script_character(
        {
        "id": character.id,
        "canonicalName": character.canonical_name,
        "aliases": character.aliases,
        "gender": character.gender,
        "ageClass": character.age_class,
        "voiceDesign": getattr(character, "voice_design", ""),
        "voiceSource": "auto",
        "confidence": character.confidence,
        }
    )


def _assign_voice(
    speaker_id: str,
    characters: list[dict],
    narrator_voice_id: str | None = None,
) -> str:
    if speaker_id == "narrator":
        return normalize_narrator_voice_id(narrator_voice_id)
    for character in characters:
        if character["id"] == speaker_id:
            return character["voiceId"]
    return "neutral_dialogue_01"


def _voice_description_for_speaker(
    speaker_id: str, characters: list[dict]
) -> str | None:
    for character in characters:
        if character["id"] == speaker_id:
            description = character.get("voiceDesign") or character.get("voiceDescription")
            return str(description) if description else None
    return None


def _fallback_voice_for_speaker(
    speaker_id: str, characters: list[dict]
) -> str | None:
    for character in characters:
        if character["id"] == speaker_id:
            fallback_voice_id = str(character.get("fallbackVoiceId") or "").strip()
            return fallback_voice_id or None
    return None


def _voice_for_gender(gender: str, character_id: str = "") -> str:
    """Legacy-compatible finite-catalog fallback helper.

    Automatic MiMo roles do not call this helper.  It remains for integrations
    that need one of the local backend's fixed voices.
    """
    return _voice_for_identity(gender, "unknown", character_id)


def _legacy_voice_for_gender(gender: str, character_id: str = "") -> str:
    """Reproduce the pre-v2 hash assignment to recognise legacy auto rows."""
    normalized_gender = str(gender or "unknown").strip().lower()
    pool = _voice_pool(normalized_gender)
    if not pool:
        return "neutral_dialogue_01"
    if not character_id:
        return pool[0]
    hash_bytes = hashlib.sha256(character_id.casefold().encode()).digest()
    index = int.from_bytes(hash_bytes[:4], "big") % len(pool)
    return pool[index]


def _voice_for_identity(gender: str, age_class: str, identity: str) -> str:
    return _fallback_voice_for_identity(
        gender,
        age_class,
        identity,
        [],
        identity,
    )


def _ensure_unique_character_voices(characters: list[dict]) -> None:
    """Give every automatic role an independent, deterministic voice design.

    MiMo creates the sound from ``voiceDescription``.  The generated ID only
    anchors that design across chapters; it is deliberately not a catalog ID.
    A finite ``fallbackVoiceId`` is retained solely for local backends.
    """
    for character in characters:
        if character.get("voiceSource") == "manual":
            character.pop("fallbackVoiceId", None)
            continue
        _finalize_automatic_voice(character)


def _finalize_automatic_voice(character: dict) -> None:
    gender = str(character.get("gender", "unknown")).strip().lower()
    age_class = str(character.get("ageClass", "unknown")).strip().lower()
    character_id = str(character.get("id") or "")
    canonical_name = str(character.get("canonicalName") or character.get("id") or "")
    aliases = character.get("aliases", [])
    character["voiceId"] = _automatic_character_voice_id(character_id, canonical_name)
    character["voiceSource"] = "auto"
    character["voiceAssignmentVersion"] = VOICE_ASSIGNMENT_VERSION
    character["voiceProfile"] = _voice_profile_key(
        gender, age_class, canonical_name, aliases
    )
    character["fallbackVoiceId"] = _fallback_voice_for_identity(
        gender,
        age_class,
        canonical_name,
        aliases,
        character_id,
    )
    description = str(character.get("voiceDesign") or "").strip()
    if not description:
        description = _voice_description_for_character(
            gender, age_class, canonical_name, aliases, character_id
        )
        character["voiceDesign"] = description
        character["voiceDesignSource"] = "fallback"
    if description:
        character["voiceDescription"] = description
    else:
        character.pop("voiceDescription", None)


def _automatic_character_voice_id(character_id: str, canonical_name: str) -> str:
    identity = _automatic_voice_identity_key(character_id, canonical_name)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{_AUTO_CHARACTER_VOICE_PREFIX}{digest}"


def _automatic_voice_identity_key(character_id: str, canonical_name: str) -> str:
    canonical_key = _identity_key(canonical_name)
    identifier_key = _identity_key(character_id)
    return ":".join(part for part in (canonical_key, identifier_key) if part) or "character"


def _fallback_voice_for_identity(
    gender: str,
    age_class: str,
    canonical_name: str,
    aliases: list[str] | tuple[str, ...],
    character_id: str,
) -> str:
    identity = _automatic_voice_identity_key(character_id, canonical_name)
    candidates = _ordered_voice_candidates(
        gender,
        age_class,
        identity,
        canonical_name,
        aliases,
    )
    return candidates[0] if candidates else "neutral_dialogue_01"


def _ordered_voice_candidates(
    gender: str,
    age_class: str,
    identity: str,
    canonical_name: str = "",
    aliases: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    normalized_gender = str(gender or "unknown").strip().lower()
    normalized_age = _normalize_age_class(age_class)
    pool = _voice_pool(normalized_gender)
    if not pool:
        return []

    role = _identity_voice_role(canonical_name or identity, aliases or [])
    primary = list(_VOICE_PREFERENCES[normalized_gender][normalized_age])
    if role == "parent" and normalized_age in {"adult", "older", "unknown"}:
        primary = list(_PARENT_VOICE_PREFERENCES[normalized_gender])
    elif role == "elder":
        primary = list(_VOICE_PREFERENCES[normalized_gender]["older"])
    elif role in {"child", "young"} and normalized_age == "unknown":
        primary = list(_VOICE_PREFERENCES[normalized_gender][role])

    fallback = [voice_id for voice_id in pool if voice_id not in primary]
    return _rotate_candidates(primary, identity) + _rotate_candidates(fallback, identity)


def _rotate_candidates(candidates: list[str], identity: str) -> list[str]:
    if len(candidates) < 2:
        return candidates
    key = str(identity or "").casefold().encode()
    start = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % len(candidates)
    return candidates[start:] + candidates[:start]


def _voice_pool(gender: str) -> list[str]:
    if gender == "female":
        return _FEMALE_VOICE_POOL
    if gender == "male":
        return _MALE_VOICE_POOL
    return []


def _effective_gender(value, canonical_name: str, aliases: list[str]) -> str:
    gender = str(value or "unknown").strip().lower()
    if gender in {"female", "male", "neutral"}:
        return gender
    identity = _identity_text(canonical_name, aliases)
    if any(term in identity for term in ("父亲", "爸爸", "爹", "儿子", "祖父", "爷爷", "外公", "公子", "少爷", "father", "dad", "son", "grandfather", "grandpa")):
        return "male"
    if any(term in identity for term in ("母亲", "妈妈", "娘", "女儿", "祖母", "奶奶", "外婆", "小姐", "姑娘", "mother", "mom", "mum", "daughter", "grandmother", "grandma")):
        return "female"
    return "unknown"


def _effective_age_class(value, canonical_name: str, aliases: list[str]) -> str:
    normalized = _normalize_age_class(value)
    if normalized != "unknown":
        return normalized
    role = _identity_voice_role(canonical_name, aliases)
    if role == "elder":
        return "older"
    if role == "parent":
        return "adult"
    if role == "child":
        return "child"
    if role == "young":
        return "young"
    return "unknown"


def _normalize_age_class(value) -> str:
    age_class = str(value or "unknown").strip().lower()
    return age_class if age_class in _AGE_CLASSES else "unknown"


def _identity_voice_role(canonical_name: str, aliases: list[str] | tuple[str, ...]) -> str:
    identity = _identity_text(canonical_name, aliases)
    # Check longer-lived family roles first: "grandfather" also contains
    # "father", and an elder should never be routed as an ordinary parent.
    for role in ("elder", "parent", "child", "young"):
        if any(term in identity for term in _IDENTITY_TERMS[role]):
            return role
    return "default"


def _identity_text(canonical_name: str, aliases: list[str] | tuple[str, ...]) -> str:
    values = [canonical_name, *(aliases or [])]
    return " ".join(str(value or "").casefold() for value in values)


def _voice_profile_key(
    gender: str,
    age_class: str,
    canonical_name: str,
    aliases: list[str] | tuple[str, ...],
) -> str:
    normalized_gender = str(gender or "unknown").strip().lower()
    normalized_age = _normalize_age_class(age_class)
    role = _identity_voice_role(canonical_name, aliases)
    return f"{normalized_gender}:{normalized_age}:{role}"


def _voice_source_for_character(
    character: dict,
    *,
    gender: str,
    canonical_name: str,
    existing_voice_id: str,
) -> str:
    requested = str(character.get("voiceSource", "")).strip().lower()
    if requested in _VOICE_SOURCES:
        return requested

    # Pre-v2 scripts and database rows did not record whether the value came
    # from the automatic hash or a user selection.  An exact old hash match is
    # safe to migrate; a different value is treated as a manual choice.
    if not existing_voice_id:
        return "auto"
    legacy_voice_id = _legacy_voice_for_gender(gender, canonical_name)
    if existing_voice_id in {legacy_voice_id, "neutral_dialogue_01"}:
        return "auto"
    return "manual"


def _is_current_auto_assignment(character: dict, profile: str) -> bool:
    if character.get("voiceSource") != "auto":
        return False
    try:
        version = int(character.get("voiceAssignmentVersion"))
    except (TypeError, ValueError):
        return False
    return (
        version == VOICE_ASSIGNMENT_VERSION
        and str(character.get("voiceProfile") or "") == profile
        and str(character.get("voiceId") or "").startswith(_AUTO_CHARACTER_VOICE_PREFIX)
        and bool(
            str(character.get("voiceDesign") or character.get("voiceDescription") or "").strip()
        )
        and bool(str(character.get("fallbackVoiceId") or "").strip())
    )


def _voice_description_for_character(
    gender: str,
    age_class: str,
    canonical_name: str,
    aliases: list[str] | tuple[str, ...],
    character_id: str,
) -> str:
    normalized_gender = str(gender or "unknown").strip().lower()
    normalized_age = _normalize_age_class(age_class)
    role = _identity_voice_role(canonical_name, aliases)
    age_descriptions = {
        "female": {
            "child": "一位八到十二岁的中文女孩，声线清亮稚嫩，避免成年人的成熟厚重感",
            "young": "一位十六到二十多岁的中文年轻女性，声线清新自然，带有青春感但不幼态夸张",
            "adult": "一位三十岁上下的中文成年女性，声线自然稳定，成熟但不过分老成",
            "older": "一位五十岁上下的中文年长女性，声线沉稳温和，保留清晰咬字，避免刻意苍老",
            "unknown": "一位中文女性，声线自然清晰，年龄感保持中性克制",
        },
        "male": {
            "child": "一位八到十二岁的中文男孩，声线清亮稚嫩，避免成年男性的低沉厚重感",
            "young": "一位十六到二十多岁的中文少年或青年男性，声线清朗，带有尚未完全成熟的青春感",
            "adult": "一位三十岁上下的中文成年男性，声线自然稳定，成熟但不过分老成",
            "older": "一位五十岁上下的中文年长男性，声线沉稳醇厚，保留清晰咬字，避免夸张老态",
            "unknown": "一位中文男性，声线自然清晰，年龄感保持中性克制",
        },
        "neutral": {
            "child": "一位八到十二岁的中文孩子，声线清亮稚嫩，性别特征保持自然中性",
            "young": "一位十六到二十多岁的中文年轻人，声线清新自然，保留青春感",
            "adult": "一位三十岁上下的中文成年人，声线自然稳定，性别特征保持克制中性",
            "older": "一位五十岁上下的中文年长者，声线沉稳温和，避免夸张老态",
            "unknown": "一位中文说话者，声线自然中性、清晰易辨，年龄感保持克制",
        },
    }
    role_direction = {
        "elder": "体现长者的从容与阅历，不要读成年轻人的声线",
        "parent": "体现父母辈的成熟可靠感，不要读成少年或少女",
        "child": "保持孩子的清亮和稚气，不要读成成人",
        "young": "保持年轻角色的清新活力，不要读成中年人",
        "default": "",
    }[role]

    voice_kind = normalized_gender if normalized_gender in {"female", "male"} else "neutral"
    timbre_options = {
        "female": {
            "child": ["音色清脆轻薄", "声音明净轻柔", "声线灵动清亮"],
            "young": ["音色明亮清新", "声线清透柔润", "声音轻快而有支撑"],
            "adult": ["音色温润清晰", "声线圆润沉稳", "声音明晰而有支撑"],
            "older": ["音色醇厚温和", "声线沉稳清晰", "声音略带岁月感但不沙哑"],
            "unknown": ["音色自然清晰", "声线柔和易辨", "声音平稳不夸张"],
        },
        "male": {
            "child": ["音色清亮轻薄", "声音明净灵动", "声线清脆自然"],
            "young": ["音色清朗明快", "声线明晰有活力", "声音清透自然"],
            "adult": ["音色沉稳自然", "声线清晰有支撑", "声音温厚而不过分低沉"],
            "older": ["音色醇厚沉稳", "声线低沉但咬字清晰", "声音略带岁月感而不显疲惫"],
            "unknown": ["音色自然清晰", "声线平稳易辨", "声音克制不夸张"],
        },
        "neutral": {
            "child": ["音色清亮轻薄", "声音明净灵动", "声线清脆自然"],
            "young": ["音色清新自然", "声线清透易辨", "声音轻快不夸张"],
            "adult": ["音色自然平稳", "声线清晰有支撑", "声音温和易辨"],
            "older": ["音色醇和沉稳", "声线温和清晰", "声音略有阅历感而不夸张"],
            "unknown": ["音色自然中性", "声线清晰易辨", "声音平稳不夸张"],
        },
    }
    resonance_options = [
        "气息平稳，共鸣自然",
        "发声位置稳定，句尾收束干净",
        "口腔共鸣清晰，避免飘忽的音色变化",
        "声音质感均衡，避免过度鼻音或压低嗓音",
    ]
    diction_options = [
        "咬字清楚，语句连接自然",
        "吐字利落，保留自然的人声起伏",
        "表达克制清晰，不使用夸张播音腔",
        "发音稳定，避免忽高忽低或忽然换声线",
    ]
    digest = hashlib.sha256(
        _automatic_voice_identity_key(character_id, canonical_name).encode("utf-8")
    ).digest()

    def choose(options: list[str], offset: int) -> str:
        return options[digest[offset] % len(options)]

    parts = [
        age_descriptions[voice_kind][normalized_age],
        choose(timbre_options[voice_kind][normalized_age], 0),
        choose(resonance_options, 1),
        choose(diction_options, 2),
        role_direction,
        "为该角色建立独立且可辨识的基础音色",
        "这是该角色全书及跨章节固定的音色设定，保持音高、共鸣和质感一致，不要借用旁白或其他角色的声线",
    ]
    return "，".join(part for part in parts if part) + "。"


def _emotion_intensity(emotion: str) -> float:
    if emotion in {"angry", "afraid", "excited", "grief", "surprised"}:
        return 0.7
    if emotion in {"tense", "sad", "happy", "teasing", "pleading", "nervous", "contemptuous", "bitter"}:
        return 0.45
    if emotion in {"whispering", "tired", "cold", "gentle", "resolute", "solemn"}:
        return 0.3
    return 0.2


def _normalize_emotion(value: str) -> str:
    emotion = str(value or "neutral").strip().lower()
    aliases = {
        "mocking": "teasing",
        "sarcastic": "teasing",
        "taunting": "teasing",
        "嘲弄": "teasing",
        "嘲讽": "teasing",
        "戏谑": "teasing",
    }
    emotion = aliases.get(emotion, emotion)
    return emotion if emotion in {
        "neutral",
        "happy",
        "sad",
        "angry",
        "afraid",
        "tense",
        "teasing",
        "whispering",
        "excited",
        "tired",
        "grief",
        "cold",
        "pleading",
        "surprised",
        "gentle",
        "resolute",
        "nervous",
        "contemptuous",
        "solemn",
        "bitter",
    } else "neutral"


def _has_teasing_cue(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(
        marker in lowered
        for marker in (
            "戏谑",
            "嘲弄",
            "嘲讽",
            "讥讽",
            "挖苦",
            "奚落",
            "取笑",
            "轻蔑",
            "哟",
            "啧",
            "还挺",
            "废物",
            "喂狗",
            "逗狗",
            "孝敬师兄",
            "正合你用",
            "mocking",
            "sarcastic",
            "taunting",
        )
    )


def _normalize_pace(value: str) -> str:
    pace = str(value or "normal").strip().lower()
    return pace if pace in {"slow", "normal", "fast"} else "normal"
