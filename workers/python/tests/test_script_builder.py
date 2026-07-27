from audiobook_worker.dialogue import segment_dialogue
from audiobook_worker.llm import CharacterAnalysis, ChapterAnalysisResult, SegmentAnnotation
from audiobook_worker.script_builder import (
    build_chapter_script,
    refresh_script_voice_assignments,
)


class CapturingAnalyzer:
    def __init__(self):
        self.request = None

    def analyze_chapter(self, request):
        self.request = request
        return ChapterAnalysisResult(characters=[], segment_annotations=[])


def test_builds_dialogue_aware_chapter_script():
    script = build_chapter_script(
        book_id="book_123",
        chapter_id="chapter_001",
        title="Chapter 1",
        text='She waited. "Come in," Elizabeth said.',
        language="en",
    )

    assert script["bookId"] == "book_123"
    assert script["chapterId"] == "chapter_001"
    character_id = script["characters"][0]["id"]
    assert character_id.startswith("char_")
    assert script["characters"][0]["gender"] == "female"
    assert [segment["type"] for segment in script["segments"]] == ["narration", "dialogue", "narration"]
    assert script["segments"][1]["speakerId"] == character_id
    assert script["segments"][1]["voiceId"].startswith("character_auto_")
    assert script["segments"][1]["fallbackVoiceId"].startswith("female_adult_")
    assert "独立且可辨识" in script["segments"][1]["voiceDescription"]
    voice_metadata = {
        voice["id"]: voice for voice in script["voices"]
    }[script["segments"][1]["voiceId"]]
    assert voice_metadata["displayName"] == "角色自动音色（身份生成）"
    assert voice_metadata["backend"] == "mimo"
    assert script["segments"][1]["emotion"] == "neutral"
    assert script["segments"][1]["confidence"] >= 0.7


def test_unknown_dialogue_uses_neutral_fallback_voice():
    script = build_chapter_script(
        book_id="book_123",
        chapter_id="chapter_001",
        title="Chapter 1",
        text='"No one knows."',
        language="en",
    )

    segment = script["segments"][0]
    assert segment["speakerId"] == "unknown"
    assert segment["voiceId"] == "neutral_dialogue_01"
    assert "speaker_unknown" in segment["warnings"]


class AnonymousSceneAnalyzer:
    def analyze_chapter(self, request):
        return ChapterAnalysisResult(
            characters=[
                CharacterAnalysis(
                    id="茶客甲",
                    canonical_name="茶客甲",
                    aliases=[],
                    gender="unknown",
                    age_class="unknown",
                    confidence=0.7,
                ),
                CharacterAnalysis(
                    id="茶客乙",
                    canonical_name="茶客乙",
                    aliases=[],
                    gender="unknown",
                    age_class="unknown",
                    confidence=0.7,
                ),
            ],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=0,
                    speaker_id="茶客甲",
                    emotion="neutral",
                    confidence=0.7,
                ),
                SegmentAnnotation(
                    segment_index=1,
                    speaker_id="茶客乙",
                    emotion="neutral",
                    confidence=0.7,
                ),
            ],
        )


def test_generic_anonymous_scene_roles_remain_separate_voice_characters():
    script = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="“白幡是做什么？”\n“官老爷都系白腰带？”",
        language="zh",
        analyzer=AnonymousSceneAnalyzer(),
    )

    assert [character["canonicalName"] for character in script["characters"]] == [
        "茶客甲",
        "茶客乙",
    ]
    assert [segment["speakerId"] for segment in script["segments"]] == [
        character["id"] for character in script["characters"]
    ]


def test_chinese_script_builder_passes_language_to_segmentation_and_analyzer():
    analyzer = CapturingAnalyzer()

    script = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="张三说道：「走吧。」",
        language="zh",
        analyzer=analyzer,
    )

    assert analyzer.request.language == "zh"
    assert [segment["type"] for segment in script["segments"]] == ["narration", "dialogue"]


class CorrectingAnalyzer:
    def analyze_chapter(self, request):
        return ChapterAnalysisResult(
            characters=[],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=0,
                    speaker_id="unknown",
                    emotion="tense",
                    confidence=0.62,
                    warnings=["inner_monologue_uncertain"],
                ),
                SegmentAnnotation(
                    segment_index=1,
                    speaker_id="narrator",
                    emotion="neutral",
                    confidence=0.96,
                    warnings=["quoted_material"],
                ),
            ],
        )


def test_llm_can_override_presegmented_segment_types():
    script = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="真的要走吗？他心里想。桌上放着《远方》：“明月照故乡。”",
        language="zh",
        analyzer=CorrectingAnalyzer(),
    )

    assert script["segments"][0]["type"] == "dialogue"
    assert script["segments"][0]["speakerId"] == "unknown"
    assert script["segments"][1]["type"] == "narration"
    assert script["segments"][1]["speakerId"] == "narrator"


class PaceAwareAnalyzer:
    def analyze_chapter(self, request):
        return ChapterAnalysisResult(
            characters=[
                CharacterAnalysis(
                    id="张三",
                    canonical_name="张三",
                    aliases=[],
                    gender="male",
                    age_class="adult",
                    confidence=0.95,
                )
            ],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=0,
                    speaker_id="narrator",
                    emotion="neutral",
                    pace="fast",
                    confidence=0.99,
                    warnings=[],
                ),
                SegmentAnnotation(
                    segment_index=1,
                    speaker_id="张三",
                    emotion="tense",
                    pace="slow",
                    confidence=0.91,
                    warnings=[],
                ),
            ],
        )


def test_narration_keeps_fixed_pace_while_character_pace_follows_analysis():
    script = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="院子里很安静。张三说道：「走吧。」",
        language="zh",
        analyzer=PaceAwareAnalyzer(),
    )

    assert script["segments"][0]["speakerId"] == "narrator"
    assert script["segments"][0]["pace"] == "normal"
    assert script["segments"][1]["speakerId"] == script["characters"][0]["id"]
    assert script["segments"][1]["pace"] == "slow"


class RenamedCharacterAnalyzer:
    def __init__(self, character_id: str):
        self.character_id = character_id

    def analyze_chapter(self, request):
        return ChapterAnalysisResult(
            characters=[
                CharacterAnalysis(
                    id=self.character_id,
                    canonical_name="李怀玉",
                    aliases=[],
                    gender="female",
                    age_class="adult",
                    confidence=0.94,
                )
            ],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=0,
                    speaker_id=self.character_id,
                    emotion="neutral",
                    confidence=0.94,
                )
            ],
        )


def test_same_canonical_character_keeps_id_and_voice_when_model_id_changes():
    first = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="“走吧。”",
        language="zh",
        analyzer=RenamedCharacterAnalyzer("li_huaiyu"),
    )
    second = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_002",
        title="第二章",
        text="“留下。”",
        language="zh",
        analyzer=RenamedCharacterAnalyzer("character_7"),
        known_characters=first["characters"],
    )

    assert second["characters"] == [
        {
            **first["characters"][0],
            "confidence": 0.94,
        }
    ]
    assert second["segments"][0]["speakerId"] == first["characters"][0]["id"]
    assert second["segments"][0]["voiceId"] == first["characters"][0]["voiceId"]


def test_new_character_id_is_worker_generated_and_deterministic_for_a_book():
    first = build_chapter_script(
        book_id="book_registry",
        chapter_id="chapter_001",
        title="第一章",
        text="“走吧。”",
        language="zh",
        analyzer=RenamedCharacterAnalyzer("model_candidate_a"),
    )
    second = build_chapter_script(
        book_id="book_registry",
        chapter_id="chapter_002",
        title="第二章",
        text="“留下。”",
        language="zh",
        analyzer=RenamedCharacterAnalyzer("model_candidate_b"),
    )

    first_id = first["characters"][0]["id"]
    second_id = second["characters"][0]["id"]
    assert first_id.startswith("char_")
    assert second_id == first_id
    assert first_id not in {"model_candidate_a", "model_candidate_b"}
    assert first["segments"][0]["speakerId"] == first_id
    assert first["characters"][0]["identityStatus"] == "confirmed"


def test_single_chapter_analysis_preserves_manual_voice_from_book_roster():
    script = build_chapter_script(
        book_id="book_registry",
        chapter_id="chapter_003",
        title="第三章",
        text="“留下。”",
        language="zh",
        analyzer=RenamedCharacterAnalyzer("another_model_id"),
        known_characters=[
            {
                "id": "char_manual_role",
                "canonicalName": "李怀玉",
                "aliases": ["怀玉"],
                "gender": "female",
                "ageClass": "adult",
                "identityStatus": "confirmed",
                "voiceId": "female_adult_05",
                "voiceSource": "manual",
                "confidence": 0.98,
            }
        ],
    )

    character = script["characters"][0]
    assert character["id"] == "char_manual_role"
    assert character["voiceId"] == "female_adult_05"
    assert character["voiceSource"] == "manual"
    assert "voiceDescription" not in character


class ParentChildAnalyzer:
    def __init__(self, *, temporary_ids: bool = False):
        self.temporary_ids = temporary_ids

    def analyze_chapter(self, request):
        father_id = "character_father" if self.temporary_ids else "父亲"
        son_id = "character_son" if self.temporary_ids else "儿子"
        return ChapterAnalysisResult(
            characters=[
                CharacterAnalysis(
                    id=father_id,
                    canonical_name="父亲",
                    aliases=["爹"],
                    gender="male",
                    age_class="adult",
                    confidence=0.96,
                ),
                CharacterAnalysis(
                    id=son_id,
                    canonical_name="儿子",
                    aliases=[],
                    gender="male",
                    age_class="young",
                    confidence=0.96,
                ),
            ],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=0,
                    speaker_id=father_id,
                    emotion="neutral",
                    confidence=0.96,
                ),
                SegmentAnnotation(
                    segment_index=1,
                    speaker_id=son_id,
                    emotion="neutral",
                    confidence=0.96,
                ),
            ],
        )


def test_identity_age_routes_parent_and_son_to_distinct_appropriate_voice_profiles():
    first = build_chapter_script(
        book_id="book_yun",
        chapter_id="chapter_001",
        title="第一章",
        text="“进来吧。”\n“爹，我知道了。”",
        language="zh",
        analyzer=ParentChildAnalyzer(),
    )

    characters = {character["canonicalName"]: character for character in first["characters"]}
    father = characters["父亲"]
    son = characters["儿子"]
    assert father["ageClass"] == "adult"
    assert son["ageClass"] == "young"
    assert father["voiceId"].startswith("character_auto_")
    assert son["voiceId"].startswith("character_auto_")
    assert father["voiceId"] != son["voiceId"]
    assert father["fallbackVoiceId"] in {"male_adult_01", "male_adult_04", "male_adult_05"}
    assert son["fallbackVoiceId"] in {"male_adult_02", "male_adult_03"}
    assert father["voiceSource"] == son["voiceSource"] == "auto"
    assert "父母辈" in father["voiceDescription"]
    assert "年轻角色" in son["voiceDescription"]
    assert "不要借用旁白或其他角色" in father["voiceDescription"]

    dialogue_segments = [segment for segment in first["segments"] if segment["type"] == "dialogue"]
    assert dialogue_segments[0]["voiceDescription"] == father["voiceDescription"]
    assert dialogue_segments[1]["voiceDescription"] == son["voiceDescription"]
    assert dialogue_segments[0]["fallbackVoiceId"] == father["fallbackVoiceId"]
    assert dialogue_segments[1]["fallbackVoiceId"] == son["fallbackVoiceId"]

    second = build_chapter_script(
        book_id="book_yun",
        chapter_id="chapter_002",
        title="第二章",
        text="“去吧。”\n“我这就去。”",
        language="zh",
        analyzer=ParentChildAnalyzer(temporary_ids=True),
        known_characters=first["characters"],
    )
    second_characters = {
        character["canonicalName"]: character for character in second["characters"]
    }
    assert second_characters["父亲"]["voiceId"] == father["voiceId"]
    assert second_characters["儿子"]["voiceId"] == son["voiceId"]
    assert second_characters["父亲"]["voiceDescription"] == father["voiceDescription"]
    assert second_characters["儿子"]["voiceDescription"] == son["voiceDescription"]
    assert second_characters["父亲"]["fallbackVoiceId"] == father["fallbackVoiceId"]
    assert second_characters["儿子"]["fallbackVoiceId"] == son["fallbackVoiceId"]


def test_legacy_hash_only_parent_child_assignments_upgrade_but_manual_selection_survives():
    script = build_chapter_script(
        book_id="book_yun",
        chapter_id="chapter_001",
        title="第一章",
        text="“进来吧。”\n“爹，我知道了。”",
        language="zh",
        analyzer=ParentChildAnalyzer(),
        known_characters=[
            {
                "id": "父亲",
                "canonicalName": "父亲",
                "aliases": ["爹"],
                "gender": "male",
                "ageClass": "adult",
                # Exact old hash result: eligible for automatic migration.
                "voiceId": "male_adult_03",
            },
            {
                "id": "儿子",
                "canonicalName": "儿子",
                "aliases": [],
                "gender": "male",
                "ageClass": "young",
                # A non-hash legacy value is treated as a saved manual choice.
                "voiceId": "male_adult_01",
            },
        ],
    )

    characters = {character["canonicalName"]: character for character in script["characters"]}
    assert characters["父亲"]["voiceSource"] == "auto"
    assert characters["父亲"]["voiceId"].startswith("character_auto_")
    assert characters["父亲"]["fallbackVoiceId"] in {
        "male_adult_01",
        "male_adult_04",
        "male_adult_05",
    }
    assert characters["儿子"]["voiceSource"] == "manual"
    assert characters["儿子"]["voiceId"] == "male_adult_01"
    assert "fallbackVoiceId" not in characters["儿子"]


def test_refresh_legacy_script_reassigns_existing_parent_child_voices_without_reanalysing_text():
    legacy_script = {
        "bookId": "book_yun",
        "chapterId": "chapter_001",
        "title": "第一章",
        "language": "zh",
        "characters": [
            {
                "id": "父亲",
                "canonicalName": "父亲",
                "aliases": ["爹"],
                "gender": "male",
                "ageClass": "adult",
                # Former collision routing; no legacy source marker existed.
                "voiceId": "male_adult_04",
                "confidence": 0.96,
            },
            {
                "id": "儿子",
                "canonicalName": "儿子",
                "aliases": [],
                "gender": "male",
                "ageClass": "young",
                "voiceId": "male_adult_03",
                "confidence": 0.96,
            },
        ],
        "segments": [
            {
                "id": "seg_0001",
                "type": "dialogue",
                "text": "进来吧。",
                "speakerId": "父亲",
                "voiceId": "male_adult_04",
                "emotion": "neutral",
                "pace": "normal",
            },
            {
                "id": "seg_0002",
                "type": "dialogue",
                "text": "爹，我知道了。",
                "speakerId": "儿子",
                "voiceId": "male_adult_03",
                "emotion": "neutral",
                "pace": "normal",
            },
        ],
    }

    refreshed = refresh_script_voice_assignments(
        legacy_script, force_legacy_auto=True
    )

    characters = {character["canonicalName"]: character for character in refreshed["characters"]}
    assert characters["父亲"]["voiceId"].startswith("character_auto_")
    assert characters["儿子"]["voiceId"].startswith("character_auto_")
    assert characters["父亲"]["voiceId"] != characters["儿子"]["voiceId"]
    assert all(character["voiceAssignmentVersion"] == 3 for character in characters.values())
    assert all("fallbackVoiceId" in character for character in characters.values())
    assert refreshed["segments"][0]["text"] == "进来吧。"
    assert refreshed["segments"][0]["voiceId"] == characters["父亲"]["voiceId"]
    assert refreshed["segments"][1]["voiceId"] == characters["儿子"]["voiceId"]
    assert refreshed["segments"][0]["fallbackVoiceId"] == characters["父亲"]["fallbackVoiceId"]
    assert "voiceDescription" in refreshed["segments"][0]


class MislabelsQuotedMaterialAnalyzer:
    def analyze_chapter(self, request):
        segment_count = len(segment_dialogue(request.text, language=request.language))
        return ChapterAnalysisResult(
            characters=[],
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=index,
                    speaker_id="character_1",
                    emotion="excited",
                    confidence=0.99,
                )
                for index in range(segment_count)
            ],
        )


def test_quoted_material_cannot_be_promoted_to_character_dialogue_by_model():
    script = build_chapter_script(
        book_id="book_zh",
        chapter_id="chapter_001",
        title="第一章",
        text="自己打生下来就被称“殿下”，何时被人称过“小姐”?",
        language="zh",
        analyzer=MislabelsQuotedMaterialAnalyzer(),
        known_characters=[
            {
                "id": "character_1",
                "canonicalName": "某角色",
                "aliases": [],
                "gender": "female",
                "voiceId": "female_adult_01",
            }
        ],
    )

    quoted_segments = [
        segment
        for segment in script["segments"]
        if "quoted_material" in segment["warnings"]
    ]
    assert [segment["text"] for segment in quoted_segments] == ["殿下", "小姐"]
    assert all(segment["type"] == "narration" for segment in quoted_segments)
    assert all(segment["speakerId"] == "narrator" for segment in quoted_segments)
    assert all(segment["voiceId"] == "narrator_default" for segment in quoted_segments)


class CollisionAndTeasingAnalyzer:
    def analyze_chapter(self, request):
        characters = [
            CharacterAnalysis(
                id=name,
                canonical_name=name,
                aliases=[],
                gender="male",
                age_class="adult",
                confidence=0.95,
            )
            for name in ("林云", "赵虎", "执事")
        ]
        return ChapterAnalysisResult(
            characters=characters,
            segment_annotations=[
                SegmentAnnotation(
                    segment_index=index,
                    speaker_id=speaker_id,
                    # The old model frequently reduced these lines to happy.
                    emotion="happy",
                    confidence=0.82,
                )
                for index, speaker_id in enumerate(("林云", "赵虎", "赵虎"))
            ],
        )


def test_same_gender_collision_is_resolved_and_mocking_lines_stay_teasing():
    script = build_chapter_script(
        book_id="book_hanguang",
        chapter_id="chapter_001",
        title="第一章",
        text='“走吧。”\n“哟，杂役也配领灵石？”\n“一个废物，拿着灵石去喂狗么？”',
        language="zh",
        analyzer=CollisionAndTeasingAnalyzer(),
    )

    voices = {
        character["canonicalName"]: character["voiceId"]
        for character in script["characters"]
    }
    assert len(voices) == 3
    assert len(set(voices.values())) == 3
    assert voices["林云"] != "male_adult_05" or voices["赵虎"] != "male_adult_05"
    assert [segment["emotion"] for segment in script["segments"]] == [
        "happy",
        "teasing",
        "teasing",
    ]
