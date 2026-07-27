import pytest

from audiobook_worker.dialogue import (
    detect_text_language,
    resolve_text_language,
    segment_dialogue,
)


def test_splits_narration_and_quoted_dialogue_in_source_order():
    text = 'She opened the door. "Who is there?" The hallway was empty.'

    segments = segment_dialogue(text)

    assert [segment.type for segment in segments] == ["narration", "dialogue", "narration"]
    assert [segment.text for segment in segments] == [
        "She opened the door.",
        "Who is there?",
        "The hallway was empty.",
    ]
    assert segments[1].start_offset == text.index('"Who')


def test_infers_speaker_from_trailing_speech_tag():
    text = '"Come in," Elizabeth said. Darcy waited.'

    segments = segment_dialogue(text)

    assert segments[0].type == "dialogue"
    assert segments[0].speaker_hint == "Elizabeth"
    assert segments[0].warnings == []


def test_infers_speaker_from_inverted_tag_with_cried():
    text = '"Do you not want to know who has taken it?" cried his wife impatiently.'

    segments = segment_dialogue(text)

    assert segments[0].type == "dialogue"
    assert segments[0].speaker_hint is not None


def test_infers_speaker_from_mrs_title():
    text = '"My dear Mr. Bennet," said Mrs. Bennet, "have you heard?"'

    segments = segment_dialogue(text)

    assert segments[0].speaker_hint == "Mrs. Bennet"


def test_alternating_dialogue_without_tags_is_marked_uncertain():
    text = '"Hello."\n"Good morning."'

    segments = segment_dialogue(text)

    assert [segment.type for segment in segments] == ["dialogue", "dialogue"]
    assert all(segment.speaker_hint is None for segment in segments)
    assert all("speaker_unknown" in segment.warnings for segment in segments)


def test_detects_chinese_novel_text():
    assert detect_text_language("院子里很安静。张三推门走了进来。") == "zh"
    assert detect_text_language("病") == "zh"


def test_keeps_english_novel_text_as_english():
    assert detect_text_language("The courtyard was quiet. Elizabeth opened the door.") == "en"


def test_language_detection_tolerates_chinese_with_ascii_names():
    assert detect_text_language("张三看了一眼 Alice，说道：你好。") == "zh"


def test_resolve_text_language_corrects_stale_english_value_for_chinese_text():
    text = "这是一段中文章节。张三说道：“走吧。”"

    assert resolve_text_language(text, "en") == "zh"
    assert resolve_text_language(text, None) == "zh"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("张三说：“走吧。”", "走吧。"),
        ("张三说：「走吧。」", "走吧。"),
        ("张三说：『走吧。』", "走吧。"),
        ('张三说："走吧。"', "走吧。"),
    ],
)
def test_splits_chinese_dialogue_quote_pairs(text: str, expected: str):
    segments = segment_dialogue(text, language="zh")
    dialogue = [segment for segment in segments if segment.type == "dialogue"]
    assert [segment.text for segment in dialogue] == [expected]


def test_chinese_dialogue_preserves_source_order_and_offsets():
    text = "风停了。张三说道：“走吧。”院门重新关上。"
    segments = segment_dialogue(text, language="zh")

    assert [segment.type for segment in segments] == ["narration", "dialogue", "narration"]
    assert text[segments[1].start_offset : segments[1].end_offset] == "“走吧。”"


@pytest.mark.parametrize(
    ("text", "speaker"),
    [
        ("张三说道：“走吧。”", "张三"),
        ("李婶问：“吃了吗？”", "李婶"),
        ("母亲叹道：『你长大了。』", "母亲"),
    ],
)
def test_infers_chinese_speaker_before_quote(text: str, speaker: str):
    dialogue = next(
        segment for segment in segment_dialogue(text, language="zh") if segment.type == "dialogue"
    )
    assert dialogue.speaker_hint == speaker
    assert dialogue.warnings == []


def test_infers_chinese_speaker_after_quote():
    dialogue = next(
        segment
        for segment in segment_dialogue("“走吧。”张三说道。", language="zh")
        if segment.type == "dialogue"
    )
    assert dialogue.speaker_hint == "张三"


def test_does_not_guess_speakers_for_untagged_chinese_turns():
    segments = segment_dialogue("“你来了？”\n“刚到。”", language="zh")
    dialogue = [segment for segment in segments if segment.type == "dialogue"]

    assert [segment.speaker_hint for segment in dialogue] == [None, None]
    assert all("speaker_unknown" in segment.warnings for segment in dialogue)


def test_splits_line_delimited_chinese_dialogue_with_malformed_curly_quotes():
    text = (
        "“这满街的白幡是做什么?“\n"
        "”嗬，官老爷都系白腰带?’\n"
        "“护国长公主薨了啊!“\n"
        "茶肆里的人议论纷纷。"
    )

    segments = segment_dialogue(text, language="zh")

    assert [segment.type for segment in segments] == [
        "dialogue",
        "dialogue",
        "dialogue",
        "narration",
    ]
    assert [segment.text for segment in segments[:3]] == [
        "这满街的白幡是做什么?",
        "嗬，官老爷都系白腰带?",
        "护国长公主薨了啊!",
    ]
    assert all("\n" not in segment.text for segment in segments[:3])


def test_splits_malformed_chinese_dialogue_after_speech_tag():
    text = "丫鬟喜道：“小姐，你终于醒了!”\n她端着水盆走近。"

    segments = segment_dialogue(text, language="zh")

    assert [segment.type for segment in segments] == ["narration", "dialogue", "narration"]
    assert segments[1].text == "小姐，你终于醒了!"
    assert segments[1].speaker_hint == "丫鬟"


def test_closes_unterminated_chinese_dialogue_at_line_boundary():
    text = (
        "笑意一顿，她回头问：“你是在喊我?\n"
        "灵秀点头：“奴婢当然是在喊您啊小姐，您不认得奴婢了?\n"
        "她认真地想了一会儿。"
    )

    segments = segment_dialogue(text, language="zh")
    dialogue = [segment for segment in segments if segment.type == "dialogue"]

    assert [segment.text for segment in dialogue] == [
        "你是在喊我?",
        "奴婢当然是在喊您啊小姐，您不认得奴婢了?",
    ]
    assert segments[-1].type == "narration"
    assert segments[-1].text == "她认真地想了一会儿。"


def test_classifies_quoted_titles_and_forms_of_address_as_narration():
    text = "自己打生下来就被称“殿下”，何时被人称过“小姐”?"

    segments = segment_dialogue(text, language="zh")
    quoted = [segment for segment in segments if "quoted_material" in segment.warnings]

    assert [segment.type for segment in segments] == [
        "narration",
        "narration",
        "narration",
        "narration",
        "narration",
    ]
    assert [segment.text for segment in quoted] == ["殿下", "小姐"]
    assert all(segment.speaker_hint is None for segment in quoted)


def test_classifies_sound_effect_and_citation_as_narration():
    sound = segment_dialogue("门“吱呀”一声被推开。", language="zh")
    citation = segment_dialogue("桌上摊着《远方》：“明月照故乡。”", language="zh")
    labels = segment_dialogue(
        "因为“谋杀重臣”被囚飞云宫，更在这一天“病”死。", language="zh"
    )

    assert sound[1].type == "narration"
    assert sound[1].warnings == ["quoted_material"]
    assert citation[-1].type == "narration"
    assert citation[-1].warnings == ["quoted_material"]
    assert [segment.text for segment in labels if "quoted_material" in segment.warnings] == [
        "谋杀重臣",
        "病",
    ]


def test_explicit_chinese_speech_with_quoted_title_inside_remains_dialogue():
    segments = segment_dialogue("丫鬟喜道：“小姐，你终于醒了!”", language="zh")

    dialogue = next(segment for segment in segments if segment.type == "dialogue")
    assert dialogue.text == "小姐，你终于醒了!"
    assert dialogue.speaker_hint == "丫鬟"
    assert "quoted_material" not in dialogue.warnings


def test_previous_quoted_material_does_not_poison_following_dialogue():
    segments = segment_dialogue(
        "她被称“殿下”。丫鬟喜道：“小姐，你终于醒了!”", language="zh"
    )

    dialogue = next(segment for segment in segments if segment.type == "dialogue")
    assert dialogue.text == "小姐，你终于醒了!"
    assert dialogue.speaker_hint == "丫鬟"
