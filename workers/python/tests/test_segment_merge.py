from audiobook_worker.segment_merge import merge_tts_segments, split_tts_segments


def test_merges_adjacent_segments_with_same_voice_emotion_and_pace():
    segments = [
        {"id": "seg_0001", "text": "Hello", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
        {"id": "seg_0002", "text": "there.", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
        {"id": "seg_0003", "text": "Stop.", "voiceId": "b", "emotion": "angry", "pace": "fast"},
    ]

    merged = merge_tts_segments(segments)

    assert len(merged) == 2
    assert merged[0]["id"] == "seg_0001"
    assert merged[0]["text"] == "Hello there."
    assert merged[0]["sourceSegmentIds"] == ["seg_0001", "seg_0002"]
    assert merged[1]["id"] == "seg_0003"


def test_protected_audio_anchor_is_not_merged_with_adjacent_segments():
    segments = [
        {"id": "seg_0001", "text": "门外有人", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
        {"id": "seg_0002", "text": "吱呀", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
        {"id": "seg_0003", "text": "门被推开", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
    ]

    merged = merge_tts_segments(
        segments,
        protected_source_segment_ids={"seg_0002"},
    )

    assert [segment["id"] for segment in merged] == [
        "seg_0001",
        "seg_0002",
        "seg_0003",
    ]
    assert [segment["sourceSegmentIds"] for segment in merged] == [
        ["seg_0001"],
        ["seg_0002"],
        ["seg_0003"],
    ]


def test_does_not_merge_when_word_limit_would_be_exceeded():
    segments = [
        {"id": "seg_0001", "text": "one two three", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
        {"id": "seg_0002", "text": "four five six", "voiceId": "a", "emotion": "neutral", "pace": "normal"},
    ]

    merged = merge_tts_segments(segments, max_words=5)

    assert [segment["id"] for segment in merged] == ["seg_0001", "seg_0002"]


def test_splits_long_chinese_text_without_dropping_any_character():
    text = "她最后的记忆停留在飞云宫里的那一天，三月二十七，她饮下了御赐的鹤顶红，吐着大口大口的血，狼狈地趴在软榻上。"
    segments = split_tts_segments(
        [{"id": "seg_0028", "text": text, "voiceId": "narrator_default"}],
        max_characters=40,
    )

    assert len(segments) > 1
    assert "".join(segment["text"] for segment in segments) == text
    assert all(len(segment["text"]) <= 40 for segment in segments)
    assert [segment["sourceSegmentIds"] for segment in segments] == [
        ["seg_0028"]
    ] * len(segments)
    assert [segment["id"] for segment in segments] == [
        "seg_0028_part_0001",
        "seg_0028_part_0002",
    ]


def test_merging_chinese_segments_does_not_insert_or_drop_text():
    segments = [
        {"id": "seg_0001", "text": "这是第一句。", "voiceId": "a"},
        {"id": "seg_0002", "text": "这是第二句。", "voiceId": "a"},
    ]

    merged = merge_tts_segments(segments, max_characters=40)

    assert len(merged) == 1
    assert merged[0]["text"] == "这是第一句。这是第二句。"


def test_does_not_merge_same_voice_id_when_character_voice_profiles_differ():
    segments = [
        {
            "id": "seg_0001",
            "text": "父亲说。",
            "voiceId": "male_adult_01",
            "voiceDescription": "成熟父亲的固定音色",
        },
        {
            "id": "seg_0002",
            "text": "儿子答。",
            "voiceId": "male_adult_01",
            "voiceDescription": "年轻儿子的固定音色",
        },
    ]

    merged = merge_tts_segments(segments, max_characters=40)

    assert [segment["id"] for segment in merged] == ["seg_0001", "seg_0002"]


def test_does_not_merge_when_local_fallback_voices_differ():
    segments = [
        {
            "id": "seg_0001",
            "text": "第一句。",
            "voiceId": "character_auto_a",
            "fallbackVoiceId": "male_adult_01",
            "voiceDescription": "角色甲的独立音色",
        },
        {
            "id": "seg_0002",
            "text": "第二句。",
            "voiceId": "character_auto_a",
            "fallbackVoiceId": "male_adult_02",
            "voiceDescription": "角色甲的独立音色",
        },
    ]

    merged = merge_tts_segments(segments, max_characters=40)

    assert [segment["id"] for segment in merged] == ["seg_0001", "seg_0002"]


def test_scene_boundaries_do_not_block_compatible_tts_merge():
    segments = [
        {
            "id": "seg_0001",
            "text": "他还在等待。",
            "voiceId": "narrator_default",
            "emotion": "neutral",
            "pace": "normal",
            "sceneId": "scene_1",
        },
        {
            "id": "seg_0002",
            "text": "门突然响了。",
            "voiceId": "narrator_default",
            "emotion": "neutral",
            "pace": "normal",
            "sceneId": "scene_2",
        },
    ]

    merged = merge_tts_segments(segments, max_characters=100)

    assert [segment["id"] for segment in merged] == ["seg_0001"]
    assert merged[0]["sourceSegmentIds"] == ["seg_0001", "seg_0002"]
