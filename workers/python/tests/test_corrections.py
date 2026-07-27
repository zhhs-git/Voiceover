from audiobook_worker.script_builder import build_chapter_script_with_corrections


def test_alias_merge_replaces_speaker_in_segments():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Over here," Lizzy called. "Coming," Elizabeth replied.',
        language="en",
        corrections={
            "aliasMerges": [{"from": "Lizzy", "to": "Elizabeth"}],
        },
    )

    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    assert len(speakers) == 1
    assert speakers == {script["characters"][0]["id"]}


def test_gender_override_changes_character_and_voice():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
        corrections={
            "genderOverrides": [{"characterId": "darcy", "gender": "female"}],
        },
    )

    character = next(c for c in script["characters"] if c["canonicalName"].startswith("Darcy"))
    assert character["gender"] == "female"
    assert character["voiceId"].startswith("character_auto_")
    assert character["fallbackVoiceId"].startswith("female_adult_")
    assert character["voiceSource"] == "auto"


def test_voice_override_changes_assigned_voice():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
        corrections={
            "voiceOverrides": [{"characterId": "darcy", "voiceId": "male_adult_01"}],
        },
    )

    character = next(c for c in script["characters"] if c["canonicalName"].startswith("Darcy"))
    assert character["voiceId"] == "male_adult_01"
    assert "fallbackVoiceId" not in character

    # segments should use the overridden voice
    segments_with_darcy = [s for s in script["segments"] if s["speakerId"] == character["id"]]
    assert all(s["voiceId"] == "male_adult_01" for s in segments_with_darcy)
    assert all("fallbackVoiceId" not in s for s in segments_with_darcy)


def test_no_corrections_returns_same_as_build_chapter_script():
    from audiobook_worker.script_builder import build_chapter_script

    kwargs = dict(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
    )
    baseline = build_chapter_script(**kwargs)
    corrected = build_chapter_script_with_corrections(**kwargs, corrections={})

    assert corrected["segments"] == baseline["segments"]
    assert corrected["characters"] == baseline["characters"]


def test_alias_with_word_boundary_prevents_partial_match():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Over here," Lizzy called. "Coming," Liz said.',
        language="en",
        corrections={
            "aliasMerges": [{"from": "Liz", "to": "Elizabeth"}],
        },
    )

    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    character_ids = {c["canonicalName"]: c["id"] for c in script["characters"]}
    assert character_ids["Lizzy"] in speakers
    assert character_ids["Elizabeth"] in speakers
    assert len(speakers) == 2


def test_multiple_aliases_applied():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Hey," said Liz. "Hello," said Lizzy.',
        language="en",
        corrections={
            "aliasMerges": [
                {"from": "Lizzy", "to": "Elizabeth"},
                {"from": "Liz", "to": "Elizabeth"},
            ],
        },
    )

    speakers = {seg["speakerId"] for seg in script["segments"] if seg["type"] == "dialogue"}
    assert speakers == {script["characters"][0]["id"]}


def test_voices_list_updated_after_gender_override():
    script = build_chapter_script_with_corrections(
        book_id="book_1",
        chapter_id="ch01",
        title="Chapter 1",
        text='"Indeed," said Darcy.',
        language="en",
        corrections={
            "genderOverrides": [{"characterId": "darcy", "gender": "female"}],
        },
    )

    voice_ids_in_script = {v["id"] for v in script["voices"]}
    character = next(c for c in script["characters"] if c["canonicalName"].startswith("Darcy"))
    assert character["voiceId"] in voice_ids_in_script
    assert "male_adult_01" not in voice_ids_in_script


def test_corrections_none_raises_value_error():
    import pytest

    with pytest.raises(ValueError, match="must be a dict"):
        build_chapter_script_with_corrections(
            book_id="book_1",
            chapter_id="ch01",
            title="Chapter 1",
            text='"Hey," said Liz.',
            language="en",
            corrections=None,
        )


def test_missing_keys_in_corrections_raises_key_error():
    import pytest

    with pytest.raises(KeyError, match="aliasMerges item missing required key 'from'"):
        build_chapter_script_with_corrections(
            book_id="book_1",
            chapter_id="ch01",
            title="Chapter 1",
            text='"Hey," said Liz.',
            language="en",
            corrections={"aliasMerges": [{"to": "Elizabeth"}]},
        )

    with pytest.raises(KeyError, match="genderOverrides item missing required key 'characterId'"):
        build_chapter_script_with_corrections(
            book_id="book_1",
            chapter_id="ch01",
            title="Chapter 1",
            text='"Hey," said Liz.',
            language="en",
            corrections={"genderOverrides": [{"gender": "female"}]},
        )

    with pytest.raises(KeyError, match="voiceOverrides item missing required key 'voiceId'"):
        build_chapter_script_with_corrections(
            book_id="book_1",
            chapter_id="ch01",
            title="Chapter 1",
            text='"Hey," said Liz.',
            language="en",
            corrections={"voiceOverrides": [{"characterId": "liz"}]},
        )
