from audiobook_worker.chapters import detect_chapters


def test_detects_arabic_number_chapter_headings():
    text = """Chapter 1
The first room was quiet.

Chapter 2
The second room was loud."""

    chapters = detect_chapters(text)

    assert [chapter.title for chapter in chapters] == ["Chapter 1", "Chapter 2"]
    assert chapters[0].text == "The first room was quiet."
    assert chapters[1].text == "The second room was loud."


def test_detects_roman_number_chapter_headings():
    text = """CHAPTER II
Elizabeth listened.

CHAPTER III
Darcy answered."""

    chapters = detect_chapters(text)

    assert [chapter.title for chapter in chapters] == ["CHAPTER II", "CHAPTER III"]


def test_returns_single_chapter_when_no_heading_is_found():
    text = "A short story without an explicit chapter heading."

    chapters = detect_chapters(text)

    assert len(chapters) == 1
    assert chapters[0].id == "chapter_001"
    assert chapters[0].title == "Chapter 1"
    assert chapters[0].confidence == 0.35


def test_detects_chinese_chapter_headings():
    text = """第一章 初见
院子里很安静。

第二章 重逢
他们再次见面。"""

    chapters = detect_chapters(text)

    assert [chapter.title for chapter in chapters] == ["第一章 初见", "第二章 重逢"]
    assert chapters[0].text == "院子里很安静。"
    assert chapters[1].text == "他们再次见面。"
