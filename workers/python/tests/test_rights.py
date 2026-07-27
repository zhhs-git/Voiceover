from pathlib import Path

from audiobook_worker.rights import classify_rights


def test_classifies_creative_commons_notice_as_allowed(tmp_path: Path):
    path = tmp_path / "book.txt"
    path.write_text("This work is licensed under CC BY 4.0.", encoding="utf-8")

    result = classify_rights(path, metadata={})

    assert result.classification == "allowed"
    assert result.reason == "creative_commons"


def test_classifies_drm_metadata_as_blocked(tmp_path: Path):
    path = tmp_path / "book.txt"
    path.write_text("Encrypted content", encoding="utf-8")

    result = classify_rights(path, metadata={"drm": True})

    assert result.classification == "blocked"
    assert result.reason == "drm_detected"


def test_classifies_missing_rights_as_unknown(tmp_path: Path):
    path = tmp_path / "book.txt"
    path.write_text("A book without rights metadata.", encoding="utf-8")

    result = classify_rights(path, metadata={})

    assert result.classification == "unknown"
    assert result.requires_attestation is True
