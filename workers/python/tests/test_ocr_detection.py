from pathlib import Path

import fitz

from audiobook_worker.extract import extract_book_text
from audiobook_worker.ocr import classify_pdf_text_layer, run_ocr


def write_blank_pdf(path: Path) -> None:
    document = fitz.open()
    document.new_page()
    document.save(path)
    document.close()


def write_mixed_pdf(path: Path) -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "This page has selectable text.")
    document.new_page()
    document.save(path)
    document.close()


def test_blank_pdf_is_classified_as_scanned(tmp_path: Path):
    input_path = tmp_path / "blank.pdf"
    write_blank_pdf(input_path)

    classification = classify_pdf_text_layer(input_path)
    extracted = extract_book_text(input_path)

    assert classification == "scanned"
    assert extracted.requires_ocr is True


def test_mixed_pdf_is_classified_as_mixed(tmp_path: Path):
    input_path = tmp_path / "mixed.pdf"
    write_mixed_pdf(input_path)

    assert classify_pdf_text_layer(input_path) == "mixed"


def test_run_ocr_returns_text_from_rendered_page(tmp_path: Path):
    """OCR extracts text from a rendered PDF page."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text(fitz.Point(20, 50), "Hello World", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    text = run_ocr(str(pdf_path))
    assert "Hello" in text or "World" in text
