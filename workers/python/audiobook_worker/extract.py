from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ebooklib
import fitz
import re
from bs4 import BeautifulSoup
from ebooklib import epub

from audiobook_worker.ocr import classify_pdf_text_layer


@dataclass(frozen=True)
class ExtractedBookText:
    kind: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    requires_ocr: bool = False
    warnings: list[str] = field(default_factory=list)


def extract_book_text(input_path: Path | str) -> ExtractedBookText:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return _extract_epub(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".txt":
        return _extract_txt(path)
    raise ValueError(f"Unsupported book format: {suffix}")


def _extract_epub(path: Path) -> ExtractedBookText:
    book = epub.read_epub(str(path))
    chunks: list[str] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n")
        normalized = _normalize_text(text)
        if normalized:
            chunks.append(normalized)

    title_values = book.get_metadata("DC", "title")
    language_values = book.get_metadata("DC", "language")
    metadata: dict[str, Any] = {}
    if title_values:
        metadata["title"] = title_values[0][0]
    if language_values:
        metadata["language"] = language_values[0][0]

    return ExtractedBookText(
        kind="epub",
        text="\n\n".join(chunks),
        metadata=metadata,
        requires_ocr=False,
    )


def _extract_pdf(path: Path) -> ExtractedBookText:
    from audiobook_worker.ocr import run_ocr

    text_layer = classify_pdf_text_layer(path)
    document = fitz.open(path)
    page_texts: list[str] = []
    ocr_warnings: list[str] = []
    uses_ocr = False
    try:
        if text_layer == "scanned":
            try:
                text = run_ocr(path)
                page_texts.append(text)
                uses_ocr = True
            except Exception as e:
                return ExtractedBookText(
                    kind="pdf",
                    text="",
                    metadata={
                        "page_count": document.page_count,
                        "text_layer": text_layer,
                        **{key: value for key, value in document.metadata.items() if value},
                    },
                    requires_ocr=True,
                    warnings=[f"OCR failed: {e}"],
                )
        elif text_layer == "mixed":
            for page in document:
                page_text = page.get_text("text").strip()
                if page_text:
                    page_texts.append(page_text)
                else:
                    ocr_warnings.append("mixed_page_required_ocr")
            if ocr_warnings:
                ocr_warnings.append("Consider re-running with full OCR for best results")
        else:
            for page in document:
                page_texts.append(page.get_text("text"))

        metadata = {
            "page_count": document.page_count,
            "text_layer": text_layer,
            **{key: value for key, value in document.metadata.items() if value},
        }
    finally:
        document.close()

    text = _normalize_text("\n\n".join(page_texts))
    return ExtractedBookText(
        kind="pdf",
        text=text,
        metadata=metadata,
        requires_ocr=text_layer in {"scanned", "mixed"},
        warnings=ocr_warnings if ocr_warnings else (
            ["requires_ocr"] if text_layer in {"scanned", "mixed"} and not uses_ocr else []
        ),
    )


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix Gutenberg drop-cap: single letter on its own line
    text = re.sub(r"^([A-Z])\n", r"\1", text, flags=re.MULTILINE)
    # Fix orphaned closing quote on its own line
    text = re.sub(r"\n([\u201d])\s*\n", r"\1\n", text)
    # Collapse whitespace per line, preserving paragraph breaks
    text = "\n".join(" ".join(line.split()) for line in text.splitlines())
    # Remove copyright/production brackets
    text = re.sub(r"\[[^\]]*?[Cc]opyright[^\]]*?\]", "", text)
    text = re.sub(r"\[[^\]]*?[Pp]roduced[^\]]*?\]", "", text)
    return text


def _extract_txt(path: Path) -> ExtractedBookText:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = data.decode("gb18030")
        encoding = "gb18030"
    return ExtractedBookText(
        kind="txt",
        text=_normalize_text(text),
        metadata={"title": path.stem, "encoding": encoding},
        requires_ocr=False,
    )
