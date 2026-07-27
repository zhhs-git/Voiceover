from __future__ import annotations

from pathlib import Path
from typing import Literal

import fitz

PDFTextLayerClassification = Literal["selectable_text", "scanned", "mixed"]


class OCRBackendNotConfigured(RuntimeError):
    code = "ocr_backend_not_configured"


def classify_pdf_text_layer(input_path: Path | str) -> PDFTextLayerClassification:
    document = fitz.open(Path(input_path))
    try:
        page_count = document.page_count
        pages_with_text = 0
        for page in document:
            if page.get_text("text").strip():
                pages_with_text += 1
    finally:
        document.close()

    if pages_with_text == 0:
        return "scanned"
    if pages_with_text < page_count:
        return "mixed"
    return "selectable_text"


def run_ocr(input_path: Path | str) -> str:
    path = Path(input_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"OCR only supports PDF files, got: {path.suffix}")

    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        raise OCRBackendNotConfigured(
            "pytesseract is not installed. Run: pip install pytesseract Pillow"
        )

    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        raise OCRBackendNotConfigured(
            "Tesseract is not installed. Run: brew install tesseract"
        )

    document = fitz.open(path)
    page_texts: list[str] = []

    try:
        for page in document:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            normalized = " ".join(text.split())
            if normalized:
                page_texts.append(normalized)
    finally:
        document.close()

    if not page_texts:
        raise OCRBackendNotConfigured(
            f"No text could be extracted from {path.name}"
        )

    return "\n\n".join(page_texts)
