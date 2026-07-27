from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import fitz

from audiobook_worker.extract import extract_book_text


def write_tiny_epub(path: Path) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>""",
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="UTF-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:identifier id="bookid">tiny-book</dc:identifier>
                <dc:title>Tiny EPUB</dc:title>
                <dc:language>en</dc:language>
              </metadata>
              <manifest>
                <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="chapter1"/>
              </spine>
            </package>""",
            compress_type=ZIP_DEFLATED,
        )
        archive.writestr(
            "OEBPS/chapter1.xhtml",
            """<?xml version="1.0" encoding="UTF-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml">
              <body>
                <h1>Chapter 1</h1>
                <p>Hello from an EPUB chapter.</p>
              </body>
            </html>""",
            compress_type=ZIP_DEFLATED,
        )


def write_tiny_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Hello from a selectable PDF.")
    document.save(path)
    document.close()


def test_extracts_text_from_epub(tmp_path: Path):
    input_path = tmp_path / "book.epub"
    write_tiny_epub(input_path)

    result = extract_book_text(input_path)

    assert result.kind == "epub"
    assert result.requires_ocr is False
    assert "Chapter 1" in result.text
    assert "Hello from an EPUB chapter." in result.text
    assert result.metadata["title"] == "Tiny EPUB"


def test_extracts_selectable_text_from_pdf(tmp_path: Path):
    input_path = tmp_path / "book.pdf"
    write_tiny_pdf(input_path)

    result = extract_book_text(input_path)

    assert result.kind == "pdf"
    assert result.requires_ocr is False
    assert "Hello from a selectable PDF." in result.text
    assert result.metadata["page_count"] == 1


def test_extracts_utf8_text_file(tmp_path: Path):
    input_path = tmp_path / "中文小说.txt"
    input_path.write_text("\ufeff第一章 开始\r\n\r\n这是第一段。\r\n这是第二段。", encoding="utf-8")

    result = extract_book_text(input_path)

    assert result.kind == "txt"
    assert result.requires_ocr is False
    assert result.text == "第一章 开始\n\n这是第一段。\n这是第二段。"
    assert result.metadata == {"title": "中文小说", "encoding": "utf-8-sig"}


def test_extracts_gb18030_text_file(tmp_path: Path):
    input_path = tmp_path / "旧版中文小说.txt"
    input_path.write_bytes("第一章\r\n这是简体中文正文。".encode("gb18030"))

    result = extract_book_text(input_path)

    assert result.text == "第一章\n这是简体中文正文。"
    assert result.metadata["encoding"] == "gb18030"
