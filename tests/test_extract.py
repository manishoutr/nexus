"""Extraction: formats produce (page, text) pairs; bad types are rejected."""
import pytest

from app.services.extract import extract_pages


def test_txt_gives_single_unpaged_block():
    pages = extract_pages(b"hello world\n\nsecond paragraph", "notes.txt")
    assert pages == [(0, "hello world\n\nsecond paragraph")]


def test_pdf_gives_real_page_numbers():
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    pdf.drawString(60, 800, "page one content")
    pdf.showPage()
    pdf.drawString(60, 800, "page two content")
    pdf.save()

    pages = extract_pages(buf.getvalue(), "doc.pdf")
    assert [p for p, _ in pages] == [1, 2]
    assert "page one content" in pages[0][1]
    assert "page two content" in pages[1][1]


def test_unsupported_type_is_rejected():
    with pytest.raises(ValueError):
        extract_pages(b"\x00\x01", "archive.zip")
