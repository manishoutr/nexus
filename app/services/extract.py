"""Text extraction per format.

Returns a list of (page_number, text) pairs. PDFs give real page numbers;
DOCX and TXT have no pages, so page=0 and citations fall back to sections.
"""
import io
from typing import List, Tuple

ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")


def extract_pages(file_bytes: bytes, filename: str) -> List[Tuple[int, str]]:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        return [(i + 1, page.extract_text() or "")
                for i, page in enumerate(reader.pages)]
    if name.endswith(".docx"):
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [(0, text)]
    if name.endswith((".txt", ".md")):
        return [(0, file_bytes.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type: {filename}. "
                     f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
