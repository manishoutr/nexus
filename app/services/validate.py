"""Upload validation by file content, not just the extension.

Each check raises UploadRejected with an HTTP-ish status and a plain message,
so the API can return it directly and the worker can record it.
"""
import io
import zipfile

from .extract import ALLOWED_EXTENSIONS


class UploadRejected(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Return the detected kind ("pdf", "docx", "text") or raise."""
    name = filename.lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise UploadRejected(415, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    if len(data) == 0:
        raise UploadRejected(422, "Empty file")
    if len(data) > max_bytes:
        raise UploadRejected(413, f"File too large (max {max_bytes // (1024 * 1024)} MB)")

    if name.endswith(".pdf"):
        if not data.lstrip()[:5] == b"%PDF-":
            raise UploadRejected(415, "File is not a PDF (bad signature)")
        _check_pdf(data)
        return "pdf"
    if name.endswith(".docx"):
        if data[:4] != b"PK\x03\x04":
            raise UploadRejected(415, "File is not a DOCX (bad signature)")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if "word/document.xml" not in z.namelist():
                    raise UploadRejected(415, "File is a zip but not a Word document")
        except zipfile.BadZipFile:
            raise UploadRejected(422, "DOCX is corrupt")
        return "docx"
    # .txt / .md
    if b"\x00" in data[:8192]:
        raise UploadRejected(415, "File looks binary, not text")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise UploadRejected(422, "Text file is not valid UTF-8")
    if not data.strip():
        raise UploadRejected(422, "File contains only whitespace")
    return "text"


def _check_pdf(data: bytes) -> None:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadRejected(422, "PDF is password-protected; upload an unlocked copy")
        if len(reader.pages) == 0:
            raise UploadRejected(422, "PDF has no pages")
    except UploadRejected:
        raise
    except (PdfReadError, Exception) as e:  # noqa: BLE001 - any parse failure = corrupt
        raise UploadRejected(422, f"PDF is corrupt or unreadable: {type(e).__name__}")
