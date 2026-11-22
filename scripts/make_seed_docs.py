"""Generate PDF and DOCX versions of a seed document.

This proves multi-format ingestion in the demo: the same content is
available as .txt (committed), .pdf and .docx (generated here).

Run: python scripts/make_seed_docs.py
"""
from pathlib import Path

SEED_DIR = Path(__file__).resolve().parent.parent / "seed_docs"


def make_pdf(text: str, out_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(out_path), pagesize=A4)
    width, height = A4
    y = height - 60
    for line in text.splitlines():
        if y < 60:
            pdf.showPage()
            y = height - 60
        # crude wrap so lines fit the page
        while len(line) > 95:
            pdf.drawString(60, y, line[:95])
            y -= 15
            line = line[95:]
            if y < 60:
                pdf.showPage()
                y = height - 60
        pdf.drawString(60, y, line)
        y -= 15
    pdf.save()


def make_docx(text: str, out_path: Path) -> None:
    import docx

    doc = docx.Document()
    for para in text.split("\n\n"):
        doc.add_paragraph(para.strip())
    doc.save(str(out_path))


def main() -> None:
    rag_text = (SEED_DIR / "retrieval-augmented-generation.txt").read_text(encoding="utf-8")
    make_pdf(rag_text, SEED_DIR / "retrieval-augmented-generation.pdf")
    city_text = (SEED_DIR / "bhubaneswar-smart-city.txt").read_text(encoding="utf-8")
    make_docx(city_text, SEED_DIR / "bhubaneswar-smart-city.docx")
    print("wrote retrieval-augmented-generation.pdf and bhubaneswar-smart-city.docx")


if __name__ == "__main__":
    main()
