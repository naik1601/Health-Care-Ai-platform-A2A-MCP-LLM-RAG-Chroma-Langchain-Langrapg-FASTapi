from pathlib import Path

from pypdf import PdfReader


# Locate the PDF relative to this script.
project_path = Path(__file__).resolve().parent
pdf_path = project_path / "data" / "Demo_Healthcare_Benefits_Plan.pdf"

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

reader = PdfReader(pdf_path)

print(f"Total pages: {len(reader.pages)}")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    print(f"\n--- Page {page_number} ---")

    if text.strip():
        print(text)
    else:
        print("No text extracted. This page may require OCR.")