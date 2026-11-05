import json
from pathlib import Path

from pypdf import PdfReader


PROJECT_PATH = Path(__file__).resolve().parent
PDF_PATH = PROJECT_PATH / "data" / "Demo_Healthcare_Benefits_Plan.pdf"
OUTPUT_PATH = PROJECT_PATH / "data" / "pdf_chunks.json"


def split_into_chunks(
    text: str,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[str]:
    """Split text into overlapping groups of words."""
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def main():
    reader = PdfReader(PDF_PATH)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Remove the repeated footer from our sample PDF.
        text = "\n".join(
            line
            for line in text.splitlines()
            if not line.strip().startswith(
                ("SYNTHETIC DEMO |", "Page ")
            )
        )

        if not text.strip():
            raise ValueError(
                f"No text extracted from page {page_number}. "
                "Check whether OCR is needed."
            )

        chunks = split_into_chunks(text)

        for chunk_number, chunk in enumerate(chunks, start=1):
            records.append(
                {
                    "id": (
                        f"demo-plan-001-pdf-v1-"
                        f"page-{page_number}-chunk-{chunk_number}"
                    ),
                    "text": chunk,
                    "metadata": {
                        "plan_id": "DEMO-PLAN-001",
                        "source": PDF_PATH.name,
                        "page": page_number,
                        "document_version": "1.0",
                        "section": f"PDF page {page_number}",
                    },
                }
            )

        print(f"Page {page_number}: {len(chunks)} chunks")

    OUTPUT_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nTotal chunks: {len(records)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFirst chunk:")
    print(json.dumps(records[0], indent=2))


if __name__ == "__main__":
    main()