"""PDF loading utilities.

Uses PyMuPDF to extract text page by page so every chunk can keep a reliable
source filename and page number for citations.
"""

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF


PageRecord = Dict[str, object]


def load_pdf_pages(pdf_path: str | Path) -> List[PageRecord]:
    """Extract non-empty text pages from a PDF.

    Returns:
        A list of dictionaries with text and citation metadata.
    """
    pdf_path = Path(pdf_path)
    pages: List[PageRecord] = []

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue

            pages.append(
                {
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page": page_index + 1,
                    },
                }
            )

    return pages
