from pathlib import Path
import argparse
import shutil

from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import add_chunks_to_chroma


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


def reset_chroma_db():
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Deleted existing chroma_db/")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing ChromaDB before ingesting."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters."
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap in characters."
    )

    args = parser.parse_args()

    if args.reset:
        reset_chroma_db()

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/. Add at least one PDF and rerun.")
        return

    all_chunks = []

    print(f"Using chunk size: {args.chunk_size}")
    print(f"Using chunk overlap: {args.chunk_overlap}")

    for pdf_file in pdf_files:
        print(f"\nLoading {pdf_file.name}...")

        pages = load_pdf_pages(str(pdf_file))
        chunks = chunk_pages(
            pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        print(f"Created {len(chunks)} chunks from {pdf_file.name}")
        all_chunks.extend(chunks)

    add_chunks_to_chroma(all_chunks)

    print(f"\nAdded {len(all_chunks)} chunks to ChromaDB.")


if __name__ == "__main__":
    main()