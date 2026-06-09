"""Text chunking utilities."""

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


ChunkRecord = Dict[str, object]


def chunk_pages(
    pages: List[Dict[str, object]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[ChunkRecord]:
    """Split page text into overlapping chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[ChunkRecord] = []

    for page in pages:
        text = str(page["text"])
        metadata = dict(page["metadata"])
        page_chunks = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            clean_text = chunk_text.strip()
            if not clean_text:
                continue

            chunks.append(
                {
                    "text": clean_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": chunk_index,
                    },
                }
            )

    return chunks
