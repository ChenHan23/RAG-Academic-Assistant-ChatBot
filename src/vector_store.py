"""Local ChromaDB vector store helpers."""

from functools import lru_cache
from typing import Dict, List

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the local embedding model once per Python process."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection() -> Collection:
    """Get or create the persistent academic papers collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def make_chunk_id(chunk: Dict[str, object]) -> str:
    metadata = chunk["metadata"]
    return f'{metadata["source"]}_p{metadata["page"]}_c{metadata["chunk_id"]}'


def add_chunks_to_chroma(chunks: List[Dict[str, object]]) -> int:
    """Upsert chunks into ChromaDB.

    Upsert makes repeated ingestion safe while you are developing.
    """
    if not chunks:
        return 0

    collection = get_collection()
    documents = [str(chunk["text"]) for chunk in chunks]
    ids = [make_chunk_id(chunk) for chunk in chunks]
    metadatas = [dict(chunk["metadata"]) for chunk in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def query_chroma(question: str, top_k: int = 5) -> Dict[str, object]:
    """Retrieve the most relevant chunks for a question."""
    collection = get_collection()
    query_embedding = embed_texts([question])[0]

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def reset_collection() -> None:
    """Delete and recreate the Chroma collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_chroma_client.cache_clear()
    get_embedding_model.cache_clear()
    get_collection()
