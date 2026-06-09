from __future__ import annotations

from typing import Any, Dict, List

import ollama
from sentence_transformers import SentenceTransformer

from src.vector_store import get_collection


DEFAULT_TOP_K = 5
OLLAMA_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


SYSTEM_PROMPT = """You are an academic research assistant.

Answer the user's question using only the provided context.

Rules:
- Use only the information in the context.
- If the context does not contain the answer, say: "I don't know based on the provided context."
- Cite sources in the answer using this format: (Source: filename, page X).
- Do not invent facts, citations, page numbers, or filenames.
- Be concise, accurate, and clear.
"""


_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_question(question: str) -> List[float]:
    return _embedding_model.encode([question]).tolist()[0]


def distance_to_relevance_percent(distance: float | int | None) -> float | None:
    """
    ChromaDB cosine distance is lower when documents are more similar.
    If distance = 0.10, relevance is approximately 90%.
    """
    if distance is None:
        return None

    relevance = (1.0 - float(distance)) * 100
    relevance = max(0.0, min(100.0, relevance))

    return relevance


def retrieve_chunks(question: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    collection = get_collection()

    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB is empty. Add PDFs to data/ and run: python ingest.py --reset"
        )

    query_embedding = embed_question(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: List[Dict[str, Any]] = []

    for i, document in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None
        relevance = distance_to_relevance_percent(distance)

        chunks.append(
            {
                "rank": i + 1,
                "text": document,
                "source": metadata.get("source", "Unknown source"),
                "page": metadata.get("page", "Unknown page"),
                "distance": distance,
                "relevance": relevance,
            }
        )

    return chunks


def format_context(chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for chunk in chunks:
        relevance_text = (
            "N/A"
            if chunk["relevance"] is None
            else f"{chunk['relevance']:.1f}%"
        )

        context_blocks.append(
            f"""[{chunk['rank']}]
Source: {chunk['source']}
Page: {chunk['page']}
Relevance: {relevance_text}

Text:
{chunk['text']}"""
        )

    return "\n\n---\n\n".join(context_blocks)


def ask_ollama(question: str, context: str) -> str:
    user_prompt = f"""Context:
{context}

Question:
{question}

Answer:"""

    response = ollama.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]


def run_rag(question: str, top_k: int = DEFAULT_TOP_K):
    chunks = retrieve_chunks(question, top_k=top_k)
    context = format_context(chunks)
    answer = ask_ollama(question, context)

    return answer, chunks


def print_citations(chunks: List[Dict[str, Any]]) -> None:
    print("\nRetrieved Sources:")
    print("-" * 60)

    for chunk in chunks:
        relevance_text = (
            "N/A"
            if chunk["relevance"] is None
            else f"{chunk['relevance']:.1f}%"
        )

        print(
            f"[{chunk['rank']}] {chunk['source']} "
            f"| page {chunk['page']} "
            f"| relevance {relevance_text}"
        )


def main() -> None:
    print("Academic RAG Query Script")
    print(f"LLM: Ollama / {OLLAMA_MODEL_NAME}")
    print(f"Embeddings: {EMBEDDING_MODEL_NAME}")
    print(f"Retriever: top {DEFAULT_TOP_K} chunks from local ChromaDB")
    print("Type 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        try:
            answer, chunks = run_rag(question)

            print("\nAnswer:")
            print(answer)

            print_citations(chunks)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()