"""Citation and retrieved-context formatting."""

from typing import Dict, List


def _first_result_list(results: Dict[str, object], key: str) -> List[object]:
    value = results.get(key, [[]])
    if not value:
        return []
    return value[0]


def format_context_for_prompt(results: Dict[str, object]) -> str:
    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")

    blocks = []
    for doc, metadata in zip(documents, metadatas):
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        blocks.append(
            f"Source: {source}\nPage: {page}\nText:\n{doc}"
        )

    return "\n\n---\n\n".join(blocks)


def unique_citations(results: Dict[str, object]) -> List[str]:
    metadatas = _first_result_list(results, "metadatas")
    seen = set()
    citations = []

    for metadata in metadatas:
        key = (metadata.get("source"), metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        citations.append(f'{metadata.get("source")}, page {metadata.get("page")}')

    return citations


def retrieved_chunks(results: Dict[str, object]) -> List[Dict[str, object]]:
    documents = _first_result_list(results, "documents")
    metadatas = _first_result_list(results, "metadatas")
    distances = _first_result_list(results, "distances")

    chunks = []
    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        distance = distances[i] if i < len(distances) else None
        chunks.append(
            {
                "text": doc,
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": distance,
            }
        )

    return chunks
