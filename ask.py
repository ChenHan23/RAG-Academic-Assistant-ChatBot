"""Terminal chat interface for the local academic RAG assistant."""

from src.config import DEFAULT_TOP_K, OLLAMA_MODEL_NAME
from src.rag_chain import answer_question


def main() -> None:
    print("Academic RAG Assistant")
    print(f"Model: Ollama / {OLLAMA_MODEL_NAME}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        try:
            answer, citations, chunks = answer_question(question, top_k=DEFAULT_TOP_K)
        except RuntimeError as exc:
            print(f"\nError: {exc}\n")
            continue

        print("\nAnswer:")
        print(answer)

        if citations:
            print("\nRetrieved sources:")
            for citation in citations:
                print(f"- {citation}")

        print("\nTop retrieved chunks:")
        for i, chunk in enumerate(chunks, start=1):
            distance = chunk["distance"]
            distance_text = f"distance={distance:.4f}" if isinstance(distance, float) else "distance=n/a"
            print(f"{i}. {chunk['source']}, page {chunk['page']} ({distance_text})")

        print()


if __name__ == "__main__":
    main()
