"""RAG question answering with local Ollama."""

from typing import Dict, List, Tuple

import ollama

from src.citations import format_context_for_prompt, retrieved_chunks, unique_citations
from src.config import DEFAULT_TOP_K, OLLAMA_MODEL_NAME
from src.vector_store import query_chroma


SYSTEM_INSTRUCTIONS = """
You are an academic research assistant.
Answer using only the provided context from the user's uploaded PDFs.

Rules:
- If the answer is not supported by the context, say: "I don't know based on the provided documents."
- Cite claims using this exact style: (Source: filename, page X).
- Do not invent sources, page numbers, quotes, authors, or facts.
- Prefer a clear, concise answer.
""".strip()


def build_prompt(question: str, context: str) -> str:
    return f"""
{SYSTEM_INSTRUCTIONS}

Context:
{context}

Question:
{question}

Answer:
""".strip()


def ask_ollama(prompt: str, model_name: str = OLLAMA_MODEL_NAME) -> str:
    """Send a prompt to Ollama and return the generated text."""
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is installed, running, "
            f"and that you pulled the model with: ollama pull {model_name}"
        ) from exc

    # The ollama package returns a dict-like object in current versions.
    return response["message"]["content"]


def answer_question(question: str, top_k: int = DEFAULT_TOP_K) -> Tuple[str, List[str], List[Dict[str, object]]]:
    """Retrieve relevant chunks and answer a question with citations."""
    results = query_chroma(question, top_k=top_k)
    context = format_context_for_prompt(results)

    if not context.strip():
        return "I don't know based on the provided documents.", [], []

    prompt = build_prompt(question, context)
    answer = ask_ollama(prompt)

    return answer, unique_citations(results), retrieved_chunks(results)
