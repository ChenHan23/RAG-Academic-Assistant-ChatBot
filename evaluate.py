from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from query import run_rag


TEST_DATASET_PATH = Path("evaluation/test_dataset.jsonl")
RESULTS_CSV_PATH = Path("evaluation/ragas_results.csv")

OLLAMA_MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_test_dataset(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Create evaluation/test_dataset.jsonl first."
        )

    rows = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if "question" not in row or "reference" not in row:
                raise ValueError(
                    f"Line {line_number} must contain 'question' and 'reference'."
                )

            rows.append(row)

    return rows


def build_ragas_dataset(test_rows: List[Dict[str, str]]) -> Dataset:
    ragas_rows = []

    for i, row in enumerate(test_rows, start=1):
        question = row["question"]
        reference = row["reference"]

        print(f"\nRunning RAG example {i}/{len(test_rows)}")
        print(f"Question: {question}")

        answer, chunks = run_rag(question)

        contexts = [chunk["text"] for chunk in chunks]

        sources = []
        for chunk in chunks:
            relevance = chunk.get("relevance")
            relevance_text = "N/A" if relevance is None else f"{relevance:.1f}%"

            sources.append(
                f"{chunk['source']} page {chunk['page']} relevance {relevance_text}"
            )

        ragas_rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": reference,
                "retrieved_sources": " | ".join(sources),
            }
        )

    return Dataset.from_list(ragas_rows)


def main() -> None:
    test_rows = load_test_dataset(TEST_DATASET_PATH)
    dataset = build_ragas_dataset(test_rows)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
        )
    )

    print("\nRunning RAGAS evaluation...")

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    print("\nFinal RAGAS Scores:")
    print(result)

    results_df = result.to_pandas()

    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\nSaved detailed results to: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()