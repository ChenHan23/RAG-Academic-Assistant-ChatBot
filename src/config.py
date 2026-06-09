"""Central project configuration.

Everything is local by default:
- PDFs live in data/
- vectors live in chroma_db/
- Ollama serves the LLM locally

For deployment, CHROMA_PATH and DATA_PATH can override the local folders.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.getenv("DATA_PATH", PROJECT_ROOT / "data")
)

CHROMA_DIR = Path(
    os.getenv("CHROMA_PATH", PROJECT_ROOT / "chroma_db")
)

COLLECTION_NAME = "academic_papers"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5