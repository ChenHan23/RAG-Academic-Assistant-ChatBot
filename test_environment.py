"""Quick environment check for Mac/VS Code setup."""

import importlib.util
import subprocess
import sys

REQUIRED_MODULES = [
    "fitz",
    "chromadb",
    "sentence_transformers",
    "langchain_text_splitters",
    "ollama",
    "streamlit",
]


def check_python() -> None:
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    if version < (3, 10):
        print("WARNING: Python 3.10+ is recommended. Python 3.11 is ideal.")


def check_modules() -> None:
    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        status = "OK" if found else "MISSING"
        print(f"{module}: {status}")


def check_ollama() -> None:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        print("Ollama CLI: MISSING")
        print("Install from https://ollama.com/download, then run: ollama pull llama3.2")
        return
    except subprocess.TimeoutExpired:
        print("Ollama CLI: TIMEOUT")
        return

    print("Ollama CLI: OK")
    if "llama3.2" not in result.stdout:
        print("WARNING: llama3.2 not found. Run: ollama pull llama3.2")


if __name__ == "__main__":
    check_python()
    check_modules()
    check_ollama()
