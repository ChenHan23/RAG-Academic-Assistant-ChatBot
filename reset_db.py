"""Reset the local ChromaDB collection."""

from src.vector_store import reset_collection

if __name__ == "__main__":
    reset_collection()
    print("Reset ChromaDB collection.")
