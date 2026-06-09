from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from src.pdf_loader import load_pdf_pages
from src.chunker import chunk_pages
from src.vector_store import add_chunks_to_chroma, get_collection
from query import run_rag

import chromadb

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


st.set_page_config(
    page_title="Academic RAG Assistant",
    page_icon="📚",
    layout="wide",
)


def reset_chroma_db() -> None:
    """
    Safely resets the Chroma collection without deleting database files
    while Streamlit is running.
    """
    client = chromadb.PersistentClient(path="chroma_db")

    try:
        client.delete_collection(name="academic_papers")
    except Exception:
        pass


def save_uploaded_pdfs(uploaded_files) -> List[Path]:
    """
    Saves uploaded PDFs into the local data/ folder.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name

        with file_path.open("wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_paths.append(file_path)

    return saved_paths


def ingest_pdfs(pdf_paths: List[Path], chunk_size: int, chunk_overlap: int) -> int:
    """
    Runs the full PDF ingestion pipeline:
    PDF -> pages -> chunks -> embeddings -> ChromaDB
    """
    all_chunks = []

    for pdf_path in pdf_paths:
        pages = load_pdf_pages(str(pdf_path))

        chunks = chunk_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        all_chunks.extend(chunks)

    if all_chunks:
        add_chunks_to_chroma(all_chunks)

    return len(all_chunks)


def get_loaded_documents() -> List[str]:
    """
    Reads ChromaDB metadata and returns unique source document names.
    """
    try:
        collection = get_collection()

        if collection.count() == 0:
            return []

        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        documents = sorted(
            {
                metadata.get("source", "Unknown source")
                for metadata in metadatas
                if metadata
            }
        )

        return documents

    except Exception:
        return []


def initialize_session_state() -> None:
    """
    Stores conversation history inside the Streamlit session.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "indexed_documents" not in st.session_state:
        st.session_state.indexed_documents = []


def show_sidebar() -> None:
    """
    Builds the sidebar:
    - PDF uploader
    - chunk size slider
    - process button
    - loaded document list
    """
    st.sidebar.title("📄 Document Settings")

    uploaded_files = st.sidebar.file_uploader(
        "Upload academic PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.sidebar.slider(
        "Chunk size",
        min_value=200,
        max_value=1500,
        value=500,
        step=100,
    )

    chunk_overlap = st.sidebar.slider(
        "Chunk overlap",
        min_value=0,
        max_value=400,
        value=100,
        step=50,
    )

    process_clicked = st.sidebar.button("Process PDFs", type="primary")

    if process_clicked:
        if not uploaded_files:
            st.sidebar.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Processing PDFs and building vector database..."):
                reset_chroma_db()
                saved_paths = save_uploaded_pdfs(uploaded_files)
                chunk_count = ingest_pdfs(
                    saved_paths,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                st.session_state.indexed_documents = get_loaded_documents()

            st.sidebar.success(f"Indexed {chunk_count} chunks.")

    st.sidebar.divider()

    st.sidebar.subheader("Loaded Documents")

    documents = st.session_state.get("indexed_documents", [])

    if documents:
        for document in documents:
            st.sidebar.write(f"• {document}")
    else:
        st.sidebar.caption("No documents indexed yet.")

    st.sidebar.divider()

    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def format_preview(text: str, max_chars: int = 500) -> str:
    """
    Creates a short preview of retrieved source text.
    """
    clean_text = " ".join(text.split())

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars] + "..."


def show_chat_history() -> None:
    """
    Displays all previous user and assistant messages.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                show_sources(message["sources"])


def show_sources(chunks: List[Dict[str, Any]]) -> None:
    """
    Displays retrieved chunks beneath an answer.
    """
    with st.expander("View Sources"):
        for chunk in chunks:
            source = chunk.get("source", "Unknown source")
            page = chunk.get("page", "Unknown page")
            relevance = chunk.get("relevance")

            if relevance is None:
                relevance_text = "N/A"
            else:
                relevance_text = f"{relevance:.1f}%"

            preview = format_preview(chunk.get("text", ""))

            st.markdown(
                f"""
**Source:** `{source}`  
**Page:** {page}  
**Relevance:** {relevance_text}

> {preview}
"""
            )

            st.divider()


def main() -> None:
    initialize_session_state()

    st.title("📚 Academic RAG Assistant")
    st.caption("Ask questions about your uploaded PDFs. Answers are grounded in retrieved document chunks.")

    show_sidebar()

    show_chat_history()

    question = st.chat_input("Ask a question about your PDFs...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving sources and generating answer..."):
                try:
                    answer, chunks = run_rag(question)

                    st.markdown(answer)
                    show_sources(chunks)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": chunks,
                        }
                    )

                except Exception as exc:
                    error_message = f"Error: {exc}"
                    st.error(error_message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )


if __name__ == "__main__":
    main()