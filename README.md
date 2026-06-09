---
title: Academic RAG Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Local Academic RAG Assistant — Phase 1

This is a fully local Retrieval-Augmented Generation starter project for academic PDFs.
It lets you place PDFs in `data/`, ingest them into a local ChromaDB vector database,
and ask questions through either a terminal script or Streamlit app.

The project uses:

- **LLM:** Ollama with `llama3.2`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector database:** ChromaDB, persisted locally in `chroma_db/`
- **PDF parsing:** PyMuPDF
- **Chunking:** LangChain text splitters
- **Frontend:** Streamlit

No API keys are required.

---

## 1. Open in VS Code on your Mac

Unzip the project, then open the folder in VS Code.

In VS Code, open the built-in terminal:

```bash
Terminal > New Terminal
```

Make sure the terminal is inside the project folder.

---

## 2. Install Ollama

Install Ollama for macOS from the Ollama website.

Then in Terminal run:

```bash
ollama pull llama3.2
```

Test it:

```bash
ollama run llama3.2
```

Type `/bye` to exit the Ollama chat.

---

## 3. Set up Python

Python 3.11 is recommended.

Fast setup:

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

Manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check your setup:

```bash
python test_environment.py
```

---

## 4. Add PDFs

Put your academic PDFs into the `data/` folder.

Example:

```text
data/my-paper.pdf
```

---

## 5. Ingest PDFs into ChromaDB

```bash
source .venv/bin/activate
python ingest.py --reset
```

Use `--reset` when you want to rebuild the database from scratch.

---

## 6. Ask questions in the terminal

```bash
python ask.py
```

Example questions:

```text
What is the main argument of this paper?
What does the author say about retrieval augmented generation?
Which page discusses evaluation?
```

The assistant will return an answer, retrieved sources, and page citations.

---

## 7. Run the Streamlit app

```bash
streamlit run app.py
```

The browser app lets you:

1. Upload PDFs.
2. Ingest them.
3. Ask questions.
4. View citations and retrieved chunks.

---

## Project structure

```text
rag-academic-assistant-mac/
├── data/                  # Add PDFs here
├── chroma_db/             # Local vector database after ingestion
├── src/
│   ├── config.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── vector_store.py
│   ├── citations.py
│   └── rag_chain.py
├── ingest.py              # Build/update vector database
├── ask.py                 # Terminal Q&A app
├── app.py                 # Streamlit app
├── reset_db.py            # Reset ChromaDB collection
├── test_environment.py    # Setup checker
├── setup_mac.sh           # Mac setup script
├── requirements.txt
└── README.md
```

---

## Common Mac fixes

### `ollama` command not found

Install Ollama, then restart your terminal.

### `llama3.2` not found

Run:

```bash
ollama pull llama3.2
```

### VS Code is not using `.venv`

Press `Cmd + Shift + P`, search for **Python: Select Interpreter**, then choose:

```text
.venv/bin/python
```

### Dependency install problems

Make sure you are using Python 3.10 or 3.11:

```bash
python3 --version
```

Then rebuild the virtual environment:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Next improvements after Phase 1

- Add better citation validation.
- Add hybrid search or reranking.
- Add conversation memory.
- Add RAGAS evaluation test sets.
- Add OCR support for scanned PDFs.
