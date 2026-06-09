#!/usr/bin/env bash
set -euo pipefail

echo "Creating Python virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Python setup complete."
echo "Next install Ollama from https://ollama.com/download if needed, then run:"
echo "  ollama pull llama3.2"
echo ""
echo "Then add PDFs to data/ and run:"
echo "  python ingest.py --reset"
echo "  python ask.py"
echo ""
echo "For the Streamlit app:"
echo "  streamlit run app.py"
