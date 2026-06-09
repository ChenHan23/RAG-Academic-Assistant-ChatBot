#!/usr/bin/env bash

set -e

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

echo "Pulling LLaMA 3.2..."
ollama pull llama3.2

echo "Starting Streamlit..."
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT:-8501} \
  --server.headless true