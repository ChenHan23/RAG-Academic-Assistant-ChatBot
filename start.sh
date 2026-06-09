#!/usr/bin/env bash

set -e

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama API..."
until curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
  sleep 1
done

echo "Pulling model: ${OLLAMA_MODEL_NAME:-llama3.2}"
ollama pull "${OLLAMA_MODEL_NAME:-llama3.2}"

echo "Starting Streamlit..."
python -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 7860 \
  --server.headless true