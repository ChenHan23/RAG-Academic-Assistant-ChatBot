FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OLLAMA_HOST=127.0.0.1:11434
ENV CHROMA_PATH=/tmp/chroma_db

EXPOSE 8501

RUN chmod +x start.sh

CMD ["./start.sh"]