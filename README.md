# Chat with PDF

A fully local, privacy-first RAG application. Upload any PDF and chat with it using a local LLM — no cloud APIs, no data leaves your machine.

## How it works

1. **Ingest** — PDF is parsed, split into chunks, embedded, and stored in a local vector database (ChromaDB)
2. **Retrieve** — your question is embedded and the most relevant chunks are fetched
3. **Generate** — a local LLM answers using only those chunks as context, with source page citations

## Stack

| Component | Library |
|-----------|---------|
| Vector DB | ChromaDB |
| LLM runtime | Ollama |
| PDF parsing | PyMuPDF |
| Embedding model | nomic-embed-text (~274 MB) |
| Chat model | llama3.1:8b (~4.7 GB) |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
git clone https://github.com/yourname/chat-with-pdf
cd chat-with-pdf
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` will:
- Create a Python virtual environment
- Install all dependencies
- Pull the embedding and chat models via Ollama (~5 GB total download)

## Usage

**Ingest a PDF**
```bash
./run.sh ingest path/to/yourfile.pdf
```

**List ingested PDFs**
```bash
./run.sh list
```

**Start chatting**
```bash
./run.sh chat
```

The chat supports conversational memory — follow-up questions like *"tell me more about that"* work correctly. Type `exit` to quit.

## Project structure

```
chat_with_pdf/
├── code/
│   ├── chat.py         # Terminal chat loop
│   ├── ingest.py       # PDF → ChromaDB pipeline
│   ├── main.py         # CLI entry point
│   └── requirements.txt
├── database/           # ChromaDB storage (auto-created, gitignored)
├── pdfs/               # Drop PDFs here (gitignored)
├── models/             # Placeholder — Ollama manages actual model files
├── setup.sh            # One-command setup (Mac/Linux)
└── run.sh              # CLI runner
