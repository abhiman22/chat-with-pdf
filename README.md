# Chat with PDF

A fully local, privacy-first RAG application. Chat with any PDF using a local LLM — no cloud APIs, no data leaves your machine.

## How it works

1. **Ingest** — PDF is parsed, split into chunks, embedded, and stored in a local vector database (ChromaDB)
2. **Retrieve** — your question is used to search for the most relevant chunks using hybrid search (vector + keyword)
3. **Generate** — a local LLM answers using only those chunks as context, with source page citations

### Features
- **Hybrid search** — combines semantic vector search with BM25 keyword search via Reciprocal Rank Fusion for better retrieval
- **Query rewriting** — follow-up questions are automatically rewritten into standalone queries before retrieval, so context from prior turns is used during search, not just during generation
- **Conversational memory** — the last 3 Q&A turns are included in every prompt so follow-up questions work correctly
- **Source citations** — every answer shows which pages the context came from

## Stack

| Component | Library |
|-----------|---------|
| Vector DB | ChromaDB |
| LLM runtime | Ollama |
| PDF parsing | PyMuPDF |
| Keyword search | rank_bm25 |
| Embedding model | nomic-embed-text (~274 MB) |
| Chat model | llama3.1:8b (~4.7 GB) |

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

## Setup

```bash
git clone https://github.com/abhiman22/chat-with-pdf
cd chat-with-pdf
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` will:
- Create a Python virtual environment
- Install all dependencies (including `rank_bm25`)
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

**Start chatting (hybrid search, default)**
```bash
./run.sh chat
```

**Start chatting with semantic-only search**
```bash
./run.sh chat --search semantic
```

**Chat with a specific PDF directly**
```bash
./run.sh chat MyBook --search hybrid
```

Type `exit` to quit the chat session.

## Search modes

| Mode | How it works | When to use |
|------|-------------|-------------|
| `hybrid` (default) | Vector search + BM25 merged via RRF | Best overall — catches both semantic and exact matches |
| `semantic` | Vector search only, no BM25 index | Faster startup on large PDFs |

## Project structure

```
chat_with_pdf/
├── code/
│   ├── chat.py         # Terminal chat loop
│   ├── ingest.py       # PDF → ChromaDB pipeline
│   ├── main.py         # CLI entry point
│   ├── search.py       # Semantic search, hybrid search, query rewriting
│   └── requirements.txt
├── database/           # ChromaDB storage (auto-created, gitignored)
├── pdfs/               # Drop PDFs here (gitignored)
├── models/             # Placeholder — Ollama manages actual model files
├── setup.sh            # One-command setup (Mac/Linux)
└── run.sh              # CLI runner
```
