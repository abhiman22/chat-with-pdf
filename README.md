# Chat with PDF

A fully local, privacy-first RAG application. Chat with any PDF — or a whole folder of PDFs — using a local LLM. No cloud APIs, no data leaves your machine.

## How it works

1. **Ingest** — PDF(s) are parsed, split into chunks, embedded, and stored in a local vector database (ChromaDB)
2. **Retrieve** — your question is used to search for the most relevant chunks using hybrid search (vector + keyword)
3. **Generate** — a local LLM answers using only those chunks as context, with source citations

### Features
- **Multi-PDF collections** — ingest an entire folder of PDFs into one collection and chat across all of them at once
- **Hybrid search** — combines semantic vector search with BM25 keyword search via Reciprocal Rank Fusion for better retrieval
- **Query rewriting** — follow-up questions are automatically rewritten into standalone queries before retrieval
- **Conversational memory** — the last 3 Q&A turns are included in every prompt so follow-up questions work correctly
- **Source citations** — every answer shows which file and page the context came from

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
- Install all dependencies
- Pull the embedding and chat models via Ollama (~5 GB total download)

## Usage

### Single PDF

**Ingest a PDF**
```bash
./run.sh ingest path/to/yourfile.pdf
```

**Chat with it**
```bash
./run.sh chat yourfile
```

---

### Multiple PDFs as one collection

Place PDFs in a subfolder under `pdfs/`, then ingest the folder:

```bash
mkdir pdfs/my_collection
cp book1.pdf book2.pdf pdfs/my_collection/
./run.sh ingest pdfs/my_collection
./run.sh chat my_collection
```

All PDFs in the folder are embedded into a single collection. Citations show both the source file and page number (e.g. `book1 p.12`).

To add more PDFs to an existing folder collection, add the files to the folder and re-run ingest — existing chunks are safely overwritten via upsert.

---

### Other commands

**List all ingested collections**
```bash
./run.sh list
```

**Chat with hybrid search (default)**
```bash
./run.sh chat
```

**Chat with semantic-only search**
```bash
./run.sh chat --search semantic
```

**Chat with a specific collection**
```bash
./run.sh chat my_collection --search hybrid
```

Type `exit` to quit the chat session. Prefix your message with `//` to start a new topic (clears conversation history).

---

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
│   ├── ingest.py       # PDF / folder → ChromaDB pipeline
│   ├── main.py         # CLI entry point
│   ├── search.py       # Semantic search, hybrid search, query rewriting
│   └── requirements.txt
├── database/           # ChromaDB storage (auto-created, gitignored)
├── pdfs/               # Drop PDFs or subfolders here (gitignored)
│   └── my_collection/  # Example: folder ingested as one collection
├── models/             # Placeholder — Ollama manages actual model files
├── setup.sh            # One-command setup (Mac/Linux)
└── run.sh              # CLI runner
```
