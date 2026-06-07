import sys
import os
import fitz  # pymupdf
import ollama
import chromadb

CHUNK_SIZE = 1200     # target characters per chunk
CHUNK_OVERLAP = 200   # overlap between consecutive chunks
EMBED_MODEL = "nomic-embed-text"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database")


def extract_text(pdf_path: str) -> list[dict]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """Split text on paragraph boundaries, falling back to character limit."""
    chunks = []
    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    chunks.append({"text": current, "page": page["page"], "chunk_index": len(chunks)})
                # paragraph itself exceeds CHUNK_SIZE — hard split with overlap
                if len(para) > CHUNK_SIZE:
                    start = 0
                    while start < len(para):
                        chunks.append({
                            "text": para[start:start + CHUNK_SIZE],
                            "page": page["page"],
                            "chunk_index": len(chunks),
                        })
                        start += CHUNK_SIZE - CHUNK_OVERLAP
                    current = para[-(CHUNK_OVERLAP):]
                else:
                    current = para
        if current:
            chunks.append({"text": current, "page": page["page"], "chunk_index": len(chunks)})
    return chunks


def embed(text: str) -> list[float]:
    """Get embedding vector from Ollama."""
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response["embeddings"][0]


def _store_chunks(collection, chunks: list[dict], id_prefix: str, source_name: str):
    """Embed and upsert chunks into a ChromaDB collection."""
    for chunk in chunks:
        vector = embed(chunk["text"])
        collection.upsert(
            ids=[f"{id_prefix}_chunk_{chunk['chunk_index']}"],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"], "source": source_name}],
        )


def ingest(pdf_path: str, collection_name_override: str = None):
    pdf_name = collection_name_override or os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")
    print(f"Reading: {pdf_path}")

    pages = extract_text(pdf_path)
    print(f"  {len(pages)} pages extracted")

    chunks = chunk_text(pages)
    print(f"  {len(chunks)} chunks created")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=pdf_name)

    print(f"  Embedding and storing chunks...")
    _store_chunks(collection, chunks, id_prefix=pdf_name, source_name=pdf_name)

    print(f"Done. Collection '{pdf_name}' stored in {DB_PATH}")


def ingest_folder(folder_path: str, collection_name_override: str = None):
    """Ingest all PDFs in a folder into a single ChromaDB collection."""
    folder_name = collection_name_override or os.path.basename(folder_path.rstrip("/\\")).replace(" ", "_")
    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))

    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    print(f"Ingesting folder '{folder_name}' ({len(pdf_files)} PDFs)...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=folder_name)

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        source_name = os.path.splitext(pdf_file)[0].replace(" ", "_")
        print(f"\n  [{source_name}]")

        pages = extract_text(pdf_path)
        print(f"    {len(pages)} pages extracted")

        chunks = chunk_text(pages)
        print(f"    {len(chunks)} chunks created")

        print(f"    Embedding and storing chunks...")
        _store_chunks(collection, chunks, id_prefix=f"{folder_name}_{source_name}", source_name=source_name)

    print(f"\nDone. Collection '{folder_name}' stored in {DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf_or_folder>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        ingest_folder(target)
    else:
        ingest(target)
