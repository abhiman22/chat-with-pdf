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
    for chunk in chunks:
        vector = embed(chunk["text"])
        collection.add(
            ids=[f"{pdf_name}_chunk_{chunk['chunk_index']}"],
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"page": chunk["page"], "source": pdf_name}],
        )

    print(f"Done. Collection '{pdf_name}' stored in {DB_PATH}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
