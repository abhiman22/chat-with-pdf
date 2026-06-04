import os
import ollama
import chromadb
from search import build_bm25_index, hybrid_search, semantic_search, rewrite_query

CHAT_MODEL = "llama3.1:8b"
TOP_K = 10
MAX_HISTORY_TURNS = 3
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database")


def list_collections(client: chromadb.PersistentClient) -> list[str]:
    return [c.name for c in client.list_collections()]


def build_prompt(chunks: list[dict], question: str, history: list[dict] = None) -> str:
    context = "\n\n---\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in chunks
    )

    history_text = ""
    if history:
        recent = history[-(MAX_HISTORY_TURNS * 2):]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_text = "\n\nConversation so far:\n" + "\n".join(lines)

    return (
        f"Use only the context below to answer the question. "
        f"If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}"
        f"{history_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def chat(collection_name: str, search_mode: str = "hybrid"):
    client = chromadb.PersistentClient(path=DB_PATH)
    available = list_collections(client)

    if not available:
        print("No ingested PDFs found. Run ingest.py first.")
        return

    if collection_name not in available:
        print(f"Collection '{collection_name}' not found.")
        print(f"Available: {', '.join(available)}")
        return

    collection = client.get_collection(name=collection_name)
    print(f"Chatting with '{collection_name}' [{search_mode} search]. Type 'exit' to quit.\n")

    bm25, all_docs, all_metas = None, None, None
    if search_mode == "hybrid":
        print("Building BM25 index...", end=" ", flush=True)
        bm25, all_docs, all_metas = build_bm25_index(collection)
        print("ready.\n")

    history = []  # list of {"role": "user"|"assistant", "content": str}

    while True:
        question = input("You: ").strip()
        if not question or question.lower() == "exit":
            break

        search_query = rewrite_query(question, history, CHAT_MODEL)
        if search_query != question:
            print(f"  [rewritten: {search_query}]")

        if search_mode == "hybrid":
            chunks = hybrid_search(collection, bm25, all_docs, all_metas, search_query, TOP_K)
        else:
            chunks = semantic_search(collection, search_query, TOP_K)
        prompt = build_prompt(chunks, question, history=history)

        print("Assistant: ", end="", flush=True)
        full_response = ""
        for part in ollama.generate(model=CHAT_MODEL, prompt=prompt, stream=True):
            print(part["response"], end="", flush=True)
            full_response += part["response"]
        print()

        sources = {f"p.{c['page']}" for c in chunks}
        print(f"  [sources: {', '.join(sorted(sources))}]\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    import sys
    client = chromadb.PersistentClient(path=DB_PATH)
    available = list_collections(client)

    if len(sys.argv) < 2:
        if not available:
            print("No ingested PDFs found. Run: python ingest.py ../pdfs/yourfile.pdf")
        else:
            print(f"Available collections: {', '.join(available)}")
            print(f"Usage: python chat.py <collection_name>")
    else:
        chat(sys.argv[1])
