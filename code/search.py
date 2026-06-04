import ollama
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
RRF_K = 60  # RRF constant — higher values reduce the impact of top ranks


def rewrite_query(question: str, history: list[dict], chat_model: str) -> str:
    """
    If there is conversation history, rewrite the follow-up question into a
    standalone search query so retrieval is not misled by pronouns or references.
    Returns the original question unchanged when there is no history.
    """
    if not history:
        return question

    lines = []
    for msg in history[-6:]:  # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    history_text = "\n".join(lines)

    prompt = (
        "Rewrite the follow-up question as a fully self-contained search query "
        "using the conversation history for context. "
        "Return only the rewritten query with no explanation.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone query:"
    )

    result = ollama.generate(model=chat_model, prompt=prompt, stream=False)
    rewritten = result["response"].strip().strip('"')
    return rewritten if rewritten else question


def semantic_search(collection, query: str, top_k: int) -> list[dict]:
    """Pure vector search — fast, no BM25 index needed."""
    vector = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    results = collection.query(query_embeddings=[vector], n_results=top_k)
    return [
        {"text": doc, "page": meta["page"], "source": meta["source"]}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


def build_bm25_index(collection) -> tuple:
    """Load all chunks from a ChromaDB collection and build a BM25 index."""
    result = collection.get(include=["documents", "metadatas"])
    docs = result["documents"]
    metas = result["metadatas"]
    tokenized = [doc.lower().split() for doc in docs]
    return BM25Okapi(tokenized), docs, metas


def hybrid_search(collection, bm25, all_docs: list, all_metas: list,
                  query: str, top_k: int) -> list[dict]:
    """
    Combine vector search (semantic) and BM25 (keyword) using
    Reciprocal Rank Fusion. Each method retrieves top_k candidates;
    RRF merges their rankings into a single score.
    """
    # --- Vector search ---
    vector = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    vec_results = collection.query(query_embeddings=[vector], n_results=top_k)
    vec_docs = vec_results["documents"][0]
    vec_metas = vec_results["metadatas"][0]

    # --- BM25 search ---
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = sorted(range(len(bm25_scores)),
                          key=lambda i: bm25_scores[i], reverse=True)[:top_k]

    # --- Reciprocal Rank Fusion ---
    rrf: dict[str, dict] = {}

    for rank, (doc, meta) in enumerate(zip(vec_docs, vec_metas)):
        key = doc[:120]
        if key not in rrf:
            rrf[key] = {"score": 0.0, "text": doc,
                        "page": meta["page"], "source": meta["source"]}
        rrf[key]["score"] += 1.0 / (RRF_K + rank + 1)

    for rank, idx in enumerate(top_bm25_idx):
        doc, meta = all_docs[idx], all_metas[idx]
        key = doc[:120]
        if key not in rrf:
            rrf[key] = {"score": 0.0, "text": doc,
                        "page": meta["page"], "source": meta["source"]}
        rrf[key]["score"] += 1.0 / (RRF_K + rank + 1)

    merged = sorted(rrf.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [{"text": c["text"], "page": c["page"], "source": c["source"]}
            for c in merged]
