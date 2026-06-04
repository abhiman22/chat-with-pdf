import ollama
from rank_bm25 import BM25Okapi

EMBED_MODEL = "nomic-embed-text"
RRF_K = 60  # RRF constant — higher values reduce the impact of top ranks


def rewrite_query(question: str, history: list[dict], chat_model: str) -> str:
    """
    Rewrite a follow-up question into a standalone search query using
    conversation history. Pass an empty history to skip rewriting.
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


def retrieval_confidence(chunks: list[dict]) -> tuple[float, str]:
    """
    Compute a confidence score from the top-3 chunks' scores.
    Returns (percentage 0-100, label: High | Medium | Low).
    Scores are already normalized to [0, 1] by the search functions.
    """
    top = [c["score"] for c in chunks[:3] if "score" in c]
    if not top:
        return 0.0, "Low"
    avg = sum(top) / len(top)
    pct = round(avg * 100, 1)
    label = "High" if avg >= 0.6 else "Medium" if avg >= 0.35 else "Low"
    return pct, label


def semantic_search(collection, query: str, top_k: int) -> list[dict]:
    """Pure vector search — fast, no BM25 index needed."""
    vector = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    results = collection.query(
        query_embeddings=[vector], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # Convert L2 distance to a 0-1 similarity score
        score = 1.0 / (1.0 + dist)
        chunks.append({"text": doc, "page": meta["page"], "source": meta["source"], "score": score})
    return chunks


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

    # Normalize RRF scores to [0, 1]: max possible = 2 / (RRF_K + 1)
    max_rrf = 2.0 / (RRF_K + 1)
    return [
        {"text": c["text"], "page": c["page"], "source": c["source"],
         "score": min(c["score"] / max_rrf, 1.0)}
        for c in merged
    ]
