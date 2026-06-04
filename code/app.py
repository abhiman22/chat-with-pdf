import os
import re
import time
import threading
import streamlit as st
import ollama
import chromadb
from ingest import ingest
from search import build_bm25_index, hybrid_search, rewrite_query

# Module-level dict for background threads to write status into.
# Background threads cannot access st.session_state (no ScriptRunContext).
_ingest_status: dict[str, str] = {}

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"
TOP_K = 10
MAX_HISTORY_TURNS = 3  # number of past Q&A pairs included in the prompt
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database")
PDFS_PATH = os.path.join(os.path.dirname(__file__), "..", "pdfs")

os.makedirs(PDFS_PATH, exist_ok=True)

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with PDF")


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_PATH)


def list_collections(client):
    return [c.name for c in client.list_collections()]


@st.cache_resource
def get_bm25_index(collection_name: str):
    """Build and cache the BM25 index for a collection. Rebuilt if collection changes."""
    col = get_chroma_client().get_collection(name=collection_name)
    return build_bm25_index(col)


def build_prompt(chunks: list[dict], question: str, history: list[dict] = None) -> str:
    context = "\n\n---\n\n".join(f"[Page {c['page']}] {c['text']}" for c in chunks)

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


def sanitize_collection_name(filename: str) -> str:
    """Strip non-ASCII and invalid chars so ChromaDB accepts the name."""
    name = filename.encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    return (name or "pdf")[:512].ljust(3, "_")


def run_ingest_thread(pdf_path: str, collection_name: str):
    try:
        ingest(pdf_path, collection_name_override=collection_name)
        _ingest_status[collection_name] = "done"
    except Exception as e:
        _ingest_status[collection_name] = f"error: {e}"


# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None
if "ingest_jobs" not in st.session_state:
    st.session_state.ingest_jobs = {}  # {collection_name: "running" | "done" | "error: ..."}
if "saved_files" not in st.session_state:
    st.session_state.saved_files = {}  # {collection_name: pdf_path}
if "pending_toasts" not in st.session_state:
    st.session_state.pending_toasts = []

# --- Sync background thread results into session state ---
for _name, _status in list(_ingest_status.items()):
    st.session_state.ingest_jobs[_name] = _status
    if _status == "done":
        st.session_state.pending_toasts.append(_name)
    del _ingest_status[_name]

# --- Show completion toasts (must run before sidebar renders) ---
for name in list(st.session_state.pending_toasts):
    st.toast(f"'{name}' is ready to chat!", icon="✅")
    st.balloons()
    st.session_state.pending_toasts.remove(name)

# --- Sidebar ---
client = get_chroma_client()

with st.sidebar:
    st.header("Library")

    # --- Upload section ---
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        collection_name = sanitize_collection_name(os.path.splitext(uploaded_file.name)[0])
        existing = list_collections(client)

        if collection_name in existing:
            st.info(f"'{collection_name}' is already ingested.")

        elif st.session_state.ingest_jobs.get(collection_name) == "running":
            st.info(f"⏳ Ingesting **{collection_name}**...")

        elif collection_name not in st.session_state.saved_files:
            # Step 1: save file to /pdfs
            pdf_path = os.path.join(PDFS_PATH, uploaded_file.name)
            with st.spinner(f"Saving '{uploaded_file.name}' to pdfs/..."):
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.session_state.saved_files[collection_name] = pdf_path
            st.rerun()

        else:
            # Step 2: file is saved, offer to ingest
            st.success(f"Saved to pdfs/{uploaded_file.name}")
            if st.button("Ingest PDF", type="primary"):
                pdf_path = st.session_state.saved_files[collection_name]
                st.session_state.ingest_jobs[collection_name] = "running"
                thread = threading.Thread(
                    target=run_ingest_thread,
                    args=(pdf_path, collection_name),
                    daemon=True,
                )
                thread.start()
                st.rerun()

    # --- Running jobs status ---
    running_jobs = [n for n, s in st.session_state.ingest_jobs.items() if s == "running"]
    if running_jobs:
        st.divider()
        for name in running_jobs:
            st.info(f"⏳ Ingesting **{name}**...\n\nThis may take a few minutes for large PDFs.")

    # --- Error jobs ---
    errored = {n: s for n, s in list(st.session_state.ingest_jobs.items()) if s.startswith("error")}
    for name, status in errored.items():
        st.error(f"Failed to ingest '{name}':\n{status.removeprefix('error: ')}")
        del st.session_state.ingest_jobs[name]

    # --- Clean up done jobs ---
    done_jobs = [n for n, s in list(st.session_state.ingest_jobs.items()) if s == "done"]
    for name in done_jobs:
        del st.session_state.ingest_jobs[name]
        st.session_state.saved_files.pop(name, None)

    st.divider()

    # --- PDF selector ---
    collections = list_collections(client)

    if not collections:
        if running_jobs:
            st.warning("Waiting for ingestion to complete...")
        else:
            st.warning("No PDFs ingested yet. Upload one above.")
        st.stop()

    selected = st.selectbox("Choose a PDF", collections)

    st.divider()
    st.caption(f"Model: `{CHAT_MODEL}`")
    st.caption(f"Embed: `{EMBED_MODEL}`")
    st.caption(f"Top-K: `{TOP_K}`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Auto-refresh while ingestion is running ---
if running_jobs:
    time.sleep(2)
    st.rerun()

# --- Reset chat on PDF switch ---
if st.session_state.active_pdf != selected:
    st.session_state.messages = []
    st.session_state.active_pdf = selected

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.caption(f"Sources: {msg['sources']}")

# --- Chat input ---
if question := st.chat_input(f"Ask about {selected}..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    collection = client.get_collection(name=selected)
    bm25, all_docs, all_metas = get_bm25_index(selected)
    history = st.session_state.messages[:-1]  # everything before the current question
    search_query = rewrite_query(question, history, CHAT_MODEL)
    chunks = hybrid_search(collection, bm25, all_docs, all_metas, search_query, TOP_K)
    prompt = build_prompt(chunks, question, history=history)
    sources = ", ".join(sorted({f"p.{c['page']}" for c in chunks}))

    with st.chat_message("assistant"):
        response_box = st.empty()
        full_response = ""
        for part in ollama.generate(model=CHAT_MODEL, prompt=prompt, stream=True):
            full_response += part["response"]
            response_box.markdown(full_response + "▌")
        response_box.markdown(full_response)
        st.caption(f"Sources: {sources}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
