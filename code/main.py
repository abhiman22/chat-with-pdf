import sys
import os
import chromadb

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database")


def list_collections():
    client = chromadb.PersistentClient(path=DB_PATH)
    names = [c.name for c in client.list_collections()]
    return names


def cmd_ingest(args):
    if not args:
        print("Error: provide a PDF path or folder.  e.g.  python main.py ingest ../pdfs/book.pdf")
        sys.exit(1)
    target = args[0]
    if os.path.isdir(target):
        from ingest import ingest_folder
        ingest_folder(target)
    else:
        from ingest import ingest
        ingest(target)


def cmd_chat(args):
    from chat import chat, list_collections as _list
    client = chromadb.PersistentClient(path=DB_PATH)
    available = _list(client)

    if not available:
        print("No ingested PDFs found. Run: python main.py ingest <path_to_pdf>")
        sys.exit(1)

    # Parse --search flag
    search_mode = "hybrid"
    if "--search" in args:
        idx = args.index("--search")
        if idx + 1 < len(args) and args[idx + 1] in ("hybrid", "semantic"):
            search_mode = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --search requires 'hybrid' or 'semantic'")
            sys.exit(1)

    if args:
        collection_name = args[0]
    elif len(available) == 1:
        collection_name = available[0]
        print(f"Auto-selected collection: '{collection_name}'")
    else:
        print("Available PDFs:")
        for i, name in enumerate(available, 1):
            print(f"  {i}. {name}")
        choice = input("Select a number: ").strip()
        try:
            collection_name = available[int(choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice.")
            sys.exit(1)

    chat(collection_name, search_mode=search_mode)


def cmd_list():
    names = list_collections()
    if not names:
        print("No ingested PDFs found.")
    else:
        print("Ingested PDFs:")
        for name in names:
            print(f"  - {name}")


USAGE = """
Chat with PDF — local LLM powered

Commands:
  python main.py ingest <path_to_pdf>                         Parse, embed, and store a single PDF
  python main.py ingest <path_to_folder>                      Ingest all PDFs in a folder as one collection
  python main.py chat [collection] [--search hybrid|semantic]  Start a chat session
  python main.py list                                         Show all ingested collections

Search modes (default: hybrid):
  hybrid    — combines vector + BM25 keyword search via RRF (better recall)
  semantic  — vector search only, faster startup

Folder ingestion example:
  mkdir ../pdfs/my_books
  cp book1.pdf book2.pdf ../pdfs/my_books/
  python main.py ingest ../pdfs/my_books
  python main.py chat my_books
"""

COMMANDS = {
    "ingest": lambda args: cmd_ingest(args),
    "chat":   lambda args: cmd_chat(args),
    "list":   lambda _:    cmd_list(),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(USAGE)
        sys.exit(0)

    command = sys.argv[1]
    rest = sys.argv[2:]
    COMMANDS[command](rest)
