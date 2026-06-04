#!/bin/bash
set -e

# Check setup has been run
if [ ! -d "code/venv" ]; then
    echo "Virtual environment not found. Run setup first:"
    echo "  ./setup.sh"
    exit 1
fi

# Check Ollama is running
if ! ollama list &>/dev/null; then
    echo "Ollama is not running. Start it first:"
    echo "  ollama serve"
    exit 1
fi

cd code
source venv/bin/activate

case "$1" in
    ingest)
        if [ -z "$2" ]; then
            echo "Usage: ./run.sh ingest <path-to-pdf>"
            exit 1
        fi
        python main.py ingest "$2"
        ;;
    list)
        python main.py list
        ;;
    chat|"")
        python main.py chat
        ;;
    *)
        echo "Usage: ./run.sh [ingest <pdf>|list|chat]"
        exit 1
        ;;
esac
