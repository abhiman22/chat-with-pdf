#!/bin/bash
set -e

echo "=== Chat with PDF — Setup ==="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.11+ from https://python.org"
    exit 1
fi

# Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "Error: Ollama not found. Install it from https://ollama.com/download, then re-run this script."
    exit 1
fi

# Create virtualenv
echo ""
echo "1/4  Creating virtual environment..."
cd code
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "2/4  Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Pull Ollama models
echo "3/4  Pulling embedding model (nomic-embed-text, ~274 MB)..."
ollama pull nomic-embed-text

echo "4/4  Pulling chat model (llama3.1:8b, ~4.7 GB)..."
ollama pull llama3.1:8b

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the app, run:"
echo "  ./run.sh"
