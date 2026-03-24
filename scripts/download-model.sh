#!/usr/bin/env bash
# scripts/download-model.sh
#
# Download GGUF models from HuggingFace to /opt/llama/models/
#
# Usage: ./scripts/download-model.sh <huggingface-repo> <filename>
# Example: ./scripts/download-model.sh TheBloke/some-model-GGUF model-q4_k_m.gguf
#
# Prerequisites: pip install huggingface-hub

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/opt/llama/models}"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <huggingface-repo> <filename>"
    echo "Example: $0 TheBloke/some-model-GGUF model-q4_k_m.gguf"
    echo "Downloads to: $MODELS_DIR"
    exit 1
fi

REPO="$1"
FILENAME="$2"

if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Install: pip install huggingface-hub"
    exit 1
fi

if [[ ! -w "$MODELS_DIR" ]]; then
    echo "ERROR: Cannot write to $MODELS_DIR"
    echo "Ensure you're in the 'llama' group, or use sudo."
    exit 1
fi

echo "Downloading: $REPO/$FILENAME → $MODELS_DIR/"
huggingface-cli download "$REPO" "$FILENAME" --local-dir "$MODELS_DIR"
echo "Done. Model available via GET /v1/models"
