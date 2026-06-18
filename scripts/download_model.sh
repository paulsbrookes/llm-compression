#!/usr/bin/env bash
# Download the Qwen2.5-0.5B weights (~988 MB) into data/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=data/qwen2.5-0.5b.safetensors
URL=https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors
if [ -f "$DEST" ]; then
  echo "already present: $DEST ($(du -h "$DEST" | cut -f1))"
else
  echo "downloading -> $DEST"
  curl -L -# -o "$DEST" "$URL"
  echo "done: $(du -h "$DEST" | cut -f1)"
fi
