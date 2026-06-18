#!/usr/bin/env bash
# Analyze LLM weight distributions -> docs/weights.json (for the dashboard).
# Usage: scripts/analyze_weights.sh [model.safetensors]
set -euo pipefail
cd "$(dirname "$0")/.."
make -s bin/weights
MODEL=${1:-data/qwen2.5-0.5b.safetensors}
if [ ! -f "$MODEL" ]; then
  echo "model not found: $MODEL  (run scripts/download_model.sh)" >&2
  exit 1
fi
bin/weights "$MODEL" docs/weights.json
echo "wrote docs/weights.json — view with scripts/serve.sh"
