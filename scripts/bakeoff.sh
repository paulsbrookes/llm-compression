#!/usr/bin/env bash
# Codec bake-off: entropy floor vs Huffman / range / rANS on Gaussian fp16 data.
# Usage: scripts/bakeoff.sh [sigma=1.0] [mu=0.0] [N=1000000]
set -euo pipefail
cd "$(dirname "$0")/.."
make -s all
SIGMA=${1:-1.0}; MU=${2:-0.0}; N=${3:-1000000}
echo "=== source: N(mu=$MU, sigma=$SIGMA), $N samples ==="
echo "--- entropy floor + per-field breakdown ---"
bin/pmf     "$SIGMA" "$MU" data/pmf.csv | sed 's/^/  /'
for tool in huffman range rans; do
  echo "--- $tool ---"
  bin/$tool "$SIGMA" "$MU" "$N" | sed 's/^/  /'
done
