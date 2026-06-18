#!/usr/bin/env bash
# Serve docs/ so the HTML dashboards can fetch their JSON (stdlib only, no pip).
set -euo pipefail
cd "$(dirname "$0")/../docs"
PORT=${1:-8000}
echo "serving docs/ at http://localhost:$PORT  (index.html, entropy_curve.html, weights_dashboard.html)"
exec python3 -m http.server "$PORT"
