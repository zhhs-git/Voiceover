#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PID=""
VITE_PID=""

cleanup() {
  [[ -z "$API_PID" ]] || kill "$API_PID" 2>/dev/null || true
  [[ -z "$VITE_PID" ]] || kill "$VITE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
"$PROJECT_DIR/workers/python/.venv/bin/python" -m audiobook_worker.web_server \
  --host "${AUDIOBOOK_WEB_HOST:-0.0.0.0}" \
  --port "${AUDIOBOOK_WEB_PORT:-8000}" \
  >"${TMPDIR:-/tmp}/audiobook-generator-web.log" 2>&1 &
API_PID=$!

npm run dev --workspace @audiobook-generator/desktop -- --host 0.0.0.0 &
VITE_PID=$!

wait "$VITE_PID"
