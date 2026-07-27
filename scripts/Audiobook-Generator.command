#!/bin/zsh

# Resolve the repository through the symlink on the Desktop, then start the
# shared LAN web application from the repository root.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"

if [[ ! -d "$PROJECT_DIR/apps/desktop" ]]; then
  print -u2 "Audiobook Generator project was not found at: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"
exec npm run web
