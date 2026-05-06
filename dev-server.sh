#!/usr/bin/env bash
# dev-server.sh — Start OpenFrontIO dev server in a tmux session.
# Usage: ./dev-server.sh [session-name]
#   session-name defaults to "openfront"
# Ports: Client → http://localhost:5173  |  Server → http://localhost:3000

set -euo pipefail

SESSION="${1:-openfront}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Prereqs ────────────────────────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
  echo "tmux not found. Installing via Homebrew..."
  if ! command -v brew &>/dev/null; then
    echo "Homebrew is required. Install it from https://brew.sh then re-run."
    exit 1
  fi
  brew install tmux
fi

if ! command -v node &>/dev/null; then
  echo "Node.js not found. Install Node.js >= 18 and re-run."
  exit 1
fi

# ── Kill existing session (if any) ────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Killing it first..."
  tmux kill-session -t "$SESSION"
fi

# ── Install deps if node_modules is missing ───────────────────────────────────
if [ ! -d "$PROJECT_DIR/node_modules" ]; then
  echo "node_modules not found — running 'npm run inst'..."
  (cd "$PROJECT_DIR" && npm run inst)
fi

# ── Create tmux session ───────────────────────────────────────────────────────
# Window layout:
#   Pane 0 (top)    — Vite client dev server  (port 5173)
#   Pane 1 (bottom) — Node game server        (port 3000, workers 3001-3002)

tmux new-session -d -s "$SESSION" -x 220 -y 50 -c "$PROJECT_DIR"

# Top pane: client
tmux send-keys -t "$SESSION:0" "npm run start:client" Enter

# Split horizontally for server pane
tmux split-window -v -t "$SESSION:0" -c "$PROJECT_DIR"

# Bottom pane: server
tmux send-keys -t "$SESSION:0.1" "npm run start:server-dev" Enter

# Optional: rename window
tmux rename-window -t "$SESSION:0" "openfront-dev"

# ── Attach ────────────────────────────────────────────────────────────────────
echo ""
echo "  Session '$SESSION' started."
echo "  Client  → http://localhost:5173"
echo "  Server  → http://localhost:3000"
echo "  Workers → ws://localhost:3001  ws://localhost:3002"
echo ""
echo "  Attaching... (detach with Ctrl-b d)"
echo ""

tmux attach-session -t "$SESSION"
