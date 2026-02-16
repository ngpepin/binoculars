#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$SCRIPT_DIR/vscode-extension"
PYTHON_BIN="${BINOCULARS_PYTHON:-$SCRIPT_DIR/venv/bin/python}"
BRIDGE_SCRIPT="$EXT_DIR/python/binoculars_bridge.py"
SOCKET_PATH="${BINOCULARS_SOCKET_PATH:-/tmp/binoculars-vscode-$(id -u).sock}"
DAEMON_LOG="${BINOCULARS_DAEMON_LOG:-/tmp/binoculars-vscode-daemon.log}"
VSIX_PATH="$EXT_DIR/binoculars-vscode-dev.vsix"
RELOAD_WINDOW=0
RESTART_DAEMON=1

usage() {
  cat <<'EOF'
Usage: ./refresh-binoculars-vscode.sh [--reload] [--no-daemon]

Builds, packages, force-installs the local Binoculars VS Code extension,
restarts the VS Code extension host process, and restarts the Binoculars daemon.

Options:
  --reload      Attempt to reload the current VS Code window after install.
  --no-daemon   Skip Binoculars daemon restart.
  -h, --help Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reload)
      RELOAD_WINDOW=1
      shift
      ;;
    --no-daemon)
      RESTART_DAEMON=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$EXT_DIR" ]]; then
  echo "ERROR: extension directory not found: $EXT_DIR" >&2
  exit 1
fi

if ! command -v code >/dev/null 2>&1; then
  echo "ERROR: VS Code CLI 'code' not found in PATH." >&2
  echo "Install it from VS Code Command Palette: 'Shell Command: Install code command in PATH'." >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$BRIDGE_SCRIPT" ]]; then
  echo "ERROR: bridge script not found: $BRIDGE_SCRIPT" >&2
  exit 1
fi

echo "[1/6] Compiling extension..."
(
  cd "$EXT_DIR"
  npm run compile
)

echo "[2/6] Packaging VSIX..."
(
  cd "$EXT_DIR"
  rm -f "$VSIX_PATH"
  npx --yes @vscode/vsce package --allow-missing-repository --out "$VSIX_PATH"
)

if [[ ! -f "$VSIX_PATH" ]]; then
  echo "ERROR: VSIX artifact not found in $EXT_DIR." >&2
  exit 1
fi

echo "[3/6] Installing VSIX..."
code --install-extension "$VSIX_PATH" --force

echo "[4/6] Restarting VS Code extension host..."
EXT_PIDS="$(code --status 2>/dev/null | awk '/extension-host/ {print $3}' | tr '\n' ' ' | xargs || true)"
if [[ -z "$EXT_PIDS" ]]; then
  echo "WARNING: could not find extension-host pid via 'code --status'." >&2
else
  for pid in $EXT_PIDS; do
    kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  sleep 1
fi

if [[ "$RESTART_DAEMON" == "1" ]]; then
  echo "[5/6] Restarting Binoculars daemon..."
  "$PYTHON_BIN" - "$SOCKET_PATH" <<'PY' >/dev/null 2>&1 || true
import json
import socket
import sys

p = sys.argv[1]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(1.5)
s.connect(p)
f = s.makefile("rwb", buffering=0)
f.write((json.dumps({"id": 1, "method": "shutdown_daemon", "params": {}}) + "\n").encode("utf-8"))
try:
    f.readline()
    f.readline()
except Exception:
    pass
s.close()
PY
  rm -f "$SOCKET_PATH"
  nohup "$PYTHON_BIN" "$BRIDGE_SCRIPT" --daemon --socket-path "$SOCKET_PATH" >"$DAEMON_LOG" 2>&1 &
  sleep 1
  if [[ ! -S "$SOCKET_PATH" ]]; then
    echo "ERROR: daemon socket not created: $SOCKET_PATH" >&2
    echo "Last daemon log lines:" >&2
    tail -n 80 "$DAEMON_LOG" >&2 || true
    exit 1
  fi
  "$PYTHON_BIN" - "$SOCKET_PATH" <<'PY' >/dev/null
import json
import socket
import sys

p = sys.argv[1]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(2.0)
s.connect(p)
f = s.makefile("rwb", buffering=0)
f.write((json.dumps({"id": 2, "method": "health", "params": {}}) + "\n").encode("utf-8"))
f.readline()
resp = f.readline().decode("utf-8", "replace").strip()
if '"ok": true' not in resp:
    raise SystemExit(f"health check failed: {resp}")
s.close()
PY
else
  echo "[5/6] Skipping daemon restart (--no-daemon)."
fi

if [[ "$RELOAD_WINDOW" == "1" ]]; then
  echo "[6/6] Attempting VS Code window reload..."
  code --reuse-window --command workbench.action.reloadWindow >/dev/null 2>&1 || true
else
  echo "[6/6] Window reload not requested."
fi

echo "Done."
echo "Installed: $VSIX_PATH"
if [[ "$RESTART_DAEMON" == "1" ]]; then
  echo "Daemon socket: $SOCKET_PATH"
fi
