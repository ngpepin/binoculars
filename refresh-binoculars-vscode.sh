#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="$SCRIPT_DIR/vscode-extension"
RELOAD_WINDOW=0

usage() {
  cat <<'EOF'
Usage: ./refresh-binoculars-vscode.sh [--reload]

Builds, packages, and force-installs the local Binoculars VS Code extension.

Options:
  --reload   Attempt to reload the current VS Code window after install.
  -h, --help Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reload)
      RELOAD_WINDOW=1
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

echo "[1/3] Compiling extension..."
(
  cd "$EXT_DIR"
  npm run compile
)

echo "[2/3] Packaging VSIX..."
(
  cd "$EXT_DIR"
  npx @vscode/vsce package
)

VSIX_PATH="$(ls -1t "$EXT_DIR"/binoculars-vscode-*.vsix 2>/dev/null | head -n1 || true)"
if [[ -z "$VSIX_PATH" ]]; then
  echo "ERROR: VSIX artifact not found in $EXT_DIR." >&2
  exit 1
fi

echo "[3/3] Installing VSIX..."
code --install-extension "$VSIX_PATH" --force

if [[ "$RELOAD_WINDOW" == "1" ]]; then
  echo "Attempting VS Code window reload..."
  if ! code --reuse-window --command workbench.action.reloadWindow >/dev/null 2>&1; then
    echo "WARNING: automatic reload command failed. Use 'Developer: Reload Window' in VS Code." >&2
  fi
fi

echo "Done."
echo "Installed: $VSIX_PATH"

