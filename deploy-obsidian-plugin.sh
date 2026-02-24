#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_SRC_DIR="$SCRIPT_DIR/obsidian-plugin"
PLUGIN_ID="binoculars-obsidian"
DEFAULT_PLUGIN_DIR="${OBSIDIAN_PLUGIN_DIR:-/mnt/drive2/Obsidian/Nico/.obsidian/plugins/binoculars-obsidian}"
DEFAULT_VAULT_PATH="$(dirname "$(dirname "$(dirname "$DEFAULT_PLUGIN_DIR")")")"

usage() {
  cat <<'EOF'
Usage: ./deploy-obsidian-plugin.sh [TARGET_PATH] [--no-build]

Builds the Obsidian plugin and copies main.js/manifest.json/styles.css
to the target vault as regular files (not symlinks).

Arguments:
  TARGET_PATH  Optional vault path OR full plugin directory path.
               Defaults to:
               $OBSIDIAN_PLUGIN_DIR or /mnt/drive2/Obsidian/Nico/.obsidian/plugins/binoculars-obsidian

Options:
  --no-build   Skip npm build and only copy files.
  -h, --help   Show this help message.
EOF
}

NO_BUILD=0
TARGET_PATH="$DEFAULT_PLUGIN_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build)
      NO_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "$TARGET_PATH" != "$DEFAULT_PLUGIN_DIR" ]]; then
        echo "ERROR: multiple target paths provided." >&2
        usage >&2
        exit 2
      fi
      TARGET_PATH="$1"
      shift
      ;;
  esac
done

if [[ "$TARGET_PATH" == */.obsidian/plugins/"$PLUGIN_ID" ]]; then
  DEST_DIR="$TARGET_PATH"
elif [[ "$(basename "$TARGET_PATH")" == "$PLUGIN_ID" && "$(basename "$(dirname "$TARGET_PATH")")" == "plugins" ]]; then
  DEST_DIR="$TARGET_PATH"
else
  DEST_DIR="$TARGET_PATH/.obsidian/plugins/$PLUGIN_ID"
fi

if [[ ! -d "$PLUGIN_SRC_DIR" ]]; then
  echo "ERROR: plugin source directory not found: $PLUGIN_SRC_DIR" >&2
  exit 1
fi

if [[ "$NO_BUILD" -eq 0 ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm is required to build the plugin." >&2
    exit 1
  fi
  echo "[1/3] Building Obsidian plugin..."
  (
    cd "$PLUGIN_SRC_DIR"
    npm run build
  )
else
  echo "[1/3] Skipping build (--no-build)."
fi

echo "[2/3] Preparing vault plugin directory..."
mkdir -p "$DEST_DIR"

for name in main.js manifest.json styles.css; do
  src="$PLUGIN_SRC_DIR/$name"
  dst="$DEST_DIR/$name"
  if [[ ! -f "$src" ]]; then
    echo "ERROR: missing source file: $src" >&2
    exit 1
  fi
  rm -f "$dst"
  install -m 0644 "$src" "$dst"
done

echo "[3/3] Deployed to: $DEST_DIR"
echo "Files now deployed as regular files:"
ls -l "$DEST_DIR"/main.js "$DEST_DIR"/manifest.json "$DEST_DIR"/styles.css
echo
echo "Next step: reload the Binoculars Obsidian plugin."
