#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGRAM_DIR="$ROOT_DIR/diagrams"
ASSET_DIR="$ROOT_DIR/assets"

if ! command -v mmdc >/dev/null 2>&1; then
  echo "mmdc not found. Install Mermaid CLI first (for example via @mermaid-js/mermaid-cli)." >&2
  exit 1
fi

mkdir -p "$ASSET_DIR"

for source in "$DIAGRAM_DIR"/*.mmd; do
  [ -e "$source" ] || continue
  name="$(basename "$source" .mmd)"
  mmdc -i "$source" -o "$ASSET_DIR/$name.svg" -b transparent
done

echo "Rendered Mermaid SVGs into $ASSET_DIR"
