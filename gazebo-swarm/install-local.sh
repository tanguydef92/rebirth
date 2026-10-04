#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  echo "Cette installation locale cible macOS Apple Silicon. Voir README.md pour les autres systèmes."
  exit 1
fi
mkdir -p "$ROOT_DIR/.tools"
if [[ ! -x "$ROOT_DIR/.tools/pixi" ]]; then
  curl -fL --connect-timeout 15 \
    https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin.tar.gz \
    -o "$ROOT_DIR/.tools/pixi.tar.gz"
  tar -xzf "$ROOT_DIR/.tools/pixi.tar.gz" -C "$ROOT_DIR/.tools" pixi
fi
export PIXI_CACHE_DIR="$ROOT_DIR/.cache/pixi"
"$ROOT_DIR/.tools/pixi" install --manifest-path "$ROOT_DIR/pixi.toml"
"$ROOT_DIR/.tools/pixi" run --manifest-path "$ROOT_DIR/pixi.toml" build
