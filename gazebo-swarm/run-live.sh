#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIVE_PYTHON="$ROOT_DIR/.learning-venv/bin/python"
if [[ ! -x "$LIVE_PYTHON" ]]; then LIVE_PYTHON="$ROOT_DIR/.pixi/envs/default/bin/python3"; fi
if [[ ! -x "$LIVE_PYTHON" ]]; then LIVE_PYTHON=python3; fi
exec "$LIVE_PYTHON" "$ROOT_DIR/live_sim.py" "$@"
