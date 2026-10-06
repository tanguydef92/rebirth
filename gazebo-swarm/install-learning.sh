#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LEARNING_PYTHON="${LEARNING_PYTHON:-$ROOT_DIR/.pixi/envs/default/bin/python3}"
if [[ ! -x "$LEARNING_PYTHON" ]]; then LEARNING_PYTHON=python3; fi
"$LEARNING_PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "PPO training requires Python >= 3.10"'
if [[ ! -x "$ROOT_DIR/.learning-venv/bin/python" ]]; then
  "$LEARNING_PYTHON" -m venv "$ROOT_DIR/.learning-venv"
fi
# Keep conda's OpenMP/NumPy libraries out of the PyTorch process.
"$LEARNING_PYTHON" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1]); p.write_text(p.read_text().replace("include-system-site-packages = true", "include-system-site-packages = false"))' "$ROOT_DIR/.learning-venv/pyvenv.cfg"
"$ROOT_DIR/.learning-venv/bin/python" -m pip install --cache-dir "$ROOT_DIR/.cache/pip" -r "$ROOT_DIR/requirements-learning.txt"
