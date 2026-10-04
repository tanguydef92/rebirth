#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "Ce script cible macOS. Pour Ubuntu, voir README.md."
  exit 1
fi
command -v brew >/dev/null || { echo "Homebrew est requis : https://brew.sh"; exit 1; }
brew tap osrf/simulation
brew trust osrf/simulation
brew install cmake osrf/simulation/gz-harmonic
bash "$ROOT_DIR/build.sh"
python3 "$ROOT_DIR/swarm.py" doctor
