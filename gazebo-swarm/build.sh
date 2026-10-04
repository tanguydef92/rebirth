#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$ROOT_DIR/.tools/pixi" && -z "${SWARM_PIXI_ACTIVE:-}" ]]; then
  export SWARM_PIXI_ACTIVE=1
  export PIXI_CACHE_DIR="$ROOT_DIR/.cache/pixi"
  exec "$ROOT_DIR/.tools/pixi" run --manifest-path "$ROOT_DIR/pixi.toml" bash "$ROOT_DIR/build.sh"
fi
if ! command -v cmake >/dev/null || ! command -v gz >/dev/null; then
  echo "Gazebo Harmonic et CMake sont requis. Voir README.md."
  exit 1
fi
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=RelWithDebInfo)
if [[ -n "${CONDA_PREFIX:-}" ]]; then CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"); fi
cmake -S "$ROOT_DIR" -B "$ROOT_DIR/build" "${CMAKE_ARGS[@]}"
cmake --build "$ROOT_DIR/build" --parallel 4
ctest --test-dir "$ROOT_DIR/build" --output-on-failure
