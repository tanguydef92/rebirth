#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$ROOT_DIR/.tools/pixi" && -z "${SWARM_PIXI_ACTIVE:-}" ]]; then
  export SWARM_PIXI_ACTIVE=1
  export PIXI_CACHE_DIR="$ROOT_DIR/.cache/pixi"
  exec "$ROOT_DIR/.tools/pixi" run --manifest-path "$ROOT_DIR/pixi.toml" bash "$ROOT_DIR/run.sh" "$@"
fi
MODE=gui
ITERATIONS=""
CONFIG_PATH="$ROOT_DIR/configs/ppo-1000-fast.json"
OUTPUT_DIR="$ROOT_DIR/output"
REPLAY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) MODE=headless; shift ;;
    --replay) REPLAY=true; shift ;;
    --config) CONFIG_PATH="${2:?--config nécessite un chemin}"; shift 2 ;;
    --output) OUTPUT_DIR="${2:?--output nécessite un dossier}"; shift 2 ;;
    --iterations) ITERATIONS="${2:?--iterations nécessite un entier}"; shift 2 ;;
    *) echo "Usage : bash run.sh [--headless] [--replay] [--config fichier.json] [--output dossier] [--iterations N]"; exit 1 ;;
  esac
done
python3 "$ROOT_DIR/swarm.py" doctor
GENERATOR_ARGS=(generate --config "$CONFIG_PATH" --output "$OUTPUT_DIR")
if [[ "$REPLAY" == true ]]; then
  if [[ ! -f "$OUTPUT_DIR/swarm.sdf" || ! -f "$OUTPUT_DIR/mission.json" ]]; then
    echo "Generate the mission first; --replay needs swarm.sdf and mission.json in --output."; exit 1
  fi
else
  GENERATOR_PYTHON="$ROOT_DIR/.learning-venv/bin/python"
  if [[ ! -x "$GENERATOR_PYTHON" ]]; then GENERATOR_PYTHON=python3; fi
  "$GENERATOR_PYTHON" "$ROOT_DIR/swarm.py" "${GENERATOR_ARGS[@]}"
fi
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
# Incremental build also picks up edits to an existing C++ plugin.
bash "$ROOT_DIR/build.sh"
# A separate partition keeps this swarm isolated from other Gazebo sessions.
export GZ_PARTITION="swarm-beta-$$"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$ROOT_DIR/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$OUTPUT_DIR${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
SERVER_ARGS=(sim -s -r -v 3 "$OUTPUT_DIR/swarm.sdf")
if [[ -n "$ITERATIONS" ]]; then SERVER_ARGS+=(--iterations "$ITERATIONS"); fi
if [[ "$MODE" == headless ]]; then
  exec gz "${SERVER_ARGS[@]}"
fi
SERVER_PID=""
cleanup() { if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
gz "${SERVER_ARGS[@]}" > "$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
# Wait for the world transport service, not a guessed GUI startup delay.
READY=false
for ((attempt=0; attempt<30; attempt++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$OUTPUT_DIR/server.log"
    echo "Le serveur Gazebo s'est arrêté."
    exit 1
  fi
  if gz service -l | grep -q '/world/swarm_beta/control'; then READY=true; break; fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  cat "$OUTPUT_DIR/server.log"
  echo "Le serveur n'a pas annoncé le monde swarm_beta après 30 secondes."
  exit 1
fi
GUI_ARGS=(sim -g -v 2 --render-engine ogre2)
if [[ "$(uname -s)" == Darwin ]]; then GUI_ARGS+=(--render-engine-gui-api-backend metal); fi
gz "${GUI_ARGS[@]}"
