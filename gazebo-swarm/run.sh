#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$ROOT_DIR/.tools/pixi" && -z "${SWARM_PIXI_ACTIVE:-}" ]]; then
  export SWARM_PIXI_ACTIVE=1
  export PIXI_CACHE_DIR="$ROOT_DIR/.cache/pixi"
  exec "$ROOT_DIR/.tools/pixi" run --manifest-path "$ROOT_DIR/pixi.toml" bash "$ROOT_DIR/run.sh" "$@"
fi
MODE=gui
PATTERN=circle
ITERATIONS=""
CONFIG_PATH="$ROOT_DIR/config.json"
PATTERN_OVERRIDE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) MODE=headless; shift ;;
    --pattern) PATTERN="${2:?--pattern nécessite circle, helix ou sweep}"; PATTERN_OVERRIDE=true; shift 2 ;;
    --config) CONFIG_PATH="${2:?--config nécessite un chemin}"; shift 2 ;;
    --iterations) ITERATIONS="${2:?--iterations nécessite un entier}"; shift 2 ;;
    *) echo "Usage : bash run.sh [--headless] [--pattern circle|helix|sweep] [--config fichier.json] [--iterations N]"; exit 1 ;;
  esac
done
python3 "$ROOT_DIR/swarm.py" doctor
GENERATOR_ARGS=(generate --config "$CONFIG_PATH")
if [[ "$PATTERN_OVERRIDE" == true ]]; then GENERATOR_ARGS+=(--pattern "$PATTERN"); fi
python3 "$ROOT_DIR/swarm.py" "${GENERATOR_ARGS[@]}"
if [[ ! -f "$ROOT_DIR/build/libSwarmPlayback.dylib" && ! -f "$ROOT_DIR/build/libSwarmPlayback.so" ]]; then
  bash "$ROOT_DIR/build.sh"
fi
# A separate partition keeps this swarm isolated from other Gazebo sessions.
export GZ_PARTITION="swarm-beta-$$"
export GZ_SIM_SYSTEM_PLUGIN_PATH="$ROOT_DIR/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$ROOT_DIR/output${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
SERVER_ARGS=(sim -s -r -v 3 "$ROOT_DIR/output/swarm.sdf")
if [[ -n "$ITERATIONS" ]]; then SERVER_ARGS+=(--iterations "$ITERATIONS"); fi
if [[ "$MODE" == headless ]]; then
  exec gz "${SERVER_ARGS[@]}"
fi
SERVER_PID=""
cleanup() { if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
gz "${SERVER_ARGS[@]}" > "$ROOT_DIR/output/server.log" 2>&1 &
SERVER_PID=$!
# Wait for the world transport service, not a guessed GUI startup delay.
READY=false
for ((attempt=0; attempt<30; attempt++)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$ROOT_DIR/output/server.log"
    echo "Le serveur Gazebo s'est arrêté."
    exit 1
  fi
  if gz service -l | grep -q '/world/swarm_beta/control'; then READY=true; break; fi
  sleep 1
done
if [[ "$READY" != true ]]; then
  cat "$ROOT_DIR/output/server.log"
  echo "Le serveur n'a pas annoncé le monde swarm_beta après 30 secondes."
  exit 1
fi
GUI_ARGS=(sim -g -v 2 --render-engine ogre2)
if [[ "$(uname -s)" == Darwin ]]; then GUI_ARGS+=(--render-engine-gui-api-backend metal); fi
gz "${GUI_ARGS[@]}"
