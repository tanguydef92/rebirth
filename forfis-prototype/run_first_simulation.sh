#!/usr/bin/env bash
# Activate local venv and run first simulation (macOS/Linux)
set -euo pipefail
cd /bin
. .venv/bin/activate
export FORFIS_CONFIG=config_first_simulation.yml
.venv/bin/python main.py > results/first_simulation/run.log 2>&1
