#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/service/logs"
PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"
OWLV2_PORT="${OWLV2_PORT:-4000}"
SAM_PORT="${SAM_PORT:-4001}"
GRASP_PORT="${GRASP_PORT:-4003}"

mkdir -p "$LOG_DIR"

if [ ! -x "$PIXI_BIN" ] && ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required but was not found."
  exit 1
fi

if [ ! -x "$PIXI_BIN" ]; then
  PIXI_BIN="$(command -v pixi)"
fi

cleanup() {
  local exit_code=$?
  jobs -pr | xargs -r kill
  wait || true
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

cd "$ROOT_DIR"

"$PIXI_BIN" run python service/owl_vit/server.py --port "$OWLV2_PORT" >"$LOG_DIR/owlv2.log" 2>&1 &
OWL_PID=$!

"$PIXI_BIN" run python service/sam/server.py --port "$SAM_PORT" >"$LOG_DIR/sam.log" 2>&1 &
SAM_PID=$!

"$PIXI_BIN" run -e grasp bash service/grasp/run_grasp_server.sh --port "$GRASP_PORT" >"$LOG_DIR/grasp.log" 2>&1 &
GRASP_PID=$!

echo "Started OWLv2 (${OWL_PID}), SAM (${SAM_PID}), and Contact-GraspNet (${GRASP_PID})."
echo "Logs: $LOG_DIR"
echo "Ports: OWLv2=${OWLV2_PORT}, SAM=${SAM_PORT}, Grasp=${GRASP_PORT}"

wait
