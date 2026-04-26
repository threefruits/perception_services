#!/bin/bash
#SBATCH --job-name=det-seg-grasp
#SBATCH --output=service/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=40000
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane7

set -euo pipefail

ROOT_DIR="/data/home/anxing/perception_services"
PIXI_BIN="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"

cd "$ROOT_DIR"
nvidia-smi

if [ ! -x "$PIXI_BIN" ] && ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required but was not found."
  exit 1
fi

if [ ! -x "$PIXI_BIN" ]; then
  PIXI_BIN="$(command -v pixi)"
fi

export PIXI_BIN
bash service/start_det_seg_grasp.sh
