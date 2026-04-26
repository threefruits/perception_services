#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTACT_GRASPNET_DIR="${ROOT_DIR}/third_party/contact_graspnet"
CHECKPOINT_ROOT="${CONTACT_GRASPNET_DIR}/checkpoints"
CHECKPOINT_DIR="${CONTACT_GRASPNET_CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/scene_test_2048_bs3_hor_sigma_001}"
CHECKPOINT_URL="${CONTACT_GRASPNET_CHECKPOINT_URL:-https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing}"

if [ ! -d "${CONTACT_GRASPNET_DIR}/.git" ]; then
  echo "contact_graspnet clone not found at ${CONTACT_GRASPNET_DIR}"
  echo "Run: pixi run clone-contact-graspnet"
  exit 1
fi

if [ -f "${CHECKPOINT_DIR}/checkpoint" ]; then
  echo "Contact-GraspNet checkpoint already present at ${CHECKPOINT_DIR}"
  exit 0
fi

mkdir -p "${CHECKPOINT_ROOT}"

echo "Checkpoint not found. Downloading Contact-GraspNet checkpoints from Google Drive..."
gdown --folder --fuzzy "${CHECKPOINT_URL}" -O "${CHECKPOINT_ROOT}"

# Some downloads include an extra nested checkpoints/ folder.
if [ -d "${CHECKPOINT_ROOT}/checkpoints" ]; then
  find "${CHECKPOINT_ROOT}/checkpoints" -mindepth 1 -maxdepth 1 -exec mv -n {} "${CHECKPOINT_ROOT}/" \;
  rmdir "${CHECKPOINT_ROOT}/checkpoints" || true
fi

if [ ! -f "${CHECKPOINT_DIR}/checkpoint" ]; then
  echo "Download finished but expected checkpoint was not found at:"
  echo "  ${CHECKPOINT_DIR}"
  echo "Set CONTACT_GRASPNET_CHECKPOINT_DIR to the downloaded checkpoint path and retry."
  exit 1
fi

echo "Contact-GraspNet checkpoint ready: ${CHECKPOINT_DIR}"
