#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required but was not found in PATH."
  echo "Install it first: curl -fsSL https://pixi.sh/install.sh | bash"
  exit 1
fi

pixi install
pixi run clone-contact-graspnet
pixi run -e grasp ensure-grasp-checkpoint

cat <<'EOF'
Workspace bootstrapped.

Next step for grasp service:
1. Compile the TensorFlow pointnet ops:
   pixi run -e grasp compile-grasp-ops
EOF
