#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

bash service/grasp/ensure_grasp_checkpoint.sh
python service/grasp/contact_graspnet_server.py "$@"
