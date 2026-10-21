#!/usr/bin/env bash
# Developer helper: push this directory to della-tridao and run a command there.
# Usage: scripts/sync_della.sh [command ...]   (default: ./run.sh)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/scratch/gpfs/MZHASAN/as0714/rackcal/calibration}"
GPU="${GPU:-3}"
tar -C "$HERE" --exclude build --exclude results -czf - . |
  ssh della "ssh della-tridao 'mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xzf -'"
CMD="${*:-./run.sh}"
ssh della "ssh della-tridao 'cd $REMOTE_DIR && export CUDA_HOME=/usr/local/cuda-13.3 PATH=/usr/local/cuda-13.3/bin:\$PATH CUDA_VISIBLE_DEVICES=$GPU && $CMD'"
