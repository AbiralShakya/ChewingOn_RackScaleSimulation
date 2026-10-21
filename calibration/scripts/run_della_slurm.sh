#!/usr/bin/env bash
# Developer helper: push this directory to Della scratch and run the suite
# inside an existing Slurm GPU allocation (e.g. an A100 from
#   salloc --no-shell --gres=gpu:1 --constraint="a100&gpu80" -t 02:00:00).
# Usage: JOBID=<id> scripts/run_della_slurm.sh [command ...]
# Default command: ./run.sh (build, run every bench, merge, ncu reference).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/scratch/gpfs/MZHASAN/as0714/rackcal-ampere/calibration}"
CUDA_MOD="${CUDA_MOD:-cudatoolkit/12.8}"
# Della compute nodes ship nvcc/ncu/nvdisasm but not cuobjdump or cu++filt;
# copies taken from the login node's toolkit live here.
TOOLS="${TOOLS:-/scratch/gpfs/MZHASAN/as0714/rackcal-ampere/tools}"
: "${JOBID:?set JOBID to a running Slurm allocation}"
COPYFILE_DISABLE=1 tar -C "$HERE" --exclude build --exclude results -czf - . |
  ssh della "mkdir -p $REMOTE_DIR && tar -C $REMOTE_DIR -xzf - 2>/dev/null"
CMD="${*:-./run.sh}"
ssh della "srun --jobid=$JOBID --overlap -N1 -n1 bash -lc 'module load $CUDA_MOD >/dev/null 2>&1; export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc))) PATH=$TOOLS:\$PATH; cd $REMOTE_DIR && $CMD'"
