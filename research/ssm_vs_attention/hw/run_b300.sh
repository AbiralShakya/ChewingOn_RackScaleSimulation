#!/usr/bin/env bash
# B300 (della-tridao GPU 3): timing sweep, then Nsight Compute + SASS. Each step takes the shared GPU-3 lock.
set -uo pipefail
cd /scratch/gpfs/MZHASAN/as0714/ssmstudy
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/scratch/gpfs/MZHASAN/as0714/ssmstudy/out_b300/triton_cache
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p out_b300
flock -w 3600 /home/as0714/.gpu3.lock python bench_kernels.py --out out_b300/timing.json 2> out_b300/timing.log
echo TIMING_DONE
flock -w 3600 /home/as0714/.gpu3.lock ./run_ncu.sh out_b300
echo ALL_DONE
