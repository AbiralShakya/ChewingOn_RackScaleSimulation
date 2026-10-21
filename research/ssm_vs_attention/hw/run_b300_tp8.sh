#!/usr/bin/env bash
# B300: the per-GPU kernel shapes of the 70B-class models at TP=8 (timing + ncu).
set -uo pipefail
cd /scratch/gpfs/MZHASAN/as0714/ssmstudy
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/scratch/gpfs/MZHASAN/as0714/ssmstudy/out_b300_tp8/triton_cache
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p out_b300_tp8
flock -w 3600 /home/as0714/.gpu3.lock python bench_kernels.py --set tp8_70b --out out_b300_tp8/timing.json 2> out_b300_tp8/timing.log
echo TIMING_DONE
flock -w 3600 /home/as0714/.gpu3.lock ./run_ncu.sh out_b300_tp8 --set tp8_70b
echo ALL_DONE
