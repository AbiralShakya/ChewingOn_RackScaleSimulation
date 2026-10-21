#!/usr/bin/env bash
# B300: re-time FA4 decode with split-KV, then Nsight Compute + SASS for every case.
set -uo pipefail
cd /scratch/gpfs/MZHASAN/as0714/ssmstudy
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=3 TRITON_CACHE_DIR=/scratch/gpfs/MZHASAN/as0714/ssmstudy/out_b300/triton_cache
export PATH=/usr/local/cuda-13.0/bin:$PATH
flock -w 3600 /home/as0714/.gpu3.lock python bench_kernels.py --only attn_decode --out out_b300/timing_decode.json 2> out_b300/timing_decode.log
echo DECODE_DONE
flock -w 3600 /home/as0714/.gpu3.lock ./run_ncu.sh out_b300
echo ALL_DONE
