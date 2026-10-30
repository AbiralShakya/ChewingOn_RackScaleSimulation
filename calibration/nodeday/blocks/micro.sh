#!/usr/bin/env bash
# The open microbenchmarks of docs/CALIBRATION.md that one GPU of the node can settle.
#
#   CUDA_VISIBLE_DEVICES=3 micro.sh <outdir>
#
#   1, 2, 4, 8, 9  calibration/src/bench_open.cu (stamps, DRAM latency histogram,
#                  MIO fairness, shared FP64); also part of run.sh's suite
#   7              FA2 s4096 per-PC stall sampling with Nsight Compute
#   10             bench_overlap (HMMA vs FFMA/FMUL/LDSM co-issue), run by run.sh
#   11             clock under attention: the power block (power.py) runs it
#
# 3 and 5 (TMA from HBM by stage count; TMA near/far partitions) and 6
# (re-recording FA4 ring steps per CTA tile count) need the B300 the
# existing FA4 recordings came from; they stay on Della's GPU 3.
set -uo pipefail
OUT=${1:?outdir}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
mkdir -p "$OUT"
cd "$REPO/calibration"
make -s build/bench_open || exit 1
./build/bench_open > "$OUT/open.json" 2> "$OUT/open.log" && echo "bench_open ok" || echo "bench_open FAILED"

# 7: FA2 s4096 (the ncu reference shape: b4 h16 s4096 d128), source-level stall sampling.
if command -v ncu > /dev/null && python -c "import flash_attn" 2> /dev/null; then
  ncu --profile-from-start off --set full --section SourceCounters --section WarpStateStats \
    --import-source yes -o "$OUT/fa2_s4096_pcsamp" -f \
    python "$REPO/research/flash_attention/remote/fa_bench.py" --impl fa2 --b 4 --h 16 --s 4096 --d 128 --once \
    > "$OUT/fa2_pcsamp.log" 2>&1 && echo "fa2 pc sampling ok" || echo "fa2 pc sampling FAILED (see fa2_pcsamp.log)"
  ncu --import "$OUT/fa2_s4096_pcsamp.ncu-rep" --page source --csv --print-source sass \
    > "$OUT/fa2_s4096_source.csv" 2>/dev/null
else
  echo "skip 7: needs ncu and flash_attn (FA2)"
fi
