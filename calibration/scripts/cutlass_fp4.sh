#!/usr/bin/env bash
# End-to-end FP4 GEMM check with CUTLASS, with SM clock and power sampled
# during the run. Writes <outdir>/raw/cutlass_fp4.{log,clk.csv}.
#
#   CUTLASS_DIR=/path/to/cutlass CUDA_VISIBLE_DEVICES=3 scripts/cutlass_fp4.sh results/<gpu>-sm<cc>
#
# Builds example 89 (sm_103 "ultra" FP4, K=96 UMMA) if needed. On shared
# machines set GPU_LOCK to a flock file to serialise with other jobs.
set -euo pipefail
OUT="$1"
CUTLASS_DIR="${CUTLASS_DIR:?set CUTLASS_DIR}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
BIN="$CUTLASS_DIR/build/examples/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm"
if [ ! -x "$BIN" ]; then
  cmake -S "$CUTLASS_DIR" -B "$CUTLASS_DIR/build" -G Ninja -DCUTLASS_NVCC_ARCHS=103a -DCUTLASS_ENABLE_TESTS=OFF \
    -DCUTLASS_ENABLE_LIBRARY=OFF -DCUTLASS_ENABLE_PROFILER=OFF >/dev/null
  ninja -C "$CUTLASS_DIR/build" 89_sm103_fp4_ultra_gemm >/dev/null
fi
mkdir -p "$OUT/raw"
LOCK=(); [ -n "${GPU_LOCK:-}" ] && LOCK=(flock "$GPU_LOCK")
"${LOCK[@]}" bash -c "
  nvidia-smi -i ${GPU%%,*} --query-gpu=timestamp,clocks.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active,utilization.gpu,clocks.max.sm \
    --format=csv,noheader -lms 100 > '$OUT/raw/cutlass_fp4.clk.csv' &
  SMI=\$!
  '$BIN' --m=2048 --n=2048 --k=2048 --iterations=5 | grep -i disposition
  # BEGIN/END stamps use nvidia-smi's timestamp format so merge.py can window the trace.
  # Iterations sized so each GEMM runs ~3 s per variant: long enough for the clock to settle.
  for si in 4096:100000 8192:20000 16384:2000; do
    s=\${si%:*}; it=\${si#*:}
    echo \"BEGIN \$s \$(date '+%Y/%m/%d %H:%M:%S.%3N')\"
    '$BIN' --m=\$s --n=\$s --k=\$s --iterations=\$it --no_verif | grep -E 'Problem|runtime|GFLOPS'
    echo \"END \$s \$(date '+%Y/%m/%d %H:%M:%S.%3N')\"
  done
  kill \$SMI" | tee "$OUT/raw/cutlass_fp4.log"
