#!/usr/bin/env bash
# SM clock, board power and throughput while CUTLASS GEMMs of each dtype run.
# Each GEMM runs for ~3 s so the clock settles under the power limit; the
# nvidia-smi trace (100 ms) is windowed per run by merge.py into model.power.
#
#   CUTLASS_DIR=/path/to/cutlass CUDA_VISIBLE_DEVICES=3 [GPU_LOCK=/path/lock] \
#     scripts/cutlass_power.sh results/<gpu>-sm<cc>
#
# Needs a CUTLASS build with CUTLASS_NVCC_ARCHS=103a (or 100a) and the targets
# below. Example 70 checks for cc 10.0 exactly; on sm_103 its check has to be
# relaxed to accept 10.3 (the kernels themselves build for the 10.x family).
set -euo pipefail
OUT="$1"
CUTLASS_DIR="${CUTLASS_DIR:?set CUTLASS_DIR}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EX="$CUTLASS_DIR/build/examples"
# label:binary
GEMMS=(
  "fp16_dense:$EX/70_blackwell_gemm/70_blackwell_fp16_gemm"
  "fp8_dense:$EX/70_blackwell_gemm/70_blackwell_fp8_gemm"
  "nvfp4_k64:$EX/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm"
  "nvfp4_ultra_k96:$EX/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm"
)
mkdir -p "$OUT/raw"
LOCK=(); [ -n "${GPU_LOCK:-}" ] && LOCK=(flock "$GPU_LOCK")
stamp() { date '+%Y/%m/%d %H:%M:%S.%3N'; }   # nvidia-smi's timestamp format
run_all() {
  nvidia-smi -i "${GPU%%,*}" --query-gpu=timestamp,clocks.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active,utilization.gpu,clocks.max.sm \
    --format=csv,noheader -lms 100 >"$OUT/raw/cutlass_power.clk.csv" &
  local smi=$!
  for g in "${GEMMS[@]}"; do
    local label=${g%%:*} bin=${g#*:}
    [ -x "$bin" ] || { echo "SKIP $label (not built)"; continue; }
    local nv=""; "$bin" --help 2>&1 | grep -q no_verif && nv="--no_verif"
    echo "VERIFY $label $("$bin" --m=2048 --n=2048 --k=2048 --iterations=2 2>&1 | grep -i disposition | head -1)"
    for s in 4096 8192 16384; do
      # size the iteration count from a short probe run so each GEMM lasts ~3 s
      local ms it
      ms=$("$bin" --m=$s --n=$s --k=$s --iterations=10 $nv 2>&1 | awk '/Avg runtime/ {print $3; exit}')
      it=$(awk -v ms="$ms" 'BEGIN { n = int(3000 / ms); if (n < 10) n = 10; print n }')
      echo "BEGIN $label $s $(stamp)"
      "$bin" --m=$s --n=$s --k=$s --iterations=$it $nv 2>&1 | grep -E 'runtime|GFLOPS'
      echo "END $label $s $(stamp)"
    done
  done
  kill $smi
}
export -f run_all stamp
export OUT GPU
GEMMS_STR=$(printf '%s\n' "${GEMMS[@]}")
"${LOCK[@]}" bash -c "$(declare -p GEMMS); $(declare -f run_all stamp); run_all" | tee "$OUT/raw/cutlass_power.log"
