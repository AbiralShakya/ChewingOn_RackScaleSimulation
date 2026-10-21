#!/usr/bin/env bash
# SM clock, board power and throughput while CUTLASS GEMMs of each dtype run.
# Each GEMM runs for ~3 s so the clock settles under the power limit; the
# nvidia-smi trace (100 ms) is windowed per run by merge.py into model.power.
#
#   CUTLASS_DIR=/path/to/cutlass CUDA_VISIBLE_DEVICES=3 [GPU_LOCK=/path/lock] \
#     scripts/cutlass_power.sh results/<gpu>-sm<cc>
#
# Needs a CUTLASS build with CUTLASS_NVCC_ARCHS=103a (or 100a) and the targets
# below. Two local edits to the CUTLASS examples are assumed on sm_103:
#  * example 70 checks for cc 10.0 exactly; relax it to accept 10.3,
#  * examples 70 and 72b always run a host reference check, which takes many
#    minutes at 16k; skip it when CAL_SKIP_VERIFY is set
#    (result.passed = getenv("CAL_SKIP_VERIFY") ? true : verify(options);).
# Correctness is checked once per GEMM at 2048^3 with the reference enabled.
#
# With GPU_LOCK set, the lock is taken per GEMM run, not for the whole sweep,
# so other jobs on a shared GPU can interleave.
set -euo pipefail
OUT="$1"
CUTLASS_DIR="${CUTLASS_DIR:?set CUTLASS_DIR}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
EX="$CUTLASS_DIR/build/examples"
GEMMS=(  # label:binary
  "fp16_dense:$EX/70_blackwell_gemm/70_blackwell_fp16_gemm"
  "fp8_dense:$EX/70_blackwell_gemm/70_blackwell_fp8_gemm"
  "nvfp4_k64:$EX/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm"
  "nvfp4_ultra_k96:$EX/89_sm103_fp4_ultra_gemm/89_sm103_fp4_ultra_gemm"
)
mkdir -p "$OUT/raw"
LOG="$OUT/raw/cutlass_power.log"
: >"$LOG"
locked() { if [ -n "${GPU_LOCK:-}" ]; then flock "$GPU_LOCK" "$@"; else "$@"; fi; }
stamp() { date '+%Y/%m/%d %H:%M:%S.%3N'; }   # nvidia-smi's timestamp format
export -f stamp

nvidia-smi -i "${GPU%%,*}" --query-gpu=timestamp,clocks.sm,power.draw,power.limit,temperature.gpu,clocks_event_reasons.active,utilization.gpu,clocks.max.sm \
  --format=csv,noheader -lms 100 >"$OUT/raw/cutlass_power.clk.csv" &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

for g in "${GEMMS[@]}"; do
  label=${g%%:*} bin=${g#*:}
  [ -x "$bin" ] || { echo "SKIP $label (not built)" | tee -a "$LOG"; continue; }
  echo "VERIFY $label $(locked "$bin" --m=2048 --n=2048 --k=2048 --iterations=2 2>&1 | grep -i disposition | head -1)" | tee -a "$LOG"
  for s in 4096 8192 16384; do
    # Size the iteration count from a short probe so each run lasts ~3 s.
    ms=$(CAL_SKIP_VERIFY=1 locked "$bin" --m=$s --n=$s --k=$s --iterations=10 --no_verif 2>&1 | awk '/Avg runtime/ {print $3; exit}')
    it=$(awk -v ms="${ms:-1}" 'BEGIN { n = int(3000 / ms); if (n < 10) n = 10; print n }')
    # BEGIN/END are written inside the lock, so the window holds only this run.
    CAL_SKIP_VERIFY=1 locked bash -c "echo BEGIN $label $s \$(stamp); '$bin' --m=$s --n=$s --k=$s --iterations=$it --no_verif 2>&1 | grep -E 'runtime|GFLOPS'; echo END $label $s \$(stamp)" |
      tee -a "$LOG"
  done
done
