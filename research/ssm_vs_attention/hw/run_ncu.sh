#!/usr/bin/env bash
# Profile every kernel inside the `case_<id>` NVTX ranges of bench_kernels.py with
# Nsight Compute, then dump the SASS of each distinct kernel.
# Usage: run_ncu.sh <outdir> [bench_kernels.py args...]
set -euo pipefail
OUT="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT/sass"
export TRITON_CACHE_DIR="$OUT/triton_cache"
METRICS=gpu__time_duration.sum,sm__cycles_elapsed.max,sm__cycles_elapsed.avg,smsp__cycles_active.avg,\
smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,\
sm__inst_executed_pipe_tensor.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
smsp__inst_executed_pipe_lsu.sum,smsp__inst_executed_pipe_xu.sum,smsp__inst_executed_pipe_fma.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed
for s in barrier branch_resolving dispatch_stall drain lg_throttle long_scoreboard math_pipe_throttle membar mio_throttle \
         misc no_instruction not_selected selected short_scoreboard sleeping tex_throttle wait; do
  METRICS="$METRICS,smsp__average_warps_issue_stalled_${s}_per_issue_active.ratio"
done
# Warm the Triton/CuTe JIT caches first so compilation is not profiled.
python "$HERE/bench_kernels.py" --out "$OUT/warm.json" --nvtx-only "$@" 2> "$OUT/warm.log"
ncu --clock-control none --nvtx --nvtx-include "regex:^case_.*/" --csv --page raw --print-nvtx-rename kernel \
    --kernel-name-base mangled --metrics "$METRICS" \
    python "$HERE/bench_kernels.py" --out "$OUT/ncu_run.json" --nvtx-only "$@" > "$OUT/ncu_raw.csv" 2> "$OUT/ncu.log" || true
# Also a version keyed by NVTX range so each kernel maps to its case.
ncu --clock-control none --nvtx --nvtx-include "regex:^case_.*/" --csv --page raw --print-nvtx-rename none \
    --kernel-name-base mangled --metrics gpu__time_duration.sum,sm__cycles_elapsed.max \
    python "$HERE/bench_kernels.py" --out "$OUT/ncu_run2.json" --nvtx-only "$@" > "$OUT/ncu_names.csv" 2> "$OUT/ncu2.log" || true
# SASS of the Triton kernels (each JIT'd kernel has a .cubin in the cache).
find "$TRITON_CACHE_DIR" -name '*.cubin' | while read -r c; do
  n=$(basename "$c" .cubin); h=$(basename "$(dirname "$c")" | cut -c1-12)
  cuobjdump -sass "$c" > "$OUT/sass/triton_${n}_${h}.sass" 2>/dev/null || true
  cp "$(dirname "$c")/${n}.json" "$OUT/sass/triton_${n}_${h}.json" 2>/dev/null || true
done
echo NCU_DONE
