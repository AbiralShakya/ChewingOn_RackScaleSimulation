#!/usr/bin/env bash
# Build and run the whole calibration suite on one GPU, then merge.
#
#   CUDA_VISIBLE_DEVICES=3 ./run.sh            # results/<gpu>-sm<cc>/calibration.json
#   NCU=0 ./run.sh                              # skip the Nsight Compute reference set
#
# Each bench runs in its own process; a failing bench is logged and skipped,
# the rest still run.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"

gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
gpu_index="${gpu_index%%,*}"
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_index" | head -1)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$gpu_index" | head -1 | tr -d '.')
slug=$(echo "$name" | sed 's/NVIDIA //; s/[^A-Za-z0-9]\+/-/g; s/-$//' | tr 'A-Z' 'a-z')
out="results/${slug}-sm${cc}"
mkdir -p "$out/raw"
echo "GPU $gpu_index: $name (sm_$cc) -> $out"

make -s all SM="$cc" || exit 1

run() {
  local tag=$1
  shift
  printf '%-40s' "$tag"
  if "$@" >"$out/raw/$tag.json" 2>"$out/raw/$tag.log"; then
    echo ok
  else
    echo "FAILED (see $out/raw/$tag.log)"
    rm -f "$out/raw/$tag.json"
  fi
}

run alu ./build/bench_alu && python3 scripts/sass_check.py build/bench_alu build/bench_alu.ptx "$out/raw/alu.json" 2>>"$out/raw/alu.log"
run mem ./build/bench_mem
run misc ./build/bench_misc
run tma ./build/bench_tma
for c in $(./build/bench_tc --list); do
  run "tc_$c" ./build/bench_tc "$c"
done

python3 scripts/merge.py "$out"
if [ "${NCU:-1}" = 1 ] && command -v ncu >/dev/null; then
  python3 scripts/ncu_reference.py "$out"
fi
