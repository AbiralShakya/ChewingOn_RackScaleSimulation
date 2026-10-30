#!/usr/bin/env bash
# Build and run the whole calibration suite on one GPU, then merge.
#
#   CUDA_VISIBLE_DEVICES=3 ./run.sh            # results/<gpu>-sm<cc>/calibration.json
#   NCU=0 ./run.sh                              # skip the Nsight Compute reference set
#   CAL_OUT=/path/to/dir ./run.sh               # write somewhere other than results/<gpu>
#
# Each bench runs in its own process; a failing bench is logged and skipped,
# the rest still run.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"

gpu_index="${CUDA_VISIBLE_DEVICES:-0}"
gpu_index="${gpu_index%%,*}"
# A MIG UUID is not a valid `nvidia-smi -i`; inside the job only the parent GPU is visible.
[[ $gpu_index == MIG-* ]] && gpu_index=0
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$gpu_index" | head -1)
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$gpu_index" | head -1 | tr -d '.')
slug=$(echo "$name" | sed 's/NVIDIA //; s/[^A-Za-z0-9]\+/-/g; s/-$//' | tr 'A-Z' 'a-z')
# A MIG slice reports the parent GPU's name; tag the profile (e.g. -mig-3g-40gb).
mig=$(nvidia-smi -L | grep -oE 'MIG [0-9]+g\.[0-9]+gb' | head -1 | tr 'A-Z' 'a-z' | tr ' .' '--')
# CAL_OUT keeps a second GPU of a model already in results/ from overwriting it (the node day).
out="${CAL_OUT:-results/${slug}-sm${cc}${mig:+-$mig}}"
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
run window ./build/bench_window
# Arch-specific benches: the Makefile builds them only where the feature exists.
[ -x build/bench_tma ] && run tma ./build/bench_tma
[ -x build/bench_tmat ] && run tmat ./build/bench_tmat
run loaded ./build/bench_loaded
for c in $(./build/bench_tc --list); do
  run "tc_$c" ./build/bench_tc "$c"
done

for b in mma_sync cpasync l2part overlap open; do
  [ -x "build/bench_$b" ] && run "$b" "./build/bench_$b"
done

python3 scripts/merge.py "$out"
if [ "${NCU:-1}" = 1 ] && command -v ncu >/dev/null; then
  python3 scripts/ncu_reference.py "$out"
  # Second merge: model.launch.kernel_window needs the ncu cycles of the window kernels.
  python3 scripts/merge.py "$out"
fi
