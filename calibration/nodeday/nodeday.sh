#!/usr/bin/env bash
# One day on an 8-GPU node: every measurement the article needs, in priority order.
#
#   ./nodeday.sh setup        # core installs (~20 min); rerunnable
#   ./nodeday.sh smoke        # ~5 min: is every block ready? what will be skipped?
#   ./nodeday.sh run          # every block in priority order; finished blocks are skipped
#   ./nodeday.sh run decode   # just the named blocks
#   ./nodeday.sh status       # done / failed / time used per block
#   ./nodeday.sh pack         # results tarball to copy off the node
#
# The order, time budget and what each block feeds are in README.md. Every
# block writes to $ND_OUT/<block>/ and a DONE or FAILED marker; a failed block
# never stops the run. Optional extras (H-Net, FlashAttention, DeepEP, NVBit,
# the rackscale package) build in the background at low priority while the
# first blocks run, and the blocks that need them wait for that build.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
B="$HERE/blocks"
ND_HOME=${ND_HOME:-$HOME/nodeday}
ND_EXT=${ND_EXT:-$ND_HOME/ext}
ND_VENV=${ND_VENV:-$ND_HOME/venv}
mkdir -p "$ND_EXT"

# ---------------------------------------------------------------------------
# environment

[ -d "$ND_VENV" ] && . "$ND_VENV/bin/activate"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
NGPU=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 0)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)
case "$GPU_NAME" in
  *B300*|*"GB300"*) GPU_KEY=b300 ;;
  *B200*|*"GB200"*) GPU_KEY=b200 ;;
  *H100*|*H200*|*H800*) GPU_KEY=h100 ;;
  *) GPU_KEY=other ;;
esac
SLUG=$(echo "$GPU_NAME" | sed 's/NVIDIA //; s/[^A-Za-z0-9]\+/-/g; s/-$//' | tr 'A-Z' 'a-z')
ND_OUT=${ND_OUT:-$ND_HOME/results/$(hostname -s)-${SLUG:-nogpu}}
mkdir -p "$ND_OUT"
ALL_GPUS=$(seq -s, 0 $((NGPU - 1)))
first() { seq -s, 0 $(($1 - 1)); }
NCCL_TESTS=$ND_EXT/nccl-tests/build
# nvidia.nccl is a namespace package (no __file__): the wheel's directory is __path__[0].
nccl_lib() { python -c "import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], 'lib'))" 2>/dev/null; }

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------------------------------------------------------------------------
# setup

setup_core() {
  log "core setup into $ND_VENV"
  [ -d "$ND_VENV" ] || python3 -m venv "$ND_VENV"
  . "$ND_VENV/bin/activate"
  pip install -q --upgrade pip
  # vLLM pins a torch build that matches it; everything else uses that torch.
  pip install -q vllm nvidia-ml-py huggingface_hub datasets numpy
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
  local nccl_home
  nccl_home=$(dirname "$(nccl_lib)")
  if [ ! -x "$NCCL_TESTS/all_reduce_perf" ]; then
    log "nccl-tests against torch's NCCL ($nccl_home)"
    [ -d "$ND_EXT/nccl-tests" ] || git clone -q https://github.com/NVIDIA/nccl-tests "$ND_EXT/nccl-tests"
    # pip's NCCL ships libnccl.so.2 only; nccl-tests links -lnccl.
    ln -sf "$nccl_home/lib/libnccl.so.2" "$nccl_home/lib/libnccl.so"
    make -s -C "$ND_EXT/nccl-tests" -j MPI=0 NCCL_HOME="$nccl_home" CUDA_HOME="$CUDA_HOME" > "$ND_EXT/nccl-tests.build.log" 2>&1 \
      || { log "nccl-tests build failed (see $ND_EXT/nccl-tests.build.log)"; }
  fi
  log "core setup done"
}

setup_extras() {
  # Each extra is independent and best effort; a failure only skips its block.
  . "$ND_VENV/bin/activate"
  local L="$ND_EXT/extras.log"
  : > "$L"
  {
    log "rackscale package (rackscale capture, NVBit converter)"
    command -v cargo > /dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q && . "$HOME/.cargo/env"; }
    pip install -q maturin && (cd "$REPO" && maturin develop --release -q) || log "rackscale build failed"

    log "NVBit (latest release) and the rsim capture tools"
    if [ ! -d "$ND_EXT/nvbit_release" ]; then
      url=$(curl -s https://api.github.com/repos/NVlabs/NVBit/releases/latest | grep -o 'https://[^"]*x86_64[^"]*\.tar\.bz2' | head -1)
      [ -n "$url" ] && curl -sL "$url" | tar xj -C "$ND_EXT" && mv "$ND_EXT"/nvbit_release* "$ND_EXT/nvbit_release" 2>/dev/null
    fi
    [ -d "$ND_EXT/nvbit_release" ] && NVBIT="$ND_EXT/nvbit_release" bash "$REPO/capture/nvbit/build.sh" || log "NVBit tools not built"

    log "FlashAttention 2 (wheel if one matches, else source build) and FlashAttention 4 (CuTe DSL)"
    pip install -q flash-attn --no-build-isolation || log "flash-attn (FA2) not installed"
    pip install -q "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=flash_attn/cute" \
      || pip install -q flash-attn-cute || log "FA4 (flash_attn.cute) not installed"

    log "H-Net (goombalab/hnet) and its kernels"
    [ -d "$ND_EXT/hnet" ] || git clone -q https://github.com/goombalab/hnet "$ND_EXT/hnet"
    pip install -q omegaconf einops optree regex
    pip install -q --no-build-isolation "causal_conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git" \
      "mamba_ssm @ git+https://github.com/state-spaces/mamba.git" || log "mamba_ssm / causal_conv1d not installed"

    log "DeepEP (intranode and low-latency tests)"
    if [ ! -d "$ND_EXT/DeepEP" ]; then
      git clone -q https://github.com/deepseek-ai/DeepEP "$ND_EXT/DeepEP"
    fi
    pip install -q nvidia-nvshmem-cu12 nvidia-nvshmem-cu13 2>/dev/null
    (cd "$ND_EXT/DeepEP" && NVSHMEM_DIR=$(python -c "import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])" 2>/dev/null) \
      pip install -q --no-build-isolation .) || log "DeepEP not installed"

    log "nvbandwidth (per-pair copy bandwidth)"
    if [ ! -x "$ND_EXT/nvbandwidth/nvbandwidth" ]; then
      [ -d "$ND_EXT/nvbandwidth" ] || git clone -q https://github.com/NVIDIA/nvbandwidth "$ND_EXT/nvbandwidth"
      (cd "$ND_EXT/nvbandwidth" && cmake . > /dev/null && make -s -j) || log "nvbandwidth not built (needs libboost-program-options-dev)"
    fi
    log "extras done"
  } >> "$L" 2>&1
  touch "$ND_EXT/EXTRAS_DONE"
}

# The extras compile on the host CPU. Blocks whose timing depends on the host
# (vLLM's per-step CPU work, eager PyTorch launches) pause that build.
pause_extras() { [ -f "$ND_EXT/extras.pid" ] && kill -STOP -- -"$(cat "$ND_EXT/extras.pid")" 2>/dev/null; }
resume_extras() { [ -f "$ND_EXT/extras.pid" ] && kill -CONT -- -"$(cat "$ND_EXT/extras.pid")" 2>/dev/null; }

wait_extras() {
  [ -f "$ND_EXT/EXTRAS_DONE" ] && return
  log "waiting for the background extras build (tail $ND_EXT/extras.log)"
  while [ ! -f "$ND_EXT/EXTRAS_DONE" ]; do sleep 20; done
}

# ---------------------------------------------------------------------------
# blocks. Each runs in $ND_OUT/<name> as its working directory.

block_env() {
  nvidia-smi > nvidia-smi.txt
  nvidia-smi -q > nvidia-smi-q.txt
  nvidia-smi topo -m > topo.txt
  nvidia-smi nvlink -s > nvlink-status.txt 2>&1
  nvidia-smi nvlink -e > nvlink-errors-start.txt 2>&1
  nvidia-smi -q -d CLOCK,POWER,PERFORMANCE,TEMPERATURE,ECC > clocks-power.txt
  nvidia-smi --query-gpu=index,name,uuid,serial,pci.bus_id,vbios_version,power.limit,clocks.max.sm,clocks.max.mem,temperature.gpu --format=csv > gpus.csv
  systemctl status nvidia-fabricmanager > fabricmanager.txt 2>&1
  lscpu > lscpu.txt; numactl -H > numa.txt 2>&1; free -g > mem.txt; uname -a > uname.txt
  cat /proc/driver/nvidia/version > driver.txt 2>&1
  nvcc --version > nvcc.txt 2>&1
  pip freeze > pip-freeze.txt 2>&1
  python - > versions.json <<'EOF'
import json, torch
v = dict(torch=torch.__version__, cuda=torch.version.cuda, nccl=".".join(map(str, torch.cuda.nccl.version())),
         gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         sms=torch.cuda.get_device_properties(0).multi_processor_count)
for m in ("vllm", "flash_attn", "deep_ep", "mamba_ssm"):
    try:
        v[m] = __import__(m).__version__
    except Exception as e:
        v[m] = None
print(json.dumps(v, indent=1))
EOF
  (cd "$REPO" && git rev-parse HEAD > "$PWD/repo-commit.txt" 2>/dev/null)
  echo "$GPU_KEY" > gpu-key.txt
}

nccl_run() {  # nccl_run <tag> <ngpus> <binary> [ENV=val ...]
  local tag=$1 n=$2 bin=$3
  shift 3
  env CUDA_VISIBLE_DEVICES="$(first "$n")" LD_LIBRARY_PATH="$(nccl_lib):${LD_LIBRARY_PATH:-}" "$@" \
    "$NCCL_TESTS/$bin" -b 8 -e 8G -f 2 -g "$n" -n 20 -w 5 -c 0 > "$tag.txt" 2>&1
  printf '  %-36s %s\n' "$tag" "$(grep -m1 'Avg bus bandwidth' "$tag.txt" || echo FAILED)"
}

block_nccl() {
  [ -x "$NCCL_TESTS/all_reduce_perf" ] || { echo "nccl-tests not built"; return 1; }
  # The algorithm NCCL picks per size, for the record (one run with tuning logged).
  nccl_run tuning_all_reduce_n8 8 all_reduce_perf NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING
  for coll in all_reduce all_gather reduce_scatter alltoall broadcast sendrecv; do
    for n in 8 4 2; do
      [ "$n" -le "$NGPU" ] && nccl_run "${coll}_n${n}" "$n" "${coll}_perf"
    done
  done
  for algo in Ring Tree NVLS; do nccl_run "all_reduce_n8_${algo}" 8 all_reduce_perf NCCL_ALGO=$algo; done
  for coll in all_gather reduce_scatter; do
    for algo in Ring NVLS; do nccl_run "${coll}_n8_${algo}" 8 "${coll}_perf" NCCL_ALGO=$algo; done
  done
}

block_variance() {
  python "$B/variance.py" --mode isolated --out isolated.json
  python "$B/variance.py" --mode concurrent --out concurrent.json
}

block_straggler() {
  torchrun --nproc-per-node "$NGPU" "$B/straggler.py" --out straggler.json
}

block_train() {
  python - "$HERE" <<'EOF' > plans.txt
import sys; sys.path.insert(0, sys.argv[1])
from cases import TRAIN
for c in TRAIN:
    for l in c["layers"]:
        n = 1 if c["script"] == "single" else 2 if c["script"].startswith("pp_") else 8
        print(c["script"], l, n)
EOF
  while read -r plan layers n; do
    local out="${plan}_L${layers}"
    [ -f "$out.json" ] && continue
    if [ "$n" = 1 ]; then
      python "$B/train_step.py" --plan "$plan" --layers "$layers" --out "$out.json" --trace "$out.trace.json"
    else
      torchrun --nproc-per-node "$n" "$B/train_step.py" --plan "$plan" --layers "$layers" --out "$out.json" --trace "$out.trace.json"
    fi
    gzip -f "$out.trace.json" 2>/dev/null
  done < plans.txt
}

block_decode() {
  local skip=""
  [ "$GPU_KEY" = h100 ] && skip="--skip h100"
  python "$B/decode.py" --all --gpus "$NGPU" $skip --outdir .
}

block_overlap() {
  torchrun --nproc-per-node "$NGPU" "$B/overlap.py" --alone --out R0.json
  for r in 2 4 8 16 24 32; do
    NCCL_MIN_CTAS=$r NCCL_MAX_CTAS=$r torchrun --nproc-per-node "$NGPU" "$B/overlap.py" --out "R$r.json"
  done
}

block_power() {
  python "$B/power.py" --gpus 0 --out gpu0.json
  python "$B/power.py" --gpus "$ALL_GPUS" --out all.json
}

block_links() {
  # Every GPU pair at 1 GiB (send/recv), so one slow link or plane shows up.
  for i in $(seq 0 $((NGPU - 1))); do
    for j in $(seq $((i + 1)) $((NGPU - 1))); do
      CUDA_VISIBLE_DEVICES=$i,$j LD_LIBRARY_PATH="$(nccl_lib):${LD_LIBRARY_PATH:-}" \
        "$NCCL_TESTS/sendrecv_perf" -b 1G -e 1G -g 2 -n 20 -w 5 -c 0 > "pair_${i}_${j}.txt" 2>&1
    done
  done
  grep -H "Avg bus bandwidth" pair_*.txt | sed 's/.txt:# Avg bus bandwidth//' > pairs.txt
  [ -x "$ND_EXT/nvbandwidth/nvbandwidth" ] && "$ND_EXT/nvbandwidth/nvbandwidth" -j > nvbandwidth.json 2>&1
  # NCCL protocols forced at 8 GPUs (LL, LL128, Simple), all-reduce.
  for p in LL LL128 Simple; do nccl_run "all_reduce_n8_proto_${p}" 8 all_reduce_perf NCCL_PROTO=$p; done
}

block_deepep() {
  wait_extras
  python -c "import deep_ep" 2>/dev/null || { echo "DeepEP not installed"; return 1; }
  (cd "$ND_EXT/DeepEP" && timeout 1200 python tests/test_intranode.py) > intranode.txt 2>&1
  (cd "$ND_EXT/DeepEP" && timeout 1200 python tests/test_low_latency.py) > low_latency.txt 2>&1
  tail -5 intranode.txt low_latency.txt
}

# Single-GPU work, one job per GPU at the same time: H-Net (GPU 0), NVBit
# recordings (GPU 1), the calibration suite on a fresh GPU (GPU 2), the open
# microbenchmarks and FA2 per-PC stalls (GPU 3).
block_fanout() {
  wait_extras
  (CUDA_VISIBLE_DEVICES=0 python "$B/hnet_bench.py" --hnet-repo "$ND_EXT/hnet" --out hnet.json > hnet.log 2>&1; echo "hnet rc=$?") &
  (CUDA_VISIBLE_DEVICES=1 bash "$B/capture.sh" "$PWD/capture" > capture.log 2>&1; echo "capture rc=$?") &
  (cd "$REPO/calibration" && CUDA_VISIBLE_DEVICES=2 CAL_OUT="$ND_OUT/fanout/calibration" ./run.sh > "$ND_OUT/fanout/calibration.log" 2>&1; echo "calibration rc=$?") &
  (CUDA_VISIBLE_DEVICES=3 bash "$B/micro.sh" "$PWD/micro" > micro.log 2>&1; echo "micro rc=$?") &
  wait
}

block_variance_end() {
  python "$B/variance.py" --mode isolated --reps 30 --out isolated_end.json
  nvidia-smi nvlink -e > nvlink-errors-end.txt 2>&1
}

# name priority minutes, in run order. P0 is the article's rung-4 and rung-5
# evidence; P1 fills rungs 2 and 3; P2 is the kernel-level and drift work.
BLOCKS="env:P0:2 nccl:P0:30 variance:P0:20 straggler:P0:10 train:P0:40 decode:P0:90
overlap:P1:20 power:P1:35 links:P1:25 fanout:P1:90 deepep:P2:20 variance_end:P2:10"

do_block() {
  local name=$1 dir="$ND_OUT/$1"
  mkdir -p "$dir"
  if [ -f "$dir/DONE" ]; then log "$name: done already"; return; fi
  rm -f "$dir/FAILED"
  log "$name: start"
  local t0=$SECONDS rc
  (cd "$dir" && "block_$name") 2>&1 | tee -a "$dir/log.txt"
  rc=${PIPESTATUS[0]}
  local dt=$((SECONDS - t0))
  echo "$name,$rc,$dt,$(date -Iseconds)" >> "$ND_OUT/timeline.csv"
  if [ "$rc" = 0 ]; then touch "$dir/DONE"; log "$name: done in $((dt / 60)) min"
  else echo "$rc" > "$dir/FAILED"; log "$name: FAILED rc=$rc after $((dt / 60)) min (continuing)"; fi
}

cmd_run() {
  local names=("$@")
  if [ ${#names[@]} = 0 ]; then
    for b in $BLOCKS; do names+=("${b%%:*}"); done
  fi
  # Extras build in the background, at the lowest CPU priority, while the
  # first blocks run (the GPU timings do not depend on host CPU).
  if [ ! -f "$ND_EXT/EXTRAS_DONE" ] && ! pgrep -f "nodeday.sh extras" > /dev/null; then
    log "starting the extras build in the background (log: $ND_EXT/extras.log)"
    setsid nohup nice -n 19 "$0" extras > /dev/null 2>&1 &
    echo $! > "$ND_EXT/extras.pid"
  fi
  for n in "${names[@]}"; do
    case "$n" in train|decode|overlap) pause_extras ;; esac
    do_block "$n"
    resume_extras
  done
  cmd_status
}

cmd_status() {
  printf '%-14s %-4s %-6s %-8s %s\n' block prio budget status minutes
  for b in $BLOCKS; do
    IFS=: read -r name prio mins <<< "$b"
    local st="-" used=""
    [ -f "$ND_OUT/$name/DONE" ] && st=done
    [ -f "$ND_OUT/$name/FAILED" ] && st=FAILED
    used=$(grep "^$name," "$ND_OUT/timeline.csv" 2>/dev/null | tail -1 | cut -d, -f3)
    printf '%-14s %-4s %-6s %-8s %s\n' "$name" "$prio" "$mins" "$st" "${used:+$((used / 60))}"
  done
  echo "results: $ND_OUT"
}

cmd_smoke() {
  log "GPUs: $NGPU x $GPU_NAME (key $GPU_KEY)"
  [ "$NGPU" -ge 8 ] || log "WARNING: fewer than 8 GPUs; multi-GPU blocks run at $NGPU"
  python -c "import torch, pynvml; assert torch.cuda.is_available(); print('torch', torch.__version__)" || log "MISSING torch / nvidia-ml-py: run setup"
  [ -x "$NCCL_TESTS/all_reduce_perf" ] && log "nccl-tests ok" || log "MISSING nccl-tests: run setup"
  command -v vllm > /dev/null && log "vllm ok" || log "MISSING vllm (decode block will fail)"
  for m in flash_attn flash_attn.cute deep_ep mamba_ssm rackscale; do
    python -c "import $m" 2>/dev/null && log "$m ok" || log "optional: $m not importable yet"
  done
  [ -f "$ND_EXT/nvbit_release/tools/build-rsim/rsim_capture.so" ] && log "NVBit tools ok" || log "optional: NVBit tools not built yet"
  log "10-second checks"
  local t="$ND_OUT/smoke"
  mkdir -p "$t"
  python "$B/train_step.py" --plan single --layers 2 --warmup 1 --steps 2 --out "$t/single.json" || log "FAIL train single"
  torchrun --nproc-per-node 2 "$B/train_step.py" --plan pp_zb --layers 2 --warmup 1 --steps 2 --out "$t/pp.json" || log "FAIL train pp_zb"
  torchrun --nproc-per-node "$NGPU" "$B/train_step.py" --plan tp --layers 2 --warmup 1 --steps 2 --out "$t/tp.json" || log "FAIL train tp"
  CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="$(nccl_lib):${LD_LIBRARY_PATH:-}" "$NCCL_TESTS/all_reduce_perf" -b 1M -e 1M -g 2 -n 5 > "$t/nccl.txt" 2>&1 \
    && log "nccl-tests run ok" || log "FAIL nccl-tests run"
  python "$B/power.py" --gpus 0 --seconds 0.5 --out "$t/power.json" | tail -3
  log "smoke done; see $t"
}

cmd_pack() {
  local tgz="$ND_HOME/$(basename "$ND_OUT").tar.gz"
  tar czf "$tgz" -C "$(dirname "$ND_OUT")" "$(basename "$ND_OUT")"
  log "wrote $tgz ($(du -h "$tgz" | cut -f1)). Copy it off the node, then: python calibration/nodeday/compare.py <dir>"
}

case "${1:-}" in
  setup) setup_core ;;
  extras) setup_extras ;;
  smoke) cmd_smoke ;;
  run) shift; cmd_run "$@" ;;
  status) cmd_status ;;
  pack) cmd_pack ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
