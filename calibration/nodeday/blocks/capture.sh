#!/usr/bin/env bash
# NVBit recordings of the real kernels behind each model step, on one GPU.
#
#   CUDA_VISIBLE_DEVICES=1 capture.sh <outdir>
#
# Each recording is a `rackscale capture` directory: the launch graph of the
# step and the per-warp SASS path of every distinct kernel. They answer the
# article plan's gap 3 (synthetic large GEMMs run 15-35% fast; replay the real
# cuBLAS / cuDNN / vLLM kernels instead) and can be replayed later on any
# system with `rackscale replay <dir> --system hgx_b300_8 --fast`.
#
#   train_L2      chapter 11's 1-GPU training step (forward + backward, 2 layers)
#   decode_8b     vLLM Llama-3-8B decode (4 layers: every distinct kernel), batch 64, 8K context,
#                 4 steps, no CUDA graphs, dummy weights
#   decode_70b    one TP8 rank's share of Llama-3-70B (the same decode, sharded shape)
#   decode_moe    one TP4 rank's share of Qwen3-30B-A3B
set -uo pipefail
OUT=${1:?outdir}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
ND_EXT=${ND_EXT:-$HOME/nodeday/ext}
export NVBIT=${NVBIT:-$ND_EXT/nvbit_release}
TOK=${NODEDAY_LLAMA_TOKENIZER:-NousResearch/Meta-Llama-3.1-8B}
mkdir -p "$OUT"
[ -f "$NVBIT/tools/build-rsim/rsim_capture.so" ] || { echo "NVBit tools not built ($NVBIT)"; exit 1; }
python -c "import rackscale" || { echo "rackscale not installed"; exit 1; }

cap() {  # cap <name> <region> -- cmd...
  local name=$1 region=$2
  shift 3
  [ -f "$OUT/$name/capture.json" ] && { echo "$name: done already"; return; }
  echo "== $name"
  timeout 5400 rackscale capture -o "$OUT/$name" --region "$region" -- "$@" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? ($(du -sh "$OUT/$name" 2>/dev/null | cut -f1))"
}

# One step, from the 6th forward on: past warm-up and cuBLAS's first heuristic picks.
cap train_L2 forward:5:1 -- python "$HERE/train_step.py" --plan single --layers 2 --warmup 5 --steps 2 --out "$OUT/train_L2.step.json"
cap decode_8b marked -- python "$REPO/examples/capture/vllm_decode.py" --model "$TOK" --layers 4 --batch 64 --prompt-len 8192 \
  --no-graphs --capture-steps 4
cap decode_70b marked -- python "$REPO/examples/capture/vllm_decode.py" --model "$TOK" --shape llama3-70b --layers 4 \
  --shard-tp 8 --batch 64 --prompt-len 8192 --no-graphs --capture-steps 4
cap decode_moe marked -- python "$REPO/examples/capture/vllm_decode.py" --model Qwen/Qwen3-30B-A3B --layers 4 \
  --shard-tp 4 --batch 64 --prompt-len 8192 --no-graphs --capture-steps 4
