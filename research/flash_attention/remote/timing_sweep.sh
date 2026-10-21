# Timing sweep with back-to-back calls (host overhead hidden): B*S = 16k tokens, 16 heads, d=128.
. "$(dirname "$0")/env.sh"
TAG=$1; shift
R=$W/results/$TAG; mkdir -p $R
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv >> $R/gpu_state.csv
for causal in "" "--causal"; do for S in 1024 2048 4096 8192 16384; do B=$((16384 / S))
  for i in "$@"; do
    timeout 600 python remote/fa_bench.py --impl $i --b $B --h 16 --s $S $causal --iters 70 2>/dev/null | tail -1 >> $R/timing2.jsonl
  done
done; done
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit --format=csv >> $R/gpu_state.csv
