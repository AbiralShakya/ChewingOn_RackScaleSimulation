# Timing sweep + ncu + NVBit traces for one GPU. Usage: bash remote/run_gpu.sh <tag> <impls...>
. "$(dirname "$0")/env.sh"
TAG=$1; shift
IMPLS="$*"
R=$W/results/$TAG; mkdir -p $R
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.sm,memory.total --format=csv > $R/gpu.csv
cd $W/repo
# timing: S sweep at 16k tokens per batch (B*S = 16384), H=16, D=128, causal and not
for causal in "" "--causal"; do
  for S in 1024 2048 4096 8192 16384; do
    B=$((16384 / S))
    for i in $IMPLS; do
      timeout 600 python remote/fa_bench.py --impl $i --b $B --h 16 --s $S $causal 2>/dev/null | tail -1 >> $R/timing.jsonl
    done
  done
done
# ncu on the validation shape
for i in $IMPLS; do
  [ $i = naive ] && continue
  timeout 1200 python remote/ncu_attn.py --impl $i --b 4 --h 16 --s 4096 --out $R/ncu_${i}_s4096.json 2>&1 | tail -3
done
