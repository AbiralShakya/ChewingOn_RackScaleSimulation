# Per-GPU ring-attention step kernels (Llama-3-70B attention: 64 q heads, 8 kv heads, d=128):
# one chunk of queries against one chunk of keys. Traced for the simulator, timed for validation.
. "$(dirname "$0")/env.sh"
TAG=$1; IMPL=$2; KSUB=$3; CTAS=${4:-0,1,2,3}
R=$W/results/$TAG; mkdir -p $R
for C in 1024 2048 4096; do
  for HKV in 8 64; do
    python remote/fa_bench.py --impl $IMPL --b 1 --h 64 --hkv $HKV --s $C --iters 70 2>/dev/null | tail -1 >> $R/timing_cp.jsonl
    timeout 900 python remote/ncu_attn.py --impl $IMPL --b 1 --h 64 --hkv $HKV --s $C --out $R/ncu_${IMPL}_cp_c${C}_kv${HKV}.json 2>&1 | tail -2
    # FA2 grid is (C/128, batch, heads): sample m-block 0 and 1 and the next head.
    [ "$CTAS" = auto ] && C2="0,1,$((C / 128))" || C2=$CTAS
    WP_TMA_ALL=1 bash remote/trace.sh $W/traces/${TAG}_${IMPL}_cp_c${C}_kv${HKV} "$KSUB" $C2 --impl $IMPL --b 1 --h 64 --hkv $HKV --s $C
  done
done
