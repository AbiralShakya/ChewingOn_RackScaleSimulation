# All B300 traces for the study (run under the GPU-3 lock).
set -x
. "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=3
T=$W/traces
S=${S:-4096}
WP_TMA_ALL=1 bash remote/trace.sh $T/b300_fa4_s$S FlashAttentionForwardSm100 0,1,2,3 --impl fa4 --b 4 --h 16 --s $S
bash remote/trace.sh $T/b300_fa2_s$S flash_fwd_kernel 0,1,32,128 --impl fa2 --b 4 --h 16 --s $S
