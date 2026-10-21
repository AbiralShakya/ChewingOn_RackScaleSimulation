# NVBit per-warp path trace of the timed call of one implementation.
#   bash remote/trace.sh <outdir> <kernel-substring> <ctas> fa_bench args...
. "$(dirname "$0")/env.sh"
OUT=$1; KSUB=$2; CTAS=$3; shift 3
mkdir -p $OUT
WP_OUT=$OUT WP_KERNEL="$KSUB" WP_LAUNCH=${WP_LAUNCH:-1} WP_CTAS=$CTAS LD_PRELOAD=$WARP_PATH_SO \
  python $(dirname "$0")/fa_bench.py --once "$@" 2>&1 | grep -E "WARP_PATH|Error|error|ERROR" | grep -v -E "launch void at::|launch .*distribution" 
ls -la $OUT
