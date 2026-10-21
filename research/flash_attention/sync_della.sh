#!/usr/bin/env bash
# Copy this directory to della-tridao and run a command there (default: build the NVBit tool).
#   ./sync_della.sh                       # sync + build nvbit_warp_path
#   ./sync_della.sh 'bash remote/run_b300.sh'
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
W="${W:-/scratch/gpfs/MZHASAN/as0714/flashattn}"
COPYFILE_DISABLE=1 tar --no-xattrs -C "$HERE" --exclude "._*" --exclude results --exclude '*.wp' -czf - . |
  ssh della "ssh della-tridao 'mkdir -p $W/repo && tar -C $W/repo -xzf -'"
CMD="${*:-bash remote/build_nvbit_tool.sh}"
ssh della "ssh della-tridao 'cd $W/repo && $CMD'"
