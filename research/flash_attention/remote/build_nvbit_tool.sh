# Build the warp_path NVBit tool (now in capture/nvbit, with rsim_capture).
# Needs the repository checked out at $W/repo-full (or set REPO).
set -e
. "$(dirname "$0")/env.sh"
NVBIT=$NVBIT bash ${REPO:-$W/repo-full}/capture/nvbit/build.sh
