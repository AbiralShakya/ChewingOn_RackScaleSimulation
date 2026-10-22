#!/bin/bash
# Build warp_path.so and rsim_capture.so against an NVBit release (1.8+).
#   NVBIT=<nvbit_release_x86_64> capture/nvbit/build.sh
# The sources are copied to $NVBIT/tools/rsim and built standalone into
# $NVBIT/tools/build-rsim/{warp_path,rsim_capture}.so, leaving the release's
# own tools/ build alone.
set -e
: "${NVBIT:?set NVBIT to the NVBit release directory}"
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$NVBIT/tools/rsim" && cp -r "$HERE" "$NVBIT/tools/rsim"
cmake -S "$NVBIT/tools/rsim" -B "$NVBIT/tools/build-rsim" -DNVBIT_TOOL_CUDA_ARCHITECTURES="${NVBIT_ARCHS:-80;90;100;103}" > /dev/null
cmake --build "$NVBIT/tools/build-rsim" -j "${JOBS:-16}" 2>&1 | grep -E "error|Built target|Error" | head -40
ls -la "$NVBIT"/tools/build-rsim/*.so
