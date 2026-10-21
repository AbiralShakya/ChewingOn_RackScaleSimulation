# Build the warp_path NVBit tool inside the NVBit release tree.
set -e
. "$(dirname "$0")/env.sh"
rm -rf $NVBIT/tools/warp_path && cp -r $W/repo/nvbit_warp_path $NVBIT/tools/warp_path
grep -q "add_subdirectory(warp_path)" $NVBIT/tools/CMakeLists.txt || echo "add_subdirectory(warp_path)" >> $NVBIT/tools/CMakeLists.txt
cd $NVBIT/tools
cmake -S . -B build -DNVBIT_TOOL_CUDA_ARCHITECTURES="${NVBIT_ARCHS:-80;90;100;103}" > /dev/null
cmake --build build --target warp_path -j 16 2>&1 | grep -E "error|warning: unused|Built target|Error" | head -30
ls -la $WARP_PATH_SO
