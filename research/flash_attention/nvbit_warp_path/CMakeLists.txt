# Built inside an NVBit release: copy this directory to nvbit_release/tools/warp_path
# and add `add_subdirectory(warp_path)` to tools/CMakeLists.txt (see ../setup_della.sh).
cmake_minimum_required(VERSION 3.29)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(warp_path LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NVBitTool.cmake")
endif()
add_nvbit_tool(warp_path)
nvbit_add_tool_kernel(warp_path
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tool_func/flush_channel.cu"
  MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/warp_path.cu"
  BIN2C_NAME flush_channel_bin
)
