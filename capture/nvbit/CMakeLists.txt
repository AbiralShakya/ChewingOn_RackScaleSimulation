# NVBit tools that record kernels for rsim (see ../README.md). build.sh copies
# this directory into an NVBit release (tools/rsim) and builds it standalone.
cmake_minimum_required(VERSION 3.29)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(rsim_nvbit LANGUAGES CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/NVBitTool.cmake")
endif()

# warp_path: the per-warp path of one chosen kernel launch.
add_nvbit_tool(warp_path SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/warp_path.cu")
nvbit_add_tool_kernel(warp_path
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tool_func/flush_channel.cu"
  MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/warp_path.cu"
  BIN2C_NAME flush_channel_bin
)

# rsim_capture: the launch graph of a whole program, and the path of every distinct kernel.
add_nvbit_tool(rsim_capture SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/rsim_capture.cu")
nvbit_add_tool_kernel(rsim_capture
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/tool_func/flush_channel.cu"
  MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/rsim_capture.cu"
  OUTPUT_NAME flush_channel_rc
  BIN2C_NAME flush_channel_bin
)
