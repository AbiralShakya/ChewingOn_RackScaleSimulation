# Sourced on the GPU node. Everything lives under $W (home quota is nearly full).
export W=${W:-/scratch/gpfs/MZHASAN/as0714/flashattn}
export PIP_CACHE_DIR=$W/cache/pip UV_CACHE_DIR=$W/cache/uv TMPDIR=$W/tmp XDG_CACHE_HOME=$W/cache
export TRITON_CACHE_DIR=$W/cache/triton CUTE_DSL_CACHE_DIR=$W/cache/cute
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-13.0}
export PATH=$CUDA_HOME/bin:$PATH
# NVBit needs cuobjdump; some Della compute nodes ship CUDA trees without it.
# Copies of cuobjdump/nvdisasm from della-tridao live in $W/bin.
command -v cuobjdump >/dev/null || export PATH=$PATH:$W/bin
export NCU=${NCU:-/opt/nvidia/nsight-compute/2026.2.1/ncu}
export NVBIT=$W/nvbit_dl/nvbit_release_x86_64
export WARP_PATH_SO=$NVBIT/tools/build/warp_path/warp_path.so
mkdir -p $W/cache $W/tmp
[ -f $W/venv/bin/activate ] && . $W/venv/bin/activate
export PYTHONPATH=$W/flash-attention:${PYTHONPATH:-}
