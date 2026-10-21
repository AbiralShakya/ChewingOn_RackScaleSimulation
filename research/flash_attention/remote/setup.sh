# One-time environment on the GPU node: Python venv with torch (CUDA 13), Dao-AILab
# flash-attention (FA2 built for sm_80/90/100, FA4 CuTe DSL), and NVBit 1.8.1.
# Everything lives under $W: Della home quotas are too small for builds and caches.
set -ex
. "$(dirname "$0")/env.sh"
cd $W
UV=${UV:-uv}
[ -d venv ] || $UV venv -p 3.12 venv
. venv/bin/activate
$UV pip install torch --index-url https://download.pytorch.org/whl/cu130
$UV pip install numpy packaging ninja einops psutil setuptools wheel
[ -d flash-attention ] || git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init csrc/cutlass
# FA2 (csrc/flash_attn): sm_100f code also runs on sm_103 (B300).
FLASH_ATTN_CUDA_ARCHS="80;90;100" MAX_JOBS=48 NVCC_THREADS=2 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  python setup.py build_ext --inplace
# FA4 (flash_attn/cute, CuTe DSL; JIT-compiled on first call)
$UV pip install -e "flash_attn/cute[cu13]"
cd $W
mkdir -p nvbit_dl && cd nvbit_dl
[ -d nvbit_release_x86_64 ] || { curl -sL -o nvbit.tar.bz2 \
  https://github.com/NVlabs/NVBit/releases/download/v1.8.1/nvbit-Linux-x86_64-1.8.1.tar.bz2 && tar xjf nvbit.tar.bz2; }
# Some compute nodes lack cuobjdump/nvdisasm, which NVBit needs: keep copies in $W/bin.
mkdir -p $W/bin && cp $CUDA_HOME/bin/cuobjdump $CUDA_HOME/bin/nvdisasm $W/bin/
