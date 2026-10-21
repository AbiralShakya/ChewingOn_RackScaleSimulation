#!/usr/bin/env bash
# Build the measurement environment: torch (cu130) + state-spaces/mamba +
# Dao-AILab/causal-conv1d + Dao-AILab/flash-attention (FA2 for sm_80, FA4/CuTe for sm_100+).
# Usage: ROOT=/scratch/... ARCHS="8.0" ./setup_env.sh      (A100)
#        ROOT=/scratch/... ARCHS="10.0;10.3" ./setup_env.sh (B300)
set -euo pipefail
ROOT="${ROOT:?}"; ARCHS="${ARCHS:?}"
CUDA="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_HOME="$CUDA" PATH="$CUDA/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCHS" MAX_JOBS="${MAX_JOBS:-32}"
mkdir -p "$ROOT" && cd "$ROOT"
[ -d .venv ] || uv venv -p 3.12 .venv
. .venv/bin/activate
uv pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install -q numpy packaging ninja wheel setuptools einops triton nvidia-cutlass-dsl
for r in state-spaces/mamba Dao-AILab/causal-conv1d Dao-AILab/flash-attention; do
  d=$(basename $r); [ -d src/$d ] || git clone -q --depth 1 https://github.com/$r src/$d
done
(cd src/causal-conv1d && CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install -v --no-build-isolation . ) > build_causal_conv1d.log 2>&1
(cd src/mamba && MAMBA_FORCE_BUILD=TRUE uv pip install -v --no-build-isolation . ) > build_mamba.log 2>&1
python - <<'PY'
import torch, mamba_ssm, causal_conv1d
print("torch", torch.__version__, "mamba_ssm", mamba_ssm.__version__, "causal_conv1d", causal_conv1d.__version__)
PY
echo SETUP_DONE
