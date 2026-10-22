#!/bin/bash
# Build each kernels/*.cu to build/<name>.cubin and its SASS listing.
set -e
ARCH=${ARCH:-sm_103a}
mkdir -p build
for f in ${@:-kernels/*.cu}; do
  n=$(basename $f .cu)
  nvcc -cubin -arch=$ARCH -O3 -std=c++17 -diag-suppress 177 $NVFLAGS -o build/$n.cubin $f
  cuobjdump -sass build/$n.cubin > build/$n.sass
done
