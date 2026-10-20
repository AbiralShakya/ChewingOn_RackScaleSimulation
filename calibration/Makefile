# Build every calibration bench for the local GPU.
# ARCH defaults to the compute capability of GPU 0 visible to this process,
# with the arch-specific "a" suffix (needed for tcgen05/TMEM on sm_100+).
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CC_RAW    := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i $${CUDA_VISIBLE_DEVICES:-0} 2>/dev/null | head -1 | tr -d '.')
SM        ?= $(CC_RAW)
SUFFIX    := $(if $(filter 90 100 101 103 110 120 121,$(SM)),a,)
ARCH      ?= -gencode arch=compute_$(SM)$(SUFFIX),code=sm_$(SM)$(SUFFIX)
NVFLAGS   ?= -O3 -std=c++17 -lineinfo $(ARCH) -Xptxas -v
BUILD     := build

BENCHES := alu mem tc tma misc
BINS    := $(addprefix $(BUILD)/bench_,$(BENCHES))

all: $(BINS) $(addsuffix .ptx,$(BINS))

$(BUILD)/bench_%: src/bench_%.cu src/common.cuh | $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ $< -lcuda 2> $@.ptxas.log || (cat $@.ptxas.log; false)

# PTX next to each binary: scripts/sass_check.py compares PTX and SASS counts.
$(BUILD)/bench_%.ptx: src/bench_%.cu src/common.cuh | $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=compute_$(SM)$(SUFFIX) -ptx -o $@ $<

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
