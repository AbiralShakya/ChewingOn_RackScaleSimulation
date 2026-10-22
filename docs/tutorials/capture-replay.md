# 3. Capture an engine, replay it on a rack

**What you will do:** replay a kernel that was recorded on a real B300 (no GPU needed), then
learn the three-command loop that captures any CUDA program (vLLM, SGLang, a training loop, a
megakernel) on one GPU and replays it on any rack. **Time:** 5 minutes to read and replay; the
capture step needs one NVIDIA GPU.

## Why capture

The built-in models (`rackscale decode llama3-70b ...`) are *lowered*: rackscale builds the
kernels an engine would launch from the model's shapes. That is fast and needs no GPU, but it is
our model of the engine. A capture records what the engine *actually* launched: every kernel with
its grid, stream and event dependencies, and for one launch of each distinct kernel the exact
per-warp SASS path (lane addresses, mbarriers, TMA, `tcgen05` operands). Replaying a capture
answers "what would *this* engine, with *these* kernels, do on a rack I do not have?"

## Replay a real recording, no GPU needed

The repository includes NVBit recordings of FlashAttention kernels taken on a B300. Replay
FlashAttention-4 on the calibrated B300 model:

```bash
rackscale path research/flash_attention/rpath/b300_fa4_cp_c1024_kv8.rpath.gz --gpu b300 \
    --calibration research/flash_attention/calibration/b300-sxm6-ac-sm103.json --regs 128
```

```
kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100_...
  time        27.50 us   (55897 cycles, 148 CTAs, 1 CTA/SM)
  dram        21.0 MB read, 15.2 MB written; L2 hit 91.0%
  scheduler-cycle anatomy:
    selected              33.12%  #################
    wait                  22.51%  ###########
    async_wait            18.38%  #########
    math_pipe_throttle    10.42%  #####
    ...
```

This is the real kernel's instruction stream, warp by warp, not a synthetic stand-in. Open it in
the viewer to see FA4's warp roles (load, MMA, softmax, correction, epilogue) hand work to each
other through mbarriers:

```bash
rackscale viz path research/flash_attention/rpath/b300_fa4_cp_c1024_kv8.rpath.gz --gpu b300 \
    --calibration research/flash_attention/calibration/b300-sxm6-ac-sm103.json --regs 128 \
    --warp-sms 0 -o fa4.html
rackscale view fa4.html
```

## Capture your own program

Requires one NVIDIA GPU, NVBit 1.8+ and `cuobjdump` on `PATH`; build the tool once with
`capture/nvbit/build.sh`. The program is not modified.

```bash
# 1. record: launch graph, warp paths of every distinct kernel, native timing (NVBit + nsys)
rackscale capture -o cap/llama1b -- \
    python examples/capture/vllm_decode.py --model meta-llama/Llama-3.2-1B-Instruct --batch 32 --eager

# 2. replay on the GPU it ran on: how close is the simulator, kernel by kernel?
rackscale replay cap/llama1b --system hgx_b300_8 --steps 1: --kernels --fast

# 3. replay on a rack with a parallelism plan
rackscale replay cap/llama1b --system gb300_nvl72 --tp 8 --dp 9 --steps 1: --fast
```

- **What gets captured.** By default, the region between `rsim_capture.region()` blocks in your
  program. For unmodified programs use `--region forward:S:N` (N whole engine steps starting at
  the S-th forward), and `--region all` for standalone binaries.
- **Replay modes.** `--mode paths` (default) re-simulates every recorded kernel at cycle level on
  the target GPU. `--mode measured` keeps each kernel's measured time, so only overlap,
  dependencies and communication are simulated. Comparing the two isolates the kernel model.
- **Scaling.** The plan (`--tp/--ep/--pp/--dp`, or `--plan plan.yaml`) inserts collectives at the
  engine's own layer boundaries (`RowParallelLinear`, `FusedMoE`, ...), sized from the captured
  module outputs.

## How close it gets today

Be precise about this: on vLLM serving Llama-3.2-1B on a B300, replaying the captured streams with
measured kernel times matches the native CUDA-graph step within 0.3%; re-simulating every kernel
from its warp path is 18% slow, mostly because of one decode-attention kernel and one split-K GEMM
that do not replay well yet. ThunderKittens GEMMs replay within 6% at small sizes. The kernel-by-
kernel table, the known gaps and what is not modelled are in
[Capture and replay](../CAPTURE.md).

## No GPU at all?

A PyTorch-profiler trace is a lighter route: it has no warp paths, so GEMMs with recoverable
shapes are re-simulated and everything else keeps its measured duration.

```bash
rackscale capture rank0.pt.trace.json.gz --world 8 --group block:8 --system gb200_nvl72 --fast
```

Next: [chapter 10 of the guide](../book/10-bring-your-own-workload.md) walks through a capture
end to end.
