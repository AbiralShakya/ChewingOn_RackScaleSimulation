# Bring your own workload

rackscale can simulate any program that runs on one NVIDIA GPU. You run it once under
`rackscale capture`, which records what the GPU actually did, and then replay the capture on
any system YAML with a parallelism plan. The program is not modified and no model description
is written by hand: vLLM, SGLang, a training loop, a Triton or CuTe DSL kernel, a
ThunderKittens binary and a megakernel all go through the same path.

```bash
# 1. record (on a real GPU)
rackscale capture -o cap/llama1b --gpu-lock /tmp/gpu.lock -- \
    python examples/capture/vllm_decode.py --model meta-llama/Llama-3.2-1B-Instruct --batch 32 --eager

# 2. replay on the same GPU: how close is the simulator?
rackscale replay cap/llama1b --system hgx_b300_8 --steps 1: --kernels --fast

# 3. replay on a rack
rackscale replay cap/llama70b-tp8 --system gb300_nvl72 --tp 8 --dp 9 --steps 1: --fast
```

## What a capture holds

`rackscale capture` runs the program twice under the NVBit tool
[`capture/nvbit/rsim_capture`](../capture/nvbit/rsim_capture.cu):

1. **Graph pass.** Every kernel launch in the capture region with its function, grid, block,
   cluster, shared memory, registers and stream; every event record and stream wait; every copy
   and memset; synchronisations. Each call has its host time, and each kernel and copy its GPU
   start and end, from CUDA events that the tool records on the launch stream around it.
   Nothing is instrumented in this pass, so the GPU timeline is native apart from those events.
   A small site package that is put on `PYTHONPATH` also marks engine steps (forwards of the
   model) and module calls (qualified name, class, output size). The parallelism plan uses those
   marks to place collectives.
2. **Paths pass.** For the first in-region launch of every distinct kernel (function plus launch
   configuration), the per-warp dynamic SASS path of a few CTAs, with lane addresses, mbarrier
   operands, TMA source runs and tcgen05 instruction descriptors. This is the same recording as
   the FlashAttention and calibration studies use (`warp_path`). An 80-layer decode step
   therefore records each GEMM, attention and norm kernel once. CTAs that were not recorded
   replay a recorded path with per-dimension address strides. Persistent kernels replay every
   CTA's real TMA tile sequence.

The result is a directory:

```text
cap/llama1b/
  capture.json        processes -> regions -> ops (kernel, copy, record, wait, push/pop, step);
                      kernels -> name, grid, block, regs, smem, launch count, measured duration, path
  kernels/k0007.rpath.gz   one per recorded kernel (rsim path format)
  graph/ paths/            raw tool output (events.jsonl and recordings)
```

`rsim path cap/llama1b/kernels/k0007.rpath.gz --gpu b300` simulates one recorded kernel, and
`rsim viz path ...` opens it in the SM/warp/SASS viewer.

### The capture region

| `--region` | what is captured | use it for |
|---|---|---|
| `marked` (default) | between `rsim_capture.region()` blocks in the program | your own scripts |
| `forward:S:N[:REGEX]` | from the S-th forward of the model to the (S+N)-th: N whole engine steps, including sampling and input preparation between forwards | unmodified programs (`vllm serve`, a training script) |
| `armed:S:N[:REGEX]` | like `forward`, counted from when the program calls `rsim_capture.arm()` | engines whose model runs in a subprocess (SGLang) |
| `all` | the whole program | standalone binaries (ThunderKittens benchmarks) |

The model is the outermost `nn.Module` call with at least a million parameters, or the
outermost module whose class matches `REGEX`.

```python
try:
    from rsim_capture import region      # on PYTHONPATH only under rackscale capture
except ImportError:
    from contextlib import nullcontext as region

with region("decode"):
    llm.generate(prompts, params)
```

### Requirements and limits

- **Eager mode.** Kernels launched inside CUDA graphs are not expanded. Capture with graphs
  disabled (vLLM `enforce_eager=True`, SGLang `--disable-cuda-graph`). The kernels are the same
  as in graph mode for a given batch size, and replaying with `--launch-gap-ns` set to the
  graph-mode gap models the graph-mode step. The tool logs graph launches, and the build warns
  when there were any.
- **Deterministic launches.** The two passes are joined by kernel key, not by position, but the
  paths pass must launch the kernels that the graph pass saw. Eager PyTorch engines with a fixed
  seed and fixed inputs do.
- **The recorded instance stands for every launch with its key.** A decode attention kernel
  whose grid does not change from step to step is recorded once, at the first step. Use
  `--key func` to record each function only once, whatever its configuration.
- **Recording cost.** An instrumented launch runs 10 to 100 times slower. `--max-mb` drops
  recordings that are too large (the kernel keeps its measured duration), and `--filter` limits
  recording to matching kernel names.
- The tool needs NVBit 1.8+ and `cuobjdump` on `PATH`. Build it with `capture/nvbit/build.sh`.

## Replay

```bash
rackscale replay CAPTURE --system SYSTEM [--plan plan.yaml | --tp N --ep N --pp N --dp N]
                 [--mode paths|measured] [--host] [--steps A:B] [--kernels] [--fast]
```

- `--mode paths` (default) re-simulates every recorded kernel at cycle level on the target GPU
  and keeps the measured duration of kernels without a recording. `--mode measured` keeps every
  kernel's own measured duration, so only overlap, dependencies and communication are
  simulated. Comparing the two isolates the kernel model from the rest.
- `--host` replays the captured host issue time of every launch: a launch cannot start before
  the host issued it. Without it, the GPU runs the captured streams as fast as dependencies
  allow (the CUDA-graph view).
- `--steps 1:` drops step 0, which is the prefill in `vllm_decode.py`'s region.
- `--kernels` also simulates every recorded kernel alone and compares it with its measured
  median duration.

The Python API is `rackscale.replay.to_workload(cap, plan)` (returns an rsim workload you can
edit or simulate) and `rackscale.replay.replay(cap, system, plan)`.

## Scaling to a rack: the parallelism plan

A capture is what one GPU ran. The plan adds communication and replicates or splits that
program. It never changes a kernel:

```yaml
tp: 8            # all-reduce after every row-parallel module, inside each block of 8 ranks
ep: 1            # all-to-all dispatch before and combine after every MoE module
pp: 1            # layers split over stages; activations sent between stages
dp: 9            # replicas; with dp_allreduce_bytes, a gradient all-reduce at the end
microbatches: 1  # with pp > 1: microbatches in flight
topk: 8          # experts per token (all-to-all bytes = MoE output bytes x topk)
comm_sms: 16
collectives:     # optional: where the collectives go
  - {after: {cls: RowParallelLinear}, kind: all_reduce, group: tp, bytes: out}
```

The default rules cover vLLM and SGLang layer classes (`RowParallelLinear`,
`VocabParallelEmbedding`, `FusedMoE`, ...) and Hugging Face module names (`*.o_proj`,
`*.down_proj`). `bytes` is evaluated with `out` = the module's output size from the capture.
Ranks are laid out TP-innermost: `rank = (dp * pp + stage) * tp + tp_rank`.

**Capture one rank, not the whole model.** For tensor or expert parallelism, every rank runs
smaller kernels than a single-GPU deployment. Capture those kernels directly: run the engine on
one GPU with the per-rank shapes and dummy weights. `rackscale.replay.shard_overrides(config,
tp, ep)` returns the `hf_overrides` (heads, KV heads, MLP width and vocabulary divided by `tp`,
routed experts by `ep`), and the example scripts take `--shard-tp`, `--shard-ep` and `--shape`.
A Llama-3.1-70B TP=8 rank fits on one GPU this way (`--shape llama3-70b --shard-tp 8`), and the
replay inserts its all-reduces.

## What is and is not modelled

Modelled: every kernel the engine launched, at cycle level from its real SASS path (or at its
measured duration); the engine's stream structure and cross-stream dependencies; copies; the
host launch timeline (with `--host`); and on the rack, collectives over the simulated NVLink
and switch fabric, sharing SMs with compute.

Not modelled:

- **The scheduler and batching policy.** A capture is one fixed batch at one point in time.
  Continuous batching, admission, preemption and KV-cache management are engine behaviour; to
  study them, capture each batch shape you care about.
- **Host overheads beyond the captured timeline.** `--host` replays measured issue times; when
  scaling out, the replay assumes each rank's host keeps the same timeline. Python overhead,
  tokenisation and RPC between processes are inside those times, not modelled separately.
- **Expert load imbalance.** EP replay assumes balanced routing (dummy weights route roughly
  uniformly).
- **Collectives the engine itself launched** on one GPU (none at TP=1) are replaced by the
  plan's collectives.
- **Other architectures.** A recorded path is the SASS of the GPU it was captured on. To
  predict a different GPU generation, capture on it, or lower the model with `rackscale.lower`.

## Torch-profiler traces

The older route still works for traces from `torch.profiler` (vLLM `VLLM_TORCH_PROFILER_DIR`,
SGLang `/start_profile`): GEMMs with recoverable shapes become synthetic `gemm` kernels, NCCL
kernels become collectives, and everything else keeps its measured duration.

```python
from rackscale import capture
import rackscale as rs
wl = capture.from_file("rank0.pt.trace.json.gz", world=8, collective_group="block:8")
print(rs.simulate("gb200_nvl72", wl, fast=True).summary())
```

It needs no GPU access but has no cycle-level detail for kernels other than GEMMs.
