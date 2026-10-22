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

## What it reproduces today (B300, October 2026)

All runs below are on one B300 SXM6 AC (sm_103, driver 610, CUDA 13), captured with
`rackscale capture` and replayed with `rackscale replay`. "Native" means Nsight Systems on the
unmodified program. Scripts: `examples/capture/`.

**vLLM 0.32, Llama-3.2-1B, batch 32, decode at ~520 tokens of context.** Captured with
`torch.compile` and without CUDA graphs (`vllm_decode.py --no-graphs`), so the capture has the
kernels vLLM serves; the reference is a native CUDA-graph run of the same script
(`--cuda-graph-trace=node`, `compare_graph_run.py`).

| | per decode step |
|---|---|
| native, CUDA graphs: kernels / GPU span / busy | 193 / 942 µs / 917 µs |
| replay, graph-run kernel durations (checks the captured streams and dependencies) | 945 µs (+0.3%) |
| replay, every kernel re-simulated from its warp path | 1112 µs (+18%) |
| native end-to-end step (`--measure`, includes the host between graph launches) | 2143 µs |

Kernel by kernel against the graph run (stream timing plus the measured 0.31 µs stream
window, see `compare_graph_run.py`):

| kernel | per step | native µs | simulated µs | error |
|---|---|---|---|---|
| cuBLAS `nvjet` 128x32 4-CTA cluster (QKV, gate/up) | 16 | 12.86 | 11.88 | −7.7% |
| TRT-LLM `fmhaSm100f` decode attention | 16 | 10.82 | 24.28 | +124% |
| cuBLAS `nvjet` split-K (down projection) | 16 | 10.43 | does not replay yet | |
| cuBLAS `nvjet` 64x8 4-CTA cluster (o_proj) | 16 | 5.22 | 4.21 | −19% |
| cuBLAS `nvjet` 64x16 2-CTA pairs | 16 | 5.09 | 4.30 | −16% |
| Inductor `triton_red_fused_add_rms_norm` | 16 + 16 | 2.94, 2.37 | 2.30, 2.23 | −22%, −6% |
| cuBLAS split-K reduce | 16 | 2.66 | 1.65 | −38% |
| vLLM `reshape_and_cache_flash` | 16 | 2.37 | 2.24 | −5.6% |
| Inductor `triton_poi_fused` (two kinds) | 31 | 1.54, 1.28 | 1.68, 1.40 | +9% |

Time-weighted mean |error| over the 82% of kernel time that replays: 39%. The two outliers
are gaps that the capture exposed: the TRT-LLM decode attention runs 2.2x slow in the SM
model (256 CTAs at one per SM, so about twice the measured time per CTA), and the split-K
GEMM's barrier protocol is not converted correctly yet. Kernels that do not replay
keep their measured duration.

**ThunderKittens bf16 GEMM** (`kernels/gemm/bf16_b200` built for sm_103; persistent, 2-CTA
clusters), the benchmark binary captured whole (`--region all`):

| M=N=K | native µs | simulated µs | error |
|---|---|---|---|
| 1024 | 6.30 | 5.94 | −5.7% |
| 2048 | 14.62 | 14.17 | −3.1% |
| 4096 | 85.8 | 159.1 | +85% (the L2 model misses 38% of a 64 MB working set) |
| 8192 | 603 | 747 | +24% |
| 16384 | 6741 | recording over the 2 GB cap | |

**Llama-3.1-70B, one TP=8 rank** (vLLM, eager, dummy weights, `--shape llama3-70b --shard-tp 8`):
10,193 launches of 50 distinct kernels per prefill and 8 decode steps; native decode step with
CUDA graphs 6.74 ms. Scaled with `rack_scale.py`, one decode step at batch 32 (two steps
replayed, kernels re-simulated, graph-mode launch gaps):

| system | GPUs | step | kernels | all-reduce |
|---|---|---|---|---|
| one rank, no communication | 1 | 6.00 ms | 4.62 ms | 0 |
| HGX B300, TP8 | 8 | 6.39 ms | 4.62 ms | 0.39 ms |
| GB300 NVL72, TP8 x DP9 | 72 | 6.39 ms | 4.62 ms | 0.39 ms |

The 162 collectives per step (two 512 KB all-reduces per layer, plus the embedding all-reduce
and the logits all-gather) cost 0.39 ms, 6% of the step, on either system: TP8 stays inside one NVLink domain, so NVL72 only adds replicas. The eager
capture has more elementwise kernels than vLLM serves with `torch.compile`; use
`--no-graphs` captures for absolute numbers and `--eager` ones when the plan needs module marks.

**Hazy Research low-latency Llama-1B megakernel** (built for sm_103a): 1,645 tokens/s native
(608 µs per token). The capture is one 148-CTA x 640-thread launch per token at its native
duration. Cycle-level replay of a megakernel needs every SM's instruction stream recorded,
which the 4-CTA sampling does not do.

**SGLang 0.5.22** (same model and batch): native decode step 1.14 ms with CUDA graphs, 15.7 ms
eager. The capture has 2,018 launches of 50 distinct kernels (FlashInfer CuTe DSL norms,
cuBLAS, TRT-LLM attention); its native comparison is pending.

**DeepInfer** (DeepSeek-V4.1-Flash): its vendored kernels need CUDA 13.1+ and GCC 13+ to build
for sm_103a; the build is in progress.

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
