# 10. Bring your own workload

Every chapter so far built its workload from a model description: a GEMM shape, a decode
step lowered from a config, a hand-written SASS kernel. Real engines do not look like their
descriptions. A vLLM decode step on Llama-3.2-1B launches 1,800 kernels from five libraries
(cuBLAS, TensorRT-LLM attention, vLLM's own CUDA ops, Triton and PyTorch), and a third of
them exist to prepare inputs and sample tokens. This chapter turns any program that runs on
one GPU into a simulator workload without describing it. You run the program once, under a
tool that records what the GPU actually did, then replay that recording on any system.

## 10.1 What is recorded

`rackscale capture -- <command>` runs the command unmodified, three times:

| pass | how | what it gives |
|---|---|---|
| graph | NVBit tool `rsim_capture`, nothing instrumented | every kernel launch (function, grid, block, cluster, shared memory, registers, stream), every event record and stream wait, every copy; engine steps and module calls from a small Python site package |
| paths | the same tool, instrumenting one launch of every distinct kernel | the per-warp dynamic SASS path of a few CTAs: lane addresses, mbarrier operands, TMA boxes, tcgen05 descriptors |
| timing | Nsight Systems, no NVBit | the native GPU start and end of every kernel and copy, and the host time of every launch |

The three passes are joined by kernel key (function plus launch configuration) and by launch
order. The capture directory holds `capture.json` (the ordered ops of the capture region per
process and stream, with module marks) and one `kernels/kNNNN.rpath.gz` per distinct kernel.

Two details make this work on real engines:

- **The region.** You rarely want the whole program (weight loading, warm-up, CUDA graph
  capture). `--region forward:S:N` starts at the S-th forward of the model and stops N
  forwards later, by counting the outermost `nn.Module` call with at least a million
  parameters. `armed:S:N` counts only after the program calls `rsim_capture.arm()`, which is
  how SGLang's scheduler subprocess is captured from the driver script. Your own script can
  wrap any block in `rsim_capture.region()`.
- **One recording per kernel, not per launch.** A decode step of an 80-layer model launches
  the same few kernels 80 times. The paths pass records each configuration once; the
  simulator replays every launch from that recording. An instrumented launch runs 10 to 100
  times slower, so this is what makes the paths pass affordable.

## 10.2 What the replay does

`rackscale replay` turns the capture into an ordinary rsim workload: one stream per captured
stream, `record`/`wait` ops for the events, a `path` kernel for every recorded launch and a
`fixed` kernel at its native duration for the rest, and `delay` ops for copies. Two switches
decide how much is re-simulated:

| switch | kernels | between kernels |
|---|---|---|
| `--mode measured` | native durations | simulated: streams, dependencies, launch gaps, SM sharing |
| `--mode paths` (default) | re-simulated at cycle level from their SASS paths | simulated |
| `--host` | either | each launch waits for its captured host issue time |

`--mode measured` is the control: it checks that the program structure was captured
correctly, since only the overlap model stands between it and the measurement. `--mode paths`
is the prediction: change the GPU's calibration, its clock or the kernel's neighbours, and
the kernels respond.

Without `--host`, the GPU runs the captured streams as fast as their dependencies allow,
which is what a CUDA graph does. With `--host`, the replay follows the host timeline of the
eager run. A small model in eager mode is host-bound: vLLM's eager decode step of
Llama-3.2-1B at batch 32 takes 10.2 ms on the B300, against 2.1 ms with CUDA graphs, while the
kernels themselves are busy for about 1.3 ms of each step.

## 10.3 From one GPU to a rack

A capture is what one GPU ran. A parallelism plan adds communication and replicates or
splits the program. It never changes a kernel:

```yaml
tp: 8          # an all-reduce after every RowParallelLinear and VocabParallelEmbedding
ep: 1          # all-to-all before and after every FusedMoE
pp: 1          # layers split over stages, activations sent between them
dp: 9          # replicas
```

The plan places collectives with the module marks of the graph pass: vLLM and SGLang call
their layer classes `RowParallelLinear`, `VocabParallelEmbedding` and `FusedMoE`, and plain
Hugging Face models are matched by module name (`*.o_proj`, `*.down_proj`). Each collective's
size is the module's output size in the capture.

For tensor or expert parallelism the kernels themselves change: a rank of a TP=8 deployment
runs GEMMs with an eighth of the columns. So the capture is taken of one rank's share. The
example scripts run the engine on one GPU with the per-rank shapes and dummy weights
(`--shape llama3-70b --shard-tp 8`: heads, KV heads, MLP width and vocabulary divided by 8),
which gives exactly the kernels a rank of the real deployment launches, then the plan adds
the all-reduces between them.

## 10.4 What this cannot tell you

- **Scheduling and batching policy.** A capture is one fixed batch. Continuous batching,
  admission and preemption are engine behaviour; capture each batch shape you care about.
- **Host overheads beyond the captured timeline.** `--host` replays measured issue times, and
  scaling out assumes each rank's host keeps them.
- **Other GPU generations.** A recording is the SASS of the GPU it was captured on. Replaying a
  Blackwell capture on an H100 preset would run Blackwell instructions on a Hopper model.
- **Megakernels at cycle level.** A persistent megakernel runs a different instruction stream
  on every SM, so replaying a few recorded CTAs for all 148 is wrong. The Hazy Research
  low-latency Llama-1B megakernel captures as one launch per token at its native duration
  (608 µs per token on the B300); cycle-level replay would need all 148 CTAs recorded.

## Exercises

1. Capture the same model under vLLM and SGLang at batch 32. Which kernels does each engine
   launch that the other does not, and how much of the decode step do they account for?
   *(Answer: compare `capture.json`'s kernel tables. SGLang's step is about half of vLLM's
   with CUDA graphs on the B300 (1.14 ms against 2.14 ms); most of the difference is outside
   the GEMMs.)*
2. Replay a capture with `--mode measured` and with `--mode measured --host`. What does the
   gap between them say about the engine?
   *(Answer: it is the time the GPU waits for the host. A large gap means CUDA graphs, or a
   faster host, would help more than a faster GPU.)*

Reproduce: [docs/CAPTURE.md](../CAPTURE.md) and `examples/capture/`.
