# One day on an 8-GPU node

Everything the article ("How to Understand a Rack") still needs from hardware, packed into
one rented day on an HGX node (B300, B200 or H100), ordered by how much each measurement is
worth to the article, so the day is still useful if it is cut short.

The rule for the day: **every number is predicted before it is measured.**
[`predict.py`](predict.py) ran the simulator on every case below and its output,
[`predictions/`](predictions/) and [`PREDICTIONS.md`](PREDICTIONS.md), is committed with the
commit it came from. [`compare.py`](compare.py) scores the measurements against those files
as committed and never re-runs the simulator. Errors are signed, (predicted − measured) /
measured; negative means the simulator is optimistic.

```bash
# on the node
git clone https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation rackscale && cd rackscale
calibration/nodeday/nodeday.sh setup     # ~20 min: venv, vLLM (brings torch), NVML, nccl-tests
calibration/nodeday/nodeday.sh smoke     # ~5 min: what works, what will be skipped
calibration/nodeday/nodeday.sh run       # all blocks in priority order (~7 h of GPU time)
calibration/nodeday/nodeday.sh status    # any time, from another shell
calibration/nodeday/nodeday.sh pack      # results tarball

# anywhere afterwards (no GPU)
python calibration/nodeday/compare.py ~/nodeday/results/<host>-<gpu>    # COMPARE.md + compare.json
```

Run it inside `tmux`: a block that finishes writes `DONE`, so `run` after a disconnect picks up
where it stopped. A failing block writes `FAILED` and the run moves on; rerun it at the end with
`run <block>`. Results go to `~/nodeday/results/<host>-<gpu>/<block>/` (`ND_OUT` overrides).

## Before the day

Ask the provider, because each one decides whether a block can run at all:

| question | why |
|---|---|
| HGX SXM board with NVSwitch, not PCIe cards? | NVLS (switch-side reduction) and the 900 GB/s fabric exist only on SXM; every rung-4 number assumes it |
| Bare metal, or a VM with all 8 GPUs and NVLink passed through? | some VMs hide NVLink or fix the clocks |
| `nvidia-fabricmanager` running? | without it, multi-GPU NCCL fails on HGX |
| CUDA toolkit (nvcc) installed, driver ≥ the toolkit's minimum? | nccl-tests, the calibration suite and NVBit build from source |
| Root, or `NVreg_RestrictProfilingToAdminUsers=0`? | Nsight Compute counters (microbenchmark 7, the calibration suite's ncu reference set) |
| Outbound access to GitHub, PyPI and Hugging Face? | setup downloads vLLM, nccl-tests, H-Net checkpoints, DeepSeek/Qwen configs |
| ≥ 500 GB of local disk | venv, builds, NVBit recordings, H-Net checkpoints |

Then tell the thread which GPU it is. The predictions exist for all three; `compare.py` picks the
file from the GPU the node reports.

## The day, in priority order

Budgets are GPU minutes on a healthy node. P0 is the article's largest gap: no network number in
it is our own measurement yet. Cut from the bottom if time runs short; never cut P0.

| # | block | prio | budget | what it measures | article rung | prediction it tests |
|---|---|---|---|---|---|---|
| 1 | `env` | P0 | 2 min | `nvidia-smi -q`, topology, NVLink status and error counters, clocks, power limits, versions | all | — |
| 2 | `nccl` | P0 | 30 min | nccl-tests, 6 collectives × 2/4/8 GPUs, 8 B–8 GiB; all-reduce forced Ring/Tree/NVLS, all-gather and reduce-scatter Ring/NVLS; NCCL's own algorithm choice logged | 4 | every point of `rsim net sweep` (fabric "7.8% vs published" becomes our own) |
| 3 | `variance` | P0 | 20 min | same 4 workloads (BF16 8192³, 4096³, 2 GiB copy, 1000 tiny kernels) on each GPU alone, then all at once: per-GPU σ, GPU-to-GPU spread, slowest GPU | 5 | replaces chapter 16's assumed 5% jitter |
| 4 | `straggler` | P0 | 10 min | BF16 GEMM loop, all-reduce after every GEMM / every 16, at 2/4/8 GPUs: the straggler tax itself | 5 | σ·E[max_N] with the measured σ; the simulator's 8.0% at N = 8 for σ = 5% |
| 5 | `train` | P0 | 40 min | chapter 11's training cases: 1 GPU, DDP, FSDP, HSDP 4×2, TP8, TP8+SP, PP2 GPipe / 1F1B / ZB-H1; 2 layers (and 8 for 1 GPU, FSDP, TP8); torch.profiler trace of rank 0 | 2, 4 | sim, roofline replay and textbook step times; the 1F1B vs ZB-H1 swing |
| 6 | `decode` | P0 | 90 min | vLLM (dummy weights), median inter-token latency: Llama-3-8B b64, Llama-3-70B TP8 b64, Qwen3-30B-A3B TP4, DeepSeek-V3 EP8 (DP attention), full depth and truncated | 3, 4 | the decode rows of chapter 11 |
| 7 | `overlap` | P1 | 20 min | BF16 4096³ GEMM while NCCL holds R = 2…32 SMs (`NCCL_MIN/MAX_CTAS`) | 2 | "16 reserved SMs are free, 24 cost +34%" (⌈512 / free SMs⌉) |
| 8 | `power` | P1 | 35 min | BF16 and FP8 GEMMs 2k–16k and attention (FA4, cuDNN, FA2), 4 s each with NVML at 10 ms: clock, power, throttle reasons; GPU 0 alone, then all 8 at once | 2 | the sustained-clock model; microbenchmark 11 (clock under attention) |
| 9 | `links` | P1 | 25 min | send/recv between every GPU pair at 1 GiB, nvbandwidth, all-reduce with each NCCL protocol | 5 | one slow pair or plane shows up here ("a link is a stripe") |
| 10 | `fanout` | P1 | 90 min | four 1-GPU jobs at once: H-Net vs BPE (GPU 0), NVBit recordings of the step kernels (GPU 1), the whole calibration suite on a fresh GPU (GPU 2), the open microbenchmarks (GPU 3) | 1, 2, 3 | H-Net ÷ BPE, kernels per byte, prefill; per-kernel replays |
| 11 | `deepep` | P2 | 20 min | DeepEP intranode and low-latency dispatch/combine tests | 4 | the EP8 all-to-all sizes |
| 12 | `variance_end` | P2 | 10 min | the isolated variance pass again, and NVLink error counters at the end | 5 | drift over the day |

About 6.5 hours of GPU time if nothing breaks, which leaves a 24-hour rental most of a day of
margin for setup trouble and reruns. Optional extras (the `rackscale` package for NVBit capture,
NVBit, FlashAttention 2 and 4, H-Net's kernels, DeepEP, nvbandwidth) build in the background at
the lowest CPU priority from the moment `run` starts, and are paused while the host-sensitive
blocks (`train`, `decode`, `overlap`) measure. `fanout` and `deepep` wait for them.

### If time runs short

Drop, in this order: `variance_end`, `deepep`, the calibration suite inside `fanout`
(`run fanout` still does the other three), `links`, the truncated-depth decode cases. Keep `nccl`,
`variance`, `straggler`, `train` and the full-depth `decode`: they are the article's rung 4 and 5,
and the node is the only place we can get them.

## What gets compared to what

The cases come from one list, [`cases.py`](cases.py), that both `predict.py` and the node scripts
read, so a measurement and its prediction have the same model, depth, batch, sequence length and
plan by construction.

- **Training step** = forward + backward of the transformer layers on one 4096-token micro-batch
  in BF16, no embedding, LM head, loss or optimizer: exactly what `rackscale.lower.train_step`
  lowers. [`common/llama.py`](common/llama.py) is a plain Llama-3 layer (SDPA attention,
  separate Q/K/V GEMMs so DTensor TP can shard them). DDP uses one bucket per layer, as the
  simulator reduces each layer when its backward ends. FSDP2 reshards after forward, so the
  backward gathers again, as lowered. The pipeline plans run **the simulator's own op order**
  (`_pp_order`, copied and checked) with B and W split by PyTorch's zero-bubble helpers, and
  activations and gradients on separate communicators. The step ends when the slowest rank ends.
- **Decode step** = median inter-token latency with every sequence decoding (vLLM `bench serve`,
  all requests at once, EOS ignored). Full depth is the comparison; at 2–4 layers the engine's
  per-step CPU work dominates, which the simulator does not model, so those runs measure that
  overhead instead.
- **Collectives**: nccl-tests' out-of-place time against `rsim net sweep` at the same size, in
  nccl-tests' size convention (the total buffer for all-gather and all-to-all).
- **GEMM beside NCCL**: scored on time relative to R = 0, because the claim is the wave step;
  the cuBLAS kernel name is recorded, since a tile other than 128×256 moves the step.
- **Stragglers**: the measured tax, the closed form σ·E[max_N]/√k with the measured σ, and a
  bootstrap from the measured per-kernel samples (no normality assumption).

## Known gaps, stated before the day

These predictions are expected to miss, and the misses are part of the result:

- **FP8 GEMMs and attention get no power-cap clock drop** in the simulator: the power model has
  measured runs for FP4 and BF16 only, and attention kernel names carry no dtype
  (CALIBRATION.md). The `power` block measures exactly this.
- **B200 and H100 are uncalibrated presets.** Only the B300 die was measured. On a B200 the
  `fanout` block's calibration suite produces the first B200 calibration (article gap 6).
- **H-Net's batched decode cannot be measured with the released code**, whose `step` takes batch
  1 only. The node measures batch-1 decode, prefill and kernels per byte for H-Net, and BPE at
  batch 1/16/64/256. The chapter's batch-64 ratio (2.17×) stays simulated, and the article
  should say so. At batch 1 the simulator predicts H-Net about level with BPE per byte
  (0.98× 1-stage, 1.14× 2-stage); H-Net's step cannot be put in a CUDA graph (its chunk sizes are
  data-dependent), so expect launch overhead to dominate there.
- **DeepSeek-V3 EP8 does not fit 8× H100** (≈690 GB of FP8 weights in 640 GB) and is skipped there.
- vLLM's FP8 block quantisation for DeepSeek-V3 with dummy weights runs FP8 GEMMs; the
  simulator's DeepSeek-V3 decode lowers BF16 GEMMs.

## The open microbenchmarks (CALIBRATION.md)

| # | what | where on the node |
|---|---|---|
| 1 | FFMA / EX2 launch and tail, per-CTA stamps | `bench_open` (`fanout`, GPU 3; also in the suite) |
| 2 | per-SM clock under tensor load | `bench_open`, on an HMMA load; FA4 itself instrumented stays on Della |
| 3 | TMA from HBM by stage count | Della B300 (needs the recorded FA4 context) |
| 4 | DRAM latency distribution, idle and loaded | `bench_open` |
| 5 | TMA near/far partition | Della B300 |
| 6 | re-record FA4 ring steps per tile count | Della B300 |
| 7 | FA2 per-PC stalls | `micro.sh`: ncu source-level sampling |
| 8 | MIO arbitration fairness | `bench_open` |
| 9 | shared FP64 unit | `bench_open` |
| 10 | HMMA operand reads vs register banks | `bench_overlap`, in the suite run |
| 11 | clock under attention | `power` block |

## Files

| file | what |
|---|---|
| `nodeday.sh` | setup, smoke test, the ordered blocks, status, pack |
| `cases.py` | the case list both sides read |
| `predict.py` | pre-registers the simulator's numbers (run before the day; committed) |
| `compare.py` | signed errors, `COMPARE.md` |
| `blocks/train_step.py` | chapter 11 training plans (runs on CPU with gloo as a smoke test) |
| `blocks/decode.py` | vLLM serve + bench per decode case |
| `blocks/variance.py`, `blocks/straggler.py` | per-GPU σ, the straggler tax |
| `blocks/overlap.py`, `blocks/power.py` | GEMM beside NCCL; clock and power under load |
| `blocks/hnet_bench.py` | H-Net vs BPE on one GPU |
| `blocks/capture.sh`, `blocks/micro.sh` | NVBit recordings; open microbenchmarks and FA2 ncu |
| `common/nvml.py`, `common/llama.py` | 10 ms NVML sampler; the Llama layer |
| `hf/` | Llama-3 8B and 70B configs, so no gated download is needed |
