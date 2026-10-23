# What model size and architecture do to a GPU and a rack

A simulator-only study of how the shape of a model (parameter count, dense vs mixture of experts, the
attention design, Mamba-2, context length) decides what a GPU and a rack spend their time on, which
parallel layout is best, and how much of that the serving stack can change.

Every point is one simulated step of one model replica in fast mode: each distinct kernel runs at
cycle level on the SM model (B300 and A100 calibrated on real GPUs, see `docs/CALIBRATION.md`), and
every collective runs on the simulated NVLink fabric. Points that do not fit in HBM
(`rackscale.lower.memory`) are recorded and not simulated.

The page with the findings and an explorer over every point is `results/report.html` (published as
an Artifact); the numbers behind its headline cards are in `results/summary.json`.

## Studies

| study | grid |
|---|---|
| `ladder` | dense Llama 3.2 1B/3B, Llama 3 8B, Qwen2.5-32B, Llama 3 70B/405B on HGX A100, HGX H100, HGX B300, GB300 NVL72; decode (8k context, batch 1-256, every TP layout), prefill (8k), training (4k, the four least-parallel TP x PP x DP layouts that fit) |
| `attention` | the Llama-3-8B body with MHA, GQA-4, MQA, MLA, and Mamba-2 / hybrid of the same size; decode 2k-128k context, prefill 2k-32k; B300 and A100 |
| `moe` | Mixtral-8x7B, Qwen3-30B-A3B, Mixtral-8x22B, Qwen3-235B-A22B, DeepSeek-V3 on HGX B300 and GB300 NVL72; TP vs EP (DeepSeek layout), batch 8-128, two-micro-batch overlap |
| `context` | Llama-3-70B, Mamba-2 70B and hybrid at TP8, Llama-3-405B at TP8, DeepSeek-V3 at EP72 on GB300 NVL72, 4k-128k context |
| `infra` | launch mode (eager / CUDA graphs / no gaps), chunked prefill in a continuous batch, BF16 vs FP8 GEMMs, and two points set against engines measured on a B300 |

## What each point reports

- step time, tokens/s per GPU, tokens/s per user (decode), TTFT (prefill), MFU, memory per GPU;
- the step split into **compute** (kernels at >= 50% of tensor peak), **HBM** (>= 50% of achievable
  bandwidth), **latency** (neither), **comm** (collective time not hidden) and **launch** (GPU idle);
- time per kernel class (attention, GEMM, LM head, MoE experts, SSM, norms and elementwise), the
  scheduler stall anatomy and the top kernels.

## Reproduce

```bash
maturin develop --release
python research/model_sweep/run.py --jobs 8          # all studies, results/<study>.json (about 2 h on 4 cores)
python research/model_sweep/run.py --quick           # smoke test of every grid
python research/model_sweep/build_page.py            # results/report.html and summary.json
```

The same machinery answers one question at a time:

```bash
rackscale plan qwen3-235b-a22b --system gb300 --context 8192 --min-tps-user 30 --json
rackscale sweep --models llama3-8b,llama3-8b-mla --systems b300 --contexts 8192,131072 --batches 1,64
```

## Measured anchors

Two simulated points are set against engines captured on a B300 (`docs/CAPTURE.md`): vLLM 0.32 serving
Llama-3.2-1B at batch 32 (942 µs GPU step with CUDA graphs) and Hazy Research's low-latency megakernel
at batch 1 (608 µs per token). The page shows both next to the simulator.

## Limits

- The A100 GEMM model (Ampere `mma.sync`) runs 1.5-2.4x slower than cuBLAS, so A100 step times are
  pessimistic; H100 is an uncalibrated datasheet preset.
- Kernels are the simulator's synthetic FlashAttention / CUTLASS-style kernels, not each engine's own;
  the measured anchors show the distance.
- Each point is one steady-state step: request arrival, scheduling over time, expert-load imbalance and
  KV paging are not modelled.
- "No launch gaps" removes gaps between kernels; it does not model the cross-op fusion and overlap a
  real megakernel also does.
- Training points with 16 or more ranks and more than 24 layers are simulated at 6 and 12 layers and extended linearly to the
  full depth (`rackscale.sweep.EXTRAPOLATE_LAYERS`): a whole 72-rank step takes hours, because the
  NCCL-faithful network model runs every chunk of every gradient collective. Against whole-model runs
  of Qwen2.5-32B on GB300 NVL72 the extrapolation is within 0.02%.
