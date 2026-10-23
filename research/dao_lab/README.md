# Mixers on the rack

A simulator-only study of the sequence mixers that came out of, or are built on, Dao AI Lab work:
FlashAttention-4-style attention, MLA, Mamba-2 (SSD), linear attention and Gated DeltaNet, and the
decode-time variants GTA and GLA from "Hardware-Efficient Attention for Fast Decoding" (Zadouri,
Strauss and Dao). It runs them on one B300, an 8-GPU HGX B300 node and a 72-GPU GB300 NVL72 rack.

Every kernel runs on the calibrated cycle-level B300 SM model and every transfer on the simulated
NVLink fabric. No GPU was used. The findings page is `results/report.html` (published as an Artifact).

## Studies

| script | what it measures |
|---|---|
| `prefill_mixers.py` | Study A: one mixer block at d_model 4096 (GQA, MLA, Mamba-2, linear attention, Gated DeltaNet), batch 1, 1k to 1M tokens, on one B300 |
| `decode_attention.py kernels` | Study B: decode attention per layer and rank for GQA-8, MQA, GTA-8, MLA, GLA-2 at TP 1-8, for the FA4 chain and the split-block (`kv_warpgroups: 2`) schedule |
| `decode_attention.py rack` | Study B: whole Llama-3-70B-body decode steps on GB300 NVL72 at 8k/32k/128k context, TP8 and TP4, at the largest batch that fits |
| `cp_prefill.py [cp ...]` | Study C: time to first token for one 144k to 1.13M token prompt with context parallelism over 1, 8 and 72 GPUs (ring attention vs state passing) |
| `check_extrap.py` | the long-context fast path against full cycle-level runs |
| `build_page.py` | builds `results/report.html` from `page_template.html` and the JSON results |

Run from the repository root after `pip install -e .`, for example
`python research/dao_lab/cp_prefill.py 72 8 1`. Long kernels are cached in `results/kernel_cache.json`
(the key includes a hash of the native library, so a rebuilt simulator never reuses stale entries).
The full grid takes about a day on 4 cores; most of that is Study C.

## What the simulator gained for this study

- `synth::attention_with(.., AttnOpts { kv_tied, kv_warpgroups })` and the matching `kv_tied` and
  `kv_warpgroups` fields on the workload `attention` spec. `kv_tied` reads one cache tensor as both K
  and V (GTA, GLA, MLA). `kv_warpgroups: 2` is a FlashMLA-style split-block decode schedule. The defaults
  are unchanged and bit-identical.
- `rackscale.models`: `GLA`, `LinearAttn` (`delta=True` for Gated DeltaNet), block letter `L`,
  `Model.gta`, and presets `llama3-70b-{mqa,gta8,mla,gla2}`, `llama3-8b-{gta8,gla2}`, `gla-8b`,
  `gdn-8b`, `gdn-hybrid-8b`, `gdn-hybrid-70b`.
- `rackscale.lower.prefill_cp(model, cp, seqlen)`: context-parallel prefill as explicit per-rank
  programs. Attention uses zig-zag ring attention; SSM and linear layers pass state along the ranks.
- `rackscale.extrap`: the long-context fast path. It simulates long kernels on reduced exact-wave
  grids, extrapolates them, and `freeze()` turns them into fixed ops in a workload. It agrees with
  full runs within 0.25% (`results/extrap_check.json`).

## Approximations

Gated DeltaNet is SSD plus the `K Kᵀ` and WY passes; the sequential delta-rule correction between
chunks is not modelled. The context-parallel lowering skips the 3-token conv halo. Attention kernels
do not get the power-cap clock drop that GEMMs get. The MLA decode kernel uses 32-wide K/V blocks
and is pessimistic against FlashMLA.
