# The simulator as a tool: JSON API and MCP server

rackscale can be driven by a program instead of a person: a script, a notebook, or an agent
that does performance engineering. Three layers share one implementation:

| layer | what it is | use it from |
|---|---|---|
| `rackscale.api` | JSON in, JSON out, schema `rackscale-api/v1` | Python |
| `rackscale.tools` | the same functions as a tool registry: name, description, JSON schema | any LLM tool-use API (`tools.specs()`, `tools.call()`) |
| `rackscale-mcp` | the registry over the Model Context Protocol | Claude Code, Claude Desktop, any MCP client |

All three return identical answers. Every time carries an **error band**, and every run carries
**provenance**.

## Contents

- [Quickstart](#quickstart)
- [The loop it is built for](#the-loop-it-is-built-for)
- [Tools](#tools)
- [Workload specs](#workload-specs)
- [Hardware knobs](#hardware-knobs)
- [How bottlenecks are classified](#how-bottlenecks-are-classified)
- [Error bands](#error-bands)
- [Provenance and caching](#provenance-and-caching)
- [Worked example: an agent tunes DeepSeek-V3 decode](#worked-example-an-agent-tunes-deepseek-v3-decode)
- [Limits](#limits)

## Quickstart

```bash
maturin develop --release && pip install mcp      # the MCP SDK is only needed for rackscale-mcp
claude mcp add rackscale -- rackscale-mcp         # register with Claude Code (stdio)
```

Then ask, in plain words: *"Simulate DeepSeek-V3 decode on HGX B200 at EP8, batch 32 per rank, first
4 layers. What bounds it, and what would you change?"*

From Python:

```python
from rackscale import api

r = api.run("hgx_b200_8", {"model": "deepseek-v3", "phase": "decode", "batch": 32, "context": 4096,
                           "parallel": {"ep": 8}, "layers": 4})
r["step_time_us"]          # {'value': 1629.6, 'lo': 1201.5, 'hi': 2057.7, 'band_pct': 26.3}
b = api.bottlenecks(r["run_id"])
b["summary"]               # 'Step 1630 us ... 62% of GPU time is in hbm-bound kernels ...'
b["suggested_what_ifs"][0] # {'what_if': {'workload': {'batch': 64}}, 'why': 'weights are streamed once per step; ...'}
api.what_if(r["run_id"], {"workload": {"dtype": "fp8"}})["comparison"]["verdict"]
                           # 'b is 35.8% faster per step; throughput up 55.6%'
api.kernel(r["run_id"], "lm_head")["role_anatomy"]   # producer / MMA / epilogue: who waits on what
```

## The loop it is built for

1. **Measure.** `simulate` gives the step time with a band, tokens/s, MFU, idle fraction, the
   scheduler-stall anatomy and the top kernels.
2. **Locate.** `find_bottlenecks` puts every kernel in a class (HBM, tensor, memory latency, tensor
   pipeline, sync, ALU) from its achieved rates and stalls. It flags wave and tile-padding waste and
   attaches what-ifs that can be run as they are.
3. **Explain.** `inspect_kernel` reruns one kernel alone on a cold GPU, the way Nsight Compute times
   it. It returns the stall anatomy per warp role (TMA producer, MMA issuer, softmax, epilogue; or per
   warp index for recorded kernels), the mbarrier handoffs that wait, and the SASS instructions with
   the most stalled warp-cycles.
4. **Test.** `what_if` and `sweep` re-simulate with a change and compare the result against the error
   band. The verdict says when a difference is too small to trust.

## Tools

| tool | API function | returns |
|---|---|---|
| `list_hardware` | `api.hardware(name=None)` | GPU and system presets, peaks, calibration status, and the knob table |
| `list_models` | `api.models_list()` | models the lowering knows, with sizes |
| `describe_workload` | `api.workload(spec)` | kernel and collective counts and tokens per step, without simulating |
| `simulate` | `api.run(system, workload, mode, overrides, launch_gap_ns)` | run id, step time ± band, tokens/s, MFU, anatomy, top kernels, collectives, warnings, provenance |
| `find_bottlenecks` | `api.bottlenecks(run_id, top)` | time split by bound, per-kernel diagnosis and levers, ranked what-ifs |
| `inspect_kernel` | `api.kernel(run_id, name)` or `api.kernel(system=, spec=)` | time ± band, role anatomy, top SASS PCs, mbarrier waits, waves, padding, diagnosis |
| `what_if` | `api.what_if(run_id, change)` | the new run plus a comparison with the significance verdict |
| `sweep` | `api.sweep(run_id, key, values, objective)` | one row per value and the best value |
| `compare_runs` | `api.compare(a, b)` | step and throughput deltas, per-op deltas, stall-anatomy deltas |
| `list_runs` | `api.runs()` | runs kept in this session |
| `accuracy` | `api.accuracy_table()` | the evidence behind every band |

Errors come back as `{"error": "..."}` and are not raised, so an agent can read the message and try
again.

## Workload specs

```jsonc
// a model, lowered to synthetic SASS kernels (Hopper: TMA + HGMMA; Blackwell: TMA + tcgen05)
{"model": "deepseek-v3", "phase": "decode", "batch": 32, "context": 4096,
 "parallel": {"tp": 1, "ep": 8, "dp": 1, "comm_sms": 20}, "layers": 4, "overlap": false, "dtype": "fp8"}
// prefill: "seqlen"; train: "micro_batch" + "seqlen"
// one kernel: a synthetic template, a recorded warp path, or SASS
{"kernel": {"type": "gemm", "m": 8192, "n": 8192, "k": 8192, "dtype": "bf16"}}
{"kernel": {"type": "path", "path": "research/flash_attention/rpath/b300_fa4_s4096.rpath.gz"}}
// a capture from `rackscale capture -- <cmd>` replayed with a parallelism plan (docs/CAPTURE.md)
{"capture": "cap/", "plan": {"tp": 8}, "mode": "paths", "steps": [0, 1]}
// anything else in the workload format (docs/WORKLOAD_FORMAT.md)
{"workload": {"name": "...", "spmd": {...}}}
```

`what_if` merges changes into the spec: `{"workload": {"parallel": {"ep": 16}}}` changes one field.
`dtype` applies to the GEMMs; attention stays BF16.

## Hardware knobs

`overrides` (and `what_if`'s `"hardware"`) take friendly names or paths. A string `"x1.5"` scales
the current value.

| knob | path |
|---|---|
| `sm_clock_mhz`, `sm_count` | `gpu.sm_clock_mhz`, `gpu.sm_count` |
| `hbm_bw_gbps`, `hbm_latency_ns`, `hbm_capacity_gb` | `gpu.hbm.*` |
| `l2_size_bytes`, `l2_hit_latency`, `l2_bytes_per_cycle` | `gpu.l2.*`, `gpu.l2_bytes_per_cycle` |
| `smem_bytes_per_cycle` | `gpu.pipes.l1_bytes_per_cycle` |
| `tensor_{bf16,fp8,fp4}_flops_per_cycle_per_sm` | `gpu.tensor.flops_per_cycle_per_sm.*` |
| `launch_overhead_ns`, `power_cap` (`null` removes the cap) | `gpu.launch_overhead_ns`, `gpu.power` |
| `nvlink_gbps_per_dir`, `nvlink_latency_ns` | scale-up links and switch ports together |
| `nic_gbps_per_dir` | scale-out NICs (multi-node systems) |

Any `gpu.<path>` reaches any field of the GPU spec (see `list_hardware("b300")`), and
`gpu.calibration` attaches a `rackscale-calibration/v1` file. `system.<path>` edits the system
spec directly.

## How bottlenecks are classified

The rules are in `python/rackscale/analysis.py` and kept simple, so they can be checked against
the numbers that come with them.

- **hbm**: at least 70% of achievable HBM bandwidth (`bw × efficiency`) and more HBM-bound than
  tensor-bound.
- **tensor**: at least 60% of the dense tensor peak for the kernel's dtype.
- Otherwise the class comes from the largest scheduler stall, once `async_wait` is set aside. A
  warp-specialised kernel parks most of its warps on mbarriers, so that stall reveals nothing. The
  classes are `tma_wait` / `long_scoreboard` / `lg_throttle` → **memory_latency**, `tensor_wait` /
  `tensor_throttle` → **tensor_pipeline**, `barrier` → **sync**, `short_scoreboard` /
  `mio_throttle` → **smem_mio**, and `math_pipe_throttle` / `wait` → **alu**.
- Flags: **tile_padding** when under half of each GEMM tile's rows are real (decode at M=32 against
  a 128-row UMMA tile uses 25%); **underfilled** when the last wave leaves more than a quarter of
  the SM slots empty.

Levers are tied to the class. HBM-bound kernels get a larger batch, FP8 weights, more TP, and HBM
sensitivity (on its own and together with L2, because streaming traffic crosses L2 and can hit that
ceiling first). Tensor-bound kernels get lower precision. Step-level idle gets CUDA-graph launch
gaps. Collectives get two-micro-batch overlap and link bandwidth. Hardware levers are labelled
*sensitivity*: they tell you what a faster part would buy, not what an operator can change.

## Error bands

The simulator is deterministic. Its bands come from how far it has been from hardware on
comparable work, measured against Nsight Compute cycles at the same clock
(`python/rackscale/accuracy.py`, `accuracy` tool):

| kernel class | calibrated GPU | typical | worst seen | evidence |
|---|---|---|---|---|
| recorded warp path (`.rpath`) | yes | 15% | +124% | B300 microbenchmarks 12/12 within 7%; FlashAttention 16 kernels within 6%; vLLM engine kernels 39% time-weighted |
| SASS listing | yes | 8% | −23% | A100 reference kernels 9/11 within 8% |
| synthetic template (model lowering) | yes | 20% | +85% | B300 GEMMs 15–35% fast before the sustained-clock fit; ThunderKittens GEMMs −6% to +85% |
| measured duration | — | 1% | 5% | vLLM replay with measured durations +0.3% |
| any kernel on an uncalibrated preset (H100, H200, B200, GB200) | no | 25% | 60% | heuristic: datasheet numbers only |
| collectives | — | 25% | 50% | heuristic: no multi-GPU system measured yet |
| idle and launch gaps | — | 30% | 100% | heuristic: host-side scheduling not modelled |

B300 and GB300 carry their calibration built in. The A100 presets have their measured calibration
files attached automatically. A run's band is the time-weighted sum over its parts, plus fast
mode's +1%. Summing assumes the errors move together, which makes the band conservative.

Comparisons are tighter than absolute numbers, because two runs of the same kernels err the same way.
`compare` puts a band on the difference, `sqrt(a² + b² − 2ρab)`. The correlation ρ is 0.9 when only a
hardware knob or a run setting changed, 0.6 for a workload change and 0.3 for a different system.
These ρ values are heuristic, and each result says so. Throughput is judged on the same band, because
token counts are exact.

## Provenance and caching

Every run records the git commit and whether the tree was dirty, the native version, hashes of the
resolved system and the workload, each calibration file with its SHA-256, the applied overrides with
from/to values, the mode, the simulated warp instructions and the wall time. The run id is a hash of
all of that input, so repeating a call returns the cached result (`"cached": true`).

## Worked example: an agent tunes DeepSeek-V3 decode

[`examples/agent/perf_agent.py`](../examples/agent/perf_agent.py) runs the loop above. With
`--policy scripted` it works through the simulator's suggestions deterministically. With
`--policy claude`, Claude drives the same tools through the Anthropic API.

```bash
python examples/agent/perf_agent.py          # DeepSeek-V3, 4 layers, EP8 on HGX B200, batch 32/rank; ~2.5 min
```

It maximises tokens/s with the step time held to at most 1.5× the baseline, and keeps a change only
when the change beats the error band. The output below is abridged:

```
## Baseline: deepseek-v3 decode b32 ctx4096 Parallel(tp=1, ep=8, dp=1, comm_sms=20) on HGX B200 8-GPU
step 1630 (±26%) us, 157093 (±26%) tokens/s, MFU 6.6%

### Round 1: ... 62% of GPU time is in hbm-bound kernels ... Biggest single cost:
sm100_gemm_bf16_32x129280x7168_tile128x256 (17%), HBM-bound: 98.0% of achievable HBM; past the
mbarrier parking, schedulers wait on tma_wait (19%); only 25% of each 128-row tile is real rows.
- {"workload": {"batch": 64}}: b is 9.2% slower per step (within the error band); throughput up 83.2% -> keep
- {"workload": {"dtype": "fp8"}}: b is 35.8% faster per step; throughput up 55.6% -> keep
- {"launch_gap_ns": 300}: b is 4.0% faster per step (within the error band) -> within error band
**Kept** {"workload": {"batch": 64}}: tokens/s +83.2%

### Round 2: ... Biggest single cost: sm100_fmha_decode_b64_hq128_hkv1_sq1_skv4096_d576 (17%) ...
**Kept** {"workload": {"batch": 128}}: tokens/s +72.5%

### Round 3: ...
- {"workload": {"batch": 256}}: b is 45.6% slower per step; throughput up 37.3%; step 3004 us -> over latency limit
- {"workload": {"dtype": "fp8"}}: b is 28.2% faster per step; throughput up 39.3% -> keep

### Final: batch 128, fp8, EP8
step 1481 (±26%) us, 691623 (±26%) tokens/s
top kernel sm100_fmha_decode_b128_...: tensor pipeline-bound: 47.8% of tensor peak and 64.5% of HBM,
past the mbarrier parking, schedulers wait on tensor_wait (27%)
  hardware sensitivity {"hbm_bw_gbps": "x1.5"}: b is 1.4% faster per step (within the error band)
```

What the agent found, in order:

1. **The baseline is weight streaming.** The batch-32 GEMMs run at 98% of HBM while three quarters of
   every UMMA tile is padding. Doubling the batch costs 9% in step time, inside the error band, and
   adds 83% throughput. That is the classic decode batching argument, and it falls out of the stall data.
2. **At batch 128 the bottleneck moves.** MLA decode attention, which is compute-heavy with 128 query
   heads sharing one latent, becomes the largest kernel. Its MMA pipeline, not HBM, sets its time.
3. **FP8 GEMMs still pay** (+39%), because the dense and expert GEMMs remain HBM-bound.
4. **HBM bandwidth no longer helps** (+1.4%, inside the band). A hardware engineer reads that as: for
   this configuration, a faster tensor pipeline for attention is worth more than more HBM.

Over the three rounds, tokens/s rose 4.4× and the step got 9% faster. The numbers carry the B200
band: B200 is an uncalibrated preset, so read the absolute values as estimates and the
comparisons as the result.

## Limits

- **Absolute accuracy follows calibration.** Off B300 and A100, trust differences more than
  absolute values. The bands say this, and the run warnings repeat it.
- **Partial models.** `layers: N` simulates the first N layers. Step time is for those layers only,
  and dense and MoE layers differ, so a warning is attached.
- **Host overhead is not modelled** for lowered models. A host-bound engine can be much slower than
  its GPU time (see docs/CAPTURE.md). Replay a capture with `host: true` to include it.
- **Run time.** Fast-mode runs take seconds to a minute. Exact mode and long-context kernels (an 8k
  causal attention on B300 takes minutes) can be slow, which matters for an interactive agent.
- **Session state.** Runs live in the server process. Restarting the MCP server clears them, and
  identical inputs simply re-simulate.
