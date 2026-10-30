# 5. Ask the rack: tools, mechanisms, numbers

**What you will do:** ask rack-scale questions the way you would ask a colleague, and read the
answers the way rackscale gives them: a number, its error bar, the textbook number beside it, and
what bounds it. **Time:** 15 minutes. **Needs:** `maturin develop --release` (or
`pip install 'rackscale[agent]'` for the MCP server).

Every call below is also a tool an LLM can use (`rackscale-mcp`, or `rackscale.tools` for any
tool-use API), and an LLM gets back exactly what you see here.

## The shape of every answer

```python
from rackscale import api

r = api.run("hgx_b300_8", {"model": "llama3-8b", "phase": "decode", "batch": 32, "context": 4096,
                           "parallel": {"tp": 8}})
r["answer"]
```

```
{'quantity': 'step time', 'value': 3151.0, 'unit': 'us',
 'band': {'lo': 2428.0, 'hi': 3874.0, 'band_pct': 22.95, 'worst_seen_pct': 83.5},
 'closed_form': {'value': 1557.0, 'formula': 'roofline replay of the same program: ...'},
 'sim_over_closed_form': 2.024,
 'bottleneck': 'collectives', 'share_of_gpu_time': 0.309,
 'why': 'communication is on the critical path: GPUs wait for all-reduces / all-to-alls whose time
         is alpha + bytes / link bandwidth. Overlap them with compute, shrink the group (less TP / EP),
         use in-switch reduction (NVLS) or a faster fabric.'}
```

Read it in four steps:

1. **The number and its band.** 3151 µs, and the band (±23%) comes from how far the simulator has
   been from measured hardware on this mix of kernels (`api.accuracy_table()` has the evidence).
2. **The textbook.** A perfect machine running the *same program* (every kernel at its roofline,
   every all-reduce at alpha + bytes / bandwidth, perfect overlap) would take 1557 µs. Reality is
   2.0x that.
3. **Where the 2x goes.** `toolkit.textbook(rec)["gap_split_us"]` splits it: kernels 356 µs slower
   than their rooflines, all-reduces 634 µs slower than their closed form (small all-reduces are
   all latency, not bandwidth), and 604 µs of gaps with neither kernels nor collectives (eager
   kernel launches).
4. **The bottleneck and why.** The largest part of GPU time is all-reduces (31%), the rest is
   spread over HBM-, latency- and ALU-bound kernels and launch gaps.

Act on it: `api.what_if(r["run_id"], {"launch_gap_ns": 300})` (CUDA graphs) makes the step 15.8%
faster, and the verdict says the difference beats the error band.

## When you don't know why: `learn`

```python
from rackscale.learn import learn
learn("all_reduce")["worked_example"]
# {'bytes': '67.1 MB', 'ranks': 72, 'link_bw_per_gpu': '900 GB/s', 'ring_us': 152.1, 'nvls_us': 79.6,
#  '1_MiB_us': 6.2, 'alpha_us': 5.0,
#  'lesson': 'big all-reduces are bandwidth terms (NVLS halves them); small ones are all alpha'}
```

Each of the 20 concepts (`learn()` lists them) has the idea in a few sentences, the formula, a
worked example computed now on a real preset, the book chapter that derives it and the tool call
that shows it in the simulator. Change the example with parameters:
`learn("pipeline_bubble", stages=8, micro_batches=32)` gives 7/39 = 0.18.

## Inside one GPU: why a decode GEMM wants split-K

```python
from rackscale import toolkit
t = toolkit.gemm_tiling("hgx_b300_8", 16, 8192, 8192, vary={"split_k": [1, 2, 4]})
[(x["variant"], x["time_us"]["value"], x["waves"], x["bound"]) for x in t["rows"]]
```

| split-K | µs | waves | bound | wave-quantised roofline µs |
|---|---|---|---|---|
| 1 | 35.8 | 0.43 | memory latency | 19.8 |
| 2 | 27.1 | 0.87 | HBM | 20.1 |
| 4 | 25.3 | 0.87 | HBM | 20.4 |

Sixteen rows give 64 output tiles of 128x128 for 148 SMs, so without a split 84 SMs have nothing
to do and the 64 that work cannot keep enough bytes in flight (Little's law:
`learn("littles_law")` says 12.5 kB per SM at 6.8 TB/s and 271 ns). Splitting K fills the GPU and
the kernel becomes HBM bound, 1.24x its roofline. Splitting further buys little, because the extra
fp32 partials cost bytes.

## The tail: stragglers and a slow link

```python
toolkit.tail_latency(ranks=72, sigma=0.05)["stragglers"]["overhead_closed_form"]   # 0.132
toolkit.tail_latency(ranks=72, sigma=0.05, simulate=True)["answer"]                # simulated, ~0.11
```

Each GPU alone loses σ·E|Z| = 4% to 5% jitter; 72 GPUs that synchronise every kernel lose the
expected *maximum*, 13% in closed form and 11% simulated (the all-reduce absorbs part of the
spread). `plane_scale=0.5` adds one NVLink at half rate: NVLS stripes every collective across all
18 links in lockstep, so the closed form says 2x on the bandwidth term (+50% simulated), while a
ring barely notices (+0.1%).

## Power: what a lower cap costs

```python
p = api.run("hgx_b300_8", {"model": "llama3-70b", "phase": "prefill", "batch": 1, "seqlen": 8192,
                           "parallel": {"tp": 8}, "layers": 4})
toolkit.power_thermal(p["run_id"], caps_w=[800], coolant_c=45)
```

Prefill GEMMs draw the 1100 W cap, so the clock follows the power budget: at 800 W the closed form
(clock ~ (cap - idle)^(1/3) on the tensor-bound share of the step) predicts +5.7%, and the
simulator, which re-runs every kernel at the clock its own power allows, gives +9.2% with 14%
less energy per step. `coolant_c` adds the thermal limit, (T_max - T_coolant) / R_th: 45 °C
coolant still allows 1125 W.

## A GPU we don't have

```python
toolkit.predict_gpu("hgx_b300_8", {"model": "llama3-70b", "phase": "decode", "batch": 64,
                                   "context": 8192, "parallel": {"tp": 8}, "layers": 4},
                    {"hbm_bw_gbps": "x1.5"})
```

Decode is "memory bound", so 1.5x HBM should be nearly 1.5x faster. The first-order prediction
(each kernel scaled by its roofline ratio) says 1.22x; the simulator says **1.06x**, inside the
error band. The two bottleneck splits say why. On the real B300, 62% of GPU time is HBM-bound
kernels. On the faster one only 5% is: decode attention becomes ALU-bound (29%, the softmax
exponentials now take longer than its bytes), and the skinny GEMMs become latency-bound (32%: they
keep too few bytes in flight to use the new bandwidth, Little's law again). All-reduces (22%) and
gaps (12%) do not change at all. A roofline sees none of this.

## Tokens or bytes

```python
toolkit.tokens_or_bytes(simulate=True)["answer"]
```

H-Net 1-stage XL generates 0.43x the text bytes per second of a BPE Transformer of similar size at
batch 64 and 8 kB of context on one B300. The closed form (HBM bytes per byte of text) says 0.76x,
because it assumes every kernel streams at the HBM roof. The BPE model nearly does: 84% of its GPU
time is HBM-bound kernels (its decode attention moves 6.3 TB/s). H-Net spends only 40% there; the
small byte-level kernels of its outer stages are latency-, ALU- and shared-memory-bound (39%) and
20% of the time no kernel runs at all.

## Writing it up

`toolkit.html_report([run_id, ...], "report.html", labels=[...])` writes a page with the time
split, kernels by bound, stalls, the roofline and collectives, for one run or several side by side.

## How much the tools help an LLM

[`benchmarks/agent_qa`](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/tree/main/benchmarks/agent_qa)
asks Claude 19 rack-scale questions twice, once with only the rackscale tools and once with no
tools, and grades both. Its README has the results and what they do and do not show.
