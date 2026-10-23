# 4. Drive it from an agent (MCP)

**What you will do:** give an AI agent the simulator as a set of tools, ask it a performance
question in plain words, and run the same loop from a script. **Time:** 5 minutes.
**Needs:** `pip install 'rackscale[agent]'` (adds the MCP SDK).

## Why a simulator makes a good tool

Performance engineering is a loop: measure, find what bounds you, change one thing, measure
again. On hardware you do not have, the simulator is the measurement. rackscale exposes that
loop as tools whose answers carry the information an agent needs to not fool itself:

- every time comes with an **error band** taken from measured validation,
- every comparison comes with a **verdict** on whether the difference beats that band,
- every run carries its **provenance** (which numbers are measured, which are presets).

| tool | what it does |
|---|---|
| `list_hardware`, `list_models` | systems, GPU knobs that can be changed, built-in architectures |
| `simulate(system, workload)` | a run id, step time with its band, top kernels |
| `find_bottlenecks(run_id)` | each kernel classed (HBM, tensor, memory latency, ...) with ready-to-run what-ifs |
| `inspect_kernel(run_id, name)` | per-warp-role stalls and the SASS instructions warps wait on |
| `what_if(run_id, change)`, `sweep(run_id, key, values)` | re-simulate with a change and compare |
| `compare_runs`, `list_runs`, `accuracy` | bookkeeping, and the validation table behind the bands |

## With Claude Code (or any MCP client)

```bash
pip install 'rackscale[agent]'
claude mcp add rackscale -- rackscale-mcp
```

Then ask, for example:

> Simulate DeepSeek-V3 decode on HGX B200 at EP8, batch 32 per rank, first 4 layers. What bounds
> it, and what would you change? Only keep a change if it beats the error band.

The server speaks stdio by default; `rackscale-mcp --transport streamable-http` serves HTTP for
clients that prefer it. For Claude Desktop, add it to `claude_desktop_config.json`:

```json
{"mcpServers": {"rackscale": {"command": "rackscale-mcp"}}}
```

## The same loop from Python

`rackscale.api` takes and returns JSON (schema `rackscale-api/v1`):

```python
from rackscale import api

r = api.run("hgx_b200_8", {"model": "deepseek-v3", "phase": "decode", "batch": 32, "context": 4096,
                           "parallel": {"ep": 8}, "layers": 4})
r["step_time_us"]
# {'value': 1672.6, 'lo': 1231.9, 'hi': 2113.4, 'band_pct': 26.35}

b = api.bottlenecks(r["run_id"])
b["summary"]
# 'Step 1673 us on HGX B200 8-GPU (8 GPUs). 60% of GPU time is in hbm-bound kernels. ...
#  Biggest single cost: sm100_gemm_bf16_32x129280x7168_tile128x256 (16%), HBM-bound'
b["suggested_what_ifs"][0]
# {'what_if': {'workload': {'batch': 64}}, 'why': 'weights are streamed once per step; ...'}

api.what_if(r["run_id"], {"workload": {"dtype": "fp8"}})["comparison"]["verdict"]
# 'b is 34.6% faster per step; throughput up 52.8%'
```

For any LLM tool-use API, `rackscale.tools.specs()` returns the tool definitions (name,
description, JSON schema) and `rackscale.tools.call(name, args)` runs one.

## A complete agent

[`examples/agent/perf_agent.py`](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/blob/main/examples/agent/perf_agent.py)
runs the loop end to end. The default policy is scripted (it follows the simulator's own
suggested what-ifs, so it needs nothing but rackscale); `--policy claude` lets Claude drive the
tools through the Anthropic API.

```bash
python examples/agent/perf_agent.py                     # DeepSeek-V3 decode on HGX B200, about a minute
python examples/agent/perf_agent.py --policy claude --goal "maximise tokens/s with step time under 2 ms"
```

The scripted run ends with what it kept and why:

```
**Kept** {"workload": {"dtype": "fp8"}}: tokens/s +33.8%

### Final: {'model': 'deepseek-v3', 'phase': 'decode', 'batch': 128, ..., 'dtype': 'fp8'}
step 1712 (±26%) us, 598258 (±26%) tokens/s
top kernel sm100_fmha_decode_b128_hq128_hkv1_sq1_skv4096_d576: tensor pipeline-bound ...
  role softmax        tensor_wait      51%
  role mma_issuer     async_wait       100%
  hardware sensitivity {"hbm_bw_gbps": "x1.5"}: b is 1.3% faster per step (within the error band: not a reliable difference)
```

Note the last line: a 1.3% gain inside a 26% band is reported as *not a reliable difference*,
which is exactly what an agent should not act on. The full reference is
[Agent API and MCP](../AGENT_API.md).
