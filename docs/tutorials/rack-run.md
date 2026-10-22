# 2. A model on an NVL72 rack

**What you will do:** simulate a DeepSeek-V3 decode step on a 72-GPU GB200 NVL72, find what
bounds it, try DeepSeek's two-micro-batch overlap, halve the NVLink bandwidth, and open the
whole rack in the viewer. **Time:** 10 minutes. **Needs:** `pip install rackscale` (no GPU).

## The run

```bash
rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 4 --fast
```

What this asks for:

| flag | meaning |
|---|---|
| `deepseek-v3` | the built-in architecture: 61 layers, MLA attention, 256 routed experts, top-8 (`rackscale models` lists all) |
| `--system gb200_nvl72` | 72 B200 GPUs, 18 NVLink5 switch chips, 900 GB/s per direction per GPU (`rackscale system gb200_nvl72`) |
| `--ep 72` | expert parallelism over all 72 GPUs: every MoE layer does an all-to-all dispatch and combine |
| `--batch 64 --context 4096` | 64 sequences per GPU, each with 4096 tokens of KV cache |
| `--layers 4` | simulate the first 4 layers (3 dense, 1 MoE). Memory is still checked for all 61 |
| `--fast` | simulate each distinct kernel once at cycle level and reuse it across ranks and layers; the fabric and every dependency are still simulated exactly |

It takes about 4 seconds:

```
== deepseek-v3 decode b64 ctx4096 Parallel(tp=1, ep=72, dp=1, comm_sms=20, pp=1) on GB200 NVL72 (72 GPUs) ==
makespan 1556.5 us   MFU 12.5%
memory (rank 0): 70.8 of 186 GB  [kv_cache 18.4, weights 52.4]
GPU idle 8.4%   tensor-core idle in kernels 62.4%
scheduler-cycle anatomy:
  async_wait            59.95%  ##############################
  tensor_wait           17.51%  #########
  tma_wait              10.57%  #####
  ...
kernel                                                       n   total_us   TFLOPS     GB/s    tc%
sm100_fmha_decode_b64_hq128_hkv1_sq1_skv4096_d576          288    27738.6    802.7     3527   56.2
sm100_gemm_bf16_64x129280x7168_tile128x256                  72    18909.9    903.3     7197   36.1
sm100_gemm_bf16_64x36864x7168_tile128x256                  216    16923.4    863.4     7010   34.5
...
collective     algo         size ranks     n   mean_us  busbw GB/s
AllToAll       Direct    3.5 MiB    72     1      23.5         154
AllToAll       Direct    7.0 MiB    72     1      26.8         270
```

## Reading it

- **`makespan`** is the step time: from the first launch to the last kernel finishing on any
  GPU. **MFU** is model FLOPs over the rack's dense BF16 peak for that time.
- **Kernels** are summed over all 72 GPUs (`n` is the launch count, `total_us` the sum of their
  durations). The LM head (`64x129280x7168`) and the dense GEMMs stream weights at about
  7.1 TB/s, close to HBM speed: **decode is weight-bandwidth bound even on GB200**. With 64 rows
  against a 128-row UMMA tile, half the tensor work is padding, so `tc%` is about 35%.
- **MLA attention** (`d576`: 128 query heads share one 576-wide latent) runs at about
  800 TFLOPS, compute heavy rather than HBM bound. It is the largest kernel.
- **The all-to-all** is about 25 µs per call on NVLink5. **GPU idle** is the time GPUs sit
  between kernels, waiting for launches or for the collective.

## What-if 1: DeepSeek-style overlap

```bash
rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 4 --fast --overlap
```

The step gets *slower*, 1556.5 → 2487.6 µs, even though GPU idle drops from 8.4% to 2.2%. Each
half-batch reads the full weights again, and the all-to-all it hides is small on NVLink. Overlap
pays when dispatch and combine are a large part of the step: over InfiniBand, at larger EP, or
with FP8 weights.

## What-if 2: your own hardware

A system is a YAML file: start from a preset and override what you want to study. Halve every
NVLink link:

```yaml title="nvl_half.yaml"
preset: gb200_nvl72
name: GB200 NVL72, half NVLink bandwidth
topology:
  link: {gbps_per_dir: 25}
```

```console
$ rackscale system nvl_half.yaml
GB200 NVL72, half NVLink bandwidth: 72 GPUs, 90 nodes, 1296 links
GPU0 injection bandwidth: 450 GB/s per direction; min link latency 80 ns
$ rackscale decode deepseek-v3 --system nvl_half.yaml --ep 72 --batch 64 --context 4096 --layers 4 --fast
makespan 1574.1 us   MFU 12.4%
AllToAll       Direct    3.5 MiB    72     1      30.5         119
AllToAll       Direct    7.0 MiB    72     1      37.4         193
```

Half the bandwidth costs only 1.1% of step time at this batch, because the all-to-all is a
small part of it. The fabric's links come from `topology`, so overriding a GPU's own `nvlink`
field does not change them.

Other things worth changing: `gpus: [{preset: gb200, count: 72, overrides: {sm_clock_mhz: 1800}}]`
for a power-capped clock, `calibration: <file>` on a GPU group to use measured numbers, or
`kind: multi_node` for InfiniBand clusters (see
[`configs/`](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/tree/main/configs)).
A misspelled key is an error that names the closest valid key.

## What-if 3: fewer, bigger GPUs

```bash
rackscale decode deepseek-v3 --system hgx_b200_8 --ep 8 --batch 64 --context 4096 --layers 4 --fast
```

```
memory (rank 0): 216.1 of 180 GB  [kv_cache 18.4, weights 197.7]
WARNING: 8 rank(s) need up to 216.1 GB but B200 (HGX, 1000 W) has 180 GB of HBM: this configuration would not fit ...
```

The simulator still runs the step, but tells you the layout cannot exist. Use a larger system,
FP8 weights, or a smaller batch.

## See the whole rack

```bash
rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 4 \
    --fast --viz ds.html --open
```

The page starts at the rack: the NVLink fabric with per-link utilisation over time, every
stream of every GPU on one timeline, and the bubble anatomy. Click a GPU to see its 148 SMs with
one bar per CTA, a kernel to see its warp roles, and an SM to see every warp cycle by cycle down
to the SASS. A prebuilt version, with both schedules side by side, is in the
[examples gallery](../examples.md).

For a Perfetto timeline instead (one track per stream, counters for tensor and DRAM
utilisation, and with `--per-sm` one track per SM), use `--trace ds.json` and open it at
[ui.perfetto.dev](https://ui.perfetto.dev).

## From Python

```python
import rackscale as rs
from rackscale import lower, models

par = rs.Parallel(ep=72, comm_sms=20)
for overlap in (False, True):
    wl = lower.decode_step(models.DEEPSEEK_V3, par, 64, 4096, overlap=overlap, layers=4)
    rep = rs.simulate("gb200_nvl72", wl, fast=True)
    print(f"overlap={overlap}: {rep.makespan_us:.1f} us, GPU idle {rep.bubbles['gpu_idle_fraction']:.1%}")
```

`prefill`, `train` (with `--pp`, `--micro-batches`, `--schedule gpipe|1f1b|zb`, `--optimizer`)
and `lower.async_rl` work the same way. To find the best layout automatically, see the
[sweep study](https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation/tree/main/research).
