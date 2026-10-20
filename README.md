# rackscale

A cycle-level GPU simulator that scales to a whole rack. You give it a hardware description
(NVL72, HGX, a multi-node IB cluster, or your own wiring) and a workload. The workload can be a
model config, a captured vLLM/SGLang/PyTorch profiler trace, real SASS, or an Accel-Sim trace.
rackscale predicts step time and explains where every bubble comes from, down to the warp
scheduler of each SM.

```
$ rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 4 --fast
== deepseek-v3 decode b64 ctx4096 Parallel(tp=1, ep=72, dp=1, comm_sms=20) on GB200 NVL72 (72 GPUs) ==
makespan 1899.3 us   MFU 10.2%
GPU idle 5.2%   tensor-core idle in kernels 75.4%
scheduler-cycle anatomy:
  async_wait            58.30%  #############################
  short_scoreboard      11.66%  ######
  tma_wait              11.11%  ######
  tensor_wait           10.46%  #####
  ...
kernel                                                       n   total_us   TFLOPS     GB/s    tc%
sm100_fmha_decode_b64_hq128_hkv1_sq1_skv4096_d576          288    22735.7    979.3     4304   39.2
sm100_gemm_bf16_64x129280x7168_tile128x256                  72    19458.2    877.8     7130   35.1
sm100_gemm_bf16_64x7168x16384_tile128x256                  288    19446.8    445.2     3595   17.8
  ...
  AllToAll       Direct      3670016 B ranks=72  busbw 509 GB/s
  AllToAll       Direct      7340032 B ranks=72  busbw 722 GB/s
```

Rust does the simulation. Python handles the front-end: model lowering, trace capture and
reports. Results are deterministic, and the simulator runs in parallel across GPUs.

## What is modelled

| Layer | Model |
|---|---|
| **SM** (`rsim-gpu`) | 4 SMSPs with GTO/LRR warp schedulers. Control-bit scoreboards (stall/yield/SB barriers decoded from real SASS) or per-register scoreboards. Per-pipe issue intervals, MIO/LSU queues, a sectored L1, shared-memory bank conflicts. CTA barriers, `cp.async` groups and mbarriers with phase tracking. **Hopper** `HGMMA` warpgroup queues and TMA (`UTMALDG`, multicast). **Blackwell** `tcgen05`/UMMA (`UTC*MMA`, `UTCBAR`, `LDTM`/`STTM`, TMEM) on an SM-wide tensor queue. Stall reasons use Nsight names: `long_scoreboard`, `mio_throttle`, `barrier`, and so on, plus `tma_wait` and `tensor_wait`. |
| **GPU** | CTA dispatcher (occupancy from regs/smem/warps). Per-die L2 with bandwidth servers, HBM, and the B200 die-to-die link. Multiple concurrent kernels and streams. Comm kernels can reserve SMs. |
| **Fabric** (`rsim-net`) | NVLink4/5 and NVSwitch (with in-network reduction), IB leaf/spine, any explicit graph. ECMP shortest paths and chunked cut-through transport with per-link FCFS. |
| **Collectives** | Ring (multi-ring across islands), NVLS/SHARP, direct all-to-all with staggered peers, broadcast and send/recv. Reported as NCCL bus bandwidth. |
| **Rack** (`rsim-rack`) | Conservative parallel discrete-event simulation. One logical process per GPU plus the network. Lookahead is the minimum link latency, and LPs run on rayon. |
| **Workloads** | SPMD or per-rank programs with streams, events, loops, collectives and p2p. Kernels can be synthetic (GEMM, grouped GEMM, FA3/FA4-style attention, MLA decode, norms, elementwise), parsed SASS (`cuobjdump -sass`), Accel-Sim/NVBit traces, or measured fixed durations. |

## Install

```bash
# Rust CLI
cargo build --release            # -> target/release/rsim
# Python package (needs maturin)
python -m venv .venv && . .venv/bin/activate
pip install maturin && maturin develop --release
```

## Use it

**From a model config** (Llama 3, Mixtral, Qwen3-MoE and DeepSeek-V3 presets are in
`rackscale.models`):

```python
import rackscale as rs
from rackscale import lower, models

wl = lower.decode_step(models.DEEPSEEK_V3, rs.Parallel(ep=72), batch=64, context=4096, layers=4, overlap=True)
rep = rs.simulate("gb200_nvl72", wl, fast=True, trace="out/ds.json")   # open in ui.perfetto.dev
print(rep.summary())
```

`lower.prefill`, `lower.train_step` (DP gradient reduce-scatter overlapped on a comm stream) and
`lower.async_rl` (rollout and trainer pools with weight push) cover the other workload families.
See [`examples/`](examples).

**From a real engine**. Profile vLLM, SGLang or a training loop with the PyTorch profiler
(`record_shapes=True`) and feed in the trace. GEMMs are re-simulated for the *target* hardware.
NCCL kernels become collectives on the target fabric. Everything else replays its measured
duration. Details are in [docs/CAPTURE.md](docs/CAPTURE.md).

```bash
rackscale capture vllm_rank0.pt.trace.json.gz --system gb200_nvl72 --world 8 --group block:8
```

**From SASS**:

```bash
rsim sass examples/sass/vecadd_sm90.sass --inspect     # decoded control bits, classes
rsim kernel --gpu h100_sxm gemm 4096 4096 4096         # one kernel, per-SM stall anatomy
```

**Your own hardware**. A system is YAML. You can start from a preset and override anything,
down to per-opcode latencies, or point at a calibration run. See [`configs/`](configs):

```yaml
preset: hgx_b200_8
gpus:
  - preset: b300
    count: 8
    calibration: calibration/results/b300-sxm6-ac-sm103/calibration.json
```

### Exact and fast mode

By default every GPU is co-simulated at cycle level. `fast=True` (`--fast`) simulates each
distinct (kernel, free-SM) pair once on a cold GPU and reuses the result on every rank. On a
symmetric workload this is 30× or more faster.

| DeepSeek-V3 decode, 4 layers, b32 | exact | fast |
|---|---|---|
| HGX B200 x8 (EP8) | 2109.9 us, 57 s | 2121.1 us (+0.5%) |
| GB200 NVL72 (EP72) | 1712.3 us, 6.5 min | 1729.0 us (+1.0%), 12 s |

Fast mode ignores L2 reuse between kernels. It also treats concurrent kernels as sharing SMs in
blocks rather than interleaving CTAs. Use exact mode when the question is about overlap.

## Validation so far

These are sanity checks against public numbers. They do not replace calibration
(see [docs/CALIBRATION.md](docs/CALIBRATION.md)).

| Check | Simulated | Reference |
|---|---|---|
| H100 BF16 GEMM 4096³ | 676 TFLOPS (68% of peak) | cuBLAS ~600-750 |
| B200 BF16 GEMM 8192³ | 1259 TFLOPS (56%) | early CUTLASS ~1.2-1.5 PF; tcgen05 timing uncalibrated |
| H100 elementwise stream | 95% of HBM peak | ~90-93% |
| H100 decode attention | ~89% of HBM | FlashDecoding ~85-90% |
| HGX H100 ring all-reduce (large) | 340 GB/s busbw | nccl-tests ~360-370 |
| GB200 NVL72 ring / NVLS all-reduce | 744 / 1462 GB/s | NVLS is optimistic (see limitations) |
| NVL72 all-to-all | 764 GB/s | |
| 2x HGX H100 over 400G IB | 340 GB/s (NVLink-bound ring) | |

Simulation speed is about 33M SM-cycles/s per core in exact mode.

## Limitations

These are the honest ones:

- **Parameters.** Hopper latencies come from microbenchmark papers. Blackwell `tcgen05`, TMEM and
  L2 slicing are estimates until you load a calibration file. B300 has measured data in
  `calibration/`. Every preset field carries a provenance tag (`spec`, `ubench`, `est`,
  `calibrated`).
- **No functional execution.** The SASS walker follows control flow using branch profiles (trip
  counts and taken probabilities) and affine address generators. It does not compute register
  values. Data-dependent control flow needs a trace (Accel-Sim/NVBit) or a profile.
- **Synthetic kernels** follow CUTLASS/FA3/FA4 structure (warp roles, stages, barriers). They are
  not bit-identical to vendor binaries. To study a specific binary, feed its SASS.
- **Network.** There is no congestion control or adaptive routing. NVLS assumes ideal switch
  reduction throughput, so it is optimistic. IB is modelled as lossless credit-free links.
- **MoE routing** is balanced by default. Skew is expressed by giving `grouped_gemm` uneven `ms`.

## Layout

```
crates/
  rsim-core   time, deterministic event queue, conservative PDES, Perfetto writer
  rsim-hw     GPU/switch/link specs, presets, topology builder, calibration loader
  rsim-sass   SASS parser, control bits, ISA classes, Accel-Sim traces, walker
  rsim-gpu    SM / memory / GPU model, synthetic kernels
  rsim-net    fabric, transport, collective algorithms
  rsim-rack   workloads, rack runner, reports
  rsim-cli    `rsim` binary
  rsim-py     Python bindings (rackscale._native)
python/rackscale   models, lowering, profiler capture, CLI
configs/           example systems     examples/   runnable demos
docs/              ARCHITECTURE, CALIBRATION, WORKLOAD_FORMAT, CAPTURE
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The code is licensed under Apache-2.0.
