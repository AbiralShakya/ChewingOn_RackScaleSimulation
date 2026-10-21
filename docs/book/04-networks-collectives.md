# 4. Networks and collectives

Once a model no longer fits on one GPU, GPUs have to exchange data every layer. This chapter
gives the cost model for the four collectives that parallelism needs (all-reduce, all-gather,
reduce-scatter, all-to-all), applies it to the fabrics that matter (NVLink with NVSwitch, the
72-GPU NVL72 domain, InfiniBand between nodes), and compares it with what the simulator
produces.

**Everything in this chapter is simulated on presets.** We have measured single GPUs only;
no NVLink or InfiniBand number below comes from hardware. The fabric model is described in
section 4.5, and chapter 9 lists what it would take to calibrate it.

## 4.1 The fabrics

| fabric | per-GPU bandwidth, each direction | topology | preset latency |
|---|---|---|---|
| HGX H100 (NVLink4 + NVSwitch3) | 18 links × 25 GB/s = **450 GB/s** | 8 GPUs, 4 switch chips | 20 ns link + 100 ns switch + 60 ns port |
| HGX B200 (NVLink5) | 18 × 50 GB/s = **900 GB/s** | 8 GPUs, 2 switch chips | same |
| GB200 / GB300 NVL72 | 18 × 50 GB/s = **900 GB/s** | 72 GPUs, 18 switch chips, one link from every GPU to every switch | same |
| InfiniBand NDR (400G) | **50 GB/s** per NIC, one NIC per GPU | rail optimised: GPU *r* of every node on leaf switch *r*, leaves joined by a spine | 600 ns NIC + 130 ns switch |

NVSwitch has a property InfiniBand does not: **in-network reduction** (NVLink SHARP, "NVLS").
A GPU can send its buffer to the switch, the switch adds the buffers of all GPUs, and every GPU
reads back the sum. NVL72 is a single NVLink domain, so all 72 GPUs see each other at 900
GB/s; across NVL72 racks, or between HGX boxes, traffic falls to InfiniBand at 50 GB/s per
GPU, 18× less.

## 4.2 The cost model

Write a transfer of *n* bytes on a link of bandwidth *W* with latency α as `α + n/W`. For a
group of *N* GPUs, each holding a buffer of *S* bytes:

| collective | result on every GPU | data each GPU must send | time (bandwidth term) |
|---|---|---|---|
| all-gather | all *N* buffers concatenated (*N·S*) | (*N*−1)·*S* | (*N*−1)·*S* / *W* |
| reduce-scatter | 1/*N* of the sum | (*N*−1)/*N* · *S* | (*N*−1)/*N* · *S* / *W* |
| all-reduce | the sum (*S*) | 2(*N*−1)/*N* · *S* | 2(*N*−1)/*N* · *S* / *W* |
| all-to-all | buffer *j* of every GPU, on GPU *j* | (*N*−1)/*N* · *S* | (*N*−1)/*N* · *S* / *W* |

The all-reduce row is the important one: **all-reduce = reduce-scatter + all-gather**, and its
bandwidth cost does not grow with *N*. Each GPU sends about twice its buffer, whether there are
8 or 72 GPUs.

NCCL reports **bus bandwidth** (busbw) so that numbers for different *N* are comparable: the
algorithm bandwidth *S*/*T* times the factor in the table (2(*N*−1)/*N* for all-reduce,
(*N*−1)/*N* for the others). For a ring algorithm, busbw tops out at the per-GPU link
bandwidth *W*. We use busbw throughout.

**The latency term.** A ring all-reduce runs 2(*N*−1) dependent steps, so it costs
`2(N−1)·α` even for a 4-byte buffer. On NVL72 that is 142 steps. Tree, NVLS and direct
algorithms exist to cut this.

**NVLS.** With reduction in the switch, each GPU sends *S* bytes up and receives *S* bytes
back, on separate directions of its links. The time is about *S*/*W* instead of
2(*N*−1)/*N*·*S*/*W*, so busbw can reach 2(*N*−1)/*N*·*W*: 1575 GB/s on an 8-GPU NVLink5 board,
above the link rate.

## 4.3 What the simulator gives

```bash
python docs/book/scripts/collectives_sweep.py              # about a minute
python docs/book/scripts/collectives_sweep.py --algo ring --kinds all_reduce
```

Selected rows (bytes are per GPU; each run is the collective alone on an idle system):

| system | GPUs | collective | 4 KB | 1 MB | 16 MB | 256 MB | busbw at 256 MB |
|---|---|---|---|---|---|---|---|
| HGX H100 | 8 | all-reduce (NVLS) | 1.1 µs | 6.5 µs | 59.5 µs | 725 µs | 648 GB/s |
| HGX B200 | 8 | all-reduce (NVLS) | 1.1 | 3.8 | 25.7 | 363 | **1295** |
| HGX B200 | 8 | all-reduce (ring) | 7.9 | 10.3 | 46.3 | 622 | 755 |
| HGX B200 | 8 | all-gather (ring) | 4.0 | 5.2 | 23.2 | 311 | 755 |
| HGX B200 | 8 | all-to-all | 0.6 | 1.8 | 19.8 | 308 | 764 |
| NVL72 | 72 | all-reduce (NVLS) | 1.1 | 3.8 | 44.9 | 377 | **1405** |
| NVL72 | 72 | all-reduce (ring) | **79.8** | 91.8 | 123 | 772 | 686 |
| NVL72 | 72 | all-gather (ring) | 39.9 | 45.9 | 61.5 | 386 | 686 |
| NVL72 | 72 | all-to-all | 0.7 | 6.7 | 22.3 | 347 | 764 |
| 2 × HGX H100, IB | 16 | all-reduce (ring) | 21.4 | 23.9 | 96.0 | 1488 | 338 |
| 2 × HGX H100, IB | 16 | all-to-all | 1.9 | 28.7 | 390 | **5923** | **42** |

What to take from it:

- **Large messages run at 80 to 85% of the link.** Ring all-reduce and all-gather on NVLink5
  reach 755 GB/s busbw against 900; all-to-all 764. The rest is chunking and pipeline fill in
  the model.
- **NVLS beats the ring by 1.7× for big all-reduces** (363 vs 622 µs on HGX B200) and by
  70× for tiny ones on 72 GPUs (1.1 vs 79.8 µs), because it has no 2(*N*−1)-step chain.
- **Ring latency grows with *N*.** A 4 KB ring all-reduce on 72 GPUs costs 80 µs, about
  0.56 µs per step for 142 steps. At decode sizes (a few hundred KB per layer) the latency term
  is the whole cost. This is why tensor parallelism across 72 GPUs only makes sense with NVLS
  or tree algorithms.
- **H100 to B200 is 2× on the link**, and the all-reduce times follow it (725 vs 363 µs).
- **InfiniBand is the cliff.** All-to-all across two 8-GPU nodes reaches 42 GB/s busbw, 18×
  slower than NVLink5. Mixture-of-experts dispatch over IB is what DeepSeek-style
  two-micro-batch overlap exists to hide (chapter 6).

Two modelling notes for the InfiniBand rows. The all-reduce uses NCCL's rail-style multi-ring:
one ring per local GPU, each entering the other node on its own NIC, so the 8 NICs together
carry the inter-node traffic and busbw reaches 338 GB/s (close to the NVLink4 side). The
all-to-all routes every inter-node message over the network, including those between
different rails, which then cross the spine; NCCL's PXN would first move such data over NVLink
to the GPU on the matching rail. The 42 GB/s is therefore a pessimistic number.

## 4.4 Worked example: the all-reduce of a TP layer

In tensor parallelism (chapter 5) each transformer layer ends its attention block and its MLP
block with an all-reduce of the activations, *T* × *D* × 2 bytes in BF16. For Llama-3-70B
(*D* = 8192):

| phase | tokens *T* | bytes per all-reduce | NVLS time on HGX B200 (table) | 
|---|---|---|---|
| decode, batch 32 | 32 | 512 KB | about 2.5 µs |
| decode, batch 256 | 256 | 4 MB | about 8 µs |
| prefill, 8k tokens | 8192 | 128 MB | about 180 µs |

Two all-reduces per layer, 80 layers: decode at batch 32 spends about 0.4 ms per step in
all-reduce latency alone, comparable to a whole decode step on B200 (chapter 6). Prefill moves
much more data but also does far more compute per byte; chapter 5 does the comparison.

## 4.5 How the fabric is simulated

The topology is a graph of GPUs and switches built from the system YAML
([`configs/gb200_nvl72.yaml`](../../configs/gb200_nvl72.yaml)). Paths are ECMP shortest paths
that never pass through a GPU. A transfer is split into chunks, and each chunk reserves every
link on its path cut-through with a FCFS server per link, so contention between collectives,
and between a collective and point-to-point traffic, is resolved at chunk level.

Collectives are planned into dependency graphs of transfers that mirror NCCL:

- **ring** (all-reduce, all-gather, reduce-scatter), with one ring per local GPU when the
  group spans several NVLink islands;
- **NVLS** all-reduce, pipelined push-reduce-pull through every switch attached to all ranks;
- **direct** all-to-all, with staggered peers (`(rank + i) mod N`) so that every sender does
  not hit the same receiver at once;
- broadcast and point-to-point send/recv.

A collective holds a number of SMs on each GPU while it runs (`sms`, default 16 for
collectives and 20 in the model lowering, like NCCL channels or DeepEP's reserved SMs), so it
competes with compute kernels for SMs as it does on hardware.

The network is one logical process in the parallel simulation, and GPUs exchange messages with
it in conservative time windows whose lookahead is the minimum GPU-to-network latency (80 ns on
NVLink presets). Runs are bit-for-bit deterministic.

## Exercises

1. What busbw would an ideal NVLS all-reduce reach on 72 GPUs with 900 GB/s links? The
   simulator gives 1405 GB/s at 256 MB. What fraction is that?
2. An expert-parallel MoE layer sends each token's 7168-wide BF16 activation to 8 experts.
   With 64 tokens per GPU on NVL72, how many bytes does each GPU send per dispatch, and how long
   should it take at 764 GB/s busbw? Over 50 GB/s InfiniBand?
3. Why does the ring all-gather on 72 GPUs cost 40 µs at 4 KB but only 386 µs at 256 MB, a
   65,000× larger message?
4. A data-parallel job reduces 140 GB of BF16 gradients (a 70B model) across 8 nodes over
   InfiniBand. Using a ring at 50 GB/s per GPU, how long is the bandwidth term? What changes
   with reduce-scatter instead of all-reduce?

<details><summary>Answers</summary>

1. 2(*N*−1)/*N* × 900 = 1775 GB/s. 1405/1775 = **79%**.
2. 64 × 8 × 7168 × 2 B = 7.3 MB per GPU per dispatch (DeepSeek dispatches in FP8, half that).
   At 764 GB/s: about **10 µs**, which matches the per-layer all-to-all of the DeepSeek-V3 run
   in chapter 6. At 50 GB/s and the 42 GB/s the IB model reaches: about **175 µs**.
3. Each of the 71 steps costs about 0.56 µs of latency regardless of size, so 4 KB is all
   latency (40 µs). At 256 MB the bandwidth term dominates: 71/72 × 256 MB / 686 GB/s ≈ 370 µs.
4. Per GPU (if the model is replicated per GPU and sharded across 8 nodes × 8 GPUs = 64 ranks):
   2 × 63/64 × 140 GB / 50 GB/s ≈ **5.5 s**. A reduce-scatter (ZeRO / FSDP) halves the
   bandwidth term, and the matching all-gather of parameters moves to the next forward pass.

</details>

## Reproduce

| number | command |
|---|---|
| all rows of the table | `python docs/book/scripts/collectives_sweep.py` |
| forced ring | `python docs/book/scripts/collectives_sweep.py --algo ring --kinds all_reduce` |
| inspect a topology | `rsim system configs/gb200_nvl72.yaml` |
| a collective overlapped with compute | recipe 5 in the [README](../../README.md#5-a-hand-written-workload) |

Next: [5. Parallelism](05-parallelism.md).
