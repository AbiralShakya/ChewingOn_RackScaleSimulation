# 4. Networks and collectives

Once a model no longer fits on one GPU, GPUs have to exchange data every layer. This chapter
gives the cost model for the four collectives that parallelism needs (all-reduce, all-gather,
reduce-scatter, all-to-all), applies it to the fabrics that matter (NVLink with NVSwitch, the
72-GPU NVL72 domain, InfiniBand between nodes), and compares it with what the simulator
produces.

**Everything in this chapter is simulated on presets.** We have measured single GPUs only, so
the fabric model is checked against *published* multi-GPU numbers instead: nccl-tests
expectations from the NCCL maintainers, TensorRT-LLM's GB200 NVL72 all-to-all tables, DeepEP's
H800 and B200 bandwidths, and NCCL's own tuning model. Section 4.5 describes the model and
[`docs/NETWORK.md`](../NETWORK.md) gives the error against every point.

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
rsim net sweep --system hgx_b200_8 -b 4K -e 256M -f 16                 # nccl-tests style
rsim net sweep --system gb200_nvl72 --algo ring -b 4K -e 256M -f 16    # force an algorithm
rsim net model --system gb200_nvl72                                    # what NCCL's tuner predicts
```

Selected rows (bytes are per GPU; each run is the collective alone on an idle system, NCCL's
algorithm and protocol choice unless forced, transfer fidelity):

| system | GPUs | collective | 4 KB | 1 MB | 16 MB | 256 MB | busbw at 256 MB |
|---|---|---|---|---|---|---|---|
| HGX H100 | 8 | all-reduce (auto: ring LL, then NVLS) | 14.8 µs | 30.7 µs | 97.6 µs | 959 µs | 490 GB/s |
| HGX B200 | 8 | all-reduce (auto) | 14.7 | 18.6 | 62.6 | 505 | **931** |
| HGX B200 | 8 | all-reduce (ring) | 14.7 | 18.6 | 75.0 | 673 | 698 |
| HGX B200 | 8 | all-gather (ring) | 11.1 | 13.2 | 46.4 | 348 | 676 |
| HGX B200 | 8 | all-to-all | 8.0 | 17.3 | 36.4 | 342 | 688 |
| NVL72 | 72 | all-reduce (NVLS) | 19.8 | 23.3 | 78.1 | 613 | **864** |
| NVL72 | 72 | all-reduce (ring) | **80.3** | 101 | 220 | 1238 | 428 |
| NVL72 | 72 | all-gather (NVLS) | 19.3 | 22.7 | 58.5 | 412 | 643 |
| NVL72 | 72 | all-to-all | 8.1 | 14.8 | 40.7 | 411 | 644 |
| 2 × HGX H100, IB | 16 | all-reduce (auto: tree, then NVLSTree) | 26.9 | 67.1 | 111 | 1191 | 423 |
| 2 × HGX H100, IB | 16 | all-to-all (PXN) | 13.4 | 37.5 | 212 | **3086** | **82** |

What to take from it:

- **Small messages cost 10 to 20 µs whatever the fabric.** That floor is kernel launch plus
  protocol handshakes, not wire time: NCCL's own model puts it at 15 µs for an 8-GPU
  all-reduce, and TensorRT-LLM's NVL72 all-to-all bottoms out at 18 µs. The simulator's
  latency terms are fitted to those numbers ([`NETWORK.md`](../NETWORK.md)).
- **NVLS beats the ring by 1.3× for big all-reduces on HGX B200** (505 vs 673 µs), and on 72
  GPUs by 4× at 4 KB (19.8 vs 80.3 µs), because it has no 2(*N*−1)-step chain.
- **Large messages do not reach the link rate.** A ring on NVLink5 gets 698 GB/s busbw out of
  900: NVLink framing alone caps goodput at 80% (NCCL assumes 40.1 GB/s of every 50), and the
  protocol, chunking and pipeline fill take the rest. NVLS lands at 931, near NCCL's own
  prediction, because the switch does half of the work.
- **Ring latency grows with *N*.** A 4 KB ring all-reduce on 72 GPUs costs 80 µs. At decode
  sizes (a few hundred KB per layer) the latency term is the whole cost. This is why tensor
  parallelism across 72 GPUs only makes sense with NVLS or tree algorithms.
- **H100 to B200 is 2× on the link**, and the large all-reduce follows it (959 vs 505 µs).
- **InfiniBand is the cliff.** All-to-all across two 8-GPU nodes reaches 82 GB/s busbw, 8×
  slower than NVLink5. Cross-node traffic goes through PXN (NVLink to the GPU on the
  destination's rail, then its own NIC), so every NIC works, but 50 GB/s per GPU is the ceiling
  for the half of the data that leaves the node. Mixture-of-experts dispatch over IB is what
  DeepSeek-style two-micro-batch overlap exists to hide (chapter 6).

To see collectives inside a real workload, open the DeepSeek-V3 NVL72 page of the viewer
([`docs/viz/deepseek_v3_nvl72.html`](../viz/deepseek_v3_nvl72.html), figure in chapter 6.3):
the fabric view colours every NVLink by utilisation over time.

## 4.4 Worked example: the all-reduce of a TP layer

In tensor parallelism (chapter 5) each transformer layer ends its attention block and its MLP
block with an all-reduce of the activations, *T* × *D* × 2 bytes in BF16. For Llama-3-70B
(*D* = 8192):

| phase | tokens *T* | bytes per all-reduce | time on HGX B200 (`rsim net sweep`) |
|---|---|---|---|
| decode, batch 32 | 32 | 512 KB | 15 µs (ring, LL) |
| decode, batch 256 | 256 | 4 MB | 32 µs (NVLS) |
| prefill, 8k tokens | 8192 | 128 MB | 271 µs (NVLS) |

Two all-reduces per layer, 80 layers: decode at batch 32 spends about 2.4 ms per step in
all-reduce latency alone, a large share of a decode step on B200 (chapter 6). This is
why inference engines replace NCCL with one-shot all-reduce kernels on symmetric memory for
small messages, and why NCCL 2.27 advertised up to 9× lower small-message latency. comparable to a whole decode step on B200 (chapter 6). Prefill moves
much more data but also does far more compute per byte; chapter 5 does the comparison.

## 4.5 How the fabric is simulated

The topology is a graph of GPUs and switches built from the system YAML
([`configs/gb200_nvl72.yaml`](../../configs/gb200_nvl72.yaml)). Collectives are planned the way
NCCL 2.32 plans them: the simulator ports NCCL's tuning model, picks the same algorithm
(ring, double binary tree, NVLS, NVLSTree, CollNet) and protocol (LL, LL128, Simple) NCCL
would for that size and topology, uses NCCL's channel counts and chunk sizes, and then
executes the plan as a dependency graph of transfers. All-to-all uses NCCL's staggered peer
order and PXN across nodes.

Links carry wire bytes, not payload: NVLink pays a header flit per 128 B write (fitted so a
link delivers NCCL's 20.6 GB/s of 25 on Hopper and 40.1 of 50 on Blackwell), InfiniBand pays
headers per 4 KB MTU, and the LL protocols pay their flag words. Every NCCL channel is a copy
engine with a protocol-dependent rate, and inside a workload a collective reserves its SMs on
the GPU, so it competes with compute kernels as on hardware.

Two levels of detail share that planner. The default **transfer** mode moves 512 KB chunks
over FCFS link servers, cut-through, which is fast enough for whole-rack runs. **Packet** mode
(`net: {fidelity: packet}`) moves 4 KB packets through switch ports with virtual output queues,
credit-based flow control and adaptive or ECMP routing, so incast, hash collisions and
oversubscribed spines show up as back-pressure the way they do on real switches.

The network is one logical process in the parallel simulation, and GPUs exchange messages with
it in conservative time windows whose lookahead is the minimum GPU-to-network latency (80 ns on
NVLink presets). Runs are bit-for-bit deterministic.

## Exercises

1. What busbw would an ideal NVLS all-reduce reach on 72 GPUs with 900 GB/s links? NVLink
   framing leaves 40.1 GB/s of every 50 for data. The simulator gives 864 GB/s at 256 MB.
   What fraction of the framed ideal is that?
2. An expert-parallel MoE layer sends each token's 7168-wide BF16 activation to 8 experts.
   With 64 tokens per GPU on NVL72, how many bytes does each GPU send per dispatch, and how long
   should it take at the 644 GB/s all-to-all busbw above? Over 50 GB/s InfiniBand?
3. Why does the ring all-reduce on 72 GPUs cost 80 µs at 4 KB but only 1238 µs at 256 MB, a
   65,000× larger message?
4. A data-parallel job reduces 140 GB of BF16 gradients (a 70B model) across 8 nodes over
   InfiniBand. Using a ring at 50 GB/s per GPU, how long is the bandwidth term? What changes
   with reduce-scatter instead of all-reduce?

<details><summary>Answers</summary>

1. 2(*N*−1)/*N* × 18 × 40.1 = 1424 GB/s. 864/1424 = **61%**. The rest is the NVLS protocol
   (the 16 B read request per 128 B, the copy into and out of the multicast buffer without
   buffer registration) and NCCL's Blackwell NVLS efficiency of 0.74.
2. 64 × 8 × 7168 × 2 B = 7.3 MB per GPU per dispatch (DeepSeek dispatches in FP8, half that).
   At 644 GB/s: about **11 µs** of wire time, but the 15 to 20 µs floor dominates at this size
   (TensorRT-LLM measures 22 µs at batch 64, EP8). At 50 GB/s: about **150 µs**.
3. Each of the 142 steps costs about 0.5 µs of latency regardless of size, so 4 KB is all
   latency. At 256 MB the bandwidth term dominates: 2 × 71/72 × 256 MB / 428 GB/s ≈ 1238 µs.
4. Per GPU (if the model is replicated per GPU and sharded across 8 nodes × 8 GPUs = 64 ranks):
   2 × 63/64 × 140 GB / 50 GB/s ≈ **5.5 s**. A reduce-scatter (ZeRO / FSDP) halves the
   bandwidth term, and the matching all-gather of parameters moves to the next forward pass.

</details>

## Reproduce

| number | command |
|---|---|
| all rows of the table | `rsim net sweep --system <system> --coll <collective> [--algo ring] -b 4K -e 256M -f 16` |
| the same inside a workload (Python) | `python docs/book/scripts/collectives_sweep.py` |
| error against published numbers | `rsim net validate calibration/net/published.json` |
| inspect a topology | `rsim system configs/gb200_nvl72.yaml` |
| a collective overlapped with compute | recipe 5 in the [README](../../README.md#5-a-hand-written-workload) |

Next: [5. Parallelism](05-parallelism.md).
