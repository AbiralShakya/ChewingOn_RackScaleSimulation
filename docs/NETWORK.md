# Network model (`rsim-net`)

`rsim-net` simulates the scale-up fabric (NVLink + NVSwitch) and the scale-out fabric
(InfiniBand / RoCE fat-trees) at two levels of detail that share one collective planner:

| fidelity | unit | use it for | cost |
|---|---|---|---|
| `transfer` (default, fast mode) | 512 KiB chunks on FCFS link servers, cut-through | whole-rack workloads, sweeps, fitting | a few seconds for a 1 GB NVL72 all-reduce |
| `packet` | 4 KiB packets with credit flow control, VOQs and adaptive routing | congestion, incast, oversubscribed fabrics, ECMP collisions | 5–20× slower |

Pick with `NetConfig::fidelity` (`net: {fidelity: packet}` in a run config, `--fidelity packet`
on the CLI). Every number in this file comes from `calibration/net/published.json`; reproduce
with the commands at the end.

## What is modelled

**Links and framing** ([`phy.rs`](../crates/rsim-net/src/phy.rs)). A link moves *wire* bytes:
payload plus per-packet headers. NVLink carries 128 B SM writes behind a 16 B header flit plus
flow-control/CRC overhead, fitted so that per-link goodput equals NCCL's own constants
(`SM90_NVLINK_BW` 20.6 GB/s of 25, `SM100_NVLINK_BW` 40.1 GB/s of 50). InfiniBand charges a
42 B header per 4 KiB MTU (LRH+BTH+ICRC+VCRC), RoCEv2 82 B. A link's technology is inferred
from the topology (GPU↔NVSwitch = NVLink of the GPU's generation, leaf/spine = IB NDR) and can
be overridden per link with `tech:` in the system YAML.

**NCCL's algorithms and protocols** ([`nccl.rs`](../crates/rsim-net/src/nccl.rs),
[`collective.rs`](../crates/rsim-net/src/collective.rs)). The tuning model of NCCL 2.32.3
(`src/graph/tuning.cc`) is ported table for table: base and hardware latencies, the LL /
LL128 / Simple bandwidth caps, per-channel limits, tree correction factors, NVLS efficiency
(0.85 Hopper, 0.74 Blackwell) and channel counts. `select()` picks (algorithm, protocol)
exactly as NCCL would for a given size and topology, `rsim net model` prints NCCL's own
prediction for every candidate, and the simulator then *executes* the choice:

- **Ring**: one ring per channel; across nodes one ring per rail so every NIC carries one
  ring's cross-node traffic. Window of 4 slices in flight (Simple) or 8 (LL/LL128), as
  `NCCL_STEPS / SLICESTEPS`.
- **Tree**: chain inside a node plus NCCL's double binary tree across nodes
  (`ncclGetDtree`: tree 1 mirrors tree 0 for even node counts, shifts it for odd), alternated
  per channel. Reduce up, broadcast down, pipelined.
- **NVLS**: each rank heads 1/N of the buffer. `multimem.ld_reduce` sends a 16 B request per
  128 B to the switch, the switch fans reads out to every GPU, the responses are reduced in the
  switch ALU and returned to the head, which then `multimem.st`s the result for the switch to
  multicast to every rank. Non-registered buffers add the local copy into and out of the
  multicast buffer, as in nccl-tests without `-R`. Per GPU this is egress S + S/N and ingress
  S + S/N + S/8.
- **NVLSTree** (NVLS inside each node, tree across nodes) and **CollNet** (NVLS inside, IB
  SHARP reduction at the leaf switch across).
- **All-to-all**: direct sends with NCCL's staggered peer order `(rank + i) mod N`, spread over
  `p2pChannels / N` channels per peer, and **PXN** across nodes (NVLink hop to the GPU on the
  destination's rail, then its NIC), so cross-rail traffic does not cross the spine.

Protocols change both the wire (LL carries 8 B of payload per 16 B store, LL128 120 B per
128 B) and the latency per step. Channel counts follow `ncclTuningGetChannels`; chunk sizes
follow `calcCollChunking`.

**SM cost of communication.** Every channel is one copy engine (an SM running the NCCL
kernel) with a protocol-dependent rate, and a rank's local copies share an HBM-bound copy
server (`hbm_bw × copy_efficiency / 2`). Inside a rack run a collective also *reserves* its
channels' SMs on the GPU, so it competes with compute kernels for SMs as on hardware.

**Switches** ([`fabric.rs`](../crates/rsim-net/src/fabric.rs),
[`packet.rs`](../crates/rsim-net/src/packet.rs)). NVSwitch planes are interleaved weighted
round-robin and every transfer starts on a different plane, which is how NVLink address
interleaving spreads one GPU's traffic over its 18 links. In packet mode every output port has
a virtual output queue per input served round-robin, packets need downstream credits
(64 KiB per NVSwitch port, 256 KiB per IB port by default), forwarding is virtual cut-through,
and scale-out switches route adaptively (least-loaded equal-cost port) or by ECMP hash
(`packet.routing: ecmp`). SHARP-capable switches (`sharp: true`) run a reduction ALU and
stream results packet by packet. Multi-node presets build rail-optimised two-level fat-trees
with `spines` and an `oversubscription` ratio.

**Faults** (`net.faults`, chapter 16 of the book). A link between two named nodes (`gpu3`,
`nvsw0.2`, `leaf1`, `spine4`, `sw0`) can be dead for the whole run (left out of routing and of
the NVLS planes, as a link that never trained), run at `rate_scale` of its rate (kept in
routing: a link retrained at lower width), or go down for windows (`start_us`, `duration_us`,
`repeat`, `period_us`). A frame that would run into a down window waits for it to end; in
packet mode a down port reads as busy, so adaptive switches route new packets elsewhere while
ECMP and NVLink's address interleave keep queuing behind it.

**What is not modelled.** Bit errors, FEC latency variance and link-level replay as such (a
fault window stands in for them); sender-side congestion control (ECN / DCQCN, Ultra Ethernet
NSCC and RCCC): the scale-out fabric is lossless and credit based; NVLink flit-level arbitration inside
the switch (packets are 4 KiB trains); PCIe between NIC and GPU; GPUDirect RDMA proxy
threads; NCCL's runtime tuner plugins; topology-aware rank reordering. Packets larger than
`max_packets_per_transfer × packet_bytes` are coarsened to bound cost and hold one nominal
packet of buffer, since they stand for a cut-through train.

## Validation

`calibration/net/published.json` holds 86 reference points, regenerated by
[`gen_points.py`](../calibration/net/gen_points.py), each with its source and the sentence it
comes from:

- **20 measured** numbers: the NCCL maintainers' expected H100 NVLS peak, TensorRT-LLM's
  GB200 NVL72 one-sided all-to-all tables (EP 8 to 64, batch 8 to 2048), DeepEP on H800
  (intranode NVLink, 2-node RDMA) and DeepEP V2 on B200.
- **1 bound**: NVIDIA's GB200 NVLS all-reduce acceptance gate (≥ 500 GB/s busbw × 0.9).
- **65 NCCL-model** points: NCCL 2.32.3's own predicted time for the algorithm it would pick,
  for all-reduce, all-gather and reduce-scatter from 8 KB to 512 MB on HGX H100, HGX B200,
  GB200 NVL72, 8 × H100 over IB and 2 × B200 over IB. This is what NCCL believes, not a
  measurement; it pins small-message latency and protocol switch points where no measurement
  is published.

Points marked `fit` (3 measured, 35 NCCL-model) were used by `rsim net fit` to set the 9
latency parameters (per-protocol launch, per-step software and network step costs, NVLS
setup). Everything else is held out. Nothing was fitted on bandwidth: link goodput comes
from NVLink framing and NCCL's per-channel limits.

| class | role | n | transfer: mean / median / max \|err\| | packet: mean / median / max \|err\| |
|---|---|---|---|---|
| measured | fit | 3 | 8.8% / 5.8% / 19.3% | 6.5% / 7.7% / 11.6% |
| measured | **holdout** | 17 | 16.0% / 9.1% / 37.2% | **9.8% / 7.8%** / 35.1% |
| NCCL model | fit | 35 | 7.2% / 3.4% / 21.9% | 7.5% / 3.8% / 42.9% |
| NCCL model | holdout | 30 | 11.8% / 11.4% / 26.2% | 10.4% / 7.8% / 26.0% |
| bound | holdout | 1 | pass (986 ≥ 450) | pass (994 ≥ 450) |

Every measured point (time in µs unless marked GB/s; error = sim / published − 1):

| point | published | transfer | packet | source |
|---|---|---|---|---|
| H100 8-GPU NVLS all-reduce peak busbw | ~480 GB/s | 508 (+5.8%) | 517 (+7.7%) | [nccl-tests #212](https://github.com/NVIDIA/nccl-tests/issues/212) |
| NVL72 a2a EP8, batch 8 | 17.6 | 17.9 (+1.4%) | 17.5 (−0.3%) | [TRT-LLM blog 18](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md) |
| NVL72 a2a EP8, batch 128 | 31.7 | 42.7 (+34.8%) | 36.8 (+15.9%) | 〃 |
| NVL72 a2a EP8, batch 512 | 89.5 | 123 (+37.2%) | 99.0 (+10.6%) | 〃 |
| NVL72 a2a EP8, batch 2048 | 311.8 | 372 (+19.3%) | 348 (+11.6%) | 〃 |
| NVL72 a2a EP16, batch 8 | 18.9 | 17.9 (−5.3%) | 17.8 (−6.1%) | 〃 |
| NVL72 a2a EP16, batch 128 | 34.6 | 44.6 (+28.8%) | 38.9 (+12.4%) | 〃 |
| NVL72 a2a EP16, batch 512 | 97.1 | 130 (+34.1%) | 108 (+10.8%) | 〃 |
| NVL72 a2a EP16, batch 2048 | 343.3 | 397 (+15.7%) | 382 (+11.3%) | 〃 |
| NVL72 a2a EP32, batch 8 | 19.4 | 18.2 (−6.4%) | 18.1 (−6.7%) | 〃 |
| NVL72 a2a EP32, batch 128 | 37.8 | 40.4 (+6.9%) | 39.0 (+3.3%) | 〃 |
| NVL72 a2a EP32, batch 512 | 109.6 | 114 (+3.7%) | 108 (−1.4%) | 〃 |
| NVL72 a2a EP32, batch 2048 | 373.7 | 406 (+8.7%) | 384 (+2.8%) | 〃 |
| NVL72 a2a EP64, batch 8 | 20.4 | 13.1 (−36.0%) | 13.3 (−35.1%) | 〃 |
| NVL72 a2a EP64, batch 128 | 41.6 | 38.3 (−8.0%) | 38.4 (−7.8%) | 〃 |
| NVL72 a2a EP64, batch 512 | 115.7 | 105 (−9.2%) | 105 (−8.9%) | 〃 |
| NVL72 a2a EP64, batch 2048 | 399.2 | 372 (−6.8%) | 374 (−6.4%) | 〃 |
| H800 EP8 dispatch, NVLink | 153 GB/s | 158 (+3.3%) | 157 (+2.3%) | [DeepEP @567632d](https://github.com/deepseek-ai/DeepEP/blob/567632d/README.md) |
| H800 EP16 dispatch, RDMA | 43 GB/s | 34.9 (−18.8%) | 34.9 (−18.7%) | 〃 |
| B200 EP8, NVLink (logical) | 726 GB/s | 792 (+9.1%) | 769 (+5.9%) | [DeepEP V2 @3e490fb](https://github.com/deepseek-ai/DeepEP/blob/3e490fb/README.md) |

The per-point NCCL-model table is in `calibration/net/results.json` and is printed by
`rsim net validate`.

**How the all-to-all points are mapped.** TensorRT-LLM reports
`bandwidth = batch × min(EP, 8) × 14336 B / latency`, logical (including the local share); we
simulate an all-to-all of exactly that many bytes per rank over `EP` ranks of an NVL72 and
compare times. DeepEP's normal kernels deduplicate tokens per node and forward over NVLink,
which a uniform NCCL all-to-all does not do; we use DeepSeek-V3's 4096 tokens × 7168 × FP8 ×
~5.25 ranks per token (154 MB per rank) and compare NVLink or RDMA bytes per second. DeepEP's
EP32 and EP64 RDMA numbers (58 and 51 GB/s) exceed a 400G NIC's 50 GB/s because they count
logical bytes, so they are not used.

**Known gaps.**

- *Custom kernels are not NCCL.* TensorRT-LLM's and DeepEP's all-to-all kernels are one-sided
  stores into symmetric memory. The simulator runs NCCL's two-sided protocol: it is 29–37%
  slow at EP8/16 mid sizes in transfer mode (11–16% in packet mode), and too fast at EP64
  batch 8, where the real kernel's synchronisation over 64 ranks costs 20 µs and the
  simulator's LL path 13 µs.
- *Transfer vs packet on all-to-all.* Packet mode is closer on the TRT-LLM table (mean
  error 9.8% vs 16.0%) because round-robin VOQs share the links between many peers, while the
  FCFS chunk servers of transfer mode let a few peers run ahead and finish the rest late.
- *Transfer mode depends on lane placement.* Each transfer starts on a pseudo-random plane.
  When that seed changed from the communicator's creation index to a hash of its GPUs (so a
  job's timing no longer depends on unrelated jobs elsewhere in the rack), transfer-mode
  all-to-alls moved by −6% to +12% and one 32 MiB GB200 all-gather by +23%, with no physical
  change; packet-mode all-to-alls did not move. The median point did not move in either mode.
  Read transfer-mode all-to-all times as ±10%, and prefer packet mode for EP studies.
- *NVLS small-message latency on NVL72* is 20 µs in the simulator against 25 µs in NCCL's
  model, and NCCL's model is the only reference; no measured GB200 small-message latency is
  public.
- *Packet mode, multi-node NVLSTree at 512 MB* is 43% slower than NCCL's model (transfer mode
  +7%). The cross-node tree steps are store-and-forward per piece, and with round-robin
  sharing the pieces drain together instead of pipelining. Use transfer mode for large
  hierarchical all-reduces until this is fixed.
- *IB NIC efficiency* is not measured: the simulator's RDMA all-to-all reaches 35 GB/s per
  GPU where DeepEP reports 43 GB/s on the same 400G NIC (−19%).
- No point is our own measurement; the next step is nccl-tests on an HGX node and two IB nodes.

## Reproduce

```bash
cargo build --release
python3 calibration/net/gen_points.py > calibration/net/published.json   # regenerate the point set
./target/release/rsim net validate calibration/net/published.json                    # both fidelities
./target/release/rsim net fit calibration/net/published.json                          # refit the latency parameters
./target/release/rsim net sweep --system gb200_nvl72 -b 8 -e 1G -f 4                  # nccl-tests-style sweep
./target/release/rsim net sweep --system h100_8node_ib --algo tree --fidelity packet
./target/release/rsim net model --system hgx_b200_2node_ib --coll all_gather          # NCCL's own prediction
```

Any `NetConfig` field can be overridden with `--set`, e.g.
`--set nccl.step_ns=[300,1200,2400]` or `--set packet.routing=\"ecmp\"`.
