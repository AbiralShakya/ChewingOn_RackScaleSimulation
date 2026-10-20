# Architecture

## Time and determinism

All time is a global `u64` in picoseconds (`rsim_core::Time`). Each component has a `Clock`
(SM clock, link clock) that converts cycles to time and back without drift. Event queues break
ties by insertion sequence. Rank LPs are stepped in a fixed order inside each window. As a
result, every run is bit-for-bit reproducible regardless of thread count.

## Parallel discrete-event simulation (`rsim-core::pdes`)

The simulator uses conservative, YAWNS-style windows. Every logical process (one per GPU, plus
one for the network) advances to `min(next event) + lookahead` in parallel (rayon). Messages are
exchanged at window boundaries. The lookahead is the minimum GPU↔network latency (80 ns on
NVLink presets), so a kernel finishing on GPU 3 can never affect the network sooner than that.

## SM model (`rsim-gpu::sm`)

Each SM has 4 SMSPs, each with its own warp list and scheduler (GTO by default, LRR optional).
The model is *event-skipped*: when no warp can issue, the SM jumps to the next cycle at which
something becomes ready. Long stalls therefore cost nothing to simulate, while every issue slot
is still accounted for.

A warp can issue only after it passes these checks, in order:

1. the control-bit `stall` count from the previous instruction, plus `yield`
2. the wait mask on SB barriers 0-5 (control-bit mode), or the per-register scoreboard
3. pipe availability (issue interval per pipe and lane count)
4. MIO/LSU queue credits
5. barrier, mbarrier, `cp.async` group and warpgroup-MMA dependencies

Each check that fails is attributed to an Nsight stall reason. Each cycle, the *critical warp*
(the one that would become ready first) decides how a bubble is labelled.

**Tensor cores.** HMMA is a pipelined fixed-latency op. Hopper `HGMMA` goes into a per-SMSP
asynchronous queue; `WARPGROUP.DEPBAR`/`wgmma.wait_group` waits on it. Blackwell `tcgen05.mma`
(`UTC*MMA`) is issued by one thread into an SM-wide queue, and `UTCBAR` (`tcgen05.commit`)
arrives on an mbarrier when the queued MMAs retire. Throughput comes from
`TensorSpec.flops_per_cycle_per_sm[dtype]` and the instruction shape.

**TMA.** `UTMALDG` reserves bytes on the mbarrier's transaction count, streams through
L2/HBM, and completes the phase when all bytes have arrived. Cluster multicast is supported.

**Memory.** The L1 is sectored LRU, and shared memory has a bank-conflict model. L2 is split
per die, with each slice served by an FCFS bandwidth server. HBM sits behind it. On B200 a
far-die access crosses the D2D link.

## Network (`rsim-net`)

The topology is a graph of GPUs and switches. Paths are ECMP shortest paths and never transit
a GPU. A transfer is split into chunks. Each chunk reserves every link on its path in
cut-through fashion, with FCFS per-link servers. Collectives are planned into dependency DAGs
of transfers: ring (with one ring per path for multi-path fabrics), NVLS (reduce in the switch,
pipelined), direct all-to-all with staggered peers, broadcast and p2p. Results are reported as
NCCL `busbw`.

## Rack runner (`rsim-rack`)

A workload is flattened per rank into streams of `Kernel | Fixed | Coll | Record | Wait | Delay`.
Communicator groups are resolved, and each collective gets a uid that matches across ranks
(`comm_id << 32 | seq`). A collective op reserves its SMs on the GPU, posts to the network LP,
and completes when the network reports done. The network then frees the stream and the SMs.

### Fast mode

`RunConfig::memoize_kernels` keys each `(KernelLaunch, reserved SMs)` pair. The first rank to
need a key simulates it on a cold GPU under a `OnceLock`, and every other rank reuses the
result. Because the result does not depend on which thread computed it, determinism holds.
