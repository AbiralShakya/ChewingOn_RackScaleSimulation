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

The topology is a graph of GPUs and switches. Collectives are planned by a port of NCCL
2.32.3's tuning model (algorithm, protocol, channels, chunking) into dependency DAGs of
transfers: multi-ring, NCCL's double binary tree, NVLS (`multimem.ld_reduce` + multicast),
NVLSTree, CollNet, direct all-to-all with PXN, broadcast and p2p. Links charge wire bytes
(NVLink flit headers, IB/RoCE MTU headers, LL/LL128 flags), and every channel is an SM copy
engine. Two fidelities share the planner: `transfer` (chunks on FCFS link servers,
cut-through; the fast default) and `packet` (VOQs, credit flow control, adaptive/ECMP routing,
streaming SHARP). See [NETWORK.md](NETWORK.md) for the model and its validation.

## Rack runner (`rsim-rack`)

A workload is flattened per rank into streams of `Kernel | Fixed | Coll | Record | Wait | Delay`.
Communicator groups are resolved, and each collective gets a uid that matches across ranks
(`comm_id << 32 | seq`). A collective op reserves its SMs on the GPU, posts to the network LP,
and completes when the network reports done. The network then frees the stream and the SMs.

### Fast mode

`RunConfig::memoize_kernels` keys each `(KernelLaunch, reserved SMs)` pair. The first rank to
need a key simulates it on a cold GPU under a `OnceLock`, and every other rank reuses the
result. Because the result does not depend on which thread computed it, determinism holds.
Before the window loop starts, the keys the workload will most likely ask for (each kernel alone,
and next to each communication reservation of its rank) are simulated in parallel. On demand
they would run one at a time, with every rank waiting on the same one.

### Rank folding (exact mode)

SPMD ranks are often identical: same GPU, same kernels, same collectives, and the same
completion times until the fabric tells them apart. A rank LP is deterministic in its own state
and the messages delivered to it, so `rsim-rack::fold` simulates one *leader* per class of
identical ranks (same GPU spec and config, same flattened streams up to each rank's index within
its collectives). Each *follower* sends the leader's messages re-addressed and reports the
leader's spans and kernels. The first window in which a follower's deliveries differ from its
leader's, the follower *forks*: its untouched initial LP replays the leader's logged windows
(the same deliveries and the same window bounds), which recreates exactly the state it would
have had, and it then runs on its own. The report is identical to simulating every rank
(`tests/golden.rs` checks both a rack that stays in lockstep and a two-node InfiniBand run in
which every rank forks); only the work saved depends on how long ranks stay in step. The report's
`sim.folded_ranks` and `sim.forked_ranks` say how much was folded. Tracing turns folding off,
since a trace names every rank's own events, and so does `RunConfig::fold_ranks = false`
(`rsim run --no-fold`).

Exact mode also gives each rank a private copy of its kernels' instruction objects
(`KernelLaunch::private_copy`): every dynamic instruction takes and drops a reference to its
static instruction, and GPUs on different host threads sharing those reference counts made the
cache lines bounce between cores.

## Simulator speed

The hot loop is `Gpu::advance_inner` -> `Sm::step` -> per-SMSP `account` / `evaluate` / `issue`.
Rules that keep it fast:

- **No allocation or string work per cycle.** Stall attribution for mbarrier waits is keyed
  `(warp index, barrier id)` and only named when the kernel finishes.
- **Caches are flat.** Each set is one contiguous block of `3 * assoc` words (tags, LRU stamps,
  `valid | dirty << 8`) in one zero-initialised array, so a cold GPU (a fresh one per memoized
  kernel in fast mode) costs only the pages its accesses touch.
- **Per-op work is done once.** `timing()` is cached on the warp for the op it is about to issue
  and in a per-GPU direct-mapped memo keyed by instruction; warp evaluation borrows the op
  instead of cloning `Arc`s.
- **Determinism is checked, not assumed.** `crates/rsim-gpu/tests/golden.rs` pins the cycle
  count and a digest of every counter of six kernels (GEMMs on three architectures,
  elementwise, attention, and the FA4 NVBit recording on the calibrated B300), and
  `crates/rsim-rack/tests/golden.rs` pins exact and fast rack reports across host thread
  counts. A change that is meant to move timing updates the digests in the same commit
  (`RSIM_GOLDEN_PRINT=1` prints the new values).

Single-kernel replays are one thread: the SMs of a GPU share the L2/HBM model, whose queueing
state depends on the order of requests, so stepping SMs in parallel would change results.
Parallelism is across independent simulations instead: rank LPs in the PDES windows, memoized
kernels in fast mode, and reference kernels in `rsim validate`.
