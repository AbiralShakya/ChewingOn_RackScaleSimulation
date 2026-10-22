//! Collective algorithms lowered to transfer DAGs, the way NCCL runs them.
//!
//! A [`Plan`] is a DAG of point-to-point [`Transfer`]s. Each transfer names
//! the SM channel that issues it (or `None` for hardware: NVSwitch SHARP
//! engines, NIC read responses), whether it pays the protocol's per-step
//! synchronisation, and whether a switch reduces it. Both the transfer-level
//! and the packet-level engine execute the same plans.
//!
//! Algorithms (NCCL names):
//! * **Ring** — all-reduce / all-gather / reduce-scatter. One ring per
//!   channel; on multi-node communicators the rings enter every node on a
//!   different GPU (rail-style). Data is pipelined in slices with at most
//!   `NCCL_STEPS / slice_steps` slices in flight per connection.
//! * **Tree** — all-reduce. A chain inside each node and a double binary
//!   tree across nodes (even channels use tree 0, odd channels its mirror);
//!   chunks are reduced up and broadcast down, pipelined.
//! * **NVLS** — NVLink SHARP inside one NVLink domain. For all-reduce each
//!   rank is the head of 1/N of the data: it issues `multimem.ld_reduce`
//!   (the switch reads that slice from every GPU, reduces, and returns it),
//!   then `multimem.st` (the switch replicates the result to every GPU).
//!   Read requests cost a header flit per 128 B and travel against the
//!   data. Non-registered buffers add a local copy in and out of the
//!   multicast buffer.
//! * **NVLSTree** — NVLS reduce-scatter inside each node, a tree across
//!   nodes between the heads that own the same slice, NVLS multicast back.
//! * **CollNet** — InfiniBand SHARP: NVLS (or NVLink scatter) inside the
//!   node, the leaf switch reduces across nodes, then the result returns.
//! * **Direct** — all-to-all with staggered peers; cross-rail messages go
//!   through the GPU on the destination's rail first (PXN).

use crate::nccl::Proto;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum CollKind {
    AllReduce,
    AllGather,
    ReduceScatter,
    AllToAll,
    Broadcast,
    /// Point-to-point: ranks[0] sends to ranks[1].
    SendRecv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum Algo {
    /// Let NCCL's tuning model choose (see [`crate::nccl::select`]).
    #[default]
    Auto,
    Ring,
    Tree,
    Nvls,
    NvlsTree,
    /// InfiniBand SHARP (NCCL CollNet Direct).
    Collnet,
    Direct,
}

/// `bytes` follows NCCL's convention per kind: all-reduce/broadcast =
/// buffer size; all-gather = *output* size; reduce-scatter = *input* size;
/// all-to-all = total bytes each rank sends; send/recv = message size.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Collective {
    pub kind: CollKind,
    pub bytes: u64,
    /// GPU ids in communicator order.
    pub ranks: Vec<usize>,
    #[serde(default)]
    pub algo: Algo,
    /// Force a protocol (else NCCL's choice).
    #[serde(default)]
    pub proto: Option<Proto>,
    /// Cap on channels (= SMs the communication kernel occupies).
    #[serde(default)]
    pub max_channels: Option<u32>,
}

impl Collective {
    pub fn new(kind: CollKind, bytes: u64, ranks: Vec<usize>, algo: Algo) -> Self {
        Self {
            kind,
            bytes,
            ranks,
            algo,
            proto: None,
            max_channels: None,
        }
    }
}

/// Endpoint of a transfer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// A GPU, by communicator index.
    Rank(usize),
    /// One switch node.
    Switch(usize),
    /// The NVLS fabric of NVLink island `i`: packets are address-interleaved
    /// over all of its SHARP switches (weighted by link share).
    Nvls(usize),
    /// Multicast to the ranks of `Plan::groups[g]` (source must be a switch
    /// or the NVLS fabric).
    Mcast(usize),
}

#[derive(Clone, Debug)]
pub struct Transfer {
    pub src: End,
    pub dst: End,
    pub bytes: u64,
    pub deps: Vec<usize>,
    /// Issuing SM channel; `None` = hardware (switch engine, NIC/HBM read response).
    pub channel: Option<u16>,
    /// Pays the protocol's per-step synchronisation when it becomes ready.
    pub sync: bool,
    /// The source switch reduces the incoming contributions (SHARP ALU).
    pub reduce: bool,
    /// Bytes are read requests (header-only flits), not payload.
    pub request: bool,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub algo: Algo,
    pub proto: Proto,
    pub channels: u32,
    pub transfers: Vec<Transfer>,
    /// Multicast destination sets (communicator indices).
    pub groups: Vec<Vec<usize>>,
    /// Number of sequential protocol steps (for reports).
    pub steps: u32,
    /// Real channels per simulated channel (>1 when channels were folded to
    /// fit the DAG budget); the engine scales the per-channel copy rate by it.
    pub channel_weight: f64,
}

/// What the planner needs to know about the fabric.
pub struct FabricView<'a> {
    /// Ranks grouped by NVLink island (scale-up domain), in comm order.
    pub islands: &'a [Vec<usize>],
    /// NVLS available in every island the communicator touches.
    pub nvls: bool,
    /// Per comm rank: SHARP-capable scale-out switch it is attached to.
    pub sharp_leaf: &'a [Option<usize>],
}

/// Planner knobs (chosen by the engine from NCCL's model + config).
#[derive(Clone, Debug)]
pub struct PlanCfg {
    pub algo: Algo,
    pub proto: Proto,
    pub channels: u32,
    /// Pipeline piece per channel per step (bytes).
    pub chunk: u64,
    pub nvls_depth: u32,
    pub nvls_registered: bool,
    pub pxn: bool,
    pub max_transfers: usize,
}

struct B {
    t: Vec<Transfer>,
    groups: Vec<Vec<usize>>,
    /// Channels actually simulated (see `sim_channels`).
    sim_ch: usize,
}

impl B {
    #[allow(clippy::too_many_arguments)]
    fn push(
        &mut self,
        src: End,
        dst: End,
        bytes: u64,
        deps: Vec<usize>,
        channel: Option<u16>,
        sync: bool,
    ) -> usize {
        self.t.push(Transfer {
            src,
            dst,
            bytes,
            deps,
            channel,
            sync,
            reduce: false,
            request: false,
        });
        self.t.len() - 1
    }
    fn last(&mut self) -> &mut Transfer {
        self.t.last_mut().unwrap()
    }
    fn group(&mut self, g: Vec<usize>) -> usize {
        if let Some(i) = self.groups.iter().position(|x| *x == g) {
            return i;
        }
        self.groups.push(g);
        self.groups.len() - 1
    }
}

fn ring_orders(n: usize, islands: &[Vec<usize>]) -> Vec<Vec<usize>> {
    // One ring per local rank when the communicator spans several islands,
    // rotating where each island is entered so that every NIC carries one
    // ring's cross-island traffic (rail-style multi-ring).
    if islands.len() <= 1 {
        return vec![(0..n).collect()];
    }
    let l = islands.iter().map(|i| i.len()).min().unwrap_or(1).max(1);
    (0..l)
        .map(|k| {
            let mut order = Vec::with_capacity(n);
            for isl in islands {
                for j in 0..isl.len() {
                    order.push(isl[(k + j) % isl.len()]);
                }
            }
            order
        })
        .collect()
}

/// NCCL's double binary tree over `m` nodes (`ncclGetDtree`): parent of
/// each node in tree `which` (0 or 1). Tree 1 mirrors tree 0 for even `m` and
/// shifts it by one for odd `m`, so every node is interior in at most one.
fn dtree(m: usize, which: usize) -> Vec<Option<usize>> {
    // ncclGetBtree: the parent of `r` flips its lowest set bit.
    let btree = |r: usize| -> Option<usize> {
        if r == 0 {
            return None;
        }
        let bit = 1usize << r.trailing_zeros();
        let up = (r ^ bit) | (bit << 1);
        Some(if up >= m { r ^ bit } else { up })
    };
    (0..m)
        .map(|r| match (which % 2, m % 2) {
            (0, _) => btree(r),
            (_, 0) => btree(m - 1 - r).map(|p| m - 1 - p),
            _ => btree((r + m - 1) % m).map(|p| (p + 1) % m),
        })
        .collect()
}

/// Parent of every comm rank in channel `c`'s tree (NCCL: chain inside a
/// node, double binary tree across nodes).
fn tree_parents(n: usize, islands: &[Vec<usize>], c: usize) -> Vec<Option<usize>> {
    let m = islands.len();
    let bt = dtree(m, c);
    let heads: Vec<usize> = islands.iter().map(|isl| isl[(c / 2) % isl.len()]).collect();
    let mut parent = vec![None; n];
    for (k, isl) in islands.iter().enumerate() {
        let l = isl.len();
        let h = (c / 2) % l;
        let chain: Vec<usize> = (0..l).map(|j| isl[(h + j) % l]).collect();
        for j in 1..l {
            parent[chain[j]] = Some(chain[j - 1]);
        }
        if let Some(p) = bt[k] {
            parent[heads[k]] = Some(heads[p]);
        }
    }
    parent
}

/// Channels to simulate. When the DAG budget would leave fewer than four
/// pipeline pieces per channel, fold channels together (each simulated channel
/// then carries the copy rate of several) so the pipeline keeps its depth
/// instead of degenerating into one store-and-forward piece per channel.
/// `cost` is transfers per piece per channel; `unit(k)` the bytes one channel
/// pipelines when there are `k` channels.
fn sim_channels(
    b: &mut B,
    cfg: &PlanCfg,
    min: usize,
    cost: usize,
    unit: impl Fn(usize) -> u64,
) -> usize {
    const MIN_PIECES: u64 = 4;
    let nch = cfg.channels.max(1) as usize;
    let mut k = nch;
    while k > min.min(nch) {
        let want = unit(k).div_ceil(cfg.chunk.max(1)).clamp(1, MIN_PIECES) as usize;
        if k * cost * want <= cfg.max_transfers {
            break;
        }
        k -= 1;
    }
    b.sim_ch = k;
    k
}

/// Split `total` into pieces of at most `piece`.
fn pieces(total: u64, piece: u64) -> Vec<u64> {
    if total == 0 {
        return vec![];
    }
    let piece = piece.max(1);
    let n = total.div_ceil(piece);
    let mut v = Vec::with_capacity(n as usize);
    let mut left = total;
    while left > 0 {
        let b = piece.min(left);
        v.push(b);
        left -= b;
    }
    v
}

/// Read-request bytes for `bytes` of `multimem.ld_reduce` (16 B per 128 B).
fn req_bytes(bytes: u64) -> u64 {
    (bytes.div_ceil(128) * 16).max(16)
}

pub fn plan(c: &Collective, view: &FabricView, cfg: &PlanCfg) -> Plan {
    let n = c.ranks.len();
    let mut b = B {
        t: Vec::new(),
        groups: Vec::new(),
        sim_ch: cfg.channels.max(1) as usize,
    };
    if n <= 1 || c.bytes == 0 {
        return Plan {
            algo: Algo::Direct,
            proto: cfg.proto,
            channels: 0,
            transfers: b.t,
            groups: b.groups,
            steps: 0,
            channel_weight: 1.0,
        };
    }
    let nch = cfg.channels.max(1) as usize;
    let algo = cfg.algo;
    let steps: u32 = match (c.kind, algo) {
        (CollKind::SendRecv, _) => {
            let per = c.bytes.div_ceil(nch as u64);
            for ch in 0..nch {
                b.push(
                    End::Rank(0),
                    End::Rank(1.min(n - 1)),
                    per,
                    vec![],
                    Some(ch as u16),
                    true,
                );
            }
            1
        }
        (CollKind::AllToAll, _) => {
            plan_alltoall(&mut b, c, view, cfg);
            1
        }
        (CollKind::Broadcast, _) => {
            // Pipelined chain from rank 0.
            let ps = pieces(c.bytes, cfg.chunk.max(c.bytes / 64));
            let mut prev: Vec<Option<usize>> = vec![None; ps.len()];
            for h in 0..n - 1 {
                for (p, &pb) in ps.iter().enumerate() {
                    let mut deps = vec![];
                    if let Some(x) = prev[p] {
                        deps.push(x);
                    }
                    let id = b.push(
                        End::Rank(h),
                        End::Rank(h + 1),
                        pb,
                        deps,
                        Some((p % nch) as u16),
                        true,
                    );
                    prev[p] = Some(id);
                }
            }
            (n - 1) as u32
        }
        (CollKind::AllReduce, Algo::Nvls)
        | (CollKind::AllGather, Algo::Nvls)
        | (CollKind::ReduceScatter, Algo::Nvls)
            if view.nvls && view.islands.len() == 1 =>
        {
            plan_nvls(&mut b, c, cfg);
            2
        }
        (CollKind::AllReduce, Algo::NvlsTree) if view.nvls && view.islands.len() > 1 => {
            plan_nvls_tree(&mut b, c, view, cfg, false);
            2 + 2 * (view.islands.len() as f64).log2().ceil() as u32
        }
        (CollKind::AllReduce, Algo::Collnet)
            if view.islands.len() > 1 && view.sharp_leaf.iter().all(|x| x.is_some()) =>
        {
            plan_nvls_tree(&mut b, c, view, cfg, true);
            4
        }
        (CollKind::AllReduce, Algo::Tree) => {
            plan_tree(&mut b, c, view, cfg);
            2 * (n - 1) as u32
        }
        (kind, _) => {
            plan_ring(&mut b, c, view, cfg);
            match kind {
                CollKind::AllReduce => 2 * (n - 1) as u32,
                _ => (n - 1) as u32,
            }
        }
    };
    // Record what actually ran (fallbacks land on Ring).
    let ran = match (c.kind, algo) {
        (CollKind::SendRecv | CollKind::AllToAll | CollKind::Broadcast, _) => Algo::Direct,
        (CollKind::AllReduce | CollKind::AllGather | CollKind::ReduceScatter, Algo::Nvls)
            if view.nvls && view.islands.len() == 1 =>
        {
            Algo::Nvls
        }
        (CollKind::AllReduce, Algo::NvlsTree) if view.nvls && view.islands.len() > 1 => {
            Algo::NvlsTree
        }
        (CollKind::AllReduce, Algo::Collnet)
            if view.islands.len() > 1 && view.sharp_leaf.iter().all(|x| x.is_some()) =>
        {
            Algo::Collnet
        }
        (CollKind::AllReduce, Algo::Tree) => Algo::Tree,
        _ => Algo::Ring,
    };
    Plan {
        algo: ran,
        proto: cfg.proto,
        channels: cfg.channels,
        channel_weight: cfg.channels.max(1) as f64 / b.sim_ch as f64,
        transfers: b.t,
        groups: b.groups,
        steps,
    }
}

fn plan_ring(b: &mut B, c: &Collective, view: &FabricView, cfg: &PlanCfg) {
    let n = c.ranks.len();
    let rings = ring_orders(n, view.islands);
    let nsteps = match c.kind {
        CollKind::AllReduce => 2 * (n - 1),
        _ => n - 1,
    };
    // Keep one simulated channel per ring so that every rail stays in use.
    let nch = sim_channels(b, cfg, rings.len(), nsteps * n, |k| {
        c.bytes / (k * n) as u64
    });
    let per_ch = c.bytes.div_ceil(nch as u64);
    let step_bytes = per_ch.div_ceil(n as u64).max(1);
    // Pipeline piece: NCCL's slice, grown if the DAG would be too large.
    let budget = (cfg.max_transfers / (nch * nsteps * n)).max(1) as u64;
    let piece = cfg.chunk.max(step_bytes.div_ceil(budget)).max(1);
    let ps = pieces(step_bytes, piece);
    // Slices in flight per connection (NCCL_STEPS / slice steps).
    let window = if cfg.proto == Proto::Simple { 4 } else { 8 };
    for ch in 0..nch {
        let order = &rings[ch % rings.len()];
        // seq[i] = transfers issued on edge i, in order.
        let mut seq: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &pb in &ps {
            let mut last: Vec<Option<usize>> = vec![None; n];
            for _s in 0..nsteps {
                let mut next: Vec<Option<usize>> = vec![None; n];
                for i in 0..n {
                    let (src, dst) = (order[i], order[(i + 1) % n]);
                    let mut deps = last[i].map(|x| vec![x]).unwrap_or_default();
                    let k = seq[i].len();
                    if k >= window {
                        deps.push(seq[i][k - window]);
                    }
                    let id = b.push(
                        End::Rank(src),
                        End::Rank(dst),
                        pb,
                        deps,
                        Some(ch as u16),
                        true,
                    );
                    seq[i].push(id);
                    next[(i + 1) % n] = Some(id);
                }
                last = next;
            }
        }
    }
}

fn plan_tree(b: &mut B, c: &Collective, view: &FabricView, cfg: &PlanCfg) {
    let n = c.ranks.len();
    let nch = sim_channels(b, cfg, 2, 2 * n, |k| c.bytes / k as u64);
    let per_ch = c.bytes.div_ceil(nch as u64);
    let budget = (cfg.max_transfers / (nch * 2 * n)).max(1) as u64;
    let piece = cfg.chunk.max(per_ch.div_ceil(budget));
    let ps = pieces(per_ch, piece);
    for ch in 0..nch {
        let parent = tree_parents(n, view.islands, ch);
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (r, p) in parent.iter().enumerate() {
            if let Some(p) = p {
                children[*p].push(r);
            }
        }
        // Topological order leaves -> root.
        let mut depth = vec![0usize; n];
        for r in 0..n {
            let mut d = 0;
            let mut x = r;
            while let Some(p) = parent[x] {
                d += 1;
                x = p;
            }
            depth[r] = d;
        }
        let mut by_depth: Vec<usize> = (0..n).collect();
        by_depth.sort_by_key(|&r| std::cmp::Reverse(depth[r]));
        for &pb in &ps {
            // up[r] = transfer r -> parent(r) carrying r's partial sum.
            let mut up: Vec<Option<usize>> = vec![None; n];
            for &r in &by_depth {
                if let Some(p) = parent[r] {
                    let deps: Vec<usize> = children[r].iter().filter_map(|&k| up[k]).collect();
                    up[r] =
                        Some(b.push(End::Rank(r), End::Rank(p), pb, deps, Some(ch as u16), true));
                }
            }
            // Down: root after all children reduced; each child forwards.
            let mut down: Vec<Option<usize>> = vec![None; n];
            for &r in by_depth.iter().rev() {
                for &k in &children[r] {
                    let deps = match (parent[r], down[r]) {
                        (_, Some(d)) => vec![d],
                        (None, None) => children[r].iter().filter_map(|&k| up[k]).collect(),
                        (Some(_), None) => vec![],
                    };
                    down[k] =
                        Some(b.push(End::Rank(r), End::Rank(k), pb, deps, Some(ch as u16), true));
                }
            }
        }
    }
}

/// One NVLS `ld_reduce` + optional `st` for a slice owned by `head`, inside
/// NVLink island `isl` over its ranks `members`. Returns (result transfer,
/// mcast transfer if `bcast`).
#[allow(clippy::too_many_arguments)]
fn nvls_slice(
    b: &mut B,
    isl: usize,
    members: &[usize],
    head: usize,
    bytes: u64,
    ch: u16,
    deps: Vec<usize>,
    data_deps: &dyn Fn(usize) -> Vec<usize>,
) -> usize {
    let share = bytes.div_ceil(members.len() as u64).max(1);
    let _ = share;
    let g = b.group(members.to_vec());
    // Head issues the load: request to the switch, switch fans the reads out.
    let req = b.push(
        End::Rank(head),
        End::Nvls(isl),
        req_bytes(bytes),
        deps,
        Some(ch),
        false,
    );
    b.last().request = true;
    let fan = b.push(
        End::Nvls(isl),
        End::Mcast(g),
        req_bytes(bytes),
        vec![req],
        None,
        false,
    );
    b.last().request = true;
    // Every member returns the slice from its multicast-buffer copy.
    let mut data = Vec::with_capacity(members.len());
    for &r in members {
        let mut d = vec![fan];
        d.extend(data_deps(r));
        data.push(b.push(End::Rank(r), End::Nvls(isl), bytes, d, None, false));
    }
    // Switch reduces and returns the sum to the head.
    let res = b.push(End::Nvls(isl), End::Rank(head), bytes, data, None, false);
    b.last().reduce = true;
    res
}

fn nvls_mcast(
    b: &mut B,
    isl: usize,
    members: &[usize],
    head: usize,
    bytes: u64,
    ch: u16,
    dep: Vec<usize>,
) -> usize {
    let g = b.group(members.to_vec());
    let st = b.push(End::Rank(head), End::Nvls(isl), bytes, dep, Some(ch), false);
    b.push(End::Nvls(isl), End::Mcast(g), bytes, vec![st], None, false)
}

fn plan_nvls(b: &mut B, c: &Collective, cfg: &PlanCfg) {
    let n = c.ranks.len();
    let members: Vec<usize> = (0..n).collect();
    let per_xfer = match c.kind {
        CollKind::AllReduce => n + 6,
        CollKind::ReduceScatter => n + 4,
        _ => 2,
    };
    let nch = sim_channels(b, cfg, 1, n * per_xfer, |k| c.bytes / (k * n) as u64);
    // Per channel, per loop: every head owns one chunk.
    let per_ch = c.bytes.div_ceil(nch as u64);
    let per_head_total = per_ch.div_ceil(n as u64).max(1);
    let budget = (cfg.max_transfers / (nch * n * per_xfer)).max(1) as u64;
    let piece = cfg.chunk.max(per_head_total.div_ceil(budget));
    let ps = pieces(per_head_total, piece);
    let depth = cfg.nvls_depth.max(1) as usize;
    for ch in 0..nch {
        let ch16 = ch as u16;
        // Local copy into the multicast buffer (non-registered buffers),
        // one per rank per loop covering every head's slice.
        let mut prev_res: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (li, &pb) in ps.iter().enumerate() {
            let scatter: Vec<Option<usize>> = (0..n)
                .map(|r| {
                    (!cfg.nvls_registered && c.kind != CollKind::AllGather).then(|| {
                        b.push(
                            End::Rank(r),
                            End::Rank(r),
                            pb * n as u64,
                            vec![],
                            Some(ch16),
                            false,
                        )
                    })
                })
                .collect();
            let mut mc = Vec::new();
            for h in 0..n {
                match c.kind {
                    CollKind::AllGather => {
                        // Each rank multicasts its own slice.
                        let st = b.push(End::Rank(h), End::Nvls(0), pb, vec![], Some(ch16), false);
                        let g = b.group(members.clone());
                        mc.push(b.push(End::Nvls(0), End::Mcast(g), pb, vec![st], None, false));
                    }
                    _ => {
                        let deps = if li >= depth {
                            vec![prev_res[h][li - depth]]
                        } else {
                            vec![]
                        };
                        let sc = scatter.clone();
                        let res = nvls_slice(b, 0, &members, h, pb, ch16, deps, &|r| {
                            sc[r].map(|x| vec![x]).unwrap_or_default()
                        });
                        prev_res[h].push(res);
                        if c.kind == CollKind::AllReduce {
                            mc.push(nvls_mcast(b, 0, &members, h, pb, ch16, vec![res]));
                        } else {
                            mc.push(res);
                        }
                    }
                }
            }
            // Copy out of the multicast buffer.
            if !cfg.nvls_registered && c.kind != CollKind::ReduceScatter {
                for r in 0..n {
                    b.push(
                        End::Rank(r),
                        End::Rank(r),
                        pb * n as u64,
                        mc.clone(),
                        Some(ch16),
                        false,
                    );
                }
            }
        }
    }
}

/// NVLSTree (`collnet = false`) and CollNet (`collnet = true`).
fn plan_nvls_tree(b: &mut B, c: &Collective, view: &FabricView, cfg: &PlanCfg, collnet: bool) {
    let isl = view.islands;
    let m = isl.len();
    let local = isl.iter().map(|x| x.len()).min().unwrap_or(1);
    let nch = sim_channels(b, cfg, 1, local * m * (local + 10), |k| {
        c.bytes / (k * local) as u64
    });
    let per_ch = c.bytes.div_ceil(nch as u64);
    let per_head_total = per_ch.div_ceil(local as u64).max(1);
    let budget = (cfg.max_transfers / (nch * local * m * (local + 10))).max(1) as u64;
    let piece = cfg.chunk.max(per_head_total.div_ceil(budget));
    let ps = pieces(per_head_total, piece);
    // Pieces in flight per (channel, head): a new piece's reduction waits for
    // the inter-node phase of the piece NCCL_STEPS (8) back, as NCCL's fixed
    // number of step buffers does. Without it every piece would start at once
    // and the intra- and inter-node phases would no longer pipeline.
    let window = 8;
    for ch in 0..nch {
        let ch16 = ch as u16;
        let mut hist: Vec<Vec<Vec<usize>>> = vec![vec![Vec::new(); m]; local];
        for (li, &pb) in ps.iter().enumerate() {
            for j in 0..local {
                // 1. Reduce slice j to head j inside every island.
                let mut red = Vec::with_capacity(m);
                for (k, members) in isl.iter().enumerate() {
                    let head = members[j];
                    let gate = if li >= window {
                        vec![hist[j][k][li - window]]
                    } else {
                        vec![]
                    };
                    let r = if view.nvls {
                        nvls_slice(b, k, members, head, pb, ch16, gate, &|_| vec![])
                    } else {
                        // NVLink scatter to the head (direct), then it holds the partial sum.
                        let mut v = Vec::new();
                        for &r in members {
                            if r != head {
                                v.push(b.push(
                                    End::Rank(r),
                                    End::Rank(head),
                                    pb,
                                    gate.clone(),
                                    Some(ch16),
                                    true,
                                ));
                            }
                        }
                        b.push(End::Rank(head), End::Rank(head), pb, v, Some(ch16), false)
                    };
                    red.push(r);
                }
                let heads: Vec<usize> = isl.iter().map(|x| x[j]).collect();
                // 2. Across islands.
                let mut fin = vec![0usize; m];
                if collnet {
                    let leaf = view.sharp_leaf[heads[0]].unwrap();
                    let ups: Vec<usize> = (0..m)
                        .map(|k| {
                            b.push(
                                End::Rank(heads[k]),
                                End::Switch(leaf),
                                pb,
                                vec![red[k]],
                                Some(ch16),
                                true,
                            )
                        })
                        .collect();
                    let g = b.group(heads.clone());
                    let back = b.push(End::Switch(leaf), End::Mcast(g), pb, ups, None, false);
                    b.last().reduce = true;
                    fin = vec![back; m];
                } else {
                    let parent = dtree(m, ch);
                    let mut children: Vec<Vec<usize>> = vec![Vec::new(); m];
                    for (k, p) in parent.iter().enumerate() {
                        if let Some(p) = p {
                            children[*p].push(k);
                        }
                    }
                    let mut depth = vec![0usize; m];
                    for k in 0..m {
                        let mut x = k;
                        while let Some(p) = parent[x] {
                            depth[k] += 1;
                            x = p;
                        }
                    }
                    let mut order: Vec<usize> = (0..m).collect();
                    order.sort_by_key(|&k| std::cmp::Reverse(depth[k]));
                    let mut up = vec![None; m];
                    for &k in &order {
                        if let Some(p) = parent[k] {
                            let mut d = vec![red[k]];
                            d.extend(children[k].iter().filter_map(|&x| up[x]));
                            up[k] = Some(b.push(
                                End::Rank(heads[k]),
                                End::Rank(heads[p]),
                                pb,
                                d,
                                Some(ch16),
                                true,
                            ));
                        }
                    }
                    let mut down: Vec<Option<usize>> = vec![None; m];
                    for &k in order.iter().rev() {
                        let ready: Vec<usize> = match (parent[k], down[k]) {
                            (_, Some(d)) => vec![d],
                            _ => {
                                let mut d = vec![red[k]];
                                d.extend(children[k].iter().filter_map(|&x| up[x]));
                                d
                            }
                        };
                        if parent[k].is_none() {
                            fin[k] = *ready.last().unwrap();
                        }
                        for &x in &children[k] {
                            let id = b.push(
                                End::Rank(heads[k]),
                                End::Rank(heads[x]),
                                pb,
                                ready.clone(),
                                Some(ch16),
                                true,
                            );
                            down[x] = Some(id);
                            fin[x] = id;
                        }
                    }
                }
                for (k, &f) in fin.iter().enumerate() {
                    hist[j][k].push(f);
                }
                // 3. Broadcast inside every island.
                for (k, members) in isl.iter().enumerate() {
                    if view.nvls {
                        nvls_mcast(b, k, members, heads[k], pb, ch16, vec![fin[k]]);
                    } else {
                        for &r in members {
                            if r != heads[k] {
                                b.push(
                                    End::Rank(heads[k]),
                                    End::Rank(r),
                                    pb,
                                    vec![fin[k]],
                                    Some(ch16),
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn plan_alltoall(b: &mut B, c: &Collective, view: &FabricView, cfg: &PlanCfg) {
    let n = c.ranks.len();
    let nch = cfg.channels.max(1) as usize;
    let per = c.bytes / n as u64;
    let mut island_of = vec![0usize; n];
    let mut local_of = vec![0usize; n];
    for (k, isl) in view.islands.iter().enumerate() {
        for (j, &r) in isl.iter().enumerate() {
            island_of[r] = k;
            local_of[r] = j;
        }
    }
    // NCCL spreads each peer over p2pnChannels / nranks channels (at least
    // one), so that small communicators still use every channel's SMs.
    let cpp = (nch / (n - 1)).max(1);
    let part = per.div_ceil(cpp as u64);
    // Staggered peer order (NCCL's (rank + i) % n) avoids every sender
    // converging on the same receiver at once.
    for i in 1..n {
        for s in 0..n {
            let d = (s + i) % n;
            let cross = island_of[s] != island_of[d];
            let relay = view.islands[island_of[s]].get(local_of[d]).copied();
            for j in 0..cpp {
                let ch = Some(((i * cpp + j) % nch) as u16);
                let bytes = part.min(per - (j as u64 * part).min(per));
                if bytes == 0 {
                    continue;
                }
                match (cross && cfg.pxn, relay) {
                    (true, Some(p)) if p != s => {
                        // PXN: NVLink to the GPU on the destination's rail, then its NIC.
                        let a = b.push(End::Rank(s), End::Rank(p), bytes, vec![], ch, true);
                        b.push(End::Rank(p), End::Rank(d), bytes, vec![a], ch, true);
                    }
                    _ => {
                        b.push(End::Rank(s), End::Rank(d), bytes, vec![], ch, true);
                    }
                }
            }
        }
    }
}

/// Bytes each rank must move over its busiest direction for an ideal
/// algorithm (NCCL "bus bandwidth" convention) — for reports/tests.
pub fn bus_bytes(kind: CollKind, bytes: u64, n: usize) -> f64 {
    let n = n as f64;
    let b = bytes as f64;
    match kind {
        CollKind::AllReduce => 2.0 * (n - 1.0) / n * b,
        CollKind::AllGather | CollKind::ReduceScatter | CollKind::AllToAll => (n - 1.0) / n * b,
        CollKind::Broadcast | CollKind::SendRecv => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(algo: Algo, ch: u32) -> PlanCfg {
        PlanCfg {
            algo,
            proto: Proto::Simple,
            channels: ch,
            chunk: 1 << 20,
            nvls_depth: 2,
            nvls_registered: false,
            pxn: true,
            max_transfers: 1 << 20,
        }
    }

    #[test]
    fn ring_allreduce_shape() {
        let c = Collective::new(CollKind::AllReduce, 8 << 20, (0..8).collect(), Algo::Ring);
        let islands = vec![(0..8).collect::<Vec<_>>()];
        let p = plan(
            &c,
            &FabricView {
                islands: &islands,
                nvls: false,
                sharp_leaf: &[None; 8],
            },
            &cfg(Algo::Ring, 1),
        );
        assert_eq!(p.transfers.len(), 8 * 14);
        assert_eq!(p.steps, 14);
        assert!(p.transfers.iter().all(|t| t.bytes == 1 << 20));
    }

    #[test]
    fn multi_ring_across_islands() {
        let islands = vec![(0..4).collect::<Vec<_>>(), (4..8).collect()];
        let r = ring_orders(8, &islands);
        assert_eq!(r.len(), 4);
        // Each ring enters island 1 at a different GPU.
        let entries: Vec<usize> = r.iter().map(|o| o[4]).collect();
        assert_eq!(entries, vec![4, 5, 6, 7]);
    }

    #[test]
    fn double_binary_tree_mirrors() {
        let islands: Vec<Vec<usize>> = (0..4).map(|k| vec![2 * k, 2 * k + 1]).collect();
        let t0 = tree_parents(8, &islands, 0);
        let t1 = tree_parents(8, &islands, 1);
        // Exactly one root per tree, and the roots differ.
        let roots0: Vec<usize> = (0..8).filter(|&r| t0[r].is_none()).collect();
        let roots1: Vec<usize> = (0..8).filter(|&r| t1[r].is_none()).collect();
        assert_eq!(roots0.len(), 1);
        assert_eq!(roots1.len(), 1);
        assert_ne!(roots0, roots1);
    }

    #[test]
    fn dtree_interior_in_one_tree() {
        for m in [2usize, 7, 8, 13, 16] {
            let kids = |w| {
                let mut k = vec![0; m];
                for p in dtree(m, w).into_iter().flatten() {
                    k[p] += 1;
                }
                k
            };
            let (a, b) = (kids(0), kids(1));
            assert!(a.iter().all(|&x| x <= 2) && b.iter().all(|&x| x <= 2));
            // Matches the diagrams in NCCL's trees.cc for m = 12 and 13.
            for r in 0..m {
                assert!(
                    a[r] == 0 || b[r] == 0 || m % 2 == 1 || m <= 2,
                    "m={m} r={r}"
                );
            }
        }
        assert_eq!(dtree(12, 0)[4], Some(8));
        assert_eq!(dtree(12, 1)[7], Some(3));
        assert_eq!(dtree(13, 1)[0], Some(9));
    }

    #[test]
    fn nvls_allreduce_traffic() {
        // Per GPU: egress S (read responses) + S/N (st); ingress S/N + S (mcast).
        let n = 8;
        let s = 64u64 << 20;
        let c = Collective::new(CollKind::AllReduce, s, (0..n).collect(), Algo::Nvls);
        let islands = vec![(0..n).collect::<Vec<_>>()];
        let mut k = cfg(Algo::Nvls, 4);
        k.nvls_registered = true;
        let p = plan(
            &c,
            &FabricView {
                islands: &islands,
                nvls: true,
                sharp_leaf: &[None; 8],
            },
            &k,
        );
        let egress: u64 = p
            .transfers
            .iter()
            .filter(|t| t.src == End::Rank(0) && !t.request)
            .map(|t| t.bytes)
            .sum();
        let want = s + s / n as u64;
        assert!(
            (egress as f64 - want as f64).abs() / (want as f64) < 0.01,
            "{egress} vs {want}"
        );
    }
}
