//! The network logical process.
//!
//! Ranks post collectives; the engine asks NCCL's tuning model which
//! algorithm, protocol and channel count NCCL would use, lowers the
//! collective to a transfer DAG ([`crate::collective::plan`]), and executes
//! the DAG at one of two fidelities:
//!
//! * [`Fidelity::Transfer`] — each transfer is cut into chunks; a chunk
//!   reserves every link of its path in cut-through fashion with FCFS
//!   per-link servers. Fast; no buffers, no arbitration.
//! * [`Fidelity::Packet`] — [`crate::packet`]: packets with link-level credit
//!   flow control, finite switch buffers, round-robin output arbitration
//!   across input ports, per-packet adaptive routing or per-flow ECMP, and
//!   SHARP engines that reduce/multicast packet by packet.
//!
//! Both fidelities share the protocol model: wire bytes = payload over the
//! protocol's payload efficiency, framed per link technology; every SM
//! channel is a rate-limited copy engine; every `sync` step pays the
//! protocol's fence + flag latency; every collective pays a launch cost.

use crate::collective::{plan, Algo, CollKind, Collective, End, FabricView, Plan, PlanCfg};
use crate::fabric::{ChannelId, Fabric};
use crate::nccl::{self, CommShape, Gen, NcclParams, Proto};
use crate::packet::{PEv, PacketConfig, PacketState};
use rsim_core::queue::EventQueue;
use rsim_core::trace::TraceSink;
use rsim_core::Time;
use rsim_hw::{NodeKind, System};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Fidelity {
    /// Chunked transfers on FCFS link servers (fast).
    #[default]
    Transfer,
    /// Packets with credits, switch buffers, arbitration, adaptive routing.
    Packet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NetConfig {
    pub fidelity: Fidelity,
    /// Transfer fidelity: chunk granularity on the links.
    pub chunk_bytes: u64,
    /// Scale on every link's raw rate (1.0 = datasheet; link-layer framing
    /// is modelled separately).
    pub link_efficiency: f64,
    /// Allow NVLink SHARP (NVLS) algorithms.
    pub nvls: bool,
    /// User buffers registered with NVLS (no copy in/out of the multicast buffer).
    pub nvls_registered: bool,
    /// SHARP ALU throughput as a fraction of the switch's port bandwidth.
    pub sharp_alu_ratio: f64,
    /// Local copy throughput as a fraction of HBM bandwidth (read + write).
    pub copy_efficiency: f64,
    /// Record per-link busy time in bins of this many ns (0 = off; `rsim viz`).
    pub link_bin_ns: f64,
    pub nccl: NcclParams,
    pub packet: PacketConfig,
    /// Links that go down or run degraded during the run.
    pub faults: Vec<LinkFault>,
}

/// A link that flaps or degrades (chapter 16 of the book). The trunk
/// between `a` and `b` (both directions) runs at `rate_scale` of its rate
/// from `start_us` for `duration_us`, `repeat` times every `period_us`.
/// `rate_scale: 0` is a link down: frames wait for it to retrain (or reroute,
/// where the routing is adaptive). Down from the start for the whole run
/// (`start_us` and `duration_us` 0) is a link that never trained: routing
/// leaves it out. A fraction is a link retrained at lower
/// width or speed, or a lane lost.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct LinkFault {
    /// Endpoints: `gpu<rank>` or a switch label (`nvsw0.3`, `leaf2`, `spine1`, `sw0`).
    pub a: String,
    pub b: String,
    pub start_us: f64,
    /// 0 = until the end of the run.
    pub duration_us: f64,
    pub rate_scale: f64,
    pub repeat: u32,
    pub period_us: f64,
}

impl Default for LinkFault {
    fn default() -> Self {
        Self {
            a: String::new(),
            b: String::new(),
            start_us: 0.0,
            duration_us: 0.0,
            rate_scale: 0.0,
            repeat: 1,
            period_us: 0.0,
        }
    }
}

/// Install `faults` on the fabric's channels; node names as in `LinkFault`.
pub(crate) fn apply_faults(fabric: &mut Fabric, sys: &System, faults: &[LinkFault]) {
    let name = |i: usize| match &sys.topology.nodes[i] {
        rsim_hw::NodeKind::Gpu { rank } => format!("gpu{rank}"),
        rsim_hw::NodeKind::Switch { label, .. } => label.clone(),
    };
    for f in faults {
        let mut hit = false;
        for c in &mut fabric.channels {
            let (x, y) = (name(c.from), name(c.to));
            if (x == f.a && y == f.b) || (x == f.b && y == f.a) {
                hit = true;
                if f.duration_us <= 0.0 && f.rate_scale <= 0.0 && f.start_us <= 0.0 {
                    c.dead = true;
                    continue;
                }
                for k in 0..f.repeat.max(1) {
                    let start = (f.start_us + k as f64 * f.period_us) * 1e6;
                    let end = if f.duration_us > 0.0 {
                        start + f.duration_us * 1e6
                    } else {
                        f64::INFINITY
                    };
                    c.faults.push(crate::fabric::FaultWindow {
                        start,
                        end,
                        scale: f.rate_scale.max(0.0),
                    });
                }
            }
        }
        assert!(hit, "link fault: no link between `{}` and `{}`", f.a, f.b);
    }
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            fidelity: Fidelity::Transfer,
            chunk_bytes: 512 * 1024,
            link_efficiency: 1.0,
            nvls: true,
            nvls_registered: false,
            sharp_alu_ratio: 1.0,
            copy_efficiency: 0.8,
            link_bin_ns: 0.0,
            nccl: NcclParams::default(),
            packet: PacketConfig::default(),
            faults: Vec::new(),
        }
    }
}

/// Output: collective `coll` completed on GPU `gpu` at `at`.
#[derive(Clone, Debug)]
pub struct NetEvent {
    pub coll: u64,
    pub gpu: usize,
    pub at: Time,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Ev {
    Ready {
        coll: u64,
        tid: u32,
    },
    Chunk {
        coll: u64,
        tid: u32,
        lane: u32,
        copy: u32,
        hop: u32,
        bytes: u64,
    },
    /// A local copy or a switch engine finished one unit.
    Unit {
        coll: u64,
        tid: u32,
    },
    DepDone {
        coll: u64,
        tid: u32,
    },
    Pk(PEv),
}

pub(crate) struct XState {
    pub deps_left: u32,
    /// Streaming transfers start once every dependency has delivered something.
    pub deps_unstarted: u32,
    pub units_left: u32,
    pub units_total: u32,
    pub delivered: u32,
    pub started: bool,
    pub stream: bool,
    pub ready_at: Time,
    pub done_at: Time,
    pub dependents: Vec<u32>,
    /// Lanes (ECMP paths / NVLS planes); each lane is a set of copy routes.
    pub lanes: Vec<Vec<Vec<ChannelId>>>,
}

pub(crate) struct CollState {
    pub desc: Collective,
    pub plan: Plan,
    pub x: Vec<XState>,
    pub posted: Vec<Option<Time>>,
    pub rank_left: Vec<u32>,
    pub rank_done: Vec<Time>,
    pub first_post: Time,
    pub launch: Time,
    /// Communicator island -> global island id.
    pub isl_global: Vec<usize>,
    pub island_of_rank: Vec<usize>,
    pub chan_rate: f64,
    pub model_us: f64,
    /// Seeds each transfer's starting lane (see [`lane_seed`]).
    pub lane_seed: u64,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct CollRecord {
    pub coll: u64,
    pub kind: String,
    pub algo: String,
    pub proto: String,
    pub channels: u32,
    pub bytes: u64,
    pub ranks: usize,
    pub first_post_ps: Time,
    pub last_post_ps: Time,
    pub done_ps: Time,
    /// NCCL-style bus bandwidth measured from last post to completion, GB/s.
    pub busbw_gbps: f64,
    /// NCCL's own predicted time for the chosen algorithm/protocol, us.
    pub nccl_model_us: f64,
}

pub struct NetEngine {
    pub fabric: Fabric,
    pub(crate) cfg: NetConfig,
    pub(crate) q: EventQueue<Ev>,
    pub(crate) colls: HashMap<u64, CollState>,
    island_of: Vec<usize>,
    island_size: Vec<usize>,
    /// Per global island: SHARP switches, repeated by link count (address interleave).
    pub(crate) planes: Vec<Vec<usize>>,
    sharp_leaf: Vec<Option<usize>>,
    nic_gbps: Vec<f64>,
    gen: Vec<Gen>,
    nvlink_ports: Vec<u32>,
    copy_rate: Vec<f64>,
    pub records: Vec<CollRecord>,
    pub trace: TraceSink,
    pub(crate) sm_free: HashMap<(usize, u16), f64>,
    copy_free: Vec<f64>,
    pub(crate) alu_free: HashMap<usize, f64>,
    pub(crate) pk: PacketState,
}

impl NetEngine {
    pub fn new(sys: &System, cfg: NetConfig, trace: bool) -> Self {
        let mut fabric = Fabric::new(sys, cfg.link_efficiency);
        apply_faults(&mut fabric, sys, &cfg.faults);
        for c in &mut fabric.channels {
            c.bin_ps = cfg.link_bin_ns * 1000.0;
        }
        let ngpu = sys.num_gpus();
        // NVLink islands: connected components of GPUs through non-scale-out switches.
        let n = sys.topology.nodes.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut x = x;
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for l in &sys.topology.links {
            if fabric.is_scale_out[l.a] || fabric.is_scale_out[l.b] {
                continue;
            }
            let (a, b) = (find(&mut parent, l.a), find(&mut parent, l.b));
            parent[a] = b;
        }
        let roots: Vec<usize> = (0..ngpu).map(|g| find(&mut parent, g)).collect();
        let mut uniq: Vec<usize> = Vec::new();
        let island_of: Vec<usize> = roots
            .iter()
            .map(|r| match uniq.iter().position(|x| x == r) {
                Some(i) => i,
                None => {
                    uniq.push(*r);
                    uniq.len() - 1
                }
            })
            .collect();
        let mut island_size = vec![0usize; uniq.len()];
        for &i in &island_of {
            island_size[i] += 1;
        }
        // NVLS planes per island: SHARP switches attached to the island's
        // first GPU, repeated by link count, kept only if every GPU of the
        // island reaches them.
        let mut planes: Vec<Vec<(usize, u32)>> = vec![Vec::new(); uniq.len()];
        for (isl, pl) in planes.iter_mut().enumerate() {
            let members: Vec<usize> = (0..ngpu).filter(|&g| island_of[g] == isl).collect();
            if members.len() < 2 {
                continue;
            }
            for &c in &fabric.adj[members[0]] {
                let sw = fabric.channels[c].to;
                if !fabric.sharp[sw] || fabric.is_scale_out[sw] || fabric.channels[c].dead {
                    continue;
                }
                if members
                    .iter()
                    .all(|&g| fabric.channel_between(g, sw).is_some())
                {
                    pl.push((sw, fabric.channels[c].count.max(1)));
                }
            }
        }
        // Weighted round-robin interleave (5-4-4-5 -> a b c d a b c d a d ...),
        // so consecutive addresses alternate switches.
        let planes: Vec<Vec<usize>> = planes
            .into_iter()
            .map(|pl: Vec<(usize, u32)>| {
                let mut left: Vec<u32> = pl.iter().map(|x| x.1).collect();
                let mut out = Vec::new();
                while left.iter().any(|&x| x > 0) {
                    for (i, l) in left.iter_mut().enumerate() {
                        if *l > 0 {
                            out.push(pl[i].0);
                            *l -= 1;
                        }
                    }
                }
                out
            })
            .collect();
        let mut sharp_leaf = vec![None; ngpu];
        let mut nic_gbps = vec![0.0; ngpu];
        for g in 0..ngpu {
            for &c in &fabric.adj[g] {
                let to = fabric.channels[c].to;
                if fabric.is_scale_out[to] {
                    let ch = &fabric.channels[c];
                    nic_gbps[g] += ch.rate * 1000.0 * ch.framing.efficiency(ch.framing.sm_payload);
                    if fabric.sharp[to] {
                        sharp_leaf[g] = Some(to);
                    }
                }
            }
        }
        let gen = sys.gpus.iter().map(|s| Gen::of(&s.arch)).collect();
        let nvlink_ports = sys.gpus.iter().map(|s| s.nvlink.ports).collect();
        let copy_rate = sys
            .gpus
            .iter()
            .map(|s| s.hbm.bw_gbps * cfg.copy_efficiency / 2.0 / 1000.0)
            .collect();
        let mut tr = TraceSink::new(trace);
        tr.name_process(0, "Network");
        let pk = PacketState::new(&fabric, &cfg.packet);
        Self {
            fabric,
            cfg,
            q: EventQueue::new(),
            colls: HashMap::new(),
            island_of,
            island_size,
            planes,
            sharp_leaf,
            nic_gbps,
            gen,
            nvlink_ports,
            copy_rate,
            records: vec![],
            trace: tr,
            sm_free: HashMap::new(),
            copy_free: vec![0.0; ngpu],
            alu_free: HashMap::new(),
            pk,
        }
    }

    pub fn config(&self) -> &NetConfig {
        &self.cfg
    }

    pub fn lookahead(&self) -> Time {
        self.fabric.min_latency()
    }

    pub fn next_event_time(&self) -> Option<Time> {
        self.q.peek_time()
    }

    pub fn idle(&self) -> bool {
        self.q.is_empty()
    }

    /// Islands of a communicator (comm indices) and their global ids.
    fn islands_for(&self, c: &Collective) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut islands: Vec<(usize, Vec<usize>)> = Vec::new();
        for (i, &g) in c.ranks.iter().enumerate() {
            let isl = self.island_of[g];
            match islands.iter_mut().find(|x| x.0 == isl) {
                Some(x) => x.1.push(i),
                None => islands.push((isl, vec![i])),
            }
        }
        let ids = islands.iter().map(|x| x.0).collect();
        (islands.into_iter().map(|x| x.1).collect(), ids)
    }

    /// What NCCL's tuner would see for communicator `c`.
    pub fn comm_shape(&self, c: &Collective) -> CommShape {
        let (islands, ids) = self.islands_for(c);
        let g0 = c.ranks[0];
        let local = islands.iter().map(|x| x.len()).min().unwrap_or(1);
        // A communicator inside one big NVLink domain (NVL72) still sees one
        // node; NVLS needs every rank's island to have SHARP planes.
        let nvls = self.cfg.nvls
            && ids.iter().all(|&i| !self.planes[i].is_empty())
            && islands.iter().all(|x| x.len() >= 2);
        CommShape {
            nranks: c.ranks.len(),
            nnodes: islands.len(),
            local,
            gen: self.gen[g0],
            nvlink: self.island_size[self.island_of[g0]] > 1,
            nvlink_links: self.nvlink_ports[g0] as f64,
            nic_gbps: c
                .ranks
                .iter()
                .map(|&g| self.nic_gbps[g])
                .fold(f64::INFINITY, f64::min)
                .min(1e9),
            nvls,
            one_nvlink_domain: islands.len() == 1,
            collnet: islands.len() > 1 && c.ranks.iter().all(|&g| self.sharp_leaf[g].is_some()),
        }
    }

    /// NCCL's choice and the plan configuration for a collective.
    pub fn choose(&self, c: &Collective) -> (PlanCfg, f64) {
        let s = self.comm_shape(c);
        let p = &self.cfg.nccl;
        let maxch = c.max_channels.or(p.channels).unwrap_or(64).max(1);
        let force_proto = c.proto.or(p.proto);
        let (algo, proto, channels, model_us) = match c.kind {
            CollKind::AllToAll | CollKind::SendRecv | CollKind::Broadcast => {
                let per = match c.kind {
                    CollKind::AllToAll => c.bytes / c.ranks.len().max(1) as u64,
                    _ => c.bytes,
                };
                let proto = force_proto.unwrap_or(if per <= 16 * 1024 {
                    Proto::Ll
                } else {
                    Proto::Simple
                });
                let ch = match c.kind {
                    CollKind::SendRecv => maxch.min(if per >= 1 << 20 { 4 } else { 1 }),
                    _ => maxch.min(nccl::default_channels(&s)),
                };
                (Algo::Direct, proto, ch, 0.0)
            }
            _ => {
                let forced = (c.algo != Algo::Auto).then_some(c.algo);
                match nccl::select(c.kind, c.bytes, &s, forced, force_proto, maxch) {
                    Some(pr) => (pr.algo, pr.proto, pr.channels, pr.time_us),
                    None => (
                        Algo::Ring,
                        force_proto.unwrap_or(Proto::Simple),
                        nccl::tuned_channels(Algo::Ring, Proto::Simple, c.bytes, &s, maxch),
                        0.0,
                    ),
                }
            }
        };
        let channels = p.channels.unwrap_or(channels).min(maxch).max(1);
        let mut chunk = nccl::chunk_bytes(algo, proto, c.bytes, channels, s.local as u32);
        if algo == Algo::Ring && proto == Proto::Simple {
            chunk /= 2; // slice = 2 of the 4 chunk steps
        }
        (
            PlanCfg {
                algo,
                proto,
                channels,
                chunk,
                nvls_depth: p.nvls_depth,
                nvls_registered: self.cfg.nvls_registered,
                pxn: p.pxn,
                max_transfers: p.max_transfers,
            },
            model_us,
        )
    }

    /// A rank (communicator index `idx`) reached collective `coll`.
    pub fn post(&mut self, coll: u64, desc: &Collective, idx: usize, at: Time) {
        if !self.colls.contains_key(&coll) {
            self.create(coll, desc, at);
        }
        let cs = self.colls.get_mut(&coll).unwrap();
        cs.posted[idx] = Some(at);
        cs.first_post = cs.first_post.min(at);
        // Ranks with nothing to do complete as soon as they post.
        if cs.rank_left[idx] == 0 {
            cs.rank_done[idx] = at;
            let gpu = cs.desc.ranks[idx];
            self.q.push(
                at,
                Ev::DepDone {
                    coll,
                    tid: u32::MAX - idx as u32,
                },
            );
            let _ = gpu;
        }
        let ntrans = cs.plan.transfers.len();
        for tid in 0..ntrans {
            self.try_ready(coll, tid as u32, at);
        }
    }

    fn create(&mut self, coll: u64, desc: &Collective, at: Time) {
        let (islands, ids) = self.islands_for(desc);
        let (pcfg, model_us) = self.choose(desc);
        let sharp_leaf: Vec<Option<usize>> =
            desc.ranks.iter().map(|&g| self.sharp_leaf[g]).collect();
        let view = FabricView {
            islands: &islands,
            nvls: self.comm_shape(desc).nvls,
            sharp_leaf: &sharp_leaf,
        };
        let p = plan(desc, &view, &pcfg);
        let n = desc.ranks.len();
        let packet = self.cfg.fidelity == Fidelity::Packet;
        let mut x: Vec<XState> = p
            .transfers
            .iter()
            .map(|t| XState {
                deps_left: t.deps.len() as u32,
                deps_unstarted: t.deps.len() as u32,
                units_left: 0,
                units_total: 0,
                delivered: 0,
                started: false,
                stream: packet
                    && t.channel.is_none()
                    && !matches!((t.src, t.dst), (End::Rank(a), End::Rank(b)) if a == b),
                ready_at: 0,
                done_at: 0,
                dependents: vec![],
                lanes: vec![],
            })
            .collect();
        let mut rank_left = vec![0u32; n];
        for (i, t) in p.transfers.iter().enumerate() {
            for &d in &t.deps {
                x[d].dependents.push(i as u32);
            }
            for e in [t.src, t.dst] {
                if let End::Rank(r) = e {
                    rank_left[r] += 1;
                }
            }
            if let End::Mcast(g) = t.dst {
                for &r in &p.groups[g] {
                    rank_left[r] += 1;
                }
            }
            if matches!((t.src, t.dst), (End::Rank(a), End::Rank(b)) if a == b) {
                if let End::Rank(r) = t.src {
                    rank_left[r] -= 1;
                }
            }
        }
        let mut island_of_rank = vec![0usize; n];
        for (k, isl) in islands.iter().enumerate() {
            for &r in isl {
                island_of_rank[r] = k;
            }
        }
        let gen = self.gen[desc.ranks[0]];
        let chan_rate =
            self.cfg.nccl.channel_gbps(gen, p.proto, p.channels) / 1000.0 * p.channel_weight;
        let launch_ns = if matches!(p.algo, Algo::Nvls | Algo::NvlsTree) {
            self.cfg.nccl.nvls_launch_ns
        } else {
            self.cfg.nccl.launch_ns[proto_idx(p.proto)]
        };
        self.colls.insert(
            coll,
            CollState {
                desc: desc.clone(),
                plan: p,
                x,
                posted: vec![None; n],
                rank_left,
                rank_done: vec![0; n],
                first_post: at,
                launch: (launch_ns * 1000.0) as Time,
                isl_global: ids,
                island_of_rank,
                chan_rate,
                model_us,
                lane_seed: lane_seed(coll, desc),
            },
        );
    }

    fn gate_time(cs: &CollState, tid: usize) -> Option<Time> {
        let t = &cs.plan.transfers[tid];
        let mut g = 0;
        let mut any = false;
        for e in [t.src, t.dst] {
            if let End::Rank(r) = e {
                g = g.max(cs.posted[r]?);
                any = true;
            }
        }
        if !any {
            // Switch-to-switch/multicast: every member must have posted.
            if let End::Mcast(gr) = t.dst {
                for &r in &cs.plan.groups[gr] {
                    g = g.max(cs.posted[r]?);
                }
            }
        }
        Some(g + cs.launch)
    }

    fn crosses_network(cs: &CollState, tid: usize) -> bool {
        let t = &cs.plan.transfers[tid];
        match (t.src, t.dst) {
            (End::Rank(a), End::Rank(b)) => cs.island_of_rank[a] != cs.island_of_rank[b],
            (End::Switch(_), _) | (_, End::Switch(_)) => true,
            _ => false,
        }
    }

    fn try_ready(&mut self, coll: u64, tid: u32, now: Time) {
        let cs = self.colls.get_mut(&coll).unwrap();
        let x = &cs.x[tid as usize];
        if x.started {
            return;
        }
        if (x.stream && x.deps_unstarted > 0) || (!x.stream && x.deps_left > 0) {
            return;
        }
        let Some(g) = Self::gate_time(cs, tid as usize) else {
            return;
        };
        let t = &cs.plan.transfers[tid as usize];
        let mut step = 0.0;
        if t.sync {
            let pi = proto_idx(cs.plan.proto);
            step = self.cfg.nccl.step_ns[pi];
            if Self::crosses_network(cs, tid as usize) {
                step += self.cfg.nccl.net_step_ns[pi];
            }
        }
        let at = g.max(x.ready_at).max(now) + (step * 1000.0) as Time;
        cs.x[tid as usize].started = true;
        self.q.push(at, Ev::Ready { coll, tid });
    }

    pub(crate) fn node_of(desc: &Collective, e: End) -> Option<usize> {
        match e {
            End::Rank(r) => Some(desc.ranks[r]),
            End::Switch(s) => Some(s),
            _ => None,
        }
    }

    /// Lanes of a transfer: alternative routes, each a set of copies.
    pub(crate) fn lanes(&mut self, coll: u64, tid: u32) -> Vec<Vec<Vec<ChannelId>>> {
        let cs = &self.colls[&coll];
        let t = cs.plan.transfers[tid as usize].clone();
        let desc = &cs.desc;
        let plane_of = |isl: usize| self.planes[cs.isl_global[isl]].clone();
        let members: Vec<usize> = match t.dst {
            End::Mcast(g) => cs.plan.groups[g].iter().map(|&r| desc.ranks[r]).collect(),
            _ => vec![],
        };
        match (t.src, t.dst) {
            (End::Nvls(i), End::Mcast(_)) => {
                let pl = plane_of(i);
                pl.iter()
                    .map(|&sw| {
                        members
                            .iter()
                            .map(|&m| vec![self.fabric.channel_between(sw, m).expect("plane link")])
                            .collect()
                    })
                    .collect()
            }
            (End::Nvls(i), d) => {
                let dn = Self::node_of(desc, d).unwrap();
                plane_of(i)
                    .iter()
                    .map(|&sw| {
                        vec![vec![self
                            .fabric
                            .channel_between(sw, dn)
                            .expect("plane link")]]
                    })
                    .collect()
            }
            (s, End::Nvls(i)) => {
                let sn = Self::node_of(desc, s).unwrap();
                plane_of(i)
                    .iter()
                    .map(|&sw| {
                        vec![vec![self
                            .fabric
                            .channel_between(sn, sw)
                            .expect("plane link")]]
                    })
                    .collect()
            }
            (s, End::Mcast(_)) => {
                let sn = Self::node_of(desc, s).unwrap();
                let copies: Vec<Vec<ChannelId>> = members
                    .iter()
                    .map(|&m| self.fabric.paths(sn, m).first().cloned().expect("route"))
                    .collect();
                vec![copies]
            }
            (s, d) => {
                let (sn, dn) = (
                    Self::node_of(desc, s).unwrap(),
                    Self::node_of(desc, d).unwrap(),
                );
                let paths = self.fabric.paths(sn, dn).clone();
                if paths.is_empty() {
                    panic!("no route from node {sn} to node {dn}");
                }
                paths.into_iter().map(|p| vec![p]).collect()
            }
        }
    }

    /// Process all events strictly before `until`.
    pub fn advance(&mut self, until: Time, out: &mut Vec<NetEvent>) {
        let mut batch: Vec<(u64, u32)> = Vec::new();
        while let Some((t, ev)) = self.q.pop_before(until) {
            // Transfers that become ready at the same instant start in plan
            // order (not in the order their participants happened to post),
            // so per-source injection order matches the algorithm's schedule.
            if let Ev::Ready { coll, tid } = ev {
                batch.clear();
                batch.push((coll, tid));
                while let Some((t2, Ev::Ready { coll, tid })) = self.q.peek().map(|(a, b)| (a, *b))
                {
                    if t2 != t {
                        break;
                    }
                    self.q.pop();
                    batch.push((coll, tid));
                }
                batch.sort_unstable();
                for &(coll, tid) in &batch.clone() {
                    self.start_transfer(coll, tid, t, out);
                }
                continue;
            }
            match ev {
                Ev::Ready { .. } => unreachable!(),
                Ev::Chunk {
                    coll,
                    tid,
                    lane,
                    copy,
                    hop,
                    bytes,
                } => self.chunk_hop(coll, tid, lane, copy, hop, bytes, t, out),
                Ev::Unit { coll, tid } => self.deliver(coll, tid, t, t, out),
                Ev::DepDone { coll, tid } => {
                    if tid > u32::MAX - 1_000_000 {
                        // A rank with no transfers.
                        let idx = (u32::MAX - tid) as usize;
                        if let Some(cs) = self.colls.get(&coll) {
                            out.push(NetEvent {
                                coll,
                                gpu: cs.desc.ranks[idx],
                                at: t,
                            });
                        }
                        continue;
                    }
                    let Some(cs) = self.colls.get(&coll) else {
                        continue;
                    };
                    let deps = cs.x[tid as usize].dependents.clone();
                    let done_at = cs.x[tid as usize].done_at;
                    for d in deps {
                        let cs = self.colls.get_mut(&coll).unwrap();
                        let x = &mut cs.x[d as usize];
                        x.deps_left -= 1;
                        x.ready_at = x.ready_at.max(done_at);
                        if x.stream && x.started {
                            self.pk_kick_stream(coll, d, t);
                        }
                        self.try_ready(coll, d, t);
                    }
                }
                Ev::Pk(e) => self.pk_event(e, t, out),
            }
        }
    }

    fn start_transfer(&mut self, coll: u64, tid: u32, t: Time, out: &mut Vec<NetEvent>) {
        let cs = &self.colls[&coll];
        let tr = cs.plan.transfers[tid as usize].clone();
        // Local copy (into/out of a multicast buffer, or a reduce in place).
        if let (End::Rank(a), End::Rank(b)) = (tr.src, tr.dst) {
            if a == b {
                let g = cs.desc.ranks[a];
                let rate = self.copy_rate[g].min(cs.chan_rate.max(1e-9));
                let start = self.copy_free[g].max(t as f64);
                let dur = tr.bytes as f64 / rate;
                self.copy_free[g] = start + tr.bytes as f64 / self.copy_rate[g];
                let cs = self.colls.get_mut(&coll).unwrap();
                cs.x[tid as usize].units_left = 1;
                cs.x[tid as usize].units_total = 1;
                self.q
                    .push((start + dur).ceil() as Time, Ev::Unit { coll, tid });
                return;
            }
        }
        if tr.bytes == 0 {
            let cs = self.colls.get_mut(&coll).unwrap();
            cs.x[tid as usize].units_left = 1;
            cs.x[tid as usize].units_total = 1;
            self.deliver(coll, tid, t, t, out);
            return;
        }
        if self.cfg.fidelity == Fidelity::Packet {
            self.pk_start(coll, tid, t, out);
            return;
        }
        let lanes = self.lanes(coll, tid);
        let nl = lanes.len().max(1) as u64;
        let chunk = self.cfg.chunk_bytes.max(1);
        // At least one chunk per lane so a transfer spreads over every plane
        // (NVLink traffic is striped across links).
        let mut n = tr
            .bytes
            .div_ceil(chunk)
            .max(nl.min(tr.bytes.div_ceil(4096)));
        if n > nl {
            n = n.div_ceil(nl) * nl;
        }
        let per = tr.bytes.div_ceil(n);
        let copies_per: Vec<u32> = lanes.iter().map(|l| l.len() as u32).collect();
        let off = lane_offset(self.colls[&coll].lane_seed, tid, nl as usize) as u64;
        let units: u64 = (0..n)
            .map(|i| copies_per[((i + off) % nl) as usize] as u64)
            .sum();
        let src_gpu = match tr.src {
            End::Rank(r) => Some(self.colls[&coll].desc.ranks[r]),
            _ => None,
        };
        let chan_rate = self.colls[&coll].chan_rate;
        let ndeps = tr.deps.len().max(1) as f64;
        let cs = self.colls.get_mut(&coll).unwrap();
        cs.x[tid as usize].units_left = units as u32;
        cs.x[tid as usize].units_total = units as u32;
        cs.x[tid as usize].lanes = lanes.clone();
        let mut left = tr.bytes;
        for i in 0..n {
            let b = per.min(left);
            left -= b;
            let lane = ((i + off) % nl) as u32;
            // SM channel: a rate-limited copy engine in front of the first link.
            let mut start = t as f64;
            if let (Some(ch), Some(g)) = (tr.channel, src_gpu) {
                let f = self.sm_free.entry((g, ch)).or_insert(0.0);
                start = f.max(start);
                *f = start + b as f64 / chan_rate;
            }
            // SHARP engine: reduce `ndeps` contributions per byte.
            if tr.reduce {
                let first = lanes[lane as usize][0][0];
                let sw = self.fabric.channels[first].from;
                let rate = self.fabric.switch_rate[sw] * self.cfg.sharp_alu_ratio;
                let f = self.alu_free.entry(sw).or_insert(0.0);
                start = f.max(start);
                *f = start + b as f64 * ndeps / rate.max(1e-9);
            }
            for copy in 0..lanes[lane as usize].len() as u32 {
                self.q.push(
                    start.ceil() as Time,
                    Ev::Chunk {
                        coll,
                        tid,
                        lane,
                        copy,
                        hop: 0,
                        bytes: b,
                    },
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn chunk_hop(
        &mut self,
        coll: u64,
        tid: u32,
        lane: u32,
        copy: u32,
        hop: u32,
        bytes: u64,
        t: Time,
        out: &mut Vec<NetEvent>,
    ) {
        let cs = &self.colls[&coll];
        let tr = &cs.plan.transfers[tid as usize];
        let route = &cs.x[tid as usize].lanes[lane as usize][copy as usize];
        let ch = route[hop as usize];
        let last = hop as usize + 1 == route.len();
        let eff = if tr.channel.is_some() {
            cs.plan.proto.payload_efficiency()
        } else {
            1.0
        };
        let request = tr.request;
        let c = &mut self.fabric.channels[ch];
        let wire = if request {
            bytes as f64 / (1.0 - c.framing.framing)
        } else {
            c.framing
                .wire_bytes((bytes as f64 / eff).ceil() as u64, c.framing.sm_payload)
        };
        let finish = c.serve(t, wire, bytes);
        let dur = (c.slot(t as f64, wire).1).ceil() as Time;
        let lat = c.latency;
        if last {
            self.deliver(coll, tid, t, finish + lat, out);
        } else {
            // Cut-through: the head reaches the next hop one link latency
            // after it starts serialising.
            let head = finish.saturating_sub(dur) + lat;
            self.q.push(
                head.max(t),
                Ev::Chunk {
                    coll,
                    tid,
                    lane,
                    copy,
                    hop: hop + 1,
                    bytes,
                },
            );
        }
    }

    /// One unit (chunk copy / packet) of transfer `tid` lands at `arrive`.
    pub(crate) fn deliver(
        &mut self,
        coll: u64,
        tid: u32,
        now: Time,
        arrive: Time,
        out: &mut Vec<NetEvent>,
    ) {
        let cs = self.colls.get_mut(&coll).unwrap();
        let x = &mut cs.x[tid as usize];
        x.units_left -= 1;
        x.delivered += 1;
        x.done_at = x.done_at.max(arrive);
        let first = x.delivered == 1;
        let done = x.units_left == 0;
        if first {
            // Streaming dependents may start.
            let deps = x.dependents.clone();
            for d in deps {
                let cs = self.colls.get_mut(&coll).unwrap();
                let y = &mut cs.x[d as usize];
                y.deps_unstarted -= 1;
                if y.stream {
                    y.ready_at = y.ready_at.max(arrive);
                    self.try_ready(coll, d, now);
                }
            }
        } else if self.cfg.fidelity == Fidelity::Packet {
            let deps = self.colls[&coll].x[tid as usize].dependents.clone();
            for d in deps {
                let y = &self.colls[&coll].x[d as usize];
                if y.stream && y.started {
                    self.pk_kick_stream(coll, d, now);
                }
            }
        }
        if !done {
            return;
        }
        let cs = self.colls.get_mut(&coll).unwrap();
        let done = cs.x[tid as usize].done_at;
        let tr = cs.plan.transfers[tid as usize].clone();
        let mut ends: Vec<usize> = Vec::new();
        for e in [tr.src, tr.dst] {
            if let End::Rank(r) = e {
                ends.push(r);
            }
        }
        if let (End::Rank(a), End::Rank(b)) = (tr.src, tr.dst) {
            if a == b {
                ends.truncate(1);
            }
        }
        if let End::Mcast(g) = tr.dst {
            ends.extend(cs.plan.groups[g].iter().copied());
        }
        let mut finished_ranks = vec![];
        for r in ends {
            cs.rank_left[r] -= 1;
            cs.rank_done[r] = cs.rank_done[r].max(done);
            if cs.rank_left[r] == 0 {
                finished_ranks.push(r);
            }
        }
        for r in finished_ranks {
            let at = cs.rank_done[r].max(now);
            out.push(NetEvent {
                coll,
                gpu: cs.desc.ranks[r],
                at,
            });
        }
        if cs.rank_left.iter().all(|&x| x == 0) {
            let cs = self.colls.remove(&coll).unwrap();
            let last_post = cs.posted.iter().flatten().copied().max().unwrap_or(0);
            let done_all = cs.rank_done.iter().copied().max().unwrap_or(0);
            let dur_ns = (done_all.saturating_sub(last_post)) as f64 / 1000.0;
            let bus =
                crate::collective::bus_bytes(cs.desc.kind, cs.desc.bytes, cs.desc.ranks.len());
            let rec = CollRecord {
                coll,
                kind: format!("{:?}", cs.desc.kind),
                algo: format!("{:?}", cs.plan.algo),
                proto: cs.plan.proto.name().into(),
                channels: cs.plan.channels,
                bytes: cs.desc.bytes,
                ranks: cs.desc.ranks.len(),
                first_post_ps: cs.first_post,
                last_post_ps: last_post,
                done_ps: done_all,
                busbw_gbps: if dur_ns > 0.0 { bus / dur_ns } else { 0.0 },
                nccl_model_us: cs.model_us,
            };
            self.trace.span(
                format!("{} {} {} {}B", rec.kind, rec.algo, rec.proto, rec.bytes),
                "collective",
                0,
                (coll >> 32) + 1,
                last_post,
                done_all,
                serde_json::json!({ "busbw_GBps": rec.busbw_gbps, "ranks": rec.ranks, "channels": rec.channels }),
            );
            self.records.push(rec);
        } else {
            self.q.push(done.max(now), Ev::DepDone { coll, tid });
        }
    }
}

/// Starting lane of a transfer, so short transfers do not all pile onto
/// the first plane.
pub(crate) fn lane_offset(seed: u64, tid: u32, nl: usize) -> usize {
    ((seed as usize).wrapping_mul(31) ^ (tid as usize).wrapping_mul(2654435761)) % nl.max(1)
}

/// The lane seed of collective `coll`: its sequence number within its
/// communicator (the low 32 bits of the id) under a hash of the
/// communicator's GPUs and kind, in place of the communicator's index. The
/// index counts communicators in the order the workload creates them, so
/// adding an unrelated job elsewhere in the rack would renumber this one's and
/// move its transfers onto other lanes; the hash depends only on the
/// collective itself.
pub(crate) fn lane_seed(coll: u64, desc: &Collective) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in desc
        .ranks
        .iter()
        .map(|&r| r as u64)
        .chain([desc.kind as u64])
    {
        for b in v.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    ((h ^ (h >> 32)) << 32) | (coll & 0xffff_ffff)
}

pub(crate) fn proto_idx(p: Proto) -> usize {
    match p {
        Proto::Ll => 0,
        Proto::Ll128 => 1,
        Proto::Simple => 2,
    }
}

/// Run one collective alone on an idle system (all ranks post at t = 0).
pub fn run_one(sys: &System, cfg: &NetConfig, c: &Collective) -> CollRecord {
    let mut e = NetEngine::new(sys, cfg.clone(), false);
    for r in 0..c.ranks.len() {
        e.post(1, c, r, 0);
    }
    let mut out = vec![];
    e.advance(Time::MAX, &mut out);
    e.records.pop().unwrap_or_default()
}

/// Name of node `i` (for diagnostics).
pub fn node_label(sys: &System, i: usize) -> String {
    match &sys.topology.nodes[i] {
        NodeKind::Gpu { rank } => format!("gpu{rank}"),
        NodeKind::Switch { label, .. } => label.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::{Algo, CollKind};
    use rsim_hw::presets;

    fn run(sys: &str, kind: CollKind, algo: Algo, bytes: u64, n: usize, f: Fidelity) -> CollRecord {
        let sys = presets::system(sys).unwrap().build().unwrap();
        let cfg = NetConfig {
            fidelity: f,
            ..Default::default()
        };
        let mut e = NetEngine::new(&sys, cfg, false);
        let c = Collective::new(kind, bytes, (0..n).collect(), algo);
        for r in 0..n {
            e.post(1, &c, r, 0);
        }
        let mut out = vec![];
        e.advance(Time::MAX, &mut out);
        assert_eq!(out.len(), n, "every rank completes");
        e.records.pop().unwrap()
    }

    #[test]
    fn hgx_h100_ring_allreduce_busbw() {
        // 8x H100, 1 GiB ring: bounded by 18 x 20.6 GB/s = 371 GB/s.
        for f in [Fidelity::Transfer, Fidelity::Packet] {
            let r = run(
                "hgx_h100_8",
                CollKind::AllReduce,
                Algo::Ring,
                256 << 20,
                8,
                f,
            );
            eprintln!("{f:?} H100 ring AR busbw {:.0} GB/s", r.busbw_gbps);
            assert!(
                r.busbw_gbps > 280.0 && r.busbw_gbps < 375.0,
                "{}",
                r.busbw_gbps
            );
        }
    }

    #[test]
    fn nvls_beats_ring_on_hgx() {
        for f in [Fidelity::Transfer, Fidelity::Packet] {
            let ring = run(
                "hgx_h100_8",
                CollKind::AllReduce,
                Algo::Ring,
                256 << 20,
                8,
                f,
            );
            let nvls = run(
                "hgx_h100_8",
                CollKind::AllReduce,
                Algo::Nvls,
                256 << 20,
                8,
                f,
            );
            eprintln!(
                "{f:?} ring {:.0} nvls {:.0} GB/s",
                ring.busbw_gbps, nvls.busbw_gbps
            );
            assert_eq!(nvls.algo, "Nvls");
            assert!(nvls.busbw_gbps > 1.15 * ring.busbw_gbps);
        }
    }

    #[test]
    fn cross_node_is_nic_bound() {
        let r = run(
            "h100_2node_ib",
            CollKind::AllReduce,
            Algo::Ring,
            256 << 20,
            16,
            Fidelity::Transfer,
        );
        eprintln!("2-node ring AR busbw {:.0} GB/s", r.busbw_gbps);
        assert!(r.busbw_gbps < 8.0 * 50.0 * 1.02, "{}", r.busbw_gbps);
        assert!(r.busbw_gbps > 150.0, "{}", r.busbw_gbps);
    }

    #[test]
    fn all_to_all_runs() {
        for f in [Fidelity::Transfer, Fidelity::Packet] {
            let r = run(
                "gb200_nvl72",
                CollKind::AllToAll,
                Algo::Auto,
                64 << 20,
                72,
                f,
            );
            eprintln!("{f:?} NVL72 a2a busbw {:.0} GB/s", r.busbw_gbps);
            assert!(r.busbw_gbps > 300.0 && r.busbw_gbps < 900.0);
        }
    }

    #[test]
    fn auto_picks_like_nccl() {
        let small = run(
            "hgx_b200_8",
            CollKind::AllReduce,
            Algo::Auto,
            4096,
            8,
            Fidelity::Transfer,
        );
        assert_eq!(small.proto, "LL");
        let big = run(
            "hgx_b200_8",
            CollKind::AllReduce,
            Algo::Auto,
            256 << 20,
            8,
            Fidelity::Transfer,
        );
        assert_eq!(big.algo, "Nvls");
    }

    #[test]
    fn tree_and_collnet_run_multinode() {
        for (sys, algo) in [
            ("h100_8node_ib", Algo::Tree),
            ("h100_8node_ib_sharp", Algo::Collnet),
            ("h100_8node_ib", Algo::NvlsTree),
        ] {
            for f in [Fidelity::Transfer, Fidelity::Packet] {
                let r = run(sys, CollKind::AllReduce, algo, 32 << 20, 64, f);
                eprintln!("{sys} {algo:?} {f:?}: {} busbw {:.0}", r.algo, r.busbw_gbps);
                assert_eq!(r.algo, format!("{algo:?}"));
                assert!(r.busbw_gbps > 50.0);
            }
        }
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use rsim_hw::presets;

    /// Link utilisation dump for debugging (ignored by default):
    /// `RSIM_SYS=gb200_nvl72 RSIM_KIND=all_to_all RSIM_ALGO=direct RSIM_MB=64 RSIM_PACKET=1
    /// cargo test --release -p rsim-net link_util -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn link_util() {
        let name = std::env::var("RSIM_SYS").unwrap_or("hgx_h100_8".into());
        let sys = presets::system(&name).unwrap().build().unwrap();
        let n: usize = std::env::var("RSIM_RANKS")
            .map(|x| x.parse().unwrap())
            .unwrap_or(sys.num_gpus());
        let algo: Algo = std::env::var("RSIM_ALGO")
            .map(|a| serde_json::from_str(&format!("\"{a}\"")).unwrap())
            .unwrap_or(Algo::Nvls);
        let mb: u64 = std::env::var("RSIM_MB")
            .map(|x| x.parse().unwrap())
            .unwrap_or(128);
        let kind: CollKind = std::env::var("RSIM_KIND")
            .map(|a| serde_json::from_str(&format!("\"{a}\"")).unwrap())
            .unwrap_or(CollKind::AllReduce);
        let fidelity = if std::env::var("RSIM_PACKET").is_ok_and(|v| !v.is_empty()) {
            Fidelity::Packet
        } else {
            Fidelity::Transfer
        };
        let cfg = NetConfig {
            nvls_registered: true,
            fidelity,
            ..Default::default()
        };
        let mut e = NetEngine::new(&sys, cfg, false);
        let c = Collective::new(kind, mb << 20, (0..n).collect(), algo);
        for r in 0..n {
            e.post(1, &c, r, 0);
        }
        let mut out = vec![];
        e.advance(Time::MAX, &mut out);
        let end = out.iter().map(|x| x.at).max().unwrap() as f64;
        let mut v: Vec<_> = e.fabric.channels.iter().enumerate().collect();
        v.sort_by(|a, b| b.1.busy_ps.partial_cmp(&a.1.busy_ps).unwrap());
        for (i, ch) in v.into_iter().take(24) {
            {
                eprintln!(
                    "ch{i} {}->{} util {:.2} bytes {}",
                    ch.from,
                    ch.to,
                    ch.busy_ps / end,
                    ch.bytes
                );
            }
        }
        eprintln!("end {:.1} us", end / 1e6);
    }
}
