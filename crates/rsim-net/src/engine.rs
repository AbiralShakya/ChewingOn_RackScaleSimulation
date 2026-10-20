//! The network logical process.

use crate::collective::{plan, Collective, End, FabricView, Plan};
use crate::fabric::Fabric;
use rsim_core::queue::EventQueue;
use rsim_core::trace::TraceSink;
use rsim_core::Time;
use rsim_hw::{NodeKind, System};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NetConfig {
    /// Transport chunk size (NCCL Simple-protocol chunk granularity).
    pub chunk_bytes: u64,
    /// Achievable fraction of raw link bandwidth (protocol/packet overhead).
    pub link_efficiency: f64,
    /// Fixed software/protocol cost per algorithm step, ns.
    pub step_overhead_ns: f64,
    /// Allow NVLink SHARP (NVLS) all-reduce when available.
    pub nvls: bool,
}

impl Default for NetConfig {
    fn default() -> Self {
        Self {
            chunk_bytes: 512 * 1024,
            link_efficiency: 0.85,
            step_overhead_ns: 400.0,
            nvls: true,
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
enum Ev {
    Ready {
        coll: u64,
        tid: u32,
    },
    Chunk {
        coll: u64,
        tid: u32,
        path: u32,
        hop: u32,
        bytes: u64,
    },
    DepDone {
        coll: u64,
        tid: u32,
    },
}

struct XState {
    deps_left: u32,
    chunks_left: u32,
    started: bool,
    ready_at: Time,
    done_at: Time,
    dependents: Vec<u32>,
}

struct CollState {
    desc: Collective,
    plan: Plan,
    x: Vec<XState>,
    posted: Vec<Option<Time>>,
    rank_left: Vec<u32>,
    rank_done: Vec<Time>,
    first_post: Time,
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct CollRecord {
    pub coll: u64,
    pub kind: String,
    pub algo: String,
    pub bytes: u64,
    pub ranks: usize,
    pub first_post_ps: Time,
    pub last_post_ps: Time,
    pub done_ps: Time,
    /// NCCL-style bus bandwidth measured from last post to completion, GB/s.
    pub busbw_gbps: f64,
}

pub struct NetEngine {
    pub fabric: Fabric,
    cfg: NetConfig,
    q: EventQueue<Ev>,
    colls: HashMap<u64, CollState>,
    island_of: Vec<usize>,
    pub records: Vec<CollRecord>,
    pub trace: TraceSink,
    nvlink_share: Vec<HashMap<usize, f64>>,
}

impl NetEngine {
    pub fn new(sys: &System, cfg: NetConfig, trace: bool) -> Self {
        let fabric = Fabric::new(sys, cfg.link_efficiency);
        // NVLink islands: connected components of GPUs through non-scale-out switches.
        let n = sys.topology.nodes.len();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            let mut x = x;
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let scale_out = |k: &NodeKind| matches!(k, NodeKind::Switch { label, .. } if label.starts_with("leaf") || label.starts_with("spine"));
        for l in &sys.topology.links {
            if scale_out(&sys.topology.nodes[l.a]) || scale_out(&sys.topology.nodes[l.b]) {
                continue;
            }
            let (a, b) = (find(&mut parent, l.a), find(&mut parent, l.b));
            parent[a] = b;
        }
        let island_of = (0..sys.num_gpus()).map(|g| find(&mut parent, g)).collect();
        // Share of each GPU's NVLink bandwidth per attached switch.
        let mut nvlink_share = vec![HashMap::new(); sys.num_gpus()];
        for (g, share) in nvlink_share.iter_mut().enumerate() {
            let att = fabric.attached_switches(g);
            let total: f64 = att.iter().map(|&(_, c)| fabric.channels[c].rate).sum();
            for (sw, c) in att {
                if fabric.sharp_switches.contains(&sw) {
                    *share.entry(sw).or_insert(0.0) += fabric.channels[c].rate / total;
                }
            }
        }
        let mut tr = TraceSink::new(trace);
        tr.name_process(0, "Network");
        Self {
            fabric,
            cfg,
            q: EventQueue::new(),
            colls: HashMap::new(),
            island_of,
            records: vec![],
            trace: tr,
            nvlink_share,
        }
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

    fn view_for(&self, c: &Collective) -> (Vec<(usize, f64)>, Vec<Vec<usize>>) {
        let mut islands: Vec<(usize, Vec<usize>)> = Vec::new();
        for (i, &g) in c.ranks.iter().enumerate() {
            let isl = self.island_of[g];
            match islands.iter_mut().find(|x| x.0 == isl) {
                Some(x) => x.1.push(i),
                None => islands.push((isl, vec![i])),
            }
        }
        let islands: Vec<Vec<usize>> = islands.into_iter().map(|x| x.1).collect();
        let mut nvls = Vec::new();
        if self.cfg.nvls && islands.len() == 1 {
            let first = &self.nvlink_share[c.ranks[0]];
            for (&sw, &share) in first {
                if c.ranks
                    .iter()
                    .all(|&g| self.nvlink_share[g].contains_key(&sw))
                {
                    nvls.push((sw, share));
                }
            }
            nvls.sort_by_key(|x| x.0);
        }
        (nvls, islands)
    }

    /// A rank (communicator index `idx`, GPU `gpu`) reached collective `coll`.
    pub fn post(&mut self, coll: u64, desc: &Collective, idx: usize, at: Time) {
        if !self.colls.contains_key(&coll) {
            let (nvls, islands) = self.view_for(desc);
            let p = plan(
                desc,
                &FabricView {
                    nvls_switches: &nvls,
                    islands: &islands,
                },
            );
            let n = desc.ranks.len();
            let mut x: Vec<XState> = p
                .transfers
                .iter()
                .map(|t| XState {
                    deps_left: t.deps.len() as u32,
                    chunks_left: 0,
                    started: false,
                    ready_at: 0,
                    done_at: 0,
                    dependents: vec![],
                })
                .collect();
            let mut rank_left = vec![0u32; n];
            for (i, t) in p.transfers.iter().enumerate() {
                for &d in &t.deps {
                    x[d].dependents.push(i as u32);
                }
                if let End::Rank(r) = t.src {
                    rank_left[r] += 1;
                }
                if let End::Rank(r) = t.dst {
                    rank_left[r] += 1;
                }
            }
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
                },
            );
        }
        let cs = self.colls.get_mut(&coll).unwrap();
        cs.posted[idx] = Some(at);
        cs.first_post = cs.first_post.min(at);
        // Ranks with nothing to do complete as soon as they post.
        let mut out_now = vec![];
        if cs.rank_left[idx] == 0 {
            cs.rank_done[idx] = at;
            out_now.push(idx);
        }
        let ntrans = cs.plan.transfers.len();
        for tid in 0..ntrans {
            self.try_ready(coll, tid as u32, at);
        }
        for idx in out_now {
            self.q.push(
                at,
                Ev::DepDone {
                    coll,
                    tid: u32::MAX - idx as u32,
                },
            );
        }
    }

    fn gate_time(cs: &CollState, tid: usize) -> Option<Time> {
        let t = &cs.plan.transfers[tid];
        let mut g = 0;
        for e in [t.src, t.dst] {
            if let End::Rank(r) = e {
                g = g.max(cs.posted[r]?);
            }
        }
        Some(g)
    }

    fn try_ready(&mut self, coll: u64, tid: u32, now: Time) {
        let step = (self.cfg.step_overhead_ns * 1000.0) as Time;
        let cs = self.colls.get_mut(&coll).unwrap();
        let x = &cs.x[tid as usize];
        if x.started || x.deps_left > 0 {
            return;
        }
        let Some(g) = Self::gate_time(cs, tid as usize) else {
            return;
        };
        let at = g.max(x.ready_at).max(now) + step;
        cs.x[tid as usize].started = true;
        self.q.push(at, Ev::Ready { coll, tid });
    }

    fn node_of(desc: &Collective, e: End) -> usize {
        match e {
            End::Rank(r) => desc.ranks[r],
            End::Switch(s) => s,
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
                    path,
                    hop,
                    bytes,
                } => {
                    let cs = &self.colls[&coll];
                    let tr = &cs.plan.transfers[tid as usize];
                    let (src, dst) = (
                        Self::node_of(&cs.desc, tr.src),
                        Self::node_of(&cs.desc, tr.dst),
                    );
                    let paths = self.fabric.paths(src, dst);
                    if paths.is_empty() {
                        panic!("no route from node {src} to node {dst}");
                    }
                    let p = &paths[path as usize % paths.len()];
                    let ch = p[hop as usize];
                    let last = hop as usize + 1 == p.len();
                    let c = &mut self.fabric.channels[ch];
                    let dur = (bytes as f64 / c.rate).ceil() as Time;
                    let finish = c.serve(t, bytes);
                    let lat = c.latency;
                    if last {
                        self.deliver(coll, tid, t, finish + lat, out);
                    } else {
                        // Cut-through: the head of the chunk reaches the next hop
                        // one link latency after it starts serialising.
                        let head = finish.saturating_sub(dur) + lat;
                        self.q.push(
                            head.max(t),
                            Ev::Chunk {
                                coll,
                                tid,
                                path,
                                hop: hop + 1,
                                bytes,
                            },
                        );
                    }
                }
                Ev::DepDone { coll, tid } => {
                    if tid > u32::MAX - 1_000_000 {
                        continue; // rank with no transfers; already reported
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
                        self.try_ready(coll, d, t);
                    }
                }
            }
        }
    }

    fn start_transfer(&mut self, coll: u64, tid: u32, t: Time, out: &mut Vec<NetEvent>) {
        let cs = self.colls.get_mut(&coll).unwrap();
        let tr = cs.plan.transfers[tid as usize].clone();
        let (src, dst) = (
            Self::node_of(&cs.desc, tr.src),
            Self::node_of(&cs.desc, tr.dst),
        );
        if tr.bytes == 0 || src == dst {
            cs.x[tid as usize].chunks_left = 1;
            self.deliver(coll, tid, t, t, out);
            return;
        }
        let npaths = self.fabric.paths(src, dst).len().max(1);
        let chunk = self.cfg.chunk_bytes.max(1);
        // At least one chunk per path so the transfer spreads over every
        // plane (NVLink traffic is striped across links).
        let mut n = tr
            .bytes
            .div_ceil(chunk)
            .max((npaths as u64).min(tr.bytes.div_ceil(4096)));
        if n > npaths as u64 {
            // Balance chunks over planes.
            n = n.div_ceil(npaths as u64) * npaths as u64;
        }
        let per = tr.bytes.div_ceil(n);
        let cs = self.colls.get_mut(&coll).unwrap();
        cs.x[tid as usize].chunks_left = n as u32;
        let mut left = tr.bytes;
        for i in 0..n {
            let b = per.min(left);
            left -= b;
            self.q.push(
                t,
                Ev::Chunk {
                    coll,
                    tid,
                    path: (i % npaths as u64) as u32,
                    hop: 0,
                    bytes: b,
                },
            );
        }
    }

    /// One chunk of transfer `tid` lands at `arrive` (processed at `now`).
    fn deliver(&mut self, coll: u64, tid: u32, now: Time, arrive: Time, out: &mut Vec<NetEvent>) {
        let cs = self.colls.get_mut(&coll).unwrap();
        let x = &mut cs.x[tid as usize];
        x.chunks_left -= 1;
        x.done_at = x.done_at.max(arrive);
        if x.chunks_left > 0 {
            return;
        }
        let done = x.done_at;
        let tr = cs.plan.transfers[tid as usize].clone();
        let mut finished_ranks = vec![];
        for e in [tr.src, tr.dst] {
            if let End::Rank(r) = e {
                cs.rank_left[r] -= 1;
                cs.rank_done[r] = cs.rank_done[r].max(done);
                if cs.rank_left[r] == 0 {
                    finished_ranks.push(r);
                }
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
                bytes: cs.desc.bytes,
                ranks: cs.desc.ranks.len(),
                first_post_ps: cs.first_post,
                last_post_ps: last_post,
                done_ps: done_all,
                busbw_gbps: if dur_ns > 0.0 { bus / dur_ns } else { 0.0 },
            };
            self.trace.span(
                format!("{} {} {}B", rec.kind, rec.algo, rec.bytes),
                "collective",
                0,
                (coll >> 32) + 1,
                last_post,
                done_all,
                serde_json::json!({ "busbw_GBps": rec.busbw_gbps, "ranks": rec.ranks }),
            );
            self.records.push(rec);
        } else {
            self.q.push(done, Ev::DepDone { coll, tid });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collective::{Algo, CollKind};
    use rsim_hw::presets;

    fn run_coll(sys: &str, kind: CollKind, algo: Algo, bytes: u64, n: usize) -> CollRecord {
        let sys = presets::system(sys).unwrap().build().unwrap();
        let mut e = NetEngine::new(&sys, NetConfig::default(), false);
        let c = Collective {
            kind,
            bytes,
            ranks: (0..n).collect(),
            algo,
        };
        for r in 0..n {
            e.post(1, &c, r, 0);
        }
        let mut out = vec![];
        e.advance(Time::MAX, &mut out);
        assert_eq!(out.len(), n);
        e.records.pop().unwrap()
    }

    #[test]
    fn hgx_h100_ring_allreduce_busbw() {
        // 8x H100, 1 GiB: ideal busbw = 450 GB/s x 0.85 efficiency ≈ 380 GB/s.
        let r = run_coll("hgx_h100_8", CollKind::AllReduce, Algo::Ring, 1 << 30, 8);
        eprintln!("H100 ring AR busbw {:.0} GB/s", r.busbw_gbps);
        assert!(
            r.busbw_gbps > 300.0 && r.busbw_gbps < 400.0,
            "{}",
            r.busbw_gbps
        );
    }

    #[test]
    fn nvls_beats_ring_on_nvl72() {
        let ring = run_coll("gb200_nvl72", CollKind::AllReduce, Algo::Ring, 1 << 30, 72);
        let nvls = run_coll("gb200_nvl72", CollKind::AllReduce, Algo::Nvls, 1 << 30, 72);
        eprintln!(
            "NVL72 AR busbw ring {:.0} GB/s, nvls {:.0} GB/s",
            ring.busbw_gbps, nvls.busbw_gbps
        );
        assert!(nvls.busbw_gbps > 1.3 * ring.busbw_gbps);
    }

    #[test]
    fn cross_node_is_nic_bound() {
        let r = run_coll(
            "h100_2node_ib",
            CollKind::AllReduce,
            Algo::Ring,
            1 << 30,
            16,
        );
        eprintln!("2-node ring AR busbw {:.0} GB/s", r.busbw_gbps);
        // Multi-ring over 8 rails: bounded by 8 NICs x 50 GB/s x 0.85.
        assert!(r.busbw_gbps < 8.0 * 50.0 * 0.85 * 1.02, "{}", r.busbw_gbps);
        assert!(r.busbw_gbps > 200.0, "{}", r.busbw_gbps);
    }

    #[test]
    fn all_to_all_runs() {
        let r = run_coll("gb200_nvl72", CollKind::AllToAll, Algo::Auto, 256 << 20, 72);
        eprintln!("NVL72 a2a busbw {:.0} GB/s", r.busbw_gbps);
        assert!(r.busbw_gbps > 100.0);
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::collective::{Algo, CollKind};
    use rsim_hw::presets;
    #[test]
    #[ignore]
    fn a2a_channel_util() {
        let sys = presets::system("gb200_nvl72").unwrap().build().unwrap();
        let mut e = NetEngine::new(&sys, NetConfig::default(), false);
        let c = Collective {
            kind: CollKind::AllToAll,
            bytes: 256 << 20,
            ranks: (0..72).collect(),
            algo: Algo::Auto,
        };
        for r in 0..72 {
            e.post(1, &c, r, 0);
        }
        let mut out = vec![];
        e.advance(Time::MAX, &mut out);
        let end = out.iter().map(|x| x.at).max().unwrap();
        let mut busy: Vec<f64> = e.fabric.channels.iter().map(|c| c.busy_ps).collect();
        busy.sort_by(|a, b| b.partial_cmp(a).unwrap());
        eprintln!(
            "end {} us, max busy {} us, min busy {} us",
            end / 1_000_000,
            busy[0] / 1e6,
            busy[busy.len() - 1] / 1e6
        );
    }
}
