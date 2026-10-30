use crate::phy::{Framing, LinkTech};
use rsim_core::Time;
use rsim_hw::{NodeKind, System};
use std::collections::{HashMap, VecDeque};

pub type ChannelId = usize;

/// One direction of a link.
#[derive(Clone, Debug)]
pub struct Channel {
    pub from: usize,
    pub to: usize,
    /// Physical links in the trunk.
    pub count: u32,
    pub tech: LinkTech,
    pub framing: Framing,
    /// Raw signalling rate of the trunk, bytes per picosecond.
    pub rate: f64,
    pub latency: Time,
    free_at: f64,
    pub bytes: u64,
    pub busy_ps: f64,
    /// Busy picoseconds per time bin of `bin_ps` (empty when binning is off).
    pub busy_bins: Vec<f32>,
    pub bin_ps: f64,
    /// Windows in which the link is down or degraded (see `LinkFault`).
    pub faults: Vec<FaultWindow>,
    /// Down for the whole run: routing leaves it out, as the fabric
    /// manager does with a link that failed to train.
    pub dead: bool,
}

/// `[start, end)` in ps, during which the link runs at `scale` of its rate
/// (0 = down: nothing starts until `end`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaultWindow {
    pub start: f64,
    pub end: f64,
    pub scale: f64,
}

impl Channel {
    /// When `wire` bytes offered at `t` can start, and how long they take,
    /// given the link's fault windows. A frame that would run into a down
    /// window is lost and resent after it (link-level replay).
    #[inline]
    pub fn slot(&self, t: f64, wire: f64) -> (f64, f64) {
        let base = wire / self.rate;
        if self.faults.is_empty() {
            return (t, base);
        }
        let mut start = t;
        loop {
            let at = self
                .faults
                .iter()
                .find(|f| start >= f.start && start < f.end);
            match at {
                Some(f) if f.scale <= 0.0 => start = f.end,
                Some(f) => return (start, base / f.scale),
                None => {
                    let cut = self
                        .faults
                        .iter()
                        .filter(|f| f.scale <= 0.0 && f.start > start && f.start < start + base)
                        .map(|f| f.end)
                        .fold(f64::NAN, f64::max);
                    if cut.is_nan() {
                        return (start, base);
                    }
                    start = cut;
                }
            }
        }
    }

    /// FCFS: serialise `wire` bytes (carrying `payload` bytes of data);
    /// returns the time the last byte leaves the sender.
    #[inline]
    pub fn serve(&mut self, t: Time, wire: f64, payload: u64) -> Time {
        let (start, dur) = self.slot(self.free_at.max(t as f64), wire);
        self.free_at = start + dur;
        self.bytes += payload;
        self.busy_ps += dur;
        if self.bin_ps > 0.0 {
            let (mut a, b) = (start, start + dur);
            while a < b {
                let i = (a / self.bin_ps) as usize;
                let edge = ((i + 1) as f64 * self.bin_ps).min(b);
                if self.busy_bins.len() <= i {
                    self.busy_bins.resize(i + 1, 0.0);
                }
                self.busy_bins[i] += (edge - a) as f32;
                a = edge;
            }
        }
        self.free_at.ceil() as Time
    }

    /// Record busy time for a packet served outside `serve` (packet engine).
    pub fn account(&mut self, start: f64, dur: f64, payload: u64) {
        self.bytes += payload;
        self.busy_ps += dur;
        if self.bin_ps > 0.0 {
            let (mut a, b) = (start, start + dur);
            while a < b {
                let i = (a / self.bin_ps) as usize;
                let edge = ((i + 1) as f64 * self.bin_ps).min(b);
                if self.busy_bins.len() <= i {
                    self.busy_bins.resize(i + 1, 0.0);
                }
                self.busy_bins[i] += (edge - a) as f32;
                a = edge;
            }
        }
    }
}

pub struct Fabric {
    pub channels: Vec<Channel>,
    /// node -> outgoing channel ids
    pub adj: Vec<Vec<ChannelId>>,
    pub is_gpu: Vec<bool>,
    /// Scale-out (InfiniBand/Ethernet) switch.
    pub is_scale_out: Vec<bool>,
    /// Switch port-to-port latency, ps (0 for GPUs).
    pub switch_latency: Vec<Time>,
    /// Switch nodes with in-network reduction.
    pub sharp: Vec<bool>,
    /// Aggregate port bandwidth of each switch, bytes/ps (SHARP ALU sizing).
    pub switch_rate: Vec<f64>,
    dist: HashMap<usize, Vec<u32>>,
    routes: HashMap<(usize, usize), Vec<Vec<ChannelId>>>,
    max_paths: usize,
    pub num_gpus: usize,
    /// node id of each switch with in-network reduction capability.
    pub sharp_switches: Vec<usize>,
}

impl Fabric {
    /// `efficiency` scales every link's raw rate (1.0 = datasheet; the
    /// link-layer overhead is modelled separately by `framing`).
    pub fn new(sys: &System, efficiency: f64) -> Self {
        let n = sys.topology.nodes.len();
        let mut channels = Vec::new();
        let mut adj = vec![Vec::new(); n];
        let scale_out = |k: &NodeKind| matches!(k, NodeKind::Switch { label, .. } if label.starts_with("leaf") || label.starts_with("spine"));
        let is_scale_out: Vec<bool> = sys.topology.nodes.iter().map(scale_out).collect();
        for l in &sys.topology.links {
            let tech = match (&sys.topology.nodes[l.a], &sys.topology.nodes[l.b]) {
                _ if is_scale_out[l.a] || is_scale_out[l.b] => LinkTech::IbNdr,
                (NodeKind::Gpu { rank }, _) | (_, NodeKind::Gpu { rank }) => {
                    LinkTech::nvlink(sys.gpus[*rank].nvlink.generation)
                }
                _ => LinkTech::Ideal,
            };
            let tech = l
                .tech
                .as_deref()
                .and_then(LinkTech::from_name)
                .unwrap_or(tech);
            for (a, b) in [(l.a, l.b), (l.b, l.a)] {
                adj[a].push(channels.len());
                // A packet's hop latency: wire + the receiving switch's
                // port-to-port latency.
                let sw_lat = match &sys.topology.nodes[b] {
                    NodeKind::Switch { spec, .. } => spec.latency_ns,
                    _ => 0.0,
                };
                channels.push(Channel {
                    from: a,
                    to: b,
                    count: l.count,
                    tech,
                    framing: tech.framing(),
                    rate: l.gbps_per_dir * efficiency / 1000.0,
                    latency: ((l.latency_ns + sw_lat) * 1000.0).round() as Time,
                    free_at: 0.0,
                    bytes: 0,
                    busy_ps: 0.0,
                    busy_bins: Vec::new(),
                    bin_ps: 0.0,
                    faults: Vec::new(),
                    dead: false,
                });
            }
        }
        let is_gpu: Vec<bool> = sys
            .topology
            .nodes
            .iter()
            .map(|k| matches!(k, NodeKind::Gpu { .. }))
            .collect();
        let sharp: Vec<bool> = sys
            .topology
            .nodes
            .iter()
            .map(|k| matches!(k, NodeKind::Switch { spec, .. } if spec.in_network_reduction))
            .collect();
        let sharp_switches = (0..n).filter(|&i| sharp[i]).collect();
        let switch_latency = sys
            .topology
            .nodes
            .iter()
            .map(|k| match k {
                NodeKind::Switch { spec, .. } => (spec.latency_ns * 1000.0) as Time,
                _ => 0,
            })
            .collect();
        let switch_rate = (0..n)
            .map(|i| adj[i].iter().map(|&c| channels[c].rate).sum())
            .collect();
        Self {
            channels,
            adj,
            is_gpu,
            is_scale_out,
            switch_latency,
            sharp,
            switch_rate,
            dist: HashMap::new(),
            routes: HashMap::new(),
            max_paths: 64,
            num_gpus: sys.num_gpus(),
            sharp_switches,
        }
    }

    pub fn min_latency(&self) -> Time {
        self.channels
            .iter()
            .map(|c| c.latency)
            .min()
            .unwrap_or(1)
            .max(1)
    }

    /// Switches directly attached to GPU `g`, with the channel to each.
    pub fn attached_switches(&self, g: usize) -> Vec<(usize, ChannelId)> {
        self.adj[g]
            .iter()
            .filter(|&&c| !self.is_gpu[self.channels[c].to] && !self.channels[c].dead)
            .map(|&c| (self.channels[c].to, c))
            .collect()
    }

    pub fn channel_between(&self, a: usize, b: usize) -> Option<ChannelId> {
        self.adj[a]
            .iter()
            .copied()
            .find(|&c| self.channels[c].to == b && !self.channels[c].dead)
    }

    /// All shortest paths (as channel lists) from `src` to `dst`, never
    /// transiting a GPU. Cached.
    pub fn paths(&mut self, src: usize, dst: usize) -> &Vec<Vec<ChannelId>> {
        if !self.routes.contains_key(&(src, dst)) {
            let p = self.compute_paths(src, dst);
            self.routes.insert((src, dst), p);
        }
        &self.routes[&(src, dst)]
    }

    fn compute_paths(&self, src: usize, dst: usize) -> Vec<Vec<ChannelId>> {
        let n = self.adj.len();
        let mut dist = vec![usize::MAX; n];
        dist[src] = 0;
        let mut q = VecDeque::from([src]);
        while let Some(u) = q.pop_front() {
            if u != src && self.is_gpu[u] {
                continue; // GPUs do not forward
            }
            for &c in &self.adj[u] {
                let v = self.channels[c].to;
                if self.channels[c].dead {
                    continue;
                }
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    q.push_back(v);
                }
            }
        }
        if dist[dst] == usize::MAX {
            return vec![];
        }
        // Enumerate shortest paths backwards from dst.
        let mut out = Vec::new();
        let mut stack: Vec<(usize, Vec<ChannelId>)> = vec![(dst, vec![])];
        while let Some((v, suffix)) = stack.pop() {
            if out.len() >= self.max_paths {
                break;
            }
            if v == src {
                let mut p = suffix.clone();
                p.reverse();
                out.push(p);
                continue;
            }
            for u in 0..n {
                if dist[u] != usize::MAX && dist[u] + 1 == dist[v] && (u == src || !self.is_gpu[u])
                {
                    for &c in &self.adj[u] {
                        if self.channels[c].to == v && !self.channels[c].dead {
                            let mut s = suffix.clone();
                            s.push(c);
                            stack.push((u, s));
                        }
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Hop distance of every node to `dst`, never transiting a GPU
    /// (`u32::MAX` = unreachable). Cached.
    pub fn dist_to(&mut self, dst: usize) -> &Vec<u32> {
        if !self.dist.contains_key(&dst) {
            let n = self.adj.len();
            let mut d = vec![u32::MAX; n];
            d[dst] = 0;
            let mut q = VecDeque::from([dst]);
            // Reverse BFS: a node u reaches v via channel u->v.
            let mut radj = vec![Vec::new(); n];
            for c in self.channels.iter().filter(|c| !c.dead) {
                radj[c.to].push(c.from);
            }
            while let Some(v) = q.pop_front() {
                if v != dst && self.is_gpu[v] {
                    continue; // GPUs do not forward
                }
                for &u in &radj[v] {
                    if d[u] == u32::MAX {
                        d[u] = d[v] + 1;
                        q.push_back(u);
                    }
                }
            }
            self.dist.insert(dst, d);
        }
        &self.dist[&dst]
    }

    /// Next-hop channels from `u` toward `dst` on shortest paths.
    pub fn next_hops(&mut self, u: usize, dst: usize) -> Vec<ChannelId> {
        let d = self.dist_to(dst).clone();
        if d[u] == u32::MAX || u == dst {
            return vec![];
        }
        self.adj[u]
            .iter()
            .copied()
            .filter(|&c| {
                let v = self.channels[c].to;
                !self.channels[c].dead
                    && d[v] != u32::MAX
                    && d[v] + 1 == d[u]
                    && (v == dst || !self.is_gpu[v])
            })
            .collect()
    }

    pub fn reset_stats(&mut self) {
        for c in &mut self.channels {
            c.bytes = 0;
            c.busy_ps = 0.0;
            c.busy_bins.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsim_hw::presets;

    #[test]
    fn nvl72_has_18_paths() {
        let sys = presets::system("gb200_nvl72").unwrap().build().unwrap();
        let mut f = Fabric::new(&sys, 1.0);
        let p = f.paths(0, 71).clone();
        assert_eq!(p.len(), 18);
        assert!(p.iter().all(|x| x.len() == 2));
    }

    #[test]
    fn dead_link_leaves_routing_and_down_window_waits() {
        let sys = presets::system("gb200_nvl72").unwrap().build().unwrap();
        let mut f = Fabric::new(&sys, 1.0);
        let c = f.adj[0][0];
        f.channels[c].dead = true;
        assert_eq!(f.paths(0, 71).len(), 17);
        let ch = &mut f.channels[c];
        ch.faults.push(FaultWindow {
            start: 100.0,
            end: 500.0,
            scale: 0.0,
        });
        ch.faults.push(FaultWindow {
            start: 1000.0,
            end: 2000.0,
            scale: 0.5,
        });
        let w = ch.rate * 50.0; // 50 ps of wire time
        assert_eq!(ch.slot(0.0, w), (0.0, 50.0));
        // Would run into the outage: resent after it.
        assert_eq!(ch.slot(80.0, w), (500.0, 50.0));
        assert_eq!(ch.slot(200.0, w), (500.0, 50.0));
        assert_eq!(ch.slot(1500.0, w), (1500.0, 100.0));
    }

    #[test]
    fn multinode_cross_rail_uses_spine() {
        let sys = presets::system("h100_2node_ib").unwrap().build().unwrap();
        let mut f = Fabric::new(&sys, 1.0);
        // Same rail (local rank 0 on both nodes): GPU -> leaf -> GPU.
        assert_eq!(f.paths(0, 8)[0].len(), 2);
        // Cross rail: GPU -> leaf -> spine -> leaf -> GPU.
        assert_eq!(f.paths(0, 9)[0].len(), 4);
        // Intra-node: via NVSwitch.
        assert_eq!(f.paths(0, 1)[0].len(), 2);
    }
}
