use rsim_core::Time;
use rsim_hw::{NodeKind, System};
use std::collections::{HashMap, VecDeque};

pub type ChannelId = usize;

/// One direction of a link.
#[derive(Clone, Debug)]
pub struct Channel {
    pub from: usize,
    pub to: usize,
    /// Bytes per picosecond.
    pub rate: f64,
    pub latency: Time,
    free_at: f64,
    pub bytes: u64,
    pub busy_ps: f64,
}

impl Channel {
    /// FCFS: returns the time the last byte leaves the sender.
    #[inline]
    pub fn serve(&mut self, t: Time, bytes: u64) -> Time {
        let start = self.free_at.max(t as f64);
        let dur = bytes as f64 / self.rate;
        self.free_at = start + dur;
        self.bytes += bytes;
        self.busy_ps += dur;
        self.free_at.ceil() as Time
    }
}

pub struct Fabric {
    pub channels: Vec<Channel>,
    /// node -> outgoing channel ids
    adj: Vec<Vec<ChannelId>>,
    is_gpu: Vec<bool>,
    routes: HashMap<(usize, usize), Vec<Vec<ChannelId>>>,
    max_paths: usize,
    pub num_gpus: usize,
    /// node id of each switch with in-network reduction capability.
    pub sharp_switches: Vec<usize>,
}

impl Fabric {
    pub fn new(sys: &System, efficiency: f64) -> Self {
        let n = sys.topology.nodes.len();
        let mut channels = Vec::new();
        let mut adj = vec![Vec::new(); n];
        for l in &sys.topology.links {
            for (a, b) in [(l.a, l.b), (l.b, l.a)] {
                adj[a].push(channels.len());
                channels.push(Channel {
                    from: a,
                    to: b,
                    rate: l.gbps_per_dir * efficiency / 1000.0,
                    latency: (l.latency_ns * 1000.0).round() as Time,
                    free_at: 0.0,
                    bytes: 0,
                    busy_ps: 0.0,
                });
            }
        }
        let is_gpu: Vec<bool> = sys
            .topology
            .nodes
            .iter()
            .map(|k| matches!(k, NodeKind::Gpu { .. }))
            .collect();
        let sharp_switches = sys
            .topology
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(i, k)| match k {
                NodeKind::Switch { spec, .. } if spec.in_network_reduction => Some(i),
                _ => None,
            })
            .collect();
        Self {
            channels,
            adj,
            is_gpu,
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
            .filter(|&&c| !self.is_gpu[self.channels[c].to])
            .map(|&c| (self.channels[c].to, c))
            .collect()
    }

    pub fn channel_between(&self, a: usize, b: usize) -> Option<ChannelId> {
        self.adj[a]
            .iter()
            .copied()
            .find(|&c| self.channels[c].to == b)
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
                        if self.channels[c].to == v {
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

    pub fn reset_stats(&mut self) {
        for c in &mut self.channels {
            c.bytes = 0;
            c.busy_ps = 0.0;
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
