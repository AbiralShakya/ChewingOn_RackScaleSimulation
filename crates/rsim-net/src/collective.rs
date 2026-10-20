//! Collective algorithms lowered to transfer DAGs.

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
    /// Choose like NCCL would (NVLS for all-reduce inside one NVLink-SHARP
    /// domain, ring otherwise; direct for all-to-all).
    #[default]
    Auto,
    Ring,
    Nvls,
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
}

/// Endpoint of a transfer: a GPU (by communicator index) or a switch node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    Rank(usize),
    Switch(usize),
}

#[derive(Clone, Debug)]
pub struct Transfer {
    pub src: End,
    pub dst: End,
    pub bytes: u64,
    pub deps: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub algo: Algo,
    pub transfers: Vec<Transfer>,
    /// Number of sequential protocol steps (for per-step overhead).
    pub steps: u32,
}

/// Inputs the planner needs about the fabric.
pub struct FabricView<'a> {
    /// SHARP-capable switches attached to *every* rank, with per-rank link
    /// share (fraction of that rank's NVLink bandwidth going to the switch).
    pub nvls_switches: &'a [(usize, f64)],
    /// Ranks grouped by NVLink island (scale-up domain), in comm order.
    pub islands: &'a [Vec<usize>],
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

pub fn plan(c: &Collective, view: &FabricView) -> Plan {
    let n = c.ranks.len();
    let mut t = Vec::new();
    if n <= 1 || c.bytes == 0 {
        return Plan {
            algo: Algo::Direct,
            transfers: t,
            steps: 0,
        };
    }
    let algo = match (c.algo, c.kind) {
        (Algo::Auto, CollKind::AllReduce)
            if !view.nvls_switches.is_empty() && view.islands.len() <= 1 =>
        {
            Algo::Nvls
        }
        (Algo::Auto, CollKind::AllToAll) | (Algo::Auto, CollKind::SendRecv) => Algo::Direct,
        (Algo::Auto, _) => Algo::Ring,
        (Algo::Nvls, _) if view.nvls_switches.is_empty() || c.kind != CollKind::AllReduce => {
            Algo::Ring
        }
        (a, _) => a,
    };
    let steps;
    match (c.kind, algo) {
        (CollKind::SendRecv, _) => {
            t.push(Transfer {
                src: End::Rank(0),
                dst: End::Rank(1.min(n - 1)),
                bytes: c.bytes,
                deps: vec![],
            });
            steps = 1;
        }
        (CollKind::AllToAll, _) => {
            let per = c.bytes / n as u64;
            // Staggered peer order (like NCCL's (rank + i) % n) avoids every
            // sender converging on the same receiver at once.
            for i in 1..n {
                for s in 0..n {
                    let d = (s + i) % n;
                    t.push(Transfer {
                        src: End::Rank(s),
                        dst: End::Rank(d),
                        bytes: per,
                        deps: vec![],
                    });
                }
            }
            steps = 1;
        }
        (CollKind::AllReduce, Algo::Nvls) => {
            // multimem.ld_reduce: every rank pushes its buffer share to every
            // switch; the switch reduces and multicasts results back.
            // Pipelined in pieces so reduction and multicast overlap.
            for &(sw, share) in view.nvls_switches {
                let b = ((c.bytes as f64) * share).ceil() as u64;
                let pieces = (b / (1 << 20)).clamp(1, 32);
                let pb = b.div_ceil(pieces);
                for _ in 0..pieces {
                    let ups: Vec<usize> = (0..n)
                        .map(|r| {
                            t.push(Transfer {
                                src: End::Rank(r),
                                dst: End::Switch(sw),
                                bytes: pb,
                                deps: vec![],
                            });
                            t.len() - 1
                        })
                        .collect();
                    for r in 0..n {
                        t.push(Transfer {
                            src: End::Switch(sw),
                            dst: End::Rank(r),
                            bytes: pb,
                            deps: ups.clone(),
                        });
                    }
                }
            }
            steps = 2;
        }
        (CollKind::Broadcast, _) => {
            // Pipelined chain from rank 0.
            let pieces = 8u64.min(c.bytes.max(1));
            let pb = c.bytes.div_ceil(pieces);
            let mut prev: Vec<Option<usize>> = vec![None; pieces as usize];
            for h in 0..n - 1 {
                for p in 0..pieces as usize {
                    let mut deps = vec![];
                    if let Some(x) = prev[p] {
                        deps.push(x);
                    }
                    if p > 0 {
                        // Same hop serialises pieces.
                        deps.push(t.len() - 1);
                    }
                    t.push(Transfer {
                        src: End::Rank(h),
                        dst: End::Rank(h + 1),
                        bytes: pb,
                        deps,
                    });
                    prev[p] = Some(t.len() - 1);
                }
            }
            steps = (n - 1) as u32 + pieces as u32 - 1;
        }
        (kind, _) => {
            // Ring family.
            let rings = ring_orders(n, view.islands);
            let nsteps = match kind {
                CollKind::AllReduce => 2 * (n - 1),
                _ => n - 1,
            };
            let per_ring = c.bytes / rings.len() as u64;
            let chunk = (per_ring / n as u64).max(1);
            for order in &rings {
                // last[i] = transfer that delivered the previous step's data to order[i].
                let mut last: Vec<Option<usize>> = vec![None; n];
                for _s in 0..nsteps {
                    let mut next: Vec<Option<usize>> = vec![None; n];
                    for i in 0..n {
                        let (src, dst) = (order[i], order[(i + 1) % n]);
                        let deps = last[i].map(|x| vec![x]).unwrap_or_default();
                        t.push(Transfer {
                            src: End::Rank(src),
                            dst: End::Rank(dst),
                            bytes: chunk,
                            deps,
                        });
                        next[(i + 1) % n] = Some(t.len() - 1);
                    }
                    last = next;
                }
            }
            steps = nsteps as u32;
        }
    }
    Plan {
        algo,
        transfers: t,
        steps,
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
    #[test]
    fn ring_allreduce_shape() {
        let c = Collective {
            kind: CollKind::AllReduce,
            bytes: 8 << 20,
            ranks: (0..8).collect(),
            algo: Algo::Ring,
        };
        let islands = vec![(0..8).collect::<Vec<_>>()];
        let p = plan(
            &c,
            &FabricView {
                nvls_switches: &[],
                islands: &islands,
            },
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
}
