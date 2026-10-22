//! NCCL: protocols, channels, chunking, and NCCL's own cost model.
//!
//! Two things live here.
//!
//! 1. **The protocol and channel model the simulators execute.** NCCL moves
//!    data with one CTA per *channel*; each channel runs one of three
//!    protocols:
//!    * `Simple` — bulk stores, then a memory fence and a flag per slice.
//!      Full payload efficiency, but every step pays a fence + flag-poll
//!      round trip.
//!    * `LL` — 8-byte stores carrying 4 B data + 4 B flag. 50% payload
//!      efficiency, no fence, lowest latency.
//!    * `LL128` — 128-byte lines with one 8-byte flag (120/128 payload),
//!      relying on NVLink delivering a 128 B store atomically.
//!
//!    A channel is an SM, so its copy throughput is capped (`per_channel_gbps`)
//!    and the number of channels decides how many SMs a collective steals
//!    from compute.
//!
//! 2. **A port of NCCL 2.32.3's tuning model** (`src/tuning/*.cc`,
//!    Apache-2.0). NCCL predicts `latency + bytes / bandwidth` for every
//!    (algorithm, protocol) and picks the fastest. We use the same model to
//!    choose the algorithm/protocol/channel count, exactly as NCCL would, and
//!    report its prediction next to ours (`rsim net model`). The constants
//!    are NVIDIA's, fitted by them against their hardware; they are the best
//!    public description of NCCL's behaviour, but they are a model, not a
//!    measurement.

use crate::collective::{Algo, CollKind};
use rsim_hw::Arch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Proto {
    Ll,
    Ll128,
    Simple,
}

impl Proto {
    pub const ALL: [Proto; 3] = [Proto::Ll, Proto::Ll128, Proto::Simple];
    fn idx(self) -> usize {
        match self {
            Proto::Ll => 0,
            Proto::Ll128 => 1,
            Proto::Simple => 2,
        }
    }
    /// Fraction of each store that is payload.
    pub fn payload_efficiency(self) -> f64 {
        match self {
            Proto::Ll => 0.5,
            Proto::Ll128 => 120.0 / 128.0,
            Proto::Simple => 1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Proto::Ll => "LL",
            Proto::Ll128 => "LL128",
            Proto::Simple => "Simple",
        }
    }
}

/// GPU generation index used by NCCL's tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gen {
    Ampere,
    Hopper,
    Blackwell,
}

impl Gen {
    pub fn of(a: &Arch) -> Self {
        match a {
            Arch::Ampere => Gen::Ampere,
            Arch::Hopper => Gen::Hopper,
            Arch::Blackwell => Gen::Blackwell,
        }
    }
    fn idx(self) -> usize {
        match self {
            Gen::Ampere => 1,
            Gen::Hopper => 2,
            Gen::Blackwell => 3,
        }
    }
    /// NCCL's per-link NVLink bandwidth (`SMxx_NVLINK_BW`, GB/s).
    pub fn nccl_nvlink_bw(self) -> f64 {
        match self {
            Gen::Ampere => 20.0,
            Gen::Hopper => 20.6,
            Gen::Blackwell => 40.1,
        }
    }
    /// Default NVLS channels (`ncclNvlsChannels`).
    pub fn nvls_channels(self, multi_node: bool) -> u32 {
        match (self, multi_node) {
            (Gen::Blackwell, false) => 24,
            (Gen::Blackwell, true) => 32,
            _ => 16,
        }
    }
    /// NCCL `nvlsEfficiency`.
    fn nvls_efficiency(self) -> f64 {
        match self {
            Gen::Ampere => 0.0,
            Gen::Hopper => 0.85,
            Gen::Blackwell => 0.74,
        }
    }
    /// Intra-node speed array searched by `ncclTopoCompute` (GB/s per channel).
    fn speed_array_intra(self) -> &'static [f64] {
        match self {
            Gen::Ampere => &[
                40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0,
            ],
            Gen::Hopper => &[
                60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0,
            ],
            Gen::Blackwell => &[
                90.0, 80.0, 70.0, 60.0, 50.0, 45.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// NCCL 2.32.3 tuner constants (src/tuning/cost_model.cc).
// Index order: algorithm [Tree, Ring, CollNetDirect, CollNetChain, NVLS,
// NVLSTree, PAT] x protocol [LL, LL128, Simple].

const A_TREE: usize = 0;
const A_RING: usize = 1;
const A_COLLNET_DIRECT: usize = 2;
const A_NVLS: usize = 4;
const A_NVLS_TREE: usize = 5;

const BASE_LAT: [[f64; 3]; 7] = [
    [6.8, 14.0, 8.4],
    [6.6, 14.0, 8.4],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [8.0, 8.0, 8.0],
];
const HW_NVLINK: usize = 0;
const HW_PCI: usize = 1;
const HW_NET: usize = 2;
const HW_LAT: [[[f64; 3]; 7]; 3] = [
    [
        [0.6, 1.25, 4.0],
        [0.6, 1.9, 3.4],
        [0.0, 0.0, 3.7],
        [0.0, 0.0, 2.8],
        [0.0, 0.0, 25.0],
        [0.0, 0.0, 25.0],
        [0.0, 0.0, 4.0],
    ],
    [
        [1.0, 1.9, 4.0],
        [1.0, 2.5, 5.7],
        [0.0, 0.0, 3.7],
        [0.0, 0.0, 2.8],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0],
    ],
    [
        [5.0, 8.5, 14.0],
        [2.7, 4.0, 14.0],
        [0.0, 0.0, 31.0],
        [0.0, 0.0, 30.0],
        [0.0, 0.0, 18.0],
        [0.0, 0.0, 20.9],
        [0.0, 0.0, 14.0],
    ],
];
// [gen idx 0..3][N1, N2, N4]
const LL_MAX_BW: [[f64; 3]; 4] = [
    [39.0, 39.0, 20.4],
    [87.7, 22.5, 19.0],
    [141.0, 45.0, 35.0],
    [282.4, 90.0, 70.0],
];
const PER_CH_RING_LL128: [[f64; 3]; 4] = [
    [20.0, 20.0, 20.0],
    [20.0, 20.0, 20.0],
    [36.7, 36.7, 36.7],
    [40.0, 40.0, 40.0],
];
const PER_CH_TREE_LL128: [[f64; 3]; 4] = [
    [20.0, 20.0, 20.0],
    [20.0, 20.0, 20.0],
    [36.7, 36.7, 29.0],
    [55.6, 36.7, 20.0],
];
const PER_CH_TREE: [[f64; 3]; 4] = [
    [26.5, 18.5, 10.0],
    [24.0, 23.6, 17.8],
    [38.7, 41.4, 36.0],
    [70.0, 42.8, 24.0],
];
const PER_CH_NVLS_TREE: [[f64; 3]; 4] = [
    [26.5, 18.5, 10.0],
    [24.0, 23.6, 17.8],
    [0.0, 57.7, 45.5],
    [0.0, 96.0, 80.0],
];
const TREE_CORRECTION: [[f64; 24]; 3] = [
    [
        1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
    [
        1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.9,
        0.9, 0.9, 0.9, 1.0, 1.0, 1.0,
    ],
    [
        0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8,
        0.7, 0.7, 0.8, 0.9, 0.9, 0.9,
    ],
];
const NCCL_STEPS: u64 = 8;
const SIMPLE_BUFF: u64 = 4 << 20;
const LL_BUFF: u64 = 8 * 512 * NCCL_STEPS * 16;
const LL128_BUFF: u64 = 120 * 640 * NCCL_STEPS * 8;

fn log2i(x: u64) -> i64 {
    if x == 0 {
        -1
    } else {
        63 - x.leading_zeros() as i64
    }
}

/// What NCCL's tuner sees of a communicator.
#[derive(Clone, Debug)]
pub struct CommShape {
    pub nranks: usize,
    pub nnodes: usize,
    /// Ranks per node (minimum over nodes).
    pub local: usize,
    pub gen: Gen,
    /// Intra-node transport is NVLink (else PCIe).
    pub nvlink: bool,
    /// NVLink links per GPU.
    pub nvlink_links: f64,
    /// NIC bandwidth per GPU (GB/s); 0 when single node.
    pub nic_gbps: f64,
    /// NVLink SHARP available across the whole communicator's NVLink domain.
    pub nvls: bool,
    /// One NVLink domain spans every rank (NVL72 / MNNVL).
    pub one_nvlink_domain: bool,
    /// In-network reduction on the scale-out network (IB SHARP / CollNet).
    pub collnet: bool,
}

impl CommShape {
    fn n_idx(&self) -> usize {
        if self.nnodes <= 2 {
            self.nnodes - 1
        } else {
            2
        }
    }
    /// Total intra-node bandwidth NCCL assigns to ring/tree graphs (GB/s).
    pub fn intra_total(&self) -> f64 {
        if self.nvlink {
            self.nvlink_links * self.gen.nccl_nvlink_bw()
        } else {
            24.0 // PCIe Gen4/5 x16 as NCCL rates it
        }
    }
    /// Total inter-node bandwidth per GPU-ring (local ranks x NIC).
    pub fn inter_total(&self) -> f64 {
        if self.nnodes <= 1 {
            self.intra_total()
        } else {
            self.local as f64 * self.nic_gbps
        }
    }
    /// NVLS graph bandwidth: the fastest speed-array entry such that
    /// `speed x channels` fits in the GPU's NVLink bandwidth.
    pub fn nvls_total(&self) -> f64 {
        let nc = self.gen.nvls_channels(self.nnodes > 1) as f64;
        let tot = self.intra_total();
        let s = self
            .gen
            .speed_array_intra()
            .iter()
            .copied()
            .find(|&s| s * nc <= tot)
            .unwrap_or(1.0);
        s * nc
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Prediction {
    pub algo: Algo,
    pub proto: Proto,
    pub time_us: f64,
    pub lat_us: f64,
    /// NCCL "algorithm bandwidth" the model uses (bytes / time), GB/s.
    pub algbw: f64,
    pub channels: u32,
}

fn nsteps(kind: CollKind, n: usize) -> usize {
    match kind {
        CollKind::AllReduce => 2 * (n - 1),
        CollKind::AllGather | CollKind::ReduceScatter => n - 1,
        _ => n,
    }
}

fn algo_idx(a: Algo) -> Option<usize> {
    Some(match a {
        Algo::Tree => A_TREE,
        Algo::Ring => A_RING,
        Algo::Collnet => A_COLLNET_DIRECT,
        Algo::Nvls => A_NVLS,
        Algo::NvlsTree => A_NVLS_TREE,
        _ => return None,
    })
}

/// Default channel count of a communicator for ring/tree (`comm->nChannels`).
pub fn default_channels(s: &CommShape) -> u32 {
    match s.gen {
        Gen::Ampere => 12,
        // NVIDIA: "traditional ring-based collectives use 16 SMs per GPU" (NCCL 2.27 blog).
        _ => 16,
    }
}

/// `ncclTuningGetChannels`: fewer channels for small messages.
pub fn tuned_channels(algo: Algo, proto: Proto, bytes: u64, s: &CommShape, max: u32) -> u32 {
    let mut nc = match algo {
        Algo::Nvls | Algo::NvlsTree => s.gen.nvls_channels(s.nnodes > 1),
        _ => default_channels(s),
    }
    .min(max.max(1));
    if !matches!(algo, Algo::Nvls | Algo::NvlsTree | Algo::Collnet) {
        let nt: u64 = match proto {
            Proto::Simple => 512,
            Proto::Ll => 512,
            Proto::Ll128 => 640,
        };
        let mut thr: u64 = match proto {
            Proto::Simple => 64,
            Proto::Ll | Proto::Ll128 => 8,
        };
        if algo == Algo::Ring && proto == Proto::Ll {
            thr *= s.nranks as u64;
        }
        while bytes < nc as u64 * nt * thr && nc >= 2 {
            nc -= 1;
        }
    }
    nc.max(1)
}

/// NCCL's predicted time for one (algorithm, protocol). `None` = NCCL would
/// not consider it (unsupported or disabled).
pub fn predict(
    kind: CollKind,
    algo: Algo,
    proto: Proto,
    bytes: u64,
    s: &CommShape,
    max_channels: u32,
) -> Option<Prediction> {
    let a = algo_idx(algo)?;
    let p = proto.idx();
    let n = s.nranks;
    if n < 2 {
        return None;
    }
    let g = s.gen.idx();
    let ni = s.n_idx();
    let hw_intra = if s.nvlink { HW_NVLINK } else { HW_PCI };
    let hw_inter = if s.nnodes == 1 { hw_intra } else { HW_NET };
    let ns = nsteps(kind, n) as f64;
    let nch = tuned_channels(algo, proto, bytes, s, max_channels);
    let nchf = default_channels(s).min(max_channels.max(1)) as f64;
    let ll_max = LL_MAX_BW[if s.nnodes == 1 { g } else { 0 }][ni];
    // LL128 needs NVLink inside the node.
    if proto == Proto::Ll128 && !s.nvlink {
        return None;
    }
    let (lat, mut bw);
    match algo {
        Algo::Ring => {
            if !matches!(
                kind,
                CollKind::AllReduce | CollKind::AllGather | CollKind::ReduceScatter
            ) {
                return None;
            }
            let total = if s.nnodes == 1 || (s.nnodes <= 2 && s.gen != Gen::Blackwell) {
                s.intra_total()
            } else {
                s.inter_total().min(s.intra_total())
            };
            let mut bus = total * nchf / default_channels(s) as f64;
            if proto == Proto::Ll {
                bus = ll_max.min(bus * 0.5);
            }
            if proto == Proto::Ll128 {
                bus = (bus * 0.92).min(nchf * PER_CH_RING_LL128[g][ni]);
            }
            bw = bus * n as f64 / ns;
            let mut intra_lat = HW_LAT[hw_intra][A_RING][p];
            let inter_lat = if s.nnodes == 1 {
                intra_lat
            } else if s.local == 1 {
                HW_LAT[HW_NET][A_TREE][p]
            } else {
                HW_LAT[hw_inter][A_RING][p]
            };
            if s.nnodes > 1 {
                let mut ovh = 1.0;
                if proto == Proto::Simple {
                    ovh *= 3.0;
                }
                intra_lat = intra_lat.max(ovh);
            }
            let n_inter = if s.nnodes == 1 {
                0.0
            } else if kind == CollKind::AllReduce {
                2.0 * (s.nnodes - 1) as f64
            } else {
                (s.nnodes - 1) as f64
            };
            let mut l = BASE_LAT[A_RING][p] + (ns - n_inter) * intra_lat + n_inter * inter_lat;
            // Ring/Simple plateau.
            let bwnvl = s.gen == Gen::Blackwell && s.nvlink;
            let plateau = (s.nnodes > 1 && kind == CollKind::AllReduce)
                || (s.nnodes == 1
                    && bwnvl
                    && matches!(
                        kind,
                        CollKind::AllReduce | CollKind::AllGather | CollKind::ReduceScatter
                    ));
            let bprc = bytes / (default_channels(s) as u64 * n as u64);
            if proto == Proto::Simple && plateau && bprc >= 64 {
                let mut f: f64 = if s.gen == Gen::Ampere { 1.9 } else { 1.4 };
                if s.nnodes == 1 && bwnvl {
                    let delta = 0.1;
                    let maxd = f - 1.0;
                    let step_scale = 1.0 - (delta / maxd).powi((ns as i32 - 1).max(0));
                    let mut eff = maxd;
                    if bprc > 256 * 1024 {
                        let sc = ((bprc - 256 * 1024) as f64 / (256.0 * 1024.0)).min(1.0);
                        eff -= delta * sc;
                    }
                    f = 1.0 + step_scale * eff;
                }
                l *= f;
            }
            lat = l;
        }
        Algo::Tree => {
            if kind != CollKind::AllReduce {
                return None;
            }
            let total = if s.gen != Gen::Blackwell {
                if s.nnodes <= 2 {
                    s.intra_total()
                } else {
                    s.inter_total()
                }
            } else {
                s.inter_total().min(s.intra_total())
            };
            let mut bus = total * nchf / default_channels(s) as f64;
            bus = (bus * 0.92).min(nchf * PER_CH_TREE[g][ni]);
            if proto == Proto::Ll {
                bus = (bus / 3.8).min(ll_max);
            }
            if proto == Proto::Ll128 {
                let f = if s.nnodes == 1 {
                    7.0 / 9.0
                } else {
                    120.0 / 128.0
                };
                bus = (bus * f).min(nchf * PER_CH_TREE_LL128[g][ni]);
            }
            bw = bus * 0.5;
            let intra_lat = HW_LAT[hw_intra][A_TREE][p];
            let inter_lat = if s.nnodes == 1 {
                intra_lat
            } else {
                HW_LAT[hw_inter][A_TREE][p]
            };
            lat = BASE_LAT[A_TREE][p]
                + 2.0
                    * ((n / s.nnodes - 1) as f64 * intra_lat
                        + log2i(s.nnodes as u64).max(0) as f64 * inter_lat);
            let ls = log2i(bytes >> 6);
            if (0..23).contains(&ls) {
                bw *= TREE_CORRECTION[p][ls as usize];
            }
        }
        Algo::Nvls | Algo::NvlsTree => {
            if !s.nvls || proto != Proto::Simple {
                return None;
            }
            if algo == Algo::NvlsTree && (s.nnodes == 1 || kind != CollKind::AllReduce) {
                return None;
            }
            if algo == Algo::Nvls && s.nnodes > 1 && !s.collnet && !s.one_nvlink_domain {
                return None;
            }
            if !matches!(
                kind,
                CollKind::AllReduce | CollKind::AllGather | CollKind::ReduceScatter
            ) {
                return None;
            }
            let nc = s.gen.nvls_channels(s.nnodes > 1) as f64;
            let per_intra = s.nvls_total() / nc;
            let mut intra = per_intra * s.gen.nvls_efficiency() * (nc - 1.0) / nc;
            if kind == CollKind::AllReduce {
                intra *= 2.0;
            } else {
                let ppn = s.local as f64;
                intra *= (ppn - 1.0) / ppn;
            }
            let mut b = intra;
            if s.nnodes > 1 && !s.one_nvlink_domain {
                let per_inter = s.inter_total() / nc
                    * if s.nnodes <= 2 && algo == Algo::NvlsTree {
                        2.0
                    } else {
                        1.0
                    };
                b = b.min(per_inter);
            }
            if algo == Algo::NvlsTree {
                b = b.min(PER_CH_NVLS_TREE[g][ni]);
            }
            b *= nc;
            bw = if kind == CollKind::AllReduce {
                b * n as f64 / ns
            } else {
                b
            };
            let intra_lat = HW_LAT[hw_intra][a][p];
            let inter_lat = if s.nnodes == 1 {
                intra_lat
            } else {
                HW_LAT[hw_inter][a][p]
            };
            lat = if algo == Algo::Nvls {
                intra_lat
                    + if s.nnodes > 1 && !s.one_nvlink_domain {
                        inter_lat
                    } else {
                        0.0
                    }
            } else {
                intra_lat + 2.0 * log2i(s.nnodes as u64).max(0) as f64 * inter_lat
            };
        }
        Algo::Collnet => {
            if !s.collnet || s.nnodes < 2 || proto != Proto::Simple || kind != CollKind::AllReduce {
                return None;
            }
            let ppn = s.local as f64;
            let mut bus = nchf * s.intra_total() / default_channels(s) as f64;
            let mut factor = ppn / nchf;
            factor -= (factor - 1.0) / 2.0;
            bus /= factor.max(1e-9);
            if s.gen != Gen::Ampere {
                bus *= 0.85;
            }
            bw = bus * 0.5;
            let intra_lat = HW_LAT[hw_intra][a][p];
            let inter_lat = HW_LAT[HW_NET][a][p];
            lat = 2.0 * ((s.local - 1).min(1) as f64 * intra_lat + (s.local - 1) as f64 * 0.4)
                + inter_lat;
        }
        _ => return None,
    }
    if bw <= 0.0 {
        return None;
    }
    let time_us = lat + bytes as f64 / (1000.0 * bw);
    Some(Prediction {
        algo,
        proto,
        time_us,
        lat_us: lat,
        algbw: bw,
        channels: nch,
    })
}

/// Every (algorithm, protocol) NCCL would consider, fastest first.
pub fn candidates(kind: CollKind, bytes: u64, s: &CommShape, max_channels: u32) -> Vec<Prediction> {
    let mut v: Vec<Prediction> = [
        Algo::Ring,
        Algo::Tree,
        Algo::Nvls,
        Algo::NvlsTree,
        Algo::Collnet,
    ]
    .iter()
    .flat_map(|&a| {
        Proto::ALL
            .iter()
            .filter_map(move |&p| predict(kind, a, p, bytes, s, max_channels))
    })
    .collect();
    v.sort_by(|a, b| a.time_us.partial_cmp(&b.time_us).unwrap());
    v
}

/// NCCL's choice. `algo`/`proto` force a choice when given (falling back to
/// the fastest supported protocol of that algorithm).
pub fn select(
    kind: CollKind,
    bytes: u64,
    s: &CommShape,
    algo: Option<Algo>,
    proto: Option<Proto>,
    max_channels: u32,
) -> Option<Prediction> {
    let c = candidates(kind, bytes, s, max_channels);
    c.iter()
        .find(|x| algo.is_none_or(|a| x.algo == a) && proto.is_none_or(|p| x.proto == p))
        .or_else(|| c.iter().find(|x| algo.is_none_or(|a| x.algo == a)))
        .copied()
}

/// Bytes moved per channel per pipeline step ("chunk") — NCCL `calcCollChunking`.
pub fn chunk_bytes(algo: Algo, proto: Proto, bytes: u64, nch: u32, heads: u32) -> u64 {
    let step = match proto {
        Proto::Simple => SIMPLE_BUFF,
        Proto::Ll => LL_BUFF,
        Proto::Ll128 => LL128_BUFF,
    } / NCCL_STEPS;
    let chunk_steps = if proto == Proto::Simple && algo == Algo::Ring {
        NCCL_STEPS / 2
    } else {
        1
    };
    let mut c = step * chunk_steps;
    match proto {
        Proto::Ll => c /= 2,
        Proto::Ll128 => c = c / 16 * 15,
        Proto::Simple => {}
    }
    let nch = nch.max(1) as u64;
    match algo {
        Algo::Nvls => {
            c = c.min(128 * 1024);
            let ops = nch * heads.max(1) as u64;
            if bytes < 64 * ops * c && c > 65536 {
                c = 65536;
            }
            if bytes < 8 * ops * c && c > 32768 {
                c = 32768;
            }
            if bytes < 2 * ops * c && c > 16384 {
                c = 16384;
            }
        }
        Algo::NvlsTree => {
            let ops = nch * heads.max(1) as u64;
            c = c.min(128 * 1024);
            for (k, lim) in [(32, 262144), (16, 131072), (4, 65536), (1, 32768)] {
                if bytes < k * ops * c && c > lim {
                    c = lim;
                }
            }
        }
        Algo::Collnet => {
            c = c.min(256 * 1024);
            while bytes / (nch * c) < 8 && c > 32768 {
                c /= 2;
            }
        }
        Algo::Tree if proto == Proto::Ll128 => {
            while bytes / (nch * c) < 16 && c > 32768 {
                c /= 2;
            }
        }
        _ => {}
    }
    c.max(1)
}

/// Mechanistic parameters the simulators use for NCCL. Defaults are
/// starting points; `rsim net fit` refits the latency terms (see
/// `calibration/net/`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct NcclParams {
    /// Software cost of one protocol step on top of wire latency (fence +
    /// flag write + remote poll), ns, for [LL, LL128, Simple].
    pub step_ns: [f64; 3],
    /// Extra per-step cost when a step crosses the network (proxy thread,
    /// RDMA post + completion), ns, [LL, LL128, Simple].
    pub net_step_ns: [f64; 3],
    /// Fixed cost per collective (kernel launch to first data), ns, [LL, LL128, Simple].
    pub launch_ns: [f64; 3],
    /// NVLS fixed cost (multicast object sync), ns.
    pub nvls_launch_ns: f64,
    /// Copy throughput of one channel (SM), GB/s, for [LL, LL128, Simple].
    /// `None` = NCCL's per-channel caps for the GPU generation.
    pub per_channel_gbps: Option<[f64; 3]>,
    /// NVLS chunks in flight per channel (multimem.ld_reduce outstanding loads).
    pub nvls_depth: u32,
    /// Route cross-rail all-to-all traffic over NVLink to the GPU on the
    /// destination's rail first (NCCL PXN).
    pub pxn: bool,
    /// Cap on DAG transfers per collective (larger messages use bigger
    /// pipeline pieces than NCCL's chunk to keep simulation cost bounded).
    pub max_transfers: usize,
    /// Force a protocol.
    pub proto: Option<Proto>,
    /// Force a channel count.
    pub channels: Option<u32>,
}

impl Default for NcclParams {
    fn default() -> Self {
        // Fitted by `rsim net fit calibration/net/published.json` (transfer
        // fidelity, 3 rounds, mean |ln err| 0.080 over the `fit` points).
        Self {
            step_ns: [250.0, 2000.0, 4500.0],
            net_step_ns: [3750.0, 3000.0, 4000.0],
            launch_ns: [7500.0, 14800.0, 11250.0],
            nvls_launch_ns: 19000.0,
            per_channel_gbps: None,
            nvls_depth: 2,
            pxn: true,
            max_transfers: 120_000,
            proto: None,
            channels: None,
        }
    }
}

impl NcclParams {
    /// Per-channel copy throughput, GB/s.
    pub fn channel_gbps(&self, gen: Gen, proto: Proto, nch: u32) -> f64 {
        if let Some(t) = self.per_channel_gbps {
            return t[proto.idx()];
        }
        let g = gen.idx();
        match proto {
            Proto::Ll => LL_MAX_BW[g][0] / nch.max(1) as f64,
            Proto::Ll128 => PER_CH_RING_LL128[g][0],
            Proto::Simple => PER_CH_TREE[g][0] * 1.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hgx(gen: Gen) -> CommShape {
        CommShape {
            nranks: 8,
            nnodes: 1,
            local: 8,
            gen,
            nvlink: true,
            nvlink_links: 18.0,
            nic_gbps: 0.0,
            nvls: true,
            one_nvlink_domain: true,
            collnet: false,
        }
    }

    fn busbw(p: &Prediction, n: usize) -> f64 {
        p.algbw * 2.0 * (n as f64 - 1.0) / n as f64
    }

    #[test]
    fn nvls_large_message_busbw() {
        // H100: 16 ch x 20 GB/s x 0.85 x 2 x 15/16 = 510 GB/s busbw.
        let p = predict(
            CollKind::AllReduce,
            Algo::Nvls,
            Proto::Simple,
            1 << 30,
            &hgx(Gen::Hopper),
            64,
        )
        .unwrap();
        assert!((busbw(&p, 8) - 510.0).abs() < 1.0, "{}", busbw(&p, 8));
        let b = predict(
            CollKind::AllReduce,
            Algo::Nvls,
            Proto::Simple,
            1 << 30,
            &hgx(Gen::Blackwell),
            64,
        )
        .unwrap();
        assert!((busbw(&b, 8) - 1021.2).abs() < 1.0, "{}", busbw(&b, 8));
    }

    #[test]
    fn small_messages_pick_ll() {
        let s = hgx(Gen::Hopper);
        let p = select(CollKind::AllReduce, 1024, &s, None, None, 64).unwrap();
        assert_eq!(p.proto, Proto::Ll);
        let big = select(CollKind::AllReduce, 1 << 30, &s, None, None, 64).unwrap();
        assert_eq!(big.algo, Algo::Nvls);
    }
}
