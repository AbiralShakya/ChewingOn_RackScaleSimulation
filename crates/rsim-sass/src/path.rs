//! Per-warp dynamic paths recorded on real hardware (NVBit `warp_path` tool).
//!
//! A [`PathKernel`] is the static SASS of a kernel plus, for a few sampled
//! CTAs, the exact sequence of instructions every warp executed, with what a
//! timing model cannot recover statically: lane addresses, TMA source runs,
//! UMMA FLOPs, and the mbarrier each async operation signals or waits on.
//!
//! Unlike a static program with a branch profile, this is exact for
//! warp-specialised kernels, where producer, MMA, softmax and correction warps
//! take different paths through the same code. Unlike an Accel-Sim trace, it
//! keeps the mbarrier semantics, so the simulator re-times the
//! producer/consumer pipeline instead of replaying the spin loops the
//! hardware happened to execute (failed `try_wait` iterations are removed when
//! the trace is converted).
//!
//! CTAs that were not recorded replay a recorded CTA's path with every memory
//! address shifted by a per-instruction stride along each grid dimension,
//! inferred from recorded CTAs that differ in that dimension.
//!
//! File format (`.rpath`, little endian), written by
//! `research/flash_attention/trace_to_rsim.py`:
//!
//! ```text
//! b"RSIMPTH1"
//! u32 header_len, header JSON (see PathHeader; includes the SASS listing)
//! u32 n_warps
//! per warp: u32 cta, u32 warp, u32 n_ops, then per op:
//!   u32 idx, u32 active, u64 payload,
//!   u8 mbar_kind, u32 mbar_id, u32 mbar_arg, u32 mbar_n, u16 mbar_ranks,
//!   u32 run_bytes, u16 n_addr, n_addr x u64
//! ```
//!
//! then (optional) u32 n_entries; per entry: u32 cta, u32 warp, u32 n_ops, and
//! per TMA op of that warp: u16 n_addr, n_addr x u64 — the real source runs of
//! every CTA's TMA copies, for persistent kernels whose CTA -> tile mapping is
//! not affine in the CTA id.
//!
//! `mbar_ranks` is a bitmask of cluster ranks whose barrier `mbar_id` the op
//! signals (0 = the issuing CTA's own barrier): a 2-CTA TMA load completes
//! bytes on the leader CTA's barrier, `tcgen05.commit.multicast` arrives on
//! both CTAs of the pair.

use crate::inst::{Inst, Program};
use crate::parse::parse_listing;
use crate::walker::DynOp;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// mbarrier semantics of one dynamic instruction. `ranks` is a bitmask of
/// the cluster ranks whose barrier is signalled (0 = the issuing CTA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MbarRef {
    /// Block until barrier `id` has completed `need` phases (0 = no wait).
    Wait { id: u32, need: u32 },
    /// `n` arrivals when the instruction issues (one per active thread for a
    /// warp-wide `mbarrier.arrive`), adding `tx` expected transaction bytes.
    Arrive {
        id: u32,
        tx: u32,
        n: u32,
        ranks: u16,
    },
    /// The instruction's async copy completes `bytes` of transaction count
    /// (TMA load, cp.async.bulk) when its data has landed.
    TxDone { id: u32, bytes: u32, ranks: u16 },
    /// One arrival when all tensor-core work issued so far by this CTA
    /// completes (`tcgen05.commit`, possibly multicast to the CTA pair).
    CommitArrive { id: u32, ranks: u16 },
    /// Hardware `try_wait.parity` semantics: pass once the barrier's completed
    /// phase count has parity different from `parity`. Recorded waits use this
    /// rather than a phase count, because a waiter may skip phases (FA4's lazy
    /// rescaling), which makes the phase it waited for ambiguous from parity alone.
    WaitParity { id: u32, parity: u32 },
    /// Not an mbarrier: the id of a named CTA barrier (`bar.sync R, n`)
    /// whose id was in a register.
    Named { id: u32 },
}

impl MbarRef {
    pub fn id(&self) -> u32 {
        match *self {
            MbarRef::Wait { id, .. }
            | MbarRef::Arrive { id, .. }
            | MbarRef::TxDone { id, .. }
            | MbarRef::CommitArrive { id, .. }
            | MbarRef::Named { id }
            | MbarRef::WaitParity { id, .. } => id,
        }
    }
    fn decode(kind: u8, id: u32, arg: u32, n: u32, ranks: u16) -> Option<Self> {
        match kind {
            1 => Some(MbarRef::Wait { id, need: arg }),
            2 => Some(MbarRef::Arrive {
                id,
                tx: arg,
                n: n.max(1),
                ranks,
            }),
            3 => Some(MbarRef::TxDone {
                id,
                bytes: arg,
                ranks,
            }),
            4 => Some(MbarRef::CommitArrive { id, ranks }),
            5 => Some(MbarRef::Named { id }),
            6 => Some(MbarRef::WaitParity {
                id,
                parity: arg & 1,
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PathOp {
    pub idx: u32,
    pub active: u32,
    pub payload: u64,
    pub mbar: Option<MbarRef>,
    /// Bytes per address for TMA runs (0 = per-lane addresses).
    pub run_bytes: u32,
    pub addrs: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PathHeader {
    pub kernel: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    #[serde(default = "one3")]
    pub cluster: [u32; 3],
    pub regs: u32,
    pub smem: u32,
    /// Recorded CTA (linear id) whose path every CTA replays.
    pub primary: u32,
    /// mbarrier id -> arrivals per phase.
    #[serde(default)]
    pub barriers: HashMap<u32, u32>,
    /// instruction index -> byte stride per unit of blockIdx.{x,y,z}.
    #[serde(default)]
    pub strides: HashMap<u32, [i64; 3]>,
    /// The function's SASS listing in cuobjdump format (with encodings).
    pub sass: String,
    #[serde(default)]
    pub flops: f64,
    #[serde(default)]
    pub bytes: f64,
    /// CTAs that really run, for kernels that schedule their own work with
    /// cluster launch control (`clusterlaunchcontrol.try_cancel`, SASS
    /// `UGETNEXTWORKID`): a running CTA cancels the launch of a later one and
    /// takes its tile, so a recorded CTA's path covers several tiles and only
    /// about `grid / tiles per CTA` CTAs run.
    #[serde(default)]
    pub launch_ctas: Option<u32>,
}

fn one3() -> [u32; 3] {
    [1, 1, 1]
}

#[derive(Debug)]
pub struct PathKernel {
    pub header: PathHeader,
    pub program: Arc<Program>,
    /// Instruction index (NVBit `getIdx`) -> parsed instruction.
    pub insts: Vec<Arc<Inst>>,
    /// (cta, warp) -> path.
    pub warps: HashMap<(u32, u32), Arc<Vec<PathOp>>>,
    /// (cta, warp) -> source runs of each TMA op, for CTAs replaying another's path.
    pub tma_addrs: HashMap<(u32, u32), Arc<Vec<Vec<u64>>>>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err(format!("truncated path file at byte {}", self.p));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

impl PathKernel {
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let mut r = Rd { b, p: 0 };
        if r.take(8)? != b"RSIMPTH1" {
            return Err("not an rsim path file (bad magic)".into());
        }
        let hl = r.u32()? as usize;
        let header: PathHeader =
            serde_json::from_slice(r.take(hl)?).map_err(|e| format!("path header: {e}"))?;
        let program = parse_listing(&header.sass)
            .map_err(|e| format!("path SASS: {e}"))?
            .into_iter()
            .next()
            .ok_or("path SASS has no function")?;
        let insts: Vec<Arc<Inst>> = program.insts.iter().cloned().map(Arc::new).collect();
        let nw = r.u32()?;
        let mut warps = HashMap::new();
        for _ in 0..nw {
            let cta = r.u32()?;
            let warp = r.u32()?;
            let n = r.u32()? as usize;
            let mut ops = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = r.u32()?;
                let active = r.u32()?;
                let payload = r.u64()?;
                let kind = r.u8()?;
                let id = r.u32()?;
                let arg = r.u32()?;
                let n = r.u32()?;
                let ranks = r.u16()?;
                let run_bytes = r.u32()?;
                let na = r.u16()? as usize;
                let mut addrs = Vec::with_capacity(na);
                for _ in 0..na {
                    addrs.push(r.u64()?);
                }
                if idx as usize >= insts.len() {
                    return Err(format!(
                        "op index {idx} outside the listing ({} insts)",
                        insts.len()
                    ));
                }
                ops.push(PathOp {
                    idx,
                    active,
                    payload,
                    mbar: MbarRef::decode(kind, id, arg, n, ranks),
                    run_bytes,
                    addrs,
                });
            }
            warps.insert((cta, warp), Arc::new(ops));
        }
        let mut tma_addrs = HashMap::new();
        if r.p < b.len() {
            for _ in 0..r.u32()? {
                let cta = r.u32()?;
                let warp = r.u32()?;
                let n = r.u32()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    let na = r.u16()? as usize;
                    let mut a = Vec::with_capacity(na);
                    for _ in 0..na {
                        a.push(r.u64()?);
                    }
                    v.push(a);
                }
                tma_addrs.insert((cta, warp), Arc::new(v));
            }
        }
        Ok(Self {
            header,
            program: Arc::new(program),
            insts,
            warps,
            tma_addrs,
        })
    }

    /// Reads a `.rpath`, or a gzip-compressed `.rpath.gz` (recordings of
    /// long loops compress ~70x, which keeps them small enough for git).
    pub fn from_file(path: &str) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if path.ends_with(".gz") {
            use std::io::Read;
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&b[..])
                .read_to_end(&mut out)
                .map_err(|e| format!("{path}: {e}"))?;
            return Self::from_bytes(&out);
        }
        Self::from_bytes(&b)
    }

    pub fn coords(&self, cta: u64) -> [i64; 3] {
        let g = self.header.grid;
        let x = cta % g[0] as u64;
        let y = (cta / g[0] as u64) % g[1] as u64;
        let z = cta / (g[0] as u64 * g[1] as u64);
        [x as i64, y as i64, z as i64]
    }

    /// The instruction stream of warp `warp` of CTA `cta`.
    pub fn stream(self: &Arc<Self>, cta: u64, warp: u32) -> PathStream {
        // Replay a recorded CTA with the same rank in its cluster: in a 2-CTA pair the
        // leader issues the MMAs and the peer does not, so their paths differ.
        let cl = self.header.cluster.iter().product::<u32>().max(1) as u64;
        let rep = if self.warps.contains_key(&(cta as u32, warp)) {
            cta as u32
        } else if cl > 1 {
            let mut recorded: Vec<u32> = self.warps.keys().map(|k| k.0).collect();
            recorded.sort_unstable();
            recorded
                .into_iter()
                .find(|&c| c as u64 % cl == cta % cl)
                .unwrap_or(self.header.primary)
        } else {
            self.header.primary
        };
        let ops = self.warps.get(&(rep, warp)).cloned().unwrap_or_default();
        let (c, r) = (self.coords(cta), self.coords(rep as u64));
        PathStream {
            k: self.clone(),
            ops,
            i: 0,
            delta: [c[0] - r[0], c[1] - r[1], c[2] - r[2]],
            tma: self.tma_addrs.get(&(cta as u32, warp)).cloned(),
            tma_i: 0,
        }
    }
}

pub struct PathStream {
    k: Arc<PathKernel>,
    ops: Arc<Vec<PathOp>>,
    i: usize,
    delta: [i64; 3],
    tma: Option<Arc<Vec<Vec<u64>>>>,
    tma_i: usize,
}

impl PathStream {
    pub fn next_op(&mut self) -> Option<DynOp> {
        let op = self.ops.get(self.i)?;
        self.i += 1;
        let inst = &self.k.insts[op.idx as usize];
        let mut addrs = op.addrs.clone();
        let real_tma = if inst.class == crate::OpClass::Tma && op.run_bytes > 0 {
            self.tma_i += 1;
            self.tma.as_ref().and_then(|t| t.get(self.tma_i - 1))
        } else {
            None
        };
        if let Some(t) = real_tma {
            addrs = t.clone();
        } else if self.delta != [0, 0, 0] && !addrs.is_empty() {
            if let Some(s) = self.k.header.strides.get(&op.idx) {
                let off = s[0] * self.delta[0] + s[1] * self.delta[1] + s[2] * self.delta[2];
                for a in &mut addrs {
                    *a = a.wrapping_add(off as u64);
                }
            }
        }
        Some(DynOp {
            inst: self.k.insts[op.idx as usize].clone(),
            active: op.active,
            addrs,
            payload: op.payload,
            mbar: op.mbar,
            run_bytes: op.run_bytes,
        })
    }
}
