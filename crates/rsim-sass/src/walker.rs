//! Static program → dynamic per-warp instruction stream.
//!
//! Without functional execution we cannot evaluate branch conditions, so
//! control flow is resolved from a [`BranchProfile`]: for each branch PC, how
//! many times a warp takes it before falling through (a loop trip count).
//! Profiles come from NVBit/Nsight instruction counts, from the synthetic
//! kernel generators (which know their loop structure), or default to
//! "conditional branches not taken, unconditional taken once-per-visit".
//! For fully exact control flow and addresses use an Accel-Sim trace.
//!
//! Memory addresses for static programs come from [`AddrGen`]s attached
//! per instruction PC (affine in CTA id, warp id, lane and loop iteration).

use crate::inst::{Inst, Program};
use crate::isa::OpClass;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Predicate register ids: P0-P6 (256..264) and UP0-UP6 (328..336).
fn is_pred(r: crate::Reg) -> bool {
    (256..264).contains(&r) || (328..336).contains(&r)
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BranchProfile {
    /// pc -> times taken per warp before falling through.
    pub taken: HashMap<u32, u64>,
    /// pc -> fraction of CTAs for which a guarded EXIT fires (tail CTAs).
    #[serde(default)]
    pub exit_prob: HashMap<u32, f64>,
}

/// Affine address generator: `base + cta*cta_stride + warp*warp_stride +
/// lane*lane_stride + iter*iter_stride`, wrapped into `footprint` bytes when
/// non-zero (models re-use / cache-resident working sets).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AddrGen {
    pub base: u64,
    pub cta_stride: i64,
    pub warp_stride: i64,
    pub lane_stride: i64,
    pub iter_stride: i64,
    #[serde(default)]
    pub footprint: u64,
    /// CTA index transform before `cta_stride`: `(cta / cta_div) % cta_mod`
    /// (0 = disabled). Maps a linear CTA id to a tile row/column.
    #[serde(default)]
    pub cta_div: u64,
    #[serde(default)]
    pub cta_mod: u64,
}

impl AddrGen {
    pub fn addr(&self, cta: u64, warp: u64, lane: u64, iter: u64) -> u64 {
        let cta = if self.cta_div > 1 {
            cta / self.cta_div
        } else {
            cta
        };
        let cta = if self.cta_mod > 0 {
            cta % self.cta_mod
        } else {
            cta
        };
        let off = cta as i64 * self.cta_stride
            + warp as i64 * self.warp_stride
            + lane as i64 * self.lane_stride
            + iter as i64 * self.iter_stride;
        let off = if self.footprint > 0 {
            off.rem_euclid(self.footprint as i64)
        } else {
            off
        };
        self.base.wrapping_add(off as u64)
    }
}

/// A dynamic instruction handed to the timing model.
#[derive(Clone, Debug)]
pub struct DynOp {
    pub inst: Arc<Inst>,
    pub active: u32,
    /// Per-lane addresses (memory ops only; may be empty = unknown/streaming).
    pub addrs: Vec<u64>,
    /// Payload for async ops: bytes for TMA, FLOPs for MMA.
    pub payload: u64,
    /// mbarrier this op waits on or signals (recorded paths only; static
    /// programs use per-PC `BarOp` annotations instead).
    pub mbar: Option<crate::path::MbarRef>,
    /// TMA: bytes per entry of `addrs` (each address starts a contiguous run).
    pub run_bytes: u32,
}

/// Per-PC side information attached to a static program.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Annotations {
    pub addr: HashMap<u32, AddrGen>,
    /// pc -> bytes (TMA) or FLOPs (MMA) per dynamic instance.
    pub payload: HashMap<u32, u64>,
    /// pc -> active lanes override (e.g. single-thread elect for tcgen05 issue).
    pub active: HashMap<u32, u32>,
}

pub struct Walker {
    prog: Arc<Program>,
    insts: Arc<Vec<Arc<Inst>>>,
    profile: Arc<BranchProfile>,
    ann: Arc<Annotations>,
    pc: usize,
    taken: HashMap<u32, u64>,
    iter: u64,
    cta: u64,
    warp: u64,
    exited: bool,
    steps: u64,
    max_steps: u64,
}

impl Walker {
    pub fn new(
        prog: Arc<Program>,
        insts: Arc<Vec<Arc<Inst>>>,
        profile: Arc<BranchProfile>,
        ann: Arc<Annotations>,
        cta: u64,
        warp: u64,
    ) -> Self {
        Self {
            prog,
            insts,
            profile,
            ann,
            pc: 0,
            taken: HashMap::new(),
            iter: 0,
            cta,
            warp,
            exited: false,
            steps: 0,
            max_steps: 1 << 32,
        }
    }

    /// Wrap a program's instructions in `Arc`s once (share across warps).
    pub fn share(prog: &Program) -> Vec<Arc<Inst>> {
        prog.insts.iter().cloned().map(Arc::new).collect()
    }

    fn exit_fires(&self, pc: u32) -> bool {
        match self.profile.exit_prob.get(&pc) {
            None => false,
            Some(&p) => {
                // Deterministic per-CTA hash in [0,1).
                let h = (self.cta.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64
                    / (1u64 << 53) as f64;
                h < p
            }
        }
    }

    pub fn next_op(&mut self) -> Option<DynOp> {
        if self.exited || self.pc >= self.insts.len() || self.steps >= self.max_steps {
            return None;
        }
        self.steps += 1;
        let inst = self.insts[self.pc].clone();
        let pc = inst.pc;
        let mut next = self.pc + 1;
        match inst.class {
            OpClass::Exit => {
                if inst.guard.is_none() || self.exit_fires(pc) {
                    self.exited = true;
                }
            }
            OpClass::Branch
                if inst.branch_target.is_some() && matches!(inst.base_opcode(), "BRA" | "JMP") =>
            {
                let target = inst.branch_target.unwrap();
                let take = match self.profile.taken.get(&pc) {
                    Some(&n) => {
                        let c = self.taken.entry(pc).or_insert(0);
                        if *c < n {
                            *c += 1;
                            true
                        } else {
                            // Reset so an enclosing loop re-runs this inner loop.
                            *c = 0;
                            false
                        }
                    }
                    // `BRA.U !UP0, tgt` carries its predicate as an operand.
                    None => inst.guard.is_none() && !inst.srcs.iter().any(|&r| is_pred(r)),
                };
                if take {
                    if let Some(t) = self.prog.index_of_pc(target) {
                        if t <= self.pc {
                            self.iter += 1; // backward edge = loop iteration
                        }
                        // Unconditional self-loop (`BRA` to itself after EXIT) => stop.
                        if t == self.pc && inst.guard.is_none() {
                            self.exited = true;
                        }
                        next = t;
                    }
                }
            }
            _ => {}
        }
        self.pc = next;
        let active = self.ann.active.get(&pc).copied().unwrap_or(u32::MAX);
        let mut addrs = Vec::new();
        {
            if let Some(g) = self.ann.addr.get(&pc) {
                let lanes = active.count_ones().min(32) as u64;
                addrs = (0..lanes)
                    .map(|l| g.addr(self.cta, self.warp, l, self.iter))
                    .collect();
            }
        }
        let payload = self.ann.payload.get(&pc).copied().unwrap_or(0);
        Some(DynOp {
            inst,
            active,
            addrs,
            payload,
            mbar: None,
            run_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::parse_listing;

    #[test]
    fn loop_trip_counts() {
        let src = r#"
        /*0000*/                   MOV R0, 0x10 ;
        /*0010*/                   IADD3 R0, R0, -0x1, RZ ;
        /*0020*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0030*/               @P0 BRA 0x10 ;
        /*0040*/                   EXIT ;
        /*0050*/                   BRA 0x50 ;
"#;
        let p = Arc::new(parse_listing(src).unwrap().remove(0));
        let mut prof = BranchProfile::default();
        prof.taken.insert(0x30, 15);
        let insts = Arc::new(Walker::share(&p));
        let mut w = Walker::new(
            p,
            insts,
            Arc::new(prof),
            Arc::new(Annotations::default()),
            0,
            0,
        );
        let mut n = 0;
        while w.next_op().is_some() {
            n += 1;
        }
        // 1 MOV + 16 x (IADD3, ISETP, BRA) + EXIT
        assert_eq!(n, 1 + 16 * 3 + 1);
    }
}
