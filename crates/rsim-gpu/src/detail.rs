//! Opt-in capture for the visualiser (`rsim viz`).
//!
//! Three levels, each off by default and free when off:
//!
//! * **CTA residency**: which SM ran each CTA, and when (the per-SM timeline).
//! * **Per-PC statistics**: issue count and warp-state samples per SASS
//!   instruction, the simulator's equivalent of Nsight Compute's source view
//!   ("warp stall sampling"). A stalled warp's cycle is charged to the
//!   instruction it is waiting to issue, as Nsight does.
//! * **Warp timelines**: for a few chosen SMs, every warp's state as
//!   run-length segments plus the PC of every issued instruction.
//!
//! Captured data is attached to each finished kernel as a [`KernelDetail`],
//! with cycles relative to the kernel's start, so fast mode can reuse one
//! capture for every launch of the same kernel.

use crate::stats::{Stall, StallCounts};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DetailConfig {
    /// Record CTA residency (SM, start, end) for every kernel.
    pub ctas: bool,
    /// Record per-PC issue counts and warp-state samples.
    pub pc_stats: bool,
    /// SMs whose warps get a full timeline.
    pub warp_sms: Vec<u32>,
    /// Cap on timeline events (state segments + issues) per traced SM.
    pub max_warp_events: usize,
}

impl Default for DetailConfig {
    fn default() -> Self {
        Self {
            ctas: false,
            pc_stats: false,
            warp_sms: Vec::new(),
            max_warp_events: 400_000,
        }
    }
}

impl DetailConfig {
    /// Everything on, with warp timelines for `warp_sms`.
    pub fn full(warp_sms: Vec<u32>) -> Self {
        Self {
            ctas: true,
            pc_stats: true,
            warp_sms,
            ..Default::default()
        }
    }
    pub fn enabled(&self) -> bool {
        self.ctas || self.pc_stats || !self.warp_sms.is_empty()
    }
}

/// One SASS instruction's share of a kernel.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PcStat {
    /// Mnemonic with modifiers and operands, e.g. `UTCHMMA gdesc[UR4], ...`.
    pub text: String,
    pub class: String,
    /// Warp instructions issued from this PC.
    pub issued: u64,
    /// Warp-cycles spent waiting to issue this PC, by state.
    pub stalls: StallCounts,
}

/// One warp's life on a traced SM.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WarpRecord {
    /// Warp slot on the SM (sub-partition = slot % 4 on 4-SMSP parts).
    pub slot: u32,
    pub smsp: u32,
    pub cta: u64,
    /// Warp index within its CTA.
    pub widx: u32,
    /// Role index for recorded warp-specialised paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<usize>,
    pub start: u64,
    pub end: u64,
    /// Flat `[state, start, end, ...]` triples (state = index into ALL_STALLS).
    pub seg: Vec<u64>,
    /// Flat `[cycle, pc, ...]` pairs of issued instructions.
    pub iss: Vec<u64>,
    #[serde(skip)]
    kid: u32,
    /// End of the last issue cycle: the state interval that the scheduler
    /// reports next starts at the issue cycle, which is already drawn as `Selected`.
    #[serde(skip)]
    cut: u64,
}

/// Every warp of one kernel on one traced SM.
#[derive(Clone, Debug, Default, Serialize)]
pub struct WarpTimeline {
    pub sm: u32,
    /// The event cap was hit; the timeline stops early.
    pub truncated: bool,
    pub warps: Vec<WarpRecord>,
}

/// Visualiser capture for one finished kernel. Cycles are relative to the
/// kernel's `start_cycle`.
#[derive(Clone, Debug, Default, Serialize)]
pub struct KernelDetail {
    /// Warp-role names of static kernels (index = `WarpRecord::role` and the
    /// high half of per-PC keys); empty for recorded paths.
    pub roles: Vec<String>,
    /// Flat `[sm, cta, start, end, ...]` quadruples.
    pub ctas: Vec<u64>,
    pub warps: Vec<WarpTimeline>,
}

/// Per-SM warp-timeline recorder.
#[derive(Debug, Default)]
pub struct WarpTrace {
    live: Vec<Option<WarpRecord>>,
    done: Vec<WarpRecord>,
    events: usize,
    max_events: usize,
    pub truncated: bool,
}

impl WarpTrace {
    pub fn new(slots: usize, max_events: usize) -> Self {
        Self {
            live: (0..slots).map(|_| None).collect(),
            max_events,
            ..Default::default()
        }
    }

    fn room(&mut self) -> bool {
        if self.events >= self.max_events {
            self.truncated = true;
            return false;
        }
        self.events += 1;
        true
    }

    pub fn begin(
        &mut self,
        slot: usize,
        smsp: usize,
        kid: u32,
        cta: u64,
        widx: u32,
        role: Option<usize>,
        at: u64,
    ) {
        if let Some(w) = self.live[slot].take() {
            self.done.push(w);
        }
        self.live[slot] = Some(WarpRecord {
            slot: slot as u32,
            smsp: smsp as u32,
            cta,
            widx,
            role,
            start: at,
            end: at,
            kid,
            ..Default::default()
        });
    }

    /// The warp in `slot` was in state `st` over `[t0, t1)`.
    #[inline]
    pub fn state(&mut self, slot: usize, st: Stall, t0: u64, t1: u64) {
        let s = st as u64;
        let (extend, t0) = match &self.live[slot] {
            Some(w) => {
                let t0 = t0.max(w.cut);
                if t0 >= t1 {
                    return;
                }
                let n = w.seg.len();
                (n >= 3 && w.seg[n - 3] == s && w.seg[n - 1] == t0, t0)
            }
            None => return,
        };
        if extend {
            let w = self.live[slot].as_mut().unwrap();
            let n = w.seg.len();
            w.seg[n - 1] = t1;
            w.end = t1;
            return;
        }
        if !self.room() {
            return;
        }
        let w = self.live[slot].as_mut().unwrap();
        w.seg.extend_from_slice(&[s, t0, t1]);
        w.end = t1;
    }

    /// The warp in `slot` issued `pc` at cycle `c` (drawn as a one-cycle `Selected` segment).
    #[inline]
    pub fn issue(&mut self, slot: usize, c: u64, pc: u32) {
        if self.live[slot].is_none() || !self.room() {
            return;
        }
        let w = self.live[slot].as_mut().unwrap();
        w.iss.extend_from_slice(&[c, pc as u64]);
        w.end = w.end.max(c + 1);
        w.cut = c + 1;
        let s = Stall::Selected as u64;
        let n = w.seg.len();
        if n >= 3 && w.seg[n - 3] == s && w.seg[n - 1] == c {
            w.seg[n - 1] = c + 1;
        } else {
            w.seg.extend_from_slice(&[s, c, c + 1]);
        }
    }

    pub fn end(&mut self, slot: usize, c: u64) {
        if let Some(mut w) = self.live[slot].take() {
            w.end = w.end.max(c);
            self.done.push(w);
        }
    }

    /// Remove and return kernel `kid`'s warps, rebased to `base`.
    pub fn take(&mut self, kid: u32, base: u64, sm: u32) -> Option<WarpTimeline> {
        let mut out = Vec::new();
        let mut keep = Vec::new();
        for w in self.done.drain(..) {
            if w.kid == kid {
                out.push(w);
            } else {
                keep.push(w);
            }
        }
        self.done = keep;
        for l in self.live.iter_mut() {
            if l.as_ref().map(|w| w.kid == kid).unwrap_or(false) {
                out.push(l.take().unwrap());
            }
        }
        if out.is_empty() {
            return None;
        }
        out.sort_by_key(|w| (w.start, w.slot));
        for w in &mut out {
            let r = |x: u64| x.saturating_sub(base);
            w.start = r(w.start);
            w.end = r(w.end);
            for i in (0..w.seg.len()).step_by(3) {
                w.seg[i + 1] = r(w.seg[i + 1]);
                w.seg[i + 2] = r(w.seg[i + 2]);
            }
            for i in (0..w.iss.len()).step_by(2) {
                w.iss[i] = r(w.iss[i]);
            }
        }
        Some(WarpTimeline {
            sm,
            truncated: self.truncated,
            warps: out,
        })
    }
}
