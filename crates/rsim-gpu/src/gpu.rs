//! Whole-GPU model: SMs + memory system + CTA work distributor.
//!
//! Time inside a [`Gpu`] is in SM clock cycles. The rack layer converts to
//! global picoseconds with the GPU's [`rsim_core::Clock`].

use crate::kernel::KernelLaunch;
use crate::mem::MemSystem;
use crate::sm::{Policy, Sm, NEVER};
use crate::stats::KernelStats;
use rsim_core::trace::TraceSink;
use rsim_hw::GpuSpec;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SchedPolicy {
    /// Greedy-then-oldest (closest to NVIDIA's documented behaviour).
    Gto,
    /// Loose round-robin.
    Lrr,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuConfig {
    pub policy: SchedPolicy,
    /// Honour compiler control bits (stall counts / SB barriers) when the
    /// SASS carries them. Otherwise use per-register scoreboarding.
    pub use_control_bits: bool,
    /// Emit GPU-level counter samples every N cycles (0 = off).
    pub sample_interval: u64,
    /// Emit per-SM counter tracks as well (large traces for big systems).
    pub per_sm_counters: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            policy: SchedPolicy::Gto,
            use_control_bits: true,
            sample_interval: 2000,
            per_sm_counters: false,
        }
    }
}

struct ActiveKernel {
    kid: u32,
    tag: u64,
    launch: Arc<KernelLaunch>,
    available_at: u64,
    next_cta: u64,
    total: u64,
    done: u64,
    started: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct KernelDone {
    pub tag: u64,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub stats: KernelStats,
}

pub struct Gpu {
    pub rank: usize,
    pub spec: Arc<GpuSpec>,
    pub cfg: GpuConfig,
    pub sms: Vec<Sm>,
    pub mem: MemSystem,
    kernels: Vec<ActiveKernel>,
    next_kid: u32,
    now: u64,
    dispatch_free: u64,
    dispatch_dirty: bool,
    rr: usize,
    pub trace: TraceSink,
    pid: u64,
    next_sample: u64,
    last_issued: u64,
    last_tensor: u64,
    last_dram: u64,
    pub reserved_sms: u32,
    pub busy_cycles: u64,
}

impl Gpu {
    pub fn new(rank: usize, spec: GpuSpec, cfg: GpuConfig, trace_enabled: bool) -> Self {
        let spec = Arc::new(spec);
        let policy = match cfg.policy {
            SchedPolicy::Gto => Policy::Gto,
            SchedPolicy::Lrr => Policy::Lrr,
        };
        let sms = (0..spec.sm_count)
            .map(|i| Sm::new(i, spec.clone(), policy, cfg.use_control_bits))
            .collect();
        let mem = MemSystem::new(&spec);
        let mut trace = TraceSink::new(trace_enabled);
        let pid = rank as u64 + 1;
        trace.name_process(pid, &format!("GPU{rank} ({})", spec.name));
        Self {
            rank,
            next_sample: cfg.sample_interval,
            cfg,
            sms,
            mem,
            kernels: Vec::new(),
            next_kid: 0,
            now: 0,
            dispatch_free: 0,
            dispatch_dirty: false,
            rr: 0,
            trace,
            pid,
            last_issued: 0,
            last_tensor: 0,
            last_dram: 0,
            reserved_sms: 0,
            busy_cycles: 0,
            spec,
        }
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    /// Queue a kernel whose CTAs become dispatchable at `at` (cycles).
    pub fn submit(&mut self, launch: Arc<KernelLaunch>, at: u64, tag: u64) -> u32 {
        let kid = self.next_kid;
        self.next_kid += 1;
        let total = launch.num_ctas();
        self.kernels.push(ActiveKernel {
            kid,
            tag,
            launch,
            available_at: at.max(self.now),
            next_cta: 0,
            total,
            done: 0,
            started: None,
        });
        self.dispatch_dirty = true;
        kid
    }

    /// Reserve `n` SMs (highest ids first) for communication kernels: no
    /// new CTAs are placed there until released. Running CTAs finish.
    pub fn reserve_sms(&mut self, n: u32) {
        let n = n.min(self.spec.sm_count - self.reserved_sms);
        let total = self.sms.len();
        for i in 0..n as usize {
            self.sms[total - 1 - self.reserved_sms as usize - i].reserved = true;
        }
        self.reserved_sms += n;
    }

    pub fn release_sms(&mut self, n: u32) {
        let n = n.min(self.reserved_sms);
        let total = self.sms.len();
        for i in 0..n as usize {
            let idx = total - self.reserved_sms as usize + i;
            self.sms[idx].reserved = false;
        }
        self.reserved_sms -= n;
        self.dispatch_dirty = true;
    }

    pub fn idle(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Next cycle at which something happens (or `None` if quiescent).
    pub fn next_event(&self) -> Option<u64> {
        let sm = self
            .sms
            .iter()
            .map(|s| s.next_event())
            .min()
            .unwrap_or(NEVER);
        let k = self
            .kernels
            .iter()
            .filter(|k| k.next_cta < k.total)
            .map(|k| k.available_at.max(self.dispatch_free))
            .min()
            .unwrap_or(NEVER);
        let t = sm.min(if self.dispatch_dirty || k > self.now {
            k
        } else {
            NEVER
        });
        (t != NEVER).then_some(t.max(self.now))
    }

    fn dispatch(&mut self, c: u64) {
        self.dispatch_dirty = false;
        let nsm = self.sms.len();
        let lat = self.spec.cta_launch_cycles as u64;
        let mut t = self.dispatch_free.max(c);
        for ki in 0..self.kernels.len() {
            let k = &self.kernels[ki];
            if k.available_at > c || k.next_cta >= k.total {
                continue;
            }
            let launch = k.launch.clone();
            let kid = k.kid;
            // Breadth-first: walk SMs round-robin, one CTA per SM per pass.
            loop {
                let k = &self.kernels[ki];
                if k.next_cta >= k.total {
                    break;
                }
                let mut placed = false;
                for i in 0..nsm {
                    let s = (self.rr + i) % nsm;
                    if self.sms[s].can_fit(&launch) {
                        let cta = self.kernels[ki].next_cta;
                        self.sms[s].launch_cta(kid, &launch, cta, t + lat);
                        self.kernels[ki].next_cta += 1;
                        if self.kernels[ki].started.is_none() {
                            self.kernels[ki].started = Some(t);
                        }
                        self.rr = (s + 1) % nsm;
                        t += 1; // one CTA per cycle from the work distributor (est)
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    break;
                }
            }
            // Later kernels may only fill leftover space (concurrent kernels).
        }
        self.dispatch_free = t;
    }

    /// Run until cycle `until` (exclusive). Completed kernels are returned.
    pub fn advance(&mut self, until: u64, out: &mut Vec<KernelDone>) {
        let mut cta_done: Vec<(u32, u64)> = Vec::new();
        loop {
            let Some(t) = self.next_event() else {
                self.now = self.now.max(until);
                break;
            };
            if t >= until {
                self.now = until;
                break;
            }
            if !self.kernels.is_empty() {
                self.busy_cycles += t - self.now;
            }
            self.now = t;
            self.sample(t);
            if self.dispatch_dirty
                || self
                    .kernels
                    .iter()
                    .any(|k| k.next_cta < k.total && k.available_at <= t && self.dispatch_free <= t)
            {
                self.dispatch(t);
            }
            for s in 0..self.sms.len() {
                if self.sms[s].next_event() <= t {
                    self.sms[s].step(t, &mut self.mem, &mut cta_done);
                }
            }
            if !cta_done.is_empty() {
                self.dispatch_dirty = true;
                for (kid, _) in cta_done.drain(..) {
                    if let Some(k) = self.kernels.iter_mut().find(|k| k.kid == kid) {
                        k.done += 1;
                    }
                }
                let mut i = 0;
                while i < self.kernels.len() {
                    if self.kernels[i].done == self.kernels[i].total {
                        let k = self.kernels.remove(i);
                        out.push(self.finish_kernel(k, t));
                    } else {
                        i += 1;
                    }
                }
            }
            // Zero-CTA kernels complete immediately.
            let mut i = 0;
            while i < self.kernels.len() {
                if self.kernels[i].total == 0 && self.kernels[i].available_at <= t {
                    let k = self.kernels.remove(i);
                    out.push(self.finish_kernel(k, t));
                } else {
                    i += 1;
                }
            }
        }
    }

    fn finish_kernel(&mut self, k: ActiveKernel, end: u64) -> KernelDone {
        let mut st = KernelStats {
            name: k.launch.name.clone(),
            tag: k.tag,
            ..Default::default()
        };
        for sm in &mut self.sms {
            if let Some(a) = sm.acc.remove(&k.kid) {
                st.ctas += a.ctas;
                st.warps += a.warps;
                st.insts_issued += a.insts_issued;
                for (c, n) in a.class_mix {
                    *st.class_mix.entry(c).or_default() += n;
                }
                for (c, n) in a.unknown_opcodes {
                    *st.unknown_opcodes.entry(c).or_default() += n;
                }
                st.sched.merge(&a.sched);
                st.warp.merge(&a.warp);
                st.tensor_busy_cycles += a.tensor_busy_cycles;
                st.tensor_flops += a.tensor_flops;
                st.dram_read_bytes += a.dram_read_bytes;
                st.dram_write_bytes += a.dram_write_bytes;
                st.l2_hits += a.l2_hits;
                st.l2_misses += a.l2_misses;
                st.l1_hits += a.l1_hits;
                st.l1_misses += a.l1_misses;
                st.smem_wavefronts += a.smem_wavefronts;
            }
        }
        let start = k.started.unwrap_or(k.available_at);
        st.start_cycle = start;
        st.end_cycle = end;
        st.ctas_per_sm = Sm::occupancy(&self.spec, &k.launch);
        KernelDone {
            tag: k.tag,
            start_cycle: start,
            end_cycle: end,
            stats: st,
        }
    }

    fn sample(&mut self, t: u64) {
        let iv = self.cfg.sample_interval;
        if iv == 0 || !self.trace.enabled {
            return;
        }
        while t >= self.next_sample {
            let issued: u64 = self.sms.iter().map(|s| s.issued_total).sum();
            let tensor: u64 = self.sms.iter().map(|s| s.tensor_busy_total).sum();
            let dram = self.mem.dram_read_bytes + self.mem.dram_write_bytes;
            let slots = (iv * self.spec.sm_count as u64 * self.spec.sub_partitions as u64) as f64;
            let ghz = self.spec.sm_clock_mhz / 1000.0;
            let ts = (self.next_sample as f64 / ghz * 1000.0) as u64; // ps
            let active = self.sms.iter().filter(|s| !s.is_empty()).count();
            self.trace.counter(
                "issue_util",
                self.pid,
                ts,
                serde_json::json!({ "issue": (issued - self.last_issued) as f64 / slots }),
            );
            self.trace.counter(
                "tensor_util",
                self.pid,
                ts,
                serde_json::json!({ "tensor": ((tensor - self.last_tensor) as f64 / slots).min(1.0) }),
            );
            self.trace.counter(
                "dram_GBps",
                self.pid,
                ts,
                serde_json::json!({ "GB/s": (dram - self.last_dram) as f64 / (iv as f64 / ghz) }),
            );
            self.trace.counter(
                "active_sms",
                self.pid,
                ts,
                serde_json::json!({ "sms": active }),
            );
            self.last_issued = issued;
            self.last_tensor = tensor;
            self.last_dram = dram;
            self.next_sample += iv;
        }
    }
}
