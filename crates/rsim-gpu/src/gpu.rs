//! Whole-GPU model: SMs + memory system + CTA work distributor.
//!
//! Time inside a [`Gpu`] is in SM clock cycles. The rack layer converts to
//! global picoseconds with the GPU's [`rsim_core::Clock`].

use crate::detail::{DetailConfig, KernelDetail, WarpTrace};
use crate::kernel::KernelLaunch;
use crate::mem::MemSystem;
use crate::sm::{Policy, RemoteMbar, Sm, NEVER};
use crate::stats::KernelStats;
use rsim_core::trace::TraceSink;
use rsim_hw::GpuSpec;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SchedPolicy {
    /// Greedy-then-oldest (closest to NVIDIA's documented behaviour).
    Gto,
    /// Loose round-robin.
    Lrr,
    /// Greedy, then the youngest ready warp.
    Gty,
}

/// `calibration/src/bench_window.cu`'s kernel as compiled for sm_103, up
/// to the exit every warp takes (the bench passes a null sink, so the
/// `@P0 EXIT` always fires; the recorded paths confirm it): the measured
/// window includes running it, so the model's time for it is subtracted.
const WINDOW_KERNEL_SASS: &str = "\t\tFunction : window
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   S2UR UR4, SR_CTAID.X ;
        /*0030*/                   LDCU.64 UR6, c[0x0][0x380] ;
        /*0040*/                   ISETP.EQ.U32.AND P1, PT, RZ, UR6, PT ;
        /*0050*/                   LOP3.LUT P0, RZ, R0, UR4, RZ, 0xfc, !PT ;
        /*0060*/                   ISETP.EQ.U32.OR.EX P0, PT, RZ, UR7, P0, P1 ;
        /*0070*/                   EXIT ;
";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    pub policy: SchedPolicy,
    /// Honour compiler control bits (stall counts / SB barriers) when the
    /// SASS carries them. Otherwise use per-register scoreboarding.
    pub use_control_bits: bool,
    /// Emit GPU-level counter samples every N cycles (0 = off).
    pub sample_interval: u64,
    /// Emit a per-SM timeline: one track per SM with a span per CTA
    /// (kernel name, CTA id). Large traces for big systems.
    pub per_sm_counters: bool,
    /// Charge the measured kernel window (`GpuSpec::kernel_window`) before
    /// each kernel's first CTA. Right for a kernel timed alone, as ncu does.
    /// Off for streams: there the windows of back-to-back kernels overlap,
    /// and the measured back-to-back launch gap (`launch_overhead_ns`)
    /// already covers them.
    pub kernel_window: bool,
    /// Visualiser capture (`rsim viz`); all off by default.
    pub detail: DetailConfig,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            policy: SchedPolicy::Gto,
            use_control_bits: true,
            sample_interval: 2000,
            per_sm_counters: false,
            kernel_window: false,
            detail: DetailConfig::default(),
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
    /// Visualiser capture, when `GpuConfig::detail` asks for it.
    #[serde(skip)]
    pub detail: Option<Arc<KernelDetail>>,
}

/// Perfetto thread id of SM 0's track (streams use small tids).
const SM_TID0: u64 = 1000;

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
    /// Cached `sms[i].next_event()`.
    sm_next: Vec<u64>,
    /// Min-heap of (next event, sm) with lazy invalidation.
    heap: BinaryHeap<Reverse<(u64, u32)>>,
    stepping: Vec<u32>,
    /// Stepping round each SM was last queued in (dedups heap entries).
    step_mark: Vec<u64>,
    round: u64,
    tmemo: crate::sm::TimingMemo,
    /// Per-SM timeline: CTA start cycle keyed by (sm, kernel id, cta id).
    cta_start: HashMap<(u32, u32, u64), u64>,
    /// Where each resident CTA of a clustered kernel lives: (kid, cta) -> SM.
    cta_sm: HashMap<(u32, u64), usize>,
    /// Cluster mbarrier events for CTAs that are not resident yet.
    pending_mbar: HashMap<(u32, u64), Vec<RemoteMbar>>,
    /// `empty_kernel_cycles` cache: (ctas, threads, smem, cluster) -> cycles.
    window_own: HashMap<(u32, u32, u32, u32), u64>,
    /// Finished CTAs for the visualiser: (kid, sm, cta, start, end).
    cta_log: Vec<(u32, u32, u64, u64, u64)>,
}

impl Gpu {
    pub fn new(rank: usize, spec: GpuSpec, cfg: GpuConfig, trace_enabled: bool) -> Self {
        let spec = Arc::new(spec);
        let mut policy = match cfg.policy {
            SchedPolicy::Gto => Policy::Gto,
            SchedPolicy::Lrr => Policy::Lrr,
            SchedPolicy::Gty => Policy::Gty,
        };
        // A measured scheduler for this GPU (calibration `issue.scheduler`)
        // replaces the default greedy-then-oldest; an explicit LRR/GTY
        // request still wins.
        if cfg.policy == SchedPolicy::Gto {
            match spec.issue.scheduler.as_str() {
                "lrr" => policy = Policy::Lrr,
                "gty" => policy = Policy::Gty,
                _ => {}
            }
        }
        let sms = (0..spec.sm_count)
            .map(|i| {
                let mut sm = Sm::new(i, spec.clone(), policy, cfg.use_control_bits);
                sm.pc_stats = cfg.detail.pc_stats;
                if cfg.detail.warp_sms.contains(&i) {
                    sm.wtrace = Some(Box::new(WarpTrace::new(
                        spec.max_warps_per_sm as usize,
                        cfg.detail.max_warp_events,
                    )));
                }
                sm
            })
            .collect();
        let mem = MemSystem::new(&spec);
        let mut trace = TraceSink::new(trace_enabled);
        let pid = rank as u64 + 1;
        trace.name_process(pid, &format!("GPU{rank} ({})", spec.name));
        if cfg.per_sm_counters {
            for s in 0..spec.sm_count as u64 {
                trace.name_thread(pid, SM_TID0 + s, &format!("SM{s:03}"));
            }
        }
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
            sm_next: vec![NEVER; spec.sm_count as usize],
            heap: BinaryHeap::new(),
            cta_start: HashMap::new(),
            cta_sm: HashMap::new(),
            pending_mbar: HashMap::new(),
            window_own: HashMap::new(),
            cta_log: Vec::new(),
            stepping: Vec::new(),
            step_mark: vec![0; spec.sm_count as usize],
            round: 0,
            tmemo: Default::default(),
            spec,
        }
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    /// Queue a kernel whose CTAs become dispatchable at `at` (cycles).
    /// Simulated cycles of the empty kernel a window point was measured
    /// with (`WINDOW_KERNEL_SASS`, same shape), cached per point.
    fn empty_kernel_cycles(&mut self, p: &rsim_hw::WindowPoint) -> u64 {
        let key = (p.ctas, p.threads, p.smem_bytes, p.cluster);
        if let Some(&c) = self.window_own.get(&key) {
            return c;
        }
        let prog = rsim_sass::parse_listing(WINDOW_KERNEL_SASS)
            .expect("static listing")
            .remove(0);
        let ctas = if p.ctas == 0 {
            self.spec.sm_count
        } else {
            p.ctas
        };
        let mut k = KernelLaunch::from_program(
            "window",
            prog,
            [ctas, 1, 1],
            [p.threads, 1, 1],
            32,
            p.smem_bytes,
            Default::default(),
        );
        k.cluster = p.cluster.max(1);
        let mut g = Gpu::new(0, (*self.spec).clone(), GpuConfig::default(), false);
        g.submit(Arc::new(k), 0, 0);
        let mut out = vec![];
        g.advance(u64::MAX / 2, &mut out);
        let c = out.pop().map_or(0, |d| d.end_cycle - d.start_cycle);
        self.window_own.insert(key, c);
        c
    }

    pub fn submit(&mut self, launch: Arc<KernelLaunch>, at: u64, tag: u64) -> u32 {
        assert!(
            crate::sm::Sm::occupancy(&self.spec, &launch) > 0,
            "kernel `{}` cannot fit one CTA on an SM ({} B smem, {} threads, {} regs/thread)",
            launch.name,
            launch.smem_per_cta,
            launch.threads_per_cta(),
            launch.regs_per_thread
        );
        let kid = self.next_kid;
        self.next_kid += 1;
        let total = launch.num_ctas();
        // The measured fixed window (rasterisation, smem carveout, cluster
        // launch) opens with the launch and delays the first CTA. The
        // measurement is of an empty kernel, so what the model itself takes
        // for that same empty kernel is subtracted: only the residual the
        // model does not have is charged.
        let at = at.max(self.now);
        let window = if self.cfg.kernel_window {
            self.spec
                .kernel_window_point(
                    total,
                    launch.threads_per_cta(),
                    launch.smem_per_cta,
                    launch.cluster,
                )
                .cloned()
                .map(|p| {
                    let own = self.empty_kernel_cycles(&p);
                    (p.cycles - own as f64).max(0.0) as u64
                })
                .unwrap_or(0)
        } else {
            0
        };
        self.kernels.push(ActiveKernel {
            kid,
            tag,
            launch,
            available_at: at + window,
            next_cta: 0,
            total,
            done: 0,
            started: (window > 0).then_some(at),
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

    #[inline]
    fn update_sm(&mut self, s: usize) {
        let n = self.sms[s].next_event();
        if n != self.sm_next[s] || n != NEVER {
            self.sm_next[s] = n;
            if n != NEVER {
                self.heap.push(Reverse((n, s as u32)));
            }
        }
    }

    /// Next cycle at which something happens (or `None` if quiescent).
    pub fn next_event(&self) -> Option<u64> {
        // Heap entries may be stale (older, earlier times): that only makes
        // this estimate conservative.
        let sm = self.heap.peek().map(|r| r.0 .0).unwrap_or(NEVER);
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
                        if launch.cluster > 1 {
                            self.cta_sm.insert((kid, cta), s);
                            for ev in self.pending_mbar.remove(&(kid, cta)).unwrap_or_default() {
                                self.sms[s].deliver(&ev);
                            }
                        }
                        if self.cta_timeline() {
                            self.cta_start.insert((s as u32, kid, cta), t + lat);
                        }
                        self.update_sm(s);
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
        self.advance_inner(until, out, false)
    }

    /// Like [`Gpu::advance`] but returns at the first cycle in which any
    /// kernel completes (so dependent launches are never late).
    pub fn advance_to_completion(&mut self, until: u64, out: &mut Vec<KernelDone>) {
        self.advance_inner(until, out, true)
    }

    fn advance_inner(&mut self, until: u64, out: &mut Vec<KernelDone>, stop_on_done: bool) {
        let mut cta_done: Vec<(u32, u64)> = Vec::new();
        loop {
            let Some(t) = self.next_event() else {
                if !self.kernels.is_empty() && std::env::var_os("RSIM_DEBUG_DEADLOCK").is_some() {
                    for sm in self.sms.iter_mut().take(4) {
                        eprint!("{}", sm.debug_dump());
                    }
                }
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
            // Pop every SM due at `t` (skipping stale heap entries), step
            // them in id order for determinism.
            self.stepping.clear();
            self.round += 1;
            while let Some(&Reverse((tt, s))) = self.heap.peek() {
                if tt > t {
                    break;
                }
                self.heap.pop();
                // A stale or duplicate entry for an SM already taken this
                // round is dropped (`step_mark` dedups in O(1)).
                if self.sm_next[s as usize] == tt && self.step_mark[s as usize] != self.round {
                    self.step_mark[s as usize] = self.round;
                    self.stepping.push(s);
                }
            }
            self.stepping.sort_unstable();
            for i in 0..self.stepping.len() {
                let s = self.stepping[i] as usize;
                let before = cta_done.len();
                std::mem::swap(&mut self.sms[s].tmemo, &mut self.tmemo);
                self.sms[s].step(t, &self.spec, &mut self.mem, &mut cta_done);
                std::mem::swap(&mut self.sms[s].tmemo, &mut self.tmemo);
                self.update_sm(s);
                if !self.sms[s].outbox.is_empty() {
                    let evs = std::mem::take(&mut self.sms[s].outbox);
                    for ev in evs {
                        match self.cta_sm.get(&(ev.kid, ev.cta)).copied() {
                            Some(d) if self.sms[d].deliver(&ev) => self.update_sm(d),
                            _ => self
                                .pending_mbar
                                .entry((ev.kid, ev.cta))
                                .or_default()
                                .push(ev),
                        }
                    }
                }
                if !self.mem.flags.wake.is_empty() {
                    for (d, at) in std::mem::take(&mut self.mem.flags.wake) {
                        self.sms[d as usize].wake(at.max(t));
                        self.update_sm(d as usize);
                    }
                }
                if self.cta_timeline() {
                    for i in before..cta_done.len() {
                        let (kid, cta) = cta_done[i];
                        self.trace_cta(s, kid, cta, t);
                    }
                }
            }
            if !cta_done.is_empty() {
                self.dispatch_dirty = true;
                for (kid, cta) in cta_done.drain(..) {
                    self.cta_sm.remove(&(kid, cta));
                    if let Some(k) = self.kernels.iter_mut().find(|k| k.kid == kid) {
                        k.done += 1;
                    }
                }
                let mut i = 0;
                while i < self.kernels.len() {
                    if self.kernels[i].done == self.kernels[i].total {
                        let k = self.kernels.remove(i);
                        self.mem.flags.retire(k.kid);
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
            if stop_on_done && !out.is_empty() {
                // Let the caller react (launch dependent work) at exactly `t`.
                self.now = t;
                break;
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
                st.probes.extend_from_slice(&a.probes);
                st.warps += a.warps;
                st.insts_issued += a.insts_issued;
                for (i, n) in a.class_counts.iter().enumerate() {
                    if *n > 0 {
                        *st.class_mix
                            .entry(format!("{:?}", rsim_sass::OpClass::ALL[i]))
                            .or_default() += n;
                    }
                }
                for (c, n) in a.unknown_opcodes {
                    *st.unknown_opcodes.entry(c).or_default() += n;
                }
                st.sched.merge(&a.sched);
                st.warp.merge(&a.warp);
                if st.warp_by_index.len() < a.warp_by_index.len() {
                    st.warp_by_index
                        .resize(a.warp_by_index.len(), Default::default());
                    st.issued_by_index.resize(a.warp_by_index.len(), 0);
                }
                for (i, c) in a.warp_by_index.iter().enumerate() {
                    st.warp_by_index[i].merge(c);
                }
                if st.issued_by_index.len() < a.issued_by_index.len() {
                    st.issued_by_index.resize(a.issued_by_index.len(), 0);
                }
                for (i, n) in a.issued_by_index.iter().enumerate() {
                    st.issued_by_index[i] += n;
                }
                for ((w, b), n) in a.mbar_wait_raw {
                    *st.mbar_wait.entry(format!("w{w}:{b:#x}")).or_default() += n;
                }
                for (pc, p) in a.pc {
                    let e = st.pc.entry(pc).or_insert_with(|| crate::detail::PcStat {
                        text: p.text.clone(),
                        class: p.class.clone(),
                        ..Default::default()
                    });
                    e.issued += p.issued;
                    e.stalls.merge(&p.stalls);
                }
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
        let detail = self.cfg.detail.enabled().then(|| {
            let mut d = KernelDetail::default();
            if let crate::kernel::KernelSource::Static { roles, .. } = &k.launch.source {
                d.roles = roles.iter().map(|r| r.name.clone()).collect();
            }
            let kid = k.kid;
            self.cta_log.retain(|&(id, sm, cta, s0, s1)| {
                if id != kid {
                    return true;
                }
                d.ctas.extend_from_slice(&[
                    sm as u64,
                    cta,
                    s0.saturating_sub(start),
                    s1.saturating_sub(start),
                ]);
                false
            });
            for p in st.probes.as_chunks::<5>().0 {
                d.probes
                    .extend_from_slice(&[p[0], p[1], p[2], p[3], p[4].saturating_sub(start)]);
            }
            for sm in &mut self.sms {
                let id = sm.id;
                if let Some(t) = sm.wtrace.as_mut().and_then(|t| t.take(kid, start, id)) {
                    d.warps.push(t);
                }
            }
            Arc::new(d)
        });
        KernelDone {
            tag: k.tag,
            start_cycle: start,
            end_cycle: end,
            stats: st,
            detail,
        }
    }

    /// Whether CTA start/end times are tracked (Perfetto per-SM tracks or `rsim viz`).
    fn cta_timeline(&self) -> bool {
        (self.cfg.per_sm_counters && self.trace.enabled) || self.cfg.detail.ctas
    }

    fn trace_cta(&mut self, s: usize, kid: u32, cta: u64, end: u64) {
        let Some(start) = self.cta_start.remove(&(s as u32, kid, cta)) else {
            return;
        };
        if self.cfg.detail.ctas {
            self.cta_log.push((kid, s as u32, cta, start, end));
        }
        if !(self.cfg.per_sm_counters && self.trace.enabled) {
            return;
        }
        let name = self
            .kernels
            .iter()
            .find(|k| k.kid == kid)
            .map(|k| k.launch.name.clone())
            .unwrap_or_default();
        let ghz = self.spec.sm_clock_mhz / 1000.0;
        let ps = |c: u64| (c as f64 / ghz * 1000.0) as u64;
        self.trace.span(
            name,
            "cta",
            self.pid,
            SM_TID0 + s as u64,
            ps(start),
            ps(end),
            serde_json::json!({ "cta": cta, "cycles": end - start }),
        );
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

/// Power-capped clock (DVFS): the SM clock a kernel sustains under the
/// board power cap, or `None` when it runs at `sm_clock_mhz` (no power
/// model, no measured run for its dtype, or not enough tensor work to
/// draw that much power). `d` is the kernel simulated at the nominal clock.
///
/// The power model gives the clock of a fully tensor-bound kernel of the
/// same size; a kernel that keeps the tensor cores busy a fraction `u` of
/// the time draws proportionally less, so its clock sits that fraction of
/// the way down (kernels at >= 70% tensor utilisation, where the measured
/// GEMMs sit, take the full drop).
pub fn dvfs_clock_mhz(spec: &GpuSpec, launch: &KernelLaunch, d: &KernelDone) -> Option<f64> {
    let p = spec.power.as_ref()?;
    let name = launch.name.to_ascii_lowercase();
    let dtype = ["fp4", "fp8", "bf16", "fp16", "tf32"]
        .into_iter()
        .find(|t| name.contains(t))?;
    let flops = if launch.flops > 0.0 {
        launch.flops
    } else {
        d.stats.tensor_flops
    };
    let mhz = p.sustained_mhz(dtype, flops)?;
    let cycles = (d.end_cycle - d.start_cycle).max(1) as f64;
    let slots = cycles * spec.sm_count as f64 * spec.sub_partitions as f64;
    let w = (d.stats.tensor_busy_cycles as f64 / slots / 0.7).min(1.0);
    let f = spec.sm_clock_mhz - w * (spec.sm_clock_mhz - mhz).max(0.0);
    (f < 0.995 * spec.sm_clock_mhz).then_some(f)
}

/// Board power while `launch` runs (W), from the power model; `None` when
/// the GPU has none. `d` is the kernel simulated at the nominal clock.
///
/// The board draws idle power, plus the tensor term of the power model
/// scaled by tensor-core utilisation (as [`dvfs_clock_mhz`] does) at the
/// clock the kernel ran at, plus HBM traffic at `hbm_pj_per_bit`, capped at
/// the board limit. A fully tensor-bound kernel the cap slows therefore
/// draws the cap (that is how its clock was found); one that keeps the
/// tensor cores busy part of the time sits part of the way down in clock
/// and draws less.
pub fn kernel_power_w(spec: &GpuSpec, launch: &KernelLaunch, d: &KernelDone) -> Option<f64> {
    let p = spec.power.as_ref()?;
    let cycles = (d.end_cycle - d.start_cycle).max(1) as f64;
    let mut w = p.idle_w;
    let name = launch.name.to_ascii_lowercase();
    let flops = if launch.flops > 0.0 {
        launch.flops
    } else {
        d.stats.tensor_flops
    };
    if let Some(dyn_w) = ["fp4", "fp8", "bf16", "fp16", "tf32"]
        .into_iter()
        .find(|t| name.contains(t))
        .and_then(|dt| p.dynamic_w(dt, flops))
    {
        let slots = cycles * spec.sm_count as f64 * spec.sub_partitions as f64;
        let u = (d.stats.tensor_busy_cycles as f64 / slots / 0.7).min(1.0);
        let f = dvfs_clock_mhz(spec, launch, d)
            .unwrap_or(spec.sm_clock_mhz)
            .min(spec.sm_clock_mhz / tcgen05_stretch(spec, d));
        w += u * dyn_w * (f / p.max_mhz).powi(3);
    }
    let secs = cycles / (spec.sm_clock_mhz * 1e6);
    let bytes = (d.stats.dram_read_bytes + d.stats.dram_write_bytes) as f64;
    w += bytes * 8.0 * p.hbm_pj_per_bit * 1e-12 / secs;
    Some(w.min(p.cap_w))
}

/// Cheap test before any extra simulation: could the power cap lower this
/// kernel's clock at all (a measured run of its dtype sustains less than
/// `sm_clock_mhz` at its size)?
pub fn dvfs_possible(spec: &GpuSpec, launch: &KernelLaunch) -> bool {
    let Some(p) = &spec.power else {
        return false;
    };
    let name = launch.name.to_ascii_lowercase();
    ["fp4", "fp8", "bf16", "fp16", "tf32"]
        .into_iter()
        .find(|t| name.contains(t))
        .and_then(|dt| p.sustained_mhz(dt, launch.flops))
        .is_some_and(|f| f < 0.995 * spec.sm_clock_mhz)
}

/// Run one kernel alone on a cold GPU with `reserved` SMs held back.
pub fn run_alone(
    spec: &GpuSpec,
    cfg: &GpuConfig,
    k: &Arc<KernelLaunch>,
    reserved: u32,
) -> KernelDone {
    let mut g = Gpu::new(0, spec.clone(), cfg.clone(), false);
    g.reserve_sms(reserved);
    g.submit(k.clone(), 0, 0);
    let mut out = Vec::new();
    g.advance(u64::MAX / 2, &mut out);
    out.pop().expect("kernel did not complete")
}

/// Wall-time factor of the power-capped clock for a kernel. `nominal` is
/// the kernel run alone at `sm_clock_mhz` with `reserved` SMs held back.
///
/// The kernel is simulated again at its sustained clock rather than its
/// time scaled by the clock ratio: HBM and L2 work in nanoseconds, so a
/// slower SM clock stretches only the part of the kernel the SMs bound.
/// Scaling all of it over-corrected large B300 BF16 GEMMs by 20-45%.
pub fn dvfs_stretch(
    spec: &GpuSpec,
    cfg: &GpuConfig,
    launch: &Arc<KernelLaunch>,
    nominal: &KernelDone,
    reserved: u32,
) -> f64 {
    let tc = tcgen05_stretch(spec, nominal);
    let Some(f) = dvfs_clock_mhz(spec, launch, nominal) else {
        return tc;
    };
    if spec.sm_clock_mhz / f <= tc {
        return tc;
    }
    let mut slow = spec.clone();
    slow.sm_clock_mhz = f;
    let cfg = GpuConfig {
        detail: DetailConfig::default(),
        ..cfg.clone()
    };
    let d = run_alone(&slow, &cfg, launch, reserved);
    let ns = |d: &KernelDone, mhz: f64| (d.end_cycle - d.start_cycle).max(1) as f64 / mhz;
    (ns(&d, f) / ns(nominal, spec.sm_clock_mhz)).max(tc)
}

/// Wall-time factor of the fixed lower SM clock that tcgen05 MMAs pin
/// (B300: 1905 MHz, `power.tcgen05_sm_mhz`) for a kernel that issued any;
/// 1.0 otherwise. Tensor-bound kernels at a fixed clock scale almost
/// entirely, so this is a plain ratio rather than a re-simulation.
pub fn tcgen05_stretch(spec: &GpuSpec, d: &KernelDone) -> f64 {
    match &spec.power {
        Some(p)
            if p.tcgen05_mhz > 0.0
                && d.stats.class_counts[rsim_sass::OpClass::MmaTcgen05 as usize] > 0 =>
        {
            (spec.sm_clock_mhz / p.tcgen05_mhz).max(1.0)
        }
        _ => 1.0,
    }
}
