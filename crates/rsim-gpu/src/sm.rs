//! One streaming multiprocessor: 4 sub-partitions (SMSPs), warp slots,
//! CTA slots, MIO/LSU, L1, shared memory banks, tensor engine, TMA unit and
//! barrier state.

use crate::detail::{PcStat, WarpTrace};
use crate::kernel::{BarOp, InstStream, KernelLaunch, KernelSource};
use crate::mem::{Cache, MemSystem, Server};
use crate::pipes::{mma_dtype, mma_flops, mma_shape, timing, umma_shape, Pipe, Sb, NUM_PIPES};
use crate::stats::{AccMap, Stall, StallCounts};
use rsim_hw::GpuSpec;
use rsim_sass::inst::flags;
use rsim_sass::walker::DynOp;
use rsim_sass::{Inst, MbarRef, MemSpace, OpClass, Reg, NO_REG};
use std::collections::HashMap;
use std::sync::Arc;

pub const NEVER: u64 = u64::MAX;
/// Backlog (cycles) the MIO queue tolerates before throttling issue.
const MIO_QUEUE_CYCLES: u64 = 16;
/// Backlog (cycles) a per-SMSP wgmma queue tolerates.
const WGMMA_QUEUE_CYCLES: u64 = 2048;
/// Backlog (cycles) the SM-wide tcgen05 queue tolerates.
const TCGEN05_QUEUE_CYCLES: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    Gto,
    Lrr,
    /// Greedy, then the youngest (most recently launched) ready warp.
    Gty,
}

struct MBar {
    count: u32,
    arrivals: u32,
    /// Expected minus completed transaction bytes of the current phase.
    tx: i64,
    max_t: u64,
    /// Completion time of every produced phase, in order.
    phases: Vec<(u64, Stall)>,
}

/// An mbarrier event for a CTA of the same cluster on another SM (a 2-CTA
/// TMA completing on the leader's barrier, a multicast `tcgen05.commit`,
/// a remote `mbarrier.arrive`). Routed by the GPU.
#[derive(Clone, Copy, Debug)]
pub struct RemoteMbar {
    pub kid: u32,
    pub cta: u64,
    pub id: u32,
    pub t: u64,
    pub arrivals: u32,
    pub tx: i64,
    pub why: Stall,
}

/// Direct-mapped memo of [`timing`] per static instruction. An entry keeps
/// its `Arc<Inst>`, so a key can never alias a freed instruction. One per
/// GPU (shared by its SMs, which run the same kernels), lent to the SM
/// being stepped.
#[derive(Default)]
pub(crate) struct TimingMemo(Vec<Option<(Arc<Inst>, crate::pipes::Timing)>>);

impl TimingMemo {
    const N: usize = 4096;
    #[inline]
    fn get(&mut self, spec: &GpuSpec, inst: &Arc<Inst>) -> crate::pipes::Timing {
        if self.0.is_empty() {
            self.0.resize(Self::N, None);
        }
        let k = Arc::as_ptr(inst) as usize;
        let i = ((k >> 4) ^ (k >> 13)) & (Self::N - 1);
        if let Some((a, t)) = &self.0[i] {
            if Arc::ptr_eq(a, inst) {
                return *t;
            }
        }
        let t = timing(spec, inst);
        self.0[i] = Some((inst.clone(), t));
        t
    }
}

struct CtaBar {
    expected: u32,
    arrived: u32,
}

struct Cta {
    kid: u32,
    kernel: Arc<KernelLaunch>,
    cta_id: u64,
    warp_slots: Vec<usize>,
    live: u32,
    /// Named barriers keyed by (id << 32 | generation).
    bars: HashMap<u64, CtaBar>,
    mbars: HashMap<u32, MBar>,
    tensor_tail: u64,
    async_tail: u64,
    regs: u32,
    smem: u32,
}

struct Warp {
    cta: usize,
    /// Warp index within its CTA.
    widx: u32,
    kid: u32,
    role: Option<usize>,
    stream: Box<dyn InstStream>,
    next: Option<DynOp>,
    /// `timing()` of `next`, filled on its first evaluation.
    next_t: Option<crate::pipes::Timing>,
    ready_at: u64,
    ready_reason: Stall,
    pending_regs: Vec<(Reg, u64, Sb)>,
    sb: [Vec<(u64, Sb)>; 6],
    async_groups: Vec<u64>,
    open_group: u64,
    wg_pending: Vec<u64>,
    mbar_idx: HashMap<u32, usize>,
    at_bar: Option<u64>,
    /// Arrivals of this warp so far at each named barrier id: the n-th arrival of
    /// every participant belongs to generation n, so a warp that arrives (bar.arrive)
    /// twice before its partner syncs cannot complete a barrier on its own.
    bar_gen: HashMap<u32, u32>,
    /// mbarrier the warp is currently blocked on (u32::MAX = none).
    wait_bar: u32,
    /// Operand reuse cache: the register each source slot (a, b, c, d)
    /// keeps for the warp's next instruction (`NO_REG` = empty).
    reuse: [Reg; 4],
    exited: bool,
    /// Issued at the start of the interval `account` has not counted yet:
    /// its first cycle is the issue cycle (Nsight's `selected`).
    just_issued: bool,
    // Cached evaluation.
    until: u64,
    state: Stall,
    age: u64,
}

#[derive(Default)]
struct Smsp {
    warps: Vec<usize>,
    pipe_free: [u64; NUM_PIPES],
    /// Accumulated fractional issue-interval debt per pipe.
    pipe_debt: [f32; NUM_PIPES],
    tensor_tail: u64,
    /// Register-file banks: each bank delivers one operand per cycle, and
    /// reads queue per bank across instructions (and warps).
    bank_free: [u64; 8],
    next_eval: u64,
    last_eval: u64,
    sched_state: Option<(Stall, u32)>,
    greedy: Option<usize>,
    rr: usize,
}

pub struct Sm {
    pub id: u32,
    spec: Arc<GpuSpec>,
    policy: Policy,
    smsps: Vec<Smsp>,
    warps: Vec<Option<Warp>>,
    ctas: Vec<Option<Cta>>,
    used_warps: u32,
    used_regs: u32,
    used_smem: u32,
    used_ctas: u32,
    /// Reserved by a communication kernel (NCCL channel); no CTAs placed.
    pub reserved: bool,
    mio: Server,
    smem_srv: Server,
    /// Shared-memory load wavefronts served so far, and the cycle and count
    /// at the last tcgen05 MMA (LDS port utilisation between MMAs; stores
    /// do not slow UMMA operand reads).
    lds_wavefronts: u64,
    umma_mark: (u64, u64),
    /// Per sub-partition TMEM port (`issue.tmem`).
    tmem_srv: Vec<Server>,
    tma: Server,
    tensor_tail: u64,
    /// Completion times of tcgen05 MMAs in flight on this SM, oldest first
    /// (for `issue.tcgen05_queue_depth`).
    tc_inflight: std::collections::VecDeque<u64>,
    /// Next free cycle and fractional interval debt of the SM-wide unit that
    /// `sm_shared` opcodes use (B300 FP64).
    shared_unit_free: u64,
    shared_unit_debt: f32,
    l1: Cache,
    use_ctrl: bool,
    age: u64,
    stream_addr: u64,
    pub acc: AccMap,
    /// SM-level counters for timeline sampling.
    pub issued_total: u64,
    pub tensor_busy_total: u64,
    pub idle_sched_cycles: u64,
    scratch_ws: Vec<usize>,
    scratch_el: Vec<usize>,
    pub(crate) tmemo: TimingMemo,
    /// mbarrier events for CTAs on other SMs, drained by the GPU each step.
    pub outbox: Vec<RemoteMbar>,
    /// Visualiser capture: per-PC statistics, warp timeline (see `detail`).
    pub pc_stats: bool,
    pub wtrace: Option<Box<WarpTrace>>,
}

impl Sm {
    pub fn new(id: u32, spec: Arc<GpuSpec>, policy: Policy, use_ctrl: bool) -> Self {
        let lsu_rate = spec.pipes.lsu_per_sm_per_cycle.max(1) as f64;
        let l1 = Cache::new(&spec.l1, spec.l1.size_bytes);
        Self {
            id,
            smsps: (0..spec.sub_partitions)
                .map(|_| Smsp {
                    next_eval: NEVER,
                    ..Default::default()
                })
                .collect(),
            warps: (0..spec.max_warps_per_sm).map(|_| None).collect(),
            ctas: (0..spec.max_ctas_per_sm).map(|_| None).collect(),
            used_warps: 0,
            used_regs: 0,
            used_smem: 0,
            used_ctas: 0,
            reserved: false,
            mio: Server::new(lsu_rate),
            // One 32-bank wavefront (128 B) per cycle at the default 128 B/clk.
            lds_wavefronts: 0,
            umma_mark: (0, 0),
            smem_srv: Server::new((spec.pipes.l1_bytes_per_cycle.max(1) as f64) / 128.0),
            tmem_srv: (0..spec.sub_partitions).map(|_| Server::new(1.0)).collect(),
            tma: Server::new(spec.tma.bytes_per_cycle),
            tensor_tail: 0,
            tc_inflight: Default::default(),
            shared_unit_free: 0,
            shared_unit_debt: 0.0,
            l1,
            use_ctrl,
            age: 0,
            stream_addr: 0x7000_0000_0000 + ((id as u64) << 36),
            acc: AccMap::default(),
            issued_total: 0,
            tensor_busy_total: 0,
            idle_sched_cycles: 0,
            scratch_ws: Vec::with_capacity(16),
            scratch_el: Vec::with_capacity(16),
            tmemo: TimingMemo::default(),
            outbox: Vec::new(),
            pc_stats: false,
            wtrace: None,
            spec,
            policy,
        }
    }

    pub fn regs_per_cta(_spec: &GpuSpec, k: &KernelLaunch) -> u32 {
        // Allocation granularity: 256 registers per warp.
        let per_warp = (k.regs_per_thread.max(1) * 32).div_ceil(256) * 256;
        per_warp * k.warps_per_cta()
    }
    pub fn smem_per_cta(k: &KernelLaunch) -> u32 {
        // 1 KiB per CTA is reserved by the system on sm_80+.
        k.smem_per_cta + 1024
    }

    /// Max resident CTAs of `k` on an empty SM.
    pub fn occupancy(spec: &GpuSpec, k: &KernelLaunch) -> u32 {
        let w = k.warps_per_cta().max(1);
        let by_warps = spec.max_warps_per_sm / w;
        let by_regs = spec.regs_per_sm / Self::regs_per_cta(spec, k).max(1);
        let by_smem = spec.smem_per_sm_bytes / Self::smem_per_cta(k).max(1);
        spec.max_ctas_per_sm.min(by_warps).min(by_regs).min(by_smem)
    }

    pub fn can_fit(&self, k: &KernelLaunch) -> bool {
        !self.reserved
            && self.used_ctas < self.spec.max_ctas_per_sm
            && self.used_warps + k.warps_per_cta() <= self.spec.max_warps_per_sm
            && self.used_regs + Self::regs_per_cta(&self.spec, k) <= self.spec.regs_per_sm
            && self.used_smem + Self::smem_per_cta(k) <= self.spec.smem_per_sm_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.used_ctas == 0
    }

    pub fn launch_cta(&mut self, kid: u32, k: &Arc<KernelLaunch>, cta_id: u64, at: u64) {
        let slot = self
            .ctas
            .iter()
            .position(|c| c.is_none())
            .expect("can_fit checked");
        let nw = k.warps_per_cta();
        let regs = Self::regs_per_cta(&self.spec, k);
        let smem = Self::smem_per_cta(k);
        let mut mbars = HashMap::new();
        if let KernelSource::Static { barriers, .. } = &k.source {
            for (&id, init) in barriers {
                mbars.insert(
                    id,
                    MBar {
                        count: init.count.max(1),
                        arrivals: 0,
                        tx: 0,
                        max_t: 0,
                        phases: vec![(at, Stall::AsyncWait); init.initial_phases as usize],
                    },
                );
            }
        }
        if let KernelSource::Path(p) = &k.source {
            for (&id, &count) in &p.header.barriers {
                mbars.insert(
                    id,
                    MBar {
                        count: count.max(1),
                        arrivals: 0,
                        tx: 0,
                        max_t: 0,
                        phases: vec![],
                    },
                );
            }
        }
        let mut warp_slots = Vec::with_capacity(nw as usize);
        for w in 0..nw {
            // Spread warps across SMSPs: pick the free slot on the least-loaded SMSP.
            let nsp = self.smsps.len();
            let target = (w as usize + slot) % nsp;
            let ws = (0..self.warps.len())
                .filter(|i| self.warps[*i].is_none())
                .min_by_key(|i| ((i % nsp) != target, self.smsps[i % nsp].warps.len(), *i))
                .expect("warp slot");
            let (stream, role) = k.stream_for(cta_id, w);
            self.age += 1;
            self.warps[ws] = Some(Warp {
                cta: slot,
                widx: w,
                kid,
                role,
                stream,
                next: None,
                next_t: None,
                ready_at: at,
                ready_reason: Stall::Wait,
                pending_regs: Vec::new(),
                sb: Default::default(),
                async_groups: Vec::new(),
                open_group: 0,
                wg_pending: Vec::new(),
                mbar_idx: HashMap::new(),
                at_bar: None,
                bar_gen: HashMap::new(),
                wait_bar: u32::MAX,
                reuse: [NO_REG; 4],
                exited: false,
                just_issued: false,
                until: 0,
                state: Stall::Wait,
                age: self.age,
            });
            let sp = ws % nsp;
            if let Some(t) = &mut self.wtrace {
                t.begin(ws, sp, kid, cta_id, w, role, at);
            }
            self.smsps[sp].warps.push(ws);
            self.smsps[sp].next_eval = self.smsps[sp].next_eval.min(at);
            if self.smsps[sp].warps.len() == 1 {
                // SMSP was idle; account idle cycles up to `at`.
                let le = self.smsps[sp].last_eval;
                if at > le {
                    self.idle_sched_cycles += at - le;
                }
                self.smsps[sp].last_eval = at;
                self.smsps[sp].sched_state = None;
            }
            warp_slots.push(ws);
        }
        self.ctas[slot] = Some(Cta {
            kid,
            kernel: k.clone(),
            cta_id,
            warp_slots,
            live: nw,
            bars: HashMap::new(),
            mbars,
            tensor_tail: at,
            async_tail: at,
            regs,
            smem,
        });
        self.used_ctas += 1;
        self.used_warps += nw;
        self.used_regs += regs;
        self.used_smem += smem;
        let st = self.acc.get(kid);
        st.ctas += 1;
        st.warps += nw as u64;
    }

    /// Describe every resident warp (for diagnosing a kernel that never finishes).
    pub fn debug_dump(&mut self) -> String {
        let mut out = String::new();
        for ws in 0..self.warps.len() {
            let Some(w) = self.warps[ws].as_mut() else {
                continue;
            };
            let cta = w.cta;
            let next = Self::peek(w).map(|o| (o.inst.pc, o.inst.opcode.to_string(), o.mbar));
            let w = self.warps[ws].as_ref().unwrap();
            let c = self.ctas[cta].as_ref().unwrap();
            let wi = c.warp_slots.iter().position(|&x| x == ws).unwrap_or(0);
            if w.exited {
                continue;
            }
            out += &format!(
                "SM{} cta{} w{} state {:?} until {} at_bar {:?} next {:?}\n",
                self.id, c.cta_id, wi, w.state, w.until, w.at_bar, next
            );
            if let Some((_, _, Some(MbarRef::Wait { id, .. } | MbarRef::WaitParity { id, .. }))) =
                next
            {
                if let Some(m) = c.mbars.get(&id) {
                    out += &format!(
                        "    bar {id:#x}: count {} arrivals {} tx {} phases {} last phase at {:?}\n",
                        m.count,
                        m.arrivals,
                        m.tx,
                        m.phases.len(),
                        m.phases.last().map(|p| p.0)
                    );
                }
            }
        }
        out
    }

    pub fn next_event(&self) -> u64 {
        self.smsps
            .iter()
            .map(|s| s.next_eval)
            .min()
            .unwrap_or(NEVER)
    }

    fn poke_cta(&mut self, cta: usize, at: u64) {
        let nsp = self.smsps.len();
        if let Some(c) = &self.ctas[cta] {
            for &ws in &c.warp_slots {
                if let Some(w) = self.warps[ws].as_mut() {
                    w.until = w.until.min(at);
                }
                let sp = &mut self.smsps[ws % nsp];
                sp.next_eval = sp.next_eval.min(at);
            }
        }
    }

    /// Advance every SMSP whose next evaluation is at `c`. Completed CTAs
    /// are appended to `done` as `(kernel id, cta id)`.
    /// `spec` is this SM's own spec, borrowed from the GPU so the hot path
    /// takes no reference count.
    pub fn step(
        &mut self,
        c: u64,
        spec: &GpuSpec,
        mem: &mut MemSystem,
        done: &mut Vec<(u32, u64)>,
    ) {
        // The SMSPs contend for SM-wide resources (MIO, the shared FP64
        // unit) in this order, so rotate it each cycle: the arbiter is fair,
        // and a fixed order starves the last SMSP and leaves the CTA waiting
        // on it at every barrier.
        let n = self.smsps.len();
        let first = (c % n.max(1) as u64) as usize;
        for i in 0..n {
            let sp = (first + i) % n;
            if self.smsps[sp].next_eval <= c {
                self.step_smsp(sp, c, spec, mem, done);
            }
        }
    }

    fn account(&mut self, sp: usize, c: u64) {
        let s = &mut self.smsps[sp];
        let dt = c.saturating_sub(s.last_eval);
        if dt == 0 {
            return;
        }
        let t0 = s.last_eval;
        s.last_eval = c;
        if s.warps.is_empty() {
            self.idle_sched_cycles += dt;
            return;
        }
        if let Some((st, kid)) = s.sched_state {
            let acc = self.acc.get(kid);
            // `Selected` only ever lasts one cycle; the rest of the gap is
            // attributed to whatever the warps were doing (re-evaluated below).
            acc.sched.add(st, dt);
        }
        for &ws in &s.warps {
            if let Some(w) = self.warps[ws].as_mut() {
                let st = self.acc.get(w.kid);
                // The issue cycle counts as `selected` (Nsight: 1.0 per
                // issue); the warp's state covers the rest of the interval.
                let dt_issue = u64::from(std::mem::take(&mut w.just_issued));
                st.warp.add(Stall::Selected, dt_issue);
                st.warp.add(w.state, dt - dt_issue);
                if w.wait_bar != u32::MAX
                    && matches!(
                        w.state,
                        Stall::AsyncWait | Stall::TmaWait | Stall::TensorWait
                    )
                {
                    *st.mbar_wait_raw.entry((w.widx, w.wait_bar)).or_default() += dt;
                }
                let i = w.widx as usize;
                if st.warp_by_index.len() <= i {
                    st.warp_by_index.resize(i + 1, StallCounts::default());
                }
                st.warp_by_index[i].add(Stall::Selected, dt_issue);
                st.warp_by_index[i].add(w.state, dt - dt_issue);
                if self.pc_stats {
                    if let Some(op) = &w.next {
                        pc_entry(&mut st.pc, w.role, &op.inst)
                            .stalls
                            .add(w.state, dt);
                    }
                }
                if let Some(t) = &mut self.wtrace {
                    t.state(ws, w.state, t0, c);
                }
            }
        }
    }

    fn step_smsp(
        &mut self,
        sp: usize,
        c: u64,
        spec: &GpuSpec,
        mem: &mut MemSystem,
        done: &mut Vec<(u32, u64)>,
    ) {
        self.account(sp, c);
        if self.smsps[sp].warps.is_empty() {
            self.smsps[sp].next_eval = NEVER;
            return;
        }
        // 1. Evaluate warp states.
        let ws_list = std::mem::take(&mut self.scratch_ws);
        let mut ws_list = ws_list;
        ws_list.clear();
        ws_list.extend_from_slice(&self.smsps[sp].warps);
        let mut eligible = std::mem::take(&mut self.scratch_el);
        eligible.clear();
        let mut min_until = NEVER;
        for &ws in &ws_list {
            let needs = self.warps[ws]
                .as_ref()
                .map(|w| w.until <= c)
                .unwrap_or(false);
            if needs {
                let (st, until) = self.evaluate(ws, sp, c, mem);
                let w = self.warps[ws].as_mut().unwrap();
                w.state = st;
                w.until = until;
            }
            let w = self.warps[ws].as_ref().unwrap();
            if w.state == Stall::Selected {
                eligible.push(ws);
            } else {
                min_until = min_until.min(w.until);
            }
        }
        // 2. Pick one warp.
        let pick = if eligible.is_empty() {
            None
        } else {
            match self.policy {
                Policy::Gto | Policy::Gty => {
                    let g = self.smsps[sp].greedy;
                    if g.map(|g| eligible.contains(&g)).unwrap_or(false) {
                        g
                    } else if self.policy == Policy::Gto {
                        eligible
                            .iter()
                            .copied()
                            .min_by_key(|&ws| self.warps[ws].as_ref().unwrap().age)
                    } else {
                        eligible
                            .iter()
                            .copied()
                            .max_by_key(|&ws| self.warps[ws].as_ref().unwrap().age)
                    }
                }
                Policy::Lrr => {
                    let n = ws_list.len();
                    let start = self.smsps[sp].rr;
                    let mut chosen = None;
                    for i in 0..n {
                        let ws = ws_list[(start + i) % n];
                        if eligible.contains(&ws) {
                            chosen = Some(ws);
                            self.smsps[sp].rr = (start + i + 1) % n;
                            break;
                        }
                    }
                    chosen
                }
            }
        };
        for &ws in &eligible {
            if Some(ws) != pick {
                let w = self.warps[ws].as_mut().unwrap();
                w.state = Stall::NotSelected;
                w.until = c + 1;
            }
        }
        // Scheduler-level attribution for this cycle.
        let sched = match pick {
            Some(ws) => (Stall::Selected, self.warps[ws].as_ref().unwrap().kid),
            None => {
                // Attribute the bubble to the warp that will unblock first (the
                // critical warp); ties go to the oldest.
                let ws = *ws_list
                    .iter()
                    .min_by_key(|&&ws| {
                        let w = self.warps[ws].as_ref().unwrap();
                        (w.until, w.age)
                    })
                    .unwrap();
                let w = self.warps[ws].as_ref().unwrap();
                (w.state, w.kid)
            }
        };
        self.smsps[sp].sched_state = Some(sched);
        // 3. Issue.
        if let Some(ws) = pick {
            self.smsps[sp].greedy = Some(ws);
            self.issue(ws, sp, c, spec, mem, done);
            self.smsps[sp].next_eval = c + 1;
        } else {
            self.smsps[sp].greedy = None;
            self.smsps[sp].next_eval = if self.smsps[sp].warps.is_empty() {
                NEVER
            } else {
                min_until.max(c + 1)
            };
        }
        if eligible.len() > 1 {
            self.smsps[sp].next_eval = c + 1;
        }
        self.scratch_ws = ws_list;
        self.scratch_el = eligible;
    }

    /// Fetch the warp's next op if needed; `None` => stream exhausted.
    fn peek(w: &mut Warp) -> Option<&DynOp> {
        if w.next.is_none() {
            w.next = w.stream.next_op();
        }
        w.next.as_ref()
    }

    /// Decide whether warp `ws` can issue at `c`: `(Selected, c)` if so,
    /// else `(reason, cycle when it might change)`.
    fn evaluate(&mut self, ws: usize, sp: usize, c: u64, mem: &mut MemSystem) -> (Stall, u64) {
        let w = self.warps[ws].as_mut().unwrap();
        if w.exited {
            return (Stall::Drain, NEVER);
        }
        if w.at_bar.is_some() {
            return (Stall::Barrier, NEVER);
        }
        if w.ready_at > c {
            return (w.ready_reason, w.ready_at);
        }
        if Self::peek(w).is_none() {
            // Implicit exit.
            return (Stall::Selected, c);
        }
        // Lend the op out of the warp so the checks below can borrow the
        // SM freely (no reference-count traffic on the hot path).
        let op = w.next.take().unwrap();
        let t = match w.next_t {
            Some(t) => t,
            None => {
                let t = self.tmemo.get(&self.spec, &op.inst);
                w.next_t = Some(t);
                t
            }
        };
        let r = self.evaluate_op(ws, sp, c, &op, t, mem);
        self.warps[ws].as_mut().unwrap().next = Some(op);
        r
    }

    fn evaluate_op(
        &mut self,
        ws: usize,
        sp: usize,
        c: u64,
        op: &DynOp,
        t: crate::pipes::Timing,
        mem: &mut MemSystem,
    ) -> (Stall, u64) {
        let spec = &*self.spec;
        let use_ctrl = self.use_ctrl;
        let mio_free = self.mio.free_at();
        let tensor_tail = self.tensor_tail;
        let w = self.warps[ws].as_mut().unwrap();
        let cta_idx = w.cta;
        let inst = &*op.inst;
        let op_mbar = op.mbar;
        // Dependency checks.
        if use_ctrl && inst.ctrl.is_some() {
            if let Some(ctrl) = inst.ctrl {
                let mut until = 0;
                let mut why = Stall::Wait;
                for b in 0..6 {
                    if ctrl.wait_mask & (1 << b) != 0 {
                        for &(t, k) in &w.sb[b] {
                            if t > c && t > until {
                                until = t;
                                why = if k == Sb::Long {
                                    Stall::LongScoreboard
                                } else {
                                    Stall::ShortScoreboard
                                };
                            }
                        }
                    }
                }
                if until > c {
                    return (why, until);
                }
            }
        } else {
            w.pending_regs.retain(|&(_, t, _)| t > c);
            let mut until = 0;
            let mut why = Stall::Wait;
            let async_mma = matches!(inst.class, OpClass::MmaWarpgroup | OpClass::MmaTcgen05);
            for &(r, t, k) in &w.pending_regs {
                let hit = inst.srcs.contains(&r) || (!async_mma && inst.dsts.contains(&r));
                if hit && t > until {
                    until = t;
                    why = match k {
                        Sb::Long => Stall::LongScoreboard,
                        Sb::Short => Stall::ShortScoreboard,
                        Sb::Fixed => Stall::Wait,
                    };
                }
            }
            if until > c {
                return (why, until);
            }
        }
        // Recorded paths carry the barrier and the phase each wait needs.
        if let Some(MbarRef::Wait { id, need }) = op_mbar {
            if need > 0 {
                let cta = self.ctas[cta_idx].as_ref().unwrap();
                match cta
                    .mbars
                    .get(&id)
                    .and_then(|m| m.phases.get(need as usize - 1))
                {
                    None => {
                        self.warps[ws].as_mut().unwrap().wait_bar = id;
                        return (Stall::AsyncWait, NEVER);
                    }
                    Some(&(t, why)) if t > c => {
                        self.warps[ws].as_mut().unwrap().wait_bar = id;
                        return (why, t);
                    }
                    _ => {}
                }
            }
        }
        // A spin on a global counter another SM increments: ready one polling
        // round trip after the increment that reaches the value lands.
        if let Some(MbarRef::GlobalWait { id, need }) = op_mbar {
            if need > 0 {
                let kid = self.ctas[cta_idx].as_ref().unwrap().kid;
                match mem.flags.reached(kid, id, need, self.id) {
                    None => return (Stall::GridSync, NEVER),
                    Some(t) if t + mem.l2_latency() > c => {
                        return (Stall::GridSync, t + mem.l2_latency())
                    }
                    _ => {}
                }
            }
        }
        if let Some(MbarRef::WaitParity { id, parity }) = op_mbar {
            let cta = self.ctas[cta_idx].as_ref().unwrap();
            let phases: &[(u64, Stall)] = cta.mbars.get(&id).map(|m| &m.phases[..]).unwrap_or(&[]);
            let done = phases.iter().filter(|p| p.0 <= c).count();
            if done % 2 == parity as usize {
                // Blocked until the next phase lands (or is produced at all).
                return match phases.iter().filter(|p| p.0 > c).min_by_key(|p| p.0) {
                    Some(&(t, why)) => (why, t),
                    None => (Stall::AsyncWait, NEVER),
                };
            }
        }
        // Class-specific waits.
        match inst.class {
            OpClass::Warpgroup if inst.meta.flags & flags::DEPBAR != 0 => {
                let n = inst.meta.imm_last.unwrap_or(0) as usize;
                let mut pend: Vec<u64> = w.wg_pending.iter().copied().filter(|&t| t > c).collect();
                if pend.len() > n {
                    pend.sort_unstable();
                    return (Stall::TensorWait, pend[pend.len() - n - 1]);
                }
            }
            OpClass::DepBar if inst.base_opcode() != "LDGDEPBAR" => {
                let n = inst.meta.imm_last.unwrap_or(0) as usize;
                if use_ctrl && inst.ctrl.is_some() {
                    let b = inst.meta.sb.unwrap_or(0) as usize;
                    let mut pend: Vec<u64> =
                        w.sb[b].iter().map(|x| x.0).filter(|&t| t > c).collect();
                    // `cp.async.wait_group N` is `DEPBAR.LE SB0, N`: LDGDEPBAR
                    // commit groups count on SB0.
                    if b == 0 {
                        pend.extend(w.async_groups.iter().copied().filter(|&t| t > c));
                    }
                    if pend.len() > n {
                        pend.sort_unstable();
                        return (Stall::LongScoreboard, pend[pend.len() - n - 1]);
                    }
                } else {
                    let mut pend: Vec<u64> =
                        w.async_groups.iter().copied().filter(|&t| t > c).collect();
                    if pend.len() > n {
                        pend.sort_unstable();
                        return (Stall::AsyncWait, pend[pend.len() - n - 1]);
                    }
                }
            }
            // A CTA cannot retire (and free its shared memory) while its
            // async copies are still landing: its last warp waits for them.
            // Other warps exit freely (a recorded EXIT may even have no lanes
            // active while the warp goes on).
            OpClass::Exit if inst.guard.is_none() => {
                let cta = self.ctas[cta_idx].as_ref().unwrap();
                if cta.live == 1 && cta.async_tail > c {
                    return (Stall::Drain, cta.async_tail);
                }
            }
            OpClass::AsyncBarrier
                if inst.meta.flags & flags::MBAR_WAIT != 0 && op_mbar.is_none() =>
            {
                let role = w.role;
                let bar = self.bar_op(cta_idx, role, inst.pc);
                let w = self.warps[ws].as_ref().unwrap();
                let cta = self.ctas[cta_idx].as_ref().unwrap();
                match bar {
                    Some(BarOp::Wait(id)) => {
                        let k = w.mbar_idx.get(&id).copied().unwrap_or(0);
                        match cta.mbars.get(&id).and_then(|m| m.phases.get(k)) {
                            None => return (Stall::AsyncWait, NEVER),
                            Some(&(t, why)) if t > c => return (why, t),
                            _ => {}
                        }
                    }
                    // A recorded path's wait without barrier semantics is a non-blocking
                    // poll that failed on hardware: it does not wait.
                    _ if matches!(cta.kernel.source, KernelSource::Path(_)) => {}
                    _ => {
                        // Un-annotated SASS: conservatively wait for all
                        // async work this CTA has issued.
                        if cta.async_tail > c {
                            return (Stall::AsyncWait, cta.async_tail);
                        }
                    }
                }
            }
            _ => {}
        }
        // Structural hazards.
        let sps = &self.smsps[sp];
        match t.pipe {
            Pipe::Mio => {
                if mio_free > c + MIO_QUEUE_CYCLES {
                    let long = matches!(
                        inst.mem.map(|m| m.space),
                        Some(MemSpace::Global | MemSpace::Local | MemSpace::Generic)
                    );
                    return (
                        if long {
                            Stall::LgThrottle
                        } else {
                            Stall::MioThrottle
                        },
                        mio_free - MIO_QUEUE_CYCLES,
                    );
                }
            }
            Pipe::Tensor => match inst.class {
                OpClass::MmaWarpgroup if sps.tensor_tail > c + WGMMA_QUEUE_CYCLES => {
                    return (Stall::TensorThrottle, sps.tensor_tail - WGMMA_QUEUE_CYCLES)
                }
                // Measured: the issuing thread blocks once a fixed number of
                // MMAs are in flight (B300: 7, whatever their size).
                OpClass::MmaTcgen05 if spec.issue.tcgen05_queue_depth > 0 => {
                    let q = &self.tc_inflight;
                    let live = q.iter().filter(|&&t| t > c).count();
                    let d = spec.issue.tcgen05_queue_depth as usize;
                    if live >= d {
                        let mut v: Vec<u64> = q.iter().copied().filter(|&t| t > c).collect();
                        v.sort_unstable();
                        return (Stall::TensorThrottle, v[live - d]);
                    }
                }
                OpClass::MmaTcgen05 if tensor_tail > c + TCGEN05_QUEUE_CYCLES => {
                    return (Stall::TensorThrottle, tensor_tail - TCGEN05_QUEUE_CYCLES)
                }
                OpClass::MmaSync if sps.pipe_free[Pipe::Tensor as usize] > c => {
                    return (
                        Stall::MathPipeThrottle,
                        sps.pipe_free[Pipe::Tensor as usize],
                    )
                }
                _ => {}
            },
            _ if t.sm_shared => {
                if self.shared_unit_free > c {
                    return (Stall::MathPipeThrottle, self.shared_unit_free);
                }
            }
            p => {
                let f = sps.pipe_free[p as usize];
                if f > c {
                    return (Stall::MathPipeThrottle, f);
                }
            }
        }
        // Register-file operand reads: a bank this instruction reads is
        // still busy with earlier (bank-conflicted) reads.
        if spec.issue.rf_banks > 0 {
            let w = self.warps[ws].as_ref().unwrap();
            let per = rf_bank_reads(spec, inst, &w.reuse);
            let busy = per
                .iter()
                .zip(&sps.bank_free)
                .filter(|(n, _)| **n > 0)
                .map(|(_, &f)| f)
                .max()
                .unwrap_or(0);
            if busy > c {
                return (Stall::Dispatch, busy);
            }
        }
        (Stall::Selected, c)
    }

    fn bar_op(&self, cta: usize, role: Option<usize>, pc: u32) -> Option<BarOp> {
        let c = self.ctas[cta].as_ref()?;
        match (&c.kernel.source, role) {
            (KernelSource::Static { roles, .. }, Some(r)) => {
                roles.get(r)?.bar_ops.get(&pc).copied()
            }
            _ => None,
        }
    }

    fn multicast(&self, cta: usize, role: Option<usize>, pc: u32) -> f64 {
        let Some(c) = self.ctas[cta].as_ref() else {
            return 1.0;
        };
        match (&c.kernel.source, role) {
            (KernelSource::Static { roles, .. }, Some(r)) => roles
                .get(r)
                .and_then(|x| x.multicast.get(&pc).copied())
                .unwrap_or(1.0),
            _ => 1.0,
        }
    }

    fn mbar_arrive(&mut self, cta: usize, id: u32, t: u64, producer: Stall) {
        self.mbar_event(cta, id, t, 1, 0, producer);
    }

    /// One mbarrier event at time `t`: `arrivals` arrivals and a change of
    /// `tx` in the pending transaction bytes (expect-tx adds, a completed
    /// async copy subtracts). The phase completes once the arrival count is
    /// reached and no transaction bytes are outstanding.
    fn mbar_event(&mut self, cta: usize, id: u32, t: u64, arrivals: u32, tx: i64, producer: Stall) {
        let c = self.ctas[cta].as_mut().unwrap();
        let m = c.mbars.entry(id).or_insert(MBar {
            count: 1,
            arrivals: 0,
            tx: 0,
            max_t: 0,
            phases: vec![],
        });
        m.arrivals += arrivals;
        m.tx += tx;
        m.max_t = m.max_t.max(t);
        if m.arrivals >= m.count && m.tx <= 0 {
            // The last event to land decides who the waiters were waiting on.
            let why = if arrivals == 0 {
                Stall::TmaWait
            } else {
                producer
            };
            m.phases.push((m.max_t, why));
            m.arrivals = 0;
            m.tx = 0;
            m.max_t = 0;
            let at = m.phases.last().unwrap().0;
            self.poke_cta(cta, at);
        }
    }

    /// Signal barrier `id` on the cluster ranks in `ranks` (0 = this CTA).
    #[allow(clippy::too_many_arguments)]
    fn signal(
        &mut self,
        cta: usize,
        id: u32,
        ranks: u16,
        t: u64,
        arrivals: u32,
        tx: i64,
        why: Stall,
    ) {
        if ranks == 0 {
            return self.mbar_event(cta, id, t, arrivals, tx, why);
        }
        let c = self.ctas[cta].as_ref().unwrap();
        let cl = c.kernel.cluster.max(1) as u64;
        let (kid, cta_id) = (c.kid, c.cta_id);
        let own = cta_id % cl;
        for r in 0..16u64 {
            if ranks & (1 << r) == 0 || r >= cl {
                continue;
            }
            if r == own {
                self.mbar_event(cta, id, t, arrivals, tx, why);
            } else {
                self.outbox.push(RemoteMbar {
                    kid,
                    cta: cta_id - own + r,
                    id,
                    t: t + self.spec.latency.dsmem as u64,
                    arrivals,
                    tx,
                    why,
                });
            }
        }
    }

    /// Log a memory op that touches a probed address range.
    fn probe(&mut self, cta: usize, widx: usize, op: &DynOp, c: u64) {
        let c_ref = self.ctas[cta].as_ref().unwrap();
        let KernelSource::Path(p) = &c_ref.kernel.source else {
            return;
        };
        let Some(&a) = op.addrs.first() else {
            return;
        };
        if p.header.probes.iter().any(|r| a >= r[0] && a < r[1]) {
            let (kid, cta_id) = (c_ref.kid, c_ref.cta_id);
            let id = self.id as u64;
            self.acc
                .get(kid)
                .probes
                .extend_from_slice(&[id, cta_id, widx as u64, a, c]);
        }
    }

    /// Re-evaluate every resident warp at `at` (a global counter changed).
    pub fn wake(&mut self, at: u64) {
        for i in 0..self.ctas.len() {
            self.poke_cta(i, at);
        }
    }

    /// Apply an event from another SM; false if the CTA is not resident here.
    pub fn deliver(&mut self, ev: &RemoteMbar) -> bool {
        let Some(slot) = self.ctas.iter().position(|c| {
            c.as_ref()
                .is_some_and(|c| c.kid == ev.kid && c.cta_id == ev.cta)
        }) else {
            return false;
        };
        self.mbar_event(slot, ev.id, ev.t, ev.arrivals, ev.tx, ev.why);
        true
    }

    fn issue(
        &mut self,
        ws: usize,
        sp: usize,
        c: u64,
        spec: &GpuSpec,
        mem: &mut MemSystem,
        done: &mut Vec<(u32, u64)>,
    ) {
        let w = self.warps[ws].as_mut().unwrap();
        let Some(op) = w.next.take() else {
            self.exit_warp(ws, sp, c, done);
            return;
        };
        let cached_t = w.next_t.take();
        let inst = &*op.inst;
        let cta = w.cta;
        let kid = w.kid;
        let role = w.role;
        let widx = w.widx as usize;
        w.wait_bar = u32::MAX;
        let t = cached_t.unwrap_or_else(|| self.tmemo.get(spec, &op.inst));
        self.issued_total += 1;
        if let Some(tr) = &mut self.wtrace {
            tr.issue(ws, c, inst.pc);
        }
        {
            let st = self.acc.get(kid);
            st.insts_issued += 1;
            if st.issued_by_index.len() <= widx {
                st.issued_by_index.resize(widx + 1, 0);
            }
            st.issued_by_index[widx] += 1;
            st.class_counts[inst.class as usize] += 1;
            if self.pc_stats {
                pc_entry(&mut st.pc, role, inst).issued += 1;
            }
            if inst.class == OpClass::Misc {
                *st.unknown_opcodes
                    .entry(inst.base_opcode().to_string())
                    .or_default() += 1;
            }
        }
        // Default: next instruction may issue next cycle (or after the
        // compiler stall count when control bits are honoured).
        let mut next_ready = c + 1;
        let mut next_reason = Stall::Wait;
        // Multi-cycle dispatch holds the warp before its stall count starts.
        let dispatch = dispatch_cycles(spec, inst);
        if self.use_ctrl {
            if let Some(ctrl) = inst.ctrl {
                let mut stall = ctrl.stall.max(1) as u64;
                // Yield bit clear: the warp gives up its slot (the scheduler
                // moves on, and the warp cannot issue on the next cycle).
                if !ctrl.yield_hint && spec.issue.yield_min_stall > 0 {
                    stall = stall.max(spec.issue.yield_min_stall as u64);
                    self.smsps[sp].greedy = None;
                }
                next_ready = c + dispatch + stall;
            }
        } else {
            next_ready = c + dispatch.max(1);
        }
        if spec.issue.warp_min_interval > 1 {
            next_ready = next_ready.max(c + spec.issue.warp_min_interval as u64);
        }
        // Operand reads through the banked register file (and reuse cache).
        if spec.issue.rf_banks > 0 {
            let w = self.warps[ws].as_mut().unwrap();
            let per = rf_bank_reads(spec, inst, &w.reuse);
            if spec.issue.reuse_cache {
                let bits = inst.ctrl.map(|x| x.reuse).unwrap_or(0);
                if !spec.issue.reuse_sticky {
                    w.reuse = [NO_REG; 4];
                }
                for o in &inst.meta.rf[..inst.meta.rf_n as usize] {
                    let e = &mut w.reuse[o.slot as usize];
                    if bits >> o.slot & 1 == 1 {
                        *e = o.reg;
                    } else if *e != o.reg {
                        *e = NO_REG;
                    }
                }
            }
            let bf = &mut self.smsps[sp].bank_free;
            for (b, &n) in per.iter().enumerate() {
                if n > 0 {
                    bf[b] = bf[b].max(c) + n as u64;
                }
            }
        }
        // Result time and scoreboard kind for destination registers.
        let mut result: Option<(u64, Sb)> = None;
        let mut read_done = c + spec.latency.store_read as u64;

        if t.sm_shared {
            self.shared_unit_debt += t.ii_frac;
            let extra = if self.shared_unit_debt >= 1.0 {
                self.shared_unit_debt -= 1.0;
                1
            } else {
                0
            };
            self.shared_unit_free = c + t.ii as u64 + extra;
        } else if t.pipe != Pipe::Mio && t.pipe != Pipe::Tensor {
            let sps = &mut self.smsps[sp];
            let debt = &mut sps.pipe_debt[t.pipe as usize];
            *debt += t.ii_frac;
            let extra = if *debt >= 1.0 {
                *debt -= 1.0;
                1
            } else {
                0
            };
            sps.pipe_free[t.pipe as usize] = c + t.ii as u64 + extra;
        }

        if let Some(MbarRef::GlobalAdd { id, n }) = op.mbar {
            if op.active != 0 {
                mem.flags.add(kid, id, n, c + mem.l2_latency());
            }
        }
        match inst.class {
            // Warps retire when their stream ends (next fetch returns
            // `None`), not at an EXIT: a recorded EXIT may have only some
            // lanes (or none) active, and the warp's remaining lanes go on.
            OpClass::Exit => {}
            OpClass::Mem | OpClass::AsyncCopy | OpClass::Ldsm | OpClass::Tmem => {
                // A recorded op with every lane predicated off moves no data.
                let r = if op.active == 0 {
                    c + 1
                } else {
                    self.mem_op(&op, c, spec, mem, kid, sp)
                };
                if op.active != 0 {
                    self.probe(cta, widx, &op, c);
                }
                result = Some((r, t.sb));
                if inst.class == OpClass::AsyncCopy {
                    let w = self.warps[ws].as_mut().unwrap();
                    w.open_group = w.open_group.max(r);
                    result = None;
                }
                read_done = read_done.max(c + 2);
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                cta_ref.async_tail = cta_ref.async_tail.max(r);
            }
            OpClass::Shfl | OpClass::S2R | OpClass::Ldc => {
                let d = self.mio.serve(c, 1);
                result = Some((d.max(c) + t.lat as u64, t.sb));
            }
            OpClass::Fence => {
                next_ready = next_ready.max(c + t.lat as u64);
                next_reason = Stall::Membar;
            }
            OpClass::MmaSync => {
                let rate =
                    spec.tensor.sync_flops_per_cycle(mma_dtype(inst)) / spec.sub_partitions as f64;
                let flops = mma_flops(inst, op.payload);
                // Measured rates give fractional intervals (A100: 8.09 cycles
                // per m16n8k16); carry the fraction instead of rounding up.
                let exact = (flops / rate).max(1.0);
                let sps = &mut self.smsps[sp];
                let debt = &mut sps.pipe_debt[Pipe::Tensor as usize];
                *debt += (exact - exact.floor()) as f32;
                let ii = exact.floor() as u64 + (*debt >= 1.0) as u64;
                if *debt >= 1.0 {
                    *debt -= 1.0;
                }
                sps.pipe_free[Pipe::Tensor as usize] = c + ii;
                let hold = spec.tensor.hmma_fma_pipe_cycles as u64;
                if hold > 0 {
                    let f = &mut sps.pipe_free[Pipe::Fma as usize];
                    *f = (*f).max(c + hold);
                }
                // The measured dependent-issue latency already includes the issue interval.
                result = Some((c + (t.lat as u64).max(ii), Sb::Short));
                self.tensor_busy_total += ii;
                let st = self.acc.get(kid);
                st.tensor_busy_cycles += ii;
                st.tensor_flops += flops;
            }
            OpClass::MmaWarpgroup => {
                let rate =
                    spec.tensor.flops_per_cycle(mma_dtype(inst)) / spec.sub_partitions as f64;
                let flops = mma_flops(inst, op.payload);
                let dur = (flops / rate).ceil().max(1.0) as u64;
                let sps = &mut self.smsps[sp];
                let start = sps.tensor_tail.max(c + t.lat as u64);
                sps.tensor_tail = start + dur;
                let fin = start + dur;
                self.warps[ws].as_mut().unwrap().wg_pending.push(fin);
                self.tensor_busy_total += dur;
                let st = self.acc.get(kid);
                st.tensor_busy_cycles += dur;
                st.tensor_flops += flops;
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                cta_ref.async_tail = cta_ref.async_tail.max(fin);
            }
            OpClass::MmaTcgen05 => {
                // A `cta_group::2` MMA runs on the tensor cores of both SMs of
                // the pair, each computing half of M; only the leader issues it.
                let pair = if inst.opcode.contains(".2CTA") {
                    2.0
                } else {
                    1.0
                };
                let k_ext = mma_shape(inst).map(|s| s.2).unwrap_or(0);
                let rate = pair * spec.tensor.mma_rate(mma_dtype(inst), k_ext);
                let flops = mma_flops(inst, op.payload);
                let dur = if spec.tensor.umma_smem_bytes_per_clk > 0.0 {
                    // Shape-aware: M padding and smem operand reads, slowed
                    // by the LDS traffic since the previous MMA.
                    let (c0, w0) = self.umma_mark;
                    let port = (spec.pipes.l1_bytes_per_cycle.max(1) as f64) / 128.0;
                    let lds = (self.lds_wavefronts - w0) as f64
                        / port
                        / c.saturating_sub(c0).max(1) as f64;
                    self.umma_mark = (c, self.lds_wavefronts);
                    let shape = umma_shape(inst, flops);
                    spec.tensor
                        .umma_cycles(mma_dtype(inst), shape, lds)
                        .ceil()
                        .max(1.0) as u64
                } else {
                    (flops / rate).ceil().max(1.0) as u64
                };
                let start = self.tensor_tail.max(c + t.lat as u64);
                self.tensor_tail = start + dur;
                // Visible to tcgen05.commit after the pipeline latency.
                let fin = start + dur + spec.tensor.tcgen05_latency as u64;
                if spec.issue.tcgen05_queue_depth > 0 {
                    while self.tc_inflight.front().is_some_and(|&t| t <= c) {
                        self.tc_inflight.pop_front();
                    }
                    self.tc_inflight.push_back(fin);
                }
                // The SM's four tensor cores work as one engine here.
                self.tensor_busy_total += dur * spec.sub_partitions as u64;
                let st = self.acc.get(kid);
                st.tensor_busy_cycles += dur * spec.sub_partitions as u64;
                st.tensor_flops += flops;
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                cta_ref.tensor_tail = cta_ref.tensor_tail.max(fin);
                cta_ref.async_tail = cta_ref.async_tail.max(fin);
            }
            OpClass::Tma => {
                if op.active != 0 {
                    self.probe(cta, widx, &op, c);
                }
                let bytes = if op.payload > 0 {
                    op.payload
                } else {
                    16 * 1024
                };
                let addr = op.addrs.first().copied().unwrap_or_else(|| {
                    self.stream_addr += bytes;
                    self.stream_addr
                });
                let store = inst.meta.flags & flags::BULK_STORE != 0;
                let frac = self.multicast(cta, role, inst.pc);
                let start = self.tma.serve(c, bytes) + spec.tma.issue_latency as u64;
                // A recorded TMA op lists the contiguous source runs of its box.
                let runs: Vec<(u64, u64)> = if op.run_bytes > 0 && op.addrs.len() > 1 {
                    op.addrs.iter().map(|&a| (a, op.run_bytes as u64)).collect()
                } else {
                    vec![(addr, bytes)]
                };
                let mut fin = start;
                for (a, n) in runs {
                    let r = mem.bulk(self.id, start, a, n, frac, store);
                    let st = self.acc.get(kid);
                    st.l2_hits += r.l2_hit_sectors as u64;
                    st.l2_misses += r.l2_miss_sectors as u64;
                    if store {
                        st.dram_write_bytes += n;
                    } else {
                        st.dram_read_bytes += r.dram_bytes;
                    }
                    fin = fin.max(r.done);
                }
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                cta_ref.async_tail = cta_ref.async_tail.max(fin);
                match op.mbar {
                    Some(MbarRef::TxDone { id, bytes, ranks }) => {
                        self.signal(cta, id, ranks, fin, 0, -(bytes as i64), Stall::TmaWait)
                    }
                    Some(_) => {}
                    None => {
                        if let Some(BarOp::Arrive(id)) = self.bar_op(cta, role, inst.pc) {
                            self.mbar_arrive(cta, id, fin, Stall::TmaWait);
                        }
                    }
                }
            }
            OpClass::AsyncBarrier if op.mbar.is_some() => match op.mbar.unwrap() {
                MbarRef::Arrive { id, tx, n, ranks } => self.signal(
                    cta,
                    id,
                    ranks,
                    c + t.lat as u64,
                    n,
                    tx as i64,
                    Stall::AsyncWait,
                ),
                MbarRef::CommitArrive { id, ranks } => {
                    let at = self.ctas[cta].as_ref().unwrap().tensor_tail.max(c);
                    self.signal(cta, id, ranks, at, 1, 0, Stall::TensorWait)
                }
                MbarRef::TxDone { .. }
                | MbarRef::Wait { .. }
                | MbarRef::WaitParity { .. }
                | MbarRef::Named { .. }
                | MbarRef::GlobalWait { .. }
                | MbarRef::GlobalAdd { .. } => {}
            },
            OpClass::AsyncBarrier => {
                let bar = self.bar_op(cta, role, inst.pc);
                match bar {
                    Some(BarOp::Arrive(id)) => {
                        let (at, why) = if inst.base_opcode() == "UTCBAR" {
                            (
                                self.ctas[cta].as_ref().unwrap().tensor_tail.max(c),
                                Stall::TensorWait,
                            )
                        } else {
                            (c + t.lat as u64, Stall::AsyncWait)
                        };
                        self.mbar_arrive(cta, id, at, why);
                    }
                    Some(BarOp::Wait(id)) => {
                        let w = self.warps[ws].as_mut().unwrap();
                        *w.mbar_idx.entry(id).or_insert(0) += 1;
                    }
                    None => {}
                }
            }
            OpClass::Warpgroup => {}
            OpClass::DepBar => {
                if inst.base_opcode() == "LDGDEPBAR" {
                    let w = self.warps[ws].as_mut().unwrap();
                    let g = w.open_group.max(c);
                    w.async_groups.push(g);
                    w.open_group = 0;
                }
            }
            OpClass::Barrier => {
                // Recorded paths carry the id of barriers named by a register.
                let bar_id = match op.mbar {
                    Some(MbarRef::Named { id }) => id,
                    _ => inst.meta.bar_id,
                };
                let w = self.warps[ws].as_mut().unwrap();
                let g = w.bar_gen.entry(bar_id).or_insert(0);
                let key = ((bar_id as u64) << 32) | *g as u64;
                *g += 1;
                let arrive_only = inst.meta.flags & flags::ARV != 0;
                if !arrive_only {
                    w.at_bar = Some(key);
                }
                let expected = inst.meta.bar_threads.map(|t| t.div_ceil(32));
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                let live = cta_ref.live;
                let b = cta_ref.bars.entry(key).or_insert(CtaBar {
                    expected: expected.unwrap_or(live),
                    arrived: 0,
                });
                if !arrive_only {
                    b.expected = expected.unwrap_or(live);
                }
                // `bar.arrive` counts toward the barrier without waiting.
                b.arrived += 1;
                self.try_release_bar(cta, key, c);
            }
            OpClass::Branch => {
                if !self.use_ctrl || inst.ctrl.is_none() {
                    next_ready = next_ready.max(c + t.lat as u64);
                    next_reason = Stall::BranchResolving;
                }
            }
            _ => {
                if t.sb != Sb::Fixed || !self.use_ctrl || inst.ctrl.is_none() {
                    result = Some((c + t.lat as u64, t.sb));
                }
            }
        }

        let w = self.warps[ws].as_mut().unwrap();
        if let Some((rt, kind)) = result {
            if let (true, Some(ctrl)) = (self.use_ctrl, inst.ctrl) {
                if let Some(b) = ctrl.wbar {
                    w.sb[b as usize].retain(|x| x.0 > c);
                    w.sb[b as usize].push((rt, kind));
                }
            } else if !matches!(inst.class, OpClass::MmaWarpgroup | OpClass::MmaTcgen05) {
                for &d in &inst.dsts {
                    w.pending_regs.retain(|x| x.0 != d);
                    w.pending_regs.push((d, rt, kind));
                }
            }
        }
        if self.use_ctrl {
            if let Some(ctrl) = inst.ctrl {
                if let Some(b) = ctrl.rbar {
                    w.sb[b as usize].retain(|x| x.0 > c);
                    w.sb[b as usize].push((read_done, Sb::Short));
                }
            }
        }
        w.ready_at = next_ready;
        w.ready_reason = next_reason;
        w.until = c + 1;
        w.state = Stall::Wait;
        w.just_issued = true;
    }

    fn try_release_bar(&mut self, cta: usize, id: u64, c: u64) {
        let lat = self.spec.latency.bar as u64;
        let cta_ref = self.ctas[cta].as_mut().unwrap();
        let Some(b) = cta_ref.bars.get(&id) else {
            return;
        };
        if b.arrived < b.expected.min(cta_ref.live) {
            return;
        }
        cta_ref.bars.remove(&id);
        let slots = cta_ref.warp_slots.clone();
        let nsp = self.smsps.len();
        for ws in slots {
            if let Some(w) = self.warps[ws].as_mut() {
                if w.at_bar == Some(id) {
                    w.at_bar = None;
                    w.ready_at = c + lat;
                    w.ready_reason = Stall::Barrier;
                    w.until = c + 1;
                    let sp = &mut self.smsps[ws % nsp];
                    sp.next_eval = sp.next_eval.min(c + 1);
                }
            }
        }
    }

    fn exit_warp(&mut self, ws: usize, sp: usize, c: u64, done: &mut Vec<(u32, u64)>) {
        if let Some(tr) = &mut self.wtrace {
            tr.end(ws, c);
        }
        let w = self.warps[ws].as_mut().unwrap();
        w.exited = true;
        w.state = Stall::Drain;
        w.until = NEVER;
        let cta = w.cta;
        // An exited warp no longer competes for issue; its slot stays
        // allocated until the whole CTA retires.
        let nsp = self.smsps.len();
        let spx = &mut self.smsps[ws % nsp];
        spx.warps.retain(|&x| x != ws);
        if spx.greedy == Some(ws) {
            spx.greedy = None;
        }
        let cta_ref = self.ctas[cta].as_mut().unwrap();
        cta_ref.live -= 1;
        // Remaining warps may now satisfy a pending CTA barrier.
        let ids: Vec<u64> = cta_ref.bars.keys().copied().collect();
        for id in ids {
            self.try_release_bar(cta, id, c);
        }
        let cta_ref = self.ctas[cta].as_ref().unwrap();
        if cta_ref.live == 0 {
            let cta_ref = self.ctas[cta].take().unwrap();
            let nsp = self.smsps.len();
            for &s in &cta_ref.warp_slots {
                self.warps[s] = None;
                let spx = &mut self.smsps[s % nsp];
                spx.warps.retain(|&x| x != s);
                if spx.greedy == Some(s) {
                    spx.greedy = None;
                }
                if spx.warps.is_empty() {
                    spx.sched_state = None;
                }
            }
            self.used_ctas -= 1;
            self.used_warps -= cta_ref.warp_slots.len() as u32;
            self.used_regs -= cta_ref.regs;
            self.used_smem -= cta_ref.smem;
            done.push((cta_ref.kid, cta_ref.cta_id));
        }
        let _ = sp;
    }

    /// LSU path for loads/stores/atomics/ldmatrix/cp.async/TMEM.
    fn mem_op(
        &mut self,
        op: &DynOp,
        c: u64,
        spec: &GpuSpec,
        mem: &mut MemSystem,
        kid: u32,
        sp: usize,
    ) -> u64 {
        let inst = &op.inst;
        let m = inst.mem.unwrap_or(rsim_sass::MemInfo {
            space: MemSpace::Global,
            store: false,
            atomic: false,
            width: 4,
        });
        let active = op.active.count_ones().clamp(1, 32);
        match m.space {
            MemSpace::Shared => {
                let ways = if inst.class == OpClass::Ldsm {
                    if inst.meta.flags & flags::LDSM_X4 != 0 {
                        4
                    } else if inst.meta.flags & flags::LDSM_X2 != 0 {
                        2
                    } else {
                        1
                    }
                } else {
                    bank_wavefronts(&op.addrs, m.width, active)
                };
                let t0 = self.mio.serve(c, 1);
                let t1 = self.smem_srv.serve(t0, ways as u64);
                self.acc.get(kid).smem_wavefronts += ways as u64;
                if !m.store {
                    self.lds_wavefronts += ways as u64;
                }
                t1 + spec.latency.lds as u64
            }
            MemSpace::Const => c + spec.latency.ldc as u64,
            MemSpace::Tensor if spec.issue.tmem.ld_base > 0.0 => {
                let (bytes, slow) = tmem_shape(&inst.opcode);
                let t = &spec.issue.tmem;
                let (base, rate) = if m.store {
                    (t.st_base, t.st_rate_32x32b / slow)
                } else {
                    (t.ld_base, t.ld_rate_32x32b / slow)
                };
                // The port time in cycles of this sub-partition (a unit-rate
                // server fed with cycles).
                let busy = (bytes as f64 / rate).ceil() as u64;
                let srv = &mut self.tmem_srv[sp];
                let done = srv.serve(c, busy);
                // Latency counts the transfer once: base + bytes / rate after
                // the port became free for this access.
                done + base as u64
            }
            MemSpace::Tensor => {
                // TMEM <-> RF: 32 lanes x width bytes; ~16 B/lane/cycle (est).
                let bytes = (m.width as u64) * active as u64;
                let t0 = self.mio.serve(c, 1);
                t0 + spec.latency.lds as u64 + bytes / 512
            }
            _ => {
                // Coalesce to 128 B lines with sector masks.
                let mut lines: Vec<(u64, u8)> = Vec::with_capacity(4);
                if op.addrs.is_empty() {
                    let bytes = (m.width * active) as u64;
                    let n = bytes.div_ceil(128).max(1);
                    for _ in 0..n {
                        self.stream_addr += 128;
                        lines.push((self.stream_addr, 0xF));
                    }
                } else {
                    for &a in &op.addrs {
                        let line = a & !127;
                        let mask = self.l1.sector_mask_of(a, m.width);
                        match lines.iter_mut().find(|x| x.0 == line) {
                            Some(x) => x.1 |= mask,
                            None => lines.push((line, mask)),
                        }
                    }
                }
                let t0 = self.mio.serve(c, lines.len() as u64);
                let l1_lat = spec.l1.hit_latency as u64;
                let mut done_t = t0 + l1_lat;
                let bypass_l1 = m.store
                    || m.atomic
                    || inst.meta.flags & flags::STRONG != 0
                    || inst.class == OpClass::AsyncCopy && inst.meta.flags & flags::BYPASS != 0;
                let st = self.acc.get(kid);
                for (line, mask) in lines {
                    let miss = if bypass_l1 {
                        mask
                    } else {
                        let a = self.l1.access(line, mask, false);
                        let hit_n = (mask & !a.miss_mask).count_ones() as u64;
                        st.l1_hits += hit_n;
                        st.l1_misses += a.miss_mask.count_ones() as u64;
                        a.miss_mask
                    };
                    if miss != 0 {
                        let r = mem.access_line(
                            self.id,
                            t0 + l1_lat / 2,
                            line,
                            miss,
                            m.store && !m.atomic,
                        );
                        st.l2_hits += r.l2_hit_sectors as u64;
                        st.l2_misses += r.l2_miss_sectors as u64;
                        st.dram_read_bytes += r.dram_bytes;
                        if m.store {
                            st.dram_write_bytes += miss.count_ones() as u64 * 32;
                        }
                        done_t = done_t.max(r.done + l1_lat / 2);
                    }
                }
                done_t
            }
        }
    }
}

/// Register-file reads `inst` makes in each bank: distinct 32-bit source
/// registers (a register named twice is read once), minus sources served by
/// the warp's reuse cache. Bank = register index mod `rf_banks`.
fn rf_bank_reads(spec: &GpuSpec, inst: &Inst, reuse: &[Reg; 4]) -> [u8; 8] {
    let mut per = [0u8; 8];
    let banks = (spec.issue.rf_banks as usize).min(8);
    if banks == 0 || inst.meta.rf_n == 0 {
        return per;
    }
    let mut seen = [NO_REG; 16];
    let mut ns = 0;
    for o in &inst.meta.rf[..inst.meta.rf_n as usize] {
        if spec.issue.reuse_cache
            && o.width == 1
            && (reuse[o.slot as usize] == o.reg
                || (spec.issue.reuse_any_slot && reuse.contains(&o.reg)))
        {
            continue;
        }
        for k in 0..o.width as u16 {
            let r = o.reg + k;
            if seen[..ns].contains(&r) {
                continue;
            }
            if ns < seen.len() {
                seen[ns] = r;
                ns += 1;
            }
            per[r as usize % banks] += 1;
        }
    }
    per
}

/// Bytes per warp moved by an `LDTM`/`STTM` and the shape's slowdown
/// relative to 32x32b: `LDTM[.16dp{64,128,256}bit][.xN]`.
fn tmem_shape(opcode: &str) -> (u64, f64) {
    let mut n = 1u64;
    let mut per = 128u64; // bytes per repeat per warp (32x32b, 16x64b)
    let mut slow = 1.0;
    for m in opcode.split('.').skip(1) {
        if let Some(x) = m.strip_prefix('x').and_then(|x| x.parse().ok()) {
            n = x;
        }
        match m {
            "16dp64bit" => slow = 2.0,
            "16dp128bit" => {
                per = 256;
                slow = 2.0
            }
            "16dp256bit" => {
                per = 512;
                slow = 4.0
            }
            _ => {}
        }
    }
    (n * per, slow)
}

/// Cycles `inst` holds its warp in dispatch before the stall count starts.
fn dispatch_cycles(spec: &GpuSpec, inst: &Inst) -> u64 {
    let d = &spec.issue.dispatch_cycles;
    if d.is_empty() {
        return 0;
    }
    d.get(&*inst.opcode)
        .or_else(|| d.get(inst.base_opcode()))
        .map(|x| x.round().max(0.0) as u64)
        .unwrap_or(0)
}

/// Shared-memory wavefronts for a warp access (32 banks x 4 B).
/// The per-PC record for `inst`, created on first touch.
fn pc_entry<'a>(
    m: &'a mut std::collections::BTreeMap<u64, PcStat>,
    role: Option<usize>,
    inst: &rsim_sass::Inst,
) -> &'a mut PcStat {
    let key = ((role.unwrap_or(0) as u64) << 32) | inst.pc as u64;
    m.entry(key).or_insert_with(|| PcStat {
        text: if inst.operands.is_empty() {
            inst.opcode.to_string()
        } else {
            format!("{} {}", inst.opcode, inst.operands)
        },
        class: format!("{:?}", inst.class),
        ..Default::default()
    })
}

pub fn bank_wavefronts(addrs: &[u64], width: u32, active: u32) -> u32 {
    if addrs.is_empty() {
        // Unknown pattern: assume conflict-free.
        return ((width * active) as u64).div_ceil(128).max(1) as u32;
    }
    let mut per_bank: [Vec<u64>; 32] = Default::default();
    for &a in addrs {
        let words = width.div_ceil(4).max(1) as u64;
        for k in 0..words {
            let word = (a >> 2) + k;
            let b = (word % 32) as usize;
            if !per_bank[b].contains(&word) {
                per_bank[b].push(word);
            }
        }
    }
    let ways = per_bank
        .iter()
        .map(|v| v.len() as u32)
        .max()
        .unwrap_or(1)
        .max(1);
    // Wide accesses need at least width*32/128 wavefronts.
    ways.max(((width * active) as u64).div_ceil(128) as u32)
}

/// `BAR.SYNC 0x1, 0x100` => (1, Some(8 warps)).
#[cfg(test)]
fn bar_operands(ops: &str) -> (u32, Option<u32>) {
    let parts: Vec<&str> = ops.split(',').map(|s| s.trim()).collect();
    let imm = |s: &str| {
        s.strip_prefix("0x")
            .and_then(|h| u32::from_str_radix(h, 16).ok())
    };
    let id = parts.first().and_then(|s| imm(s)).unwrap_or(0);
    let threads = parts.get(1).and_then(|s| imm(s));
    (id, threads.map(|t| t.div_ceil(32)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn banks() {
        // Unit-stride 4 B: conflict free.
        let a: Vec<u64> = (0..32).map(|i| i * 4).collect();
        assert_eq!(bank_wavefronts(&a, 4, 32), 1);
        // Stride 128 B: all lanes hit bank 0 => 32-way.
        let a: Vec<u64> = (0..32).map(|i| i * 128).collect();
        assert_eq!(bank_wavefronts(&a, 4, 32), 32);
        // Broadcast: same word => 1.
        let a: Vec<u64> = vec![64; 32];
        assert_eq!(bank_wavefronts(&a, 4, 32), 1);
        // 16 B per lane, contiguous: 4 wavefronts.
        let a: Vec<u64> = (0..32).map(|i| i * 16).collect();
        assert_eq!(bank_wavefronts(&a, 16, 32), 4);
    }
    #[test]
    fn bar_ops() {
        assert_eq!(bar_operands("0x1, 0x100"), (1, Some(8)));
        assert_eq!(bar_operands(""), (0, None));
    }
}
