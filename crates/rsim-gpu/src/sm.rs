//! One streaming multiprocessor: 4 sub-partitions (SMSPs), warp slots,
//! CTA slots, MIO/LSU, L1, shared memory banks, tensor engine, TMA unit and
//! barrier state.

use crate::kernel::{BarOp, InstStream, KernelLaunch, KernelSource};
use crate::mem::{Cache, MemSystem, Server};
use crate::pipes::{mma_dtype, mma_flops, mma_shape, timing, Pipe, Sb, NUM_PIPES};
use crate::stats::{AccMap, Stall};
use rsim_hw::GpuSpec;
use rsim_sass::inst::flags;
use rsim_sass::walker::DynOp;
use rsim_sass::{MemSpace, OpClass, Reg};
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
}

struct MBar {
    count: u32,
    arrivals: u32,
    max_t: u64,
    /// Completion time of every produced phase, in order.
    phases: Vec<(u64, Stall)>,
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
    bars: HashMap<u32, CtaBar>,
    mbars: HashMap<u32, MBar>,
    tensor_tail: u64,
    async_tail: u64,
    regs: u32,
    smem: u32,
}

struct Warp {
    cta: usize,
    kid: u32,
    role: Option<usize>,
    stream: Box<dyn InstStream>,
    next: Option<DynOp>,
    ready_at: u64,
    ready_reason: Stall,
    pending_regs: Vec<(Reg, u64, Sb)>,
    sb: [Vec<(u64, Sb)>; 6],
    async_groups: Vec<u64>,
    open_group: u64,
    wg_pending: Vec<u64>,
    mbar_idx: HashMap<u32, usize>,
    at_bar: Option<u32>,
    exited: bool,
    // Cached evaluation.
    until: u64,
    state: Stall,
    age: u64,
}

#[derive(Default)]
struct Smsp {
    warps: Vec<usize>,
    pipe_free: [u64; NUM_PIPES],
    tensor_tail: u64,
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
    tma: Server,
    tensor_tail: u64,
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
            smem_srv: Server::new(1.0),
            tma: Server::new(spec.tma.bytes_per_cycle),
            tensor_tail: 0,
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
                        max_t: 0,
                        phases: vec![(at, Stall::AsyncWait); init.initial_phases as usize],
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
                kid,
                role,
                stream,
                next: None,
                ready_at: at,
                ready_reason: Stall::Wait,
                pending_regs: Vec::new(),
                sb: Default::default(),
                async_groups: Vec::new(),
                open_group: 0,
                wg_pending: Vec::new(),
                mbar_idx: HashMap::new(),
                at_bar: None,
                exited: false,
                until: 0,
                state: Stall::Wait,
                age: self.age,
            });
            let sp = ws % nsp;
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
    pub fn step(&mut self, c: u64, mem: &mut MemSystem, done: &mut Vec<(u32, u64)>) {
        for sp in 0..self.smsps.len() {
            if self.smsps[sp].next_eval <= c {
                self.step_smsp(sp, c, mem, done);
            }
        }
    }

    fn account(&mut self, sp: usize, c: u64) {
        let s = &mut self.smsps[sp];
        let dt = c.saturating_sub(s.last_eval);
        if dt == 0 {
            return;
        }
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
            if let Some(w) = &self.warps[ws] {
                self.acc.get(w.kid).warp.add(w.state, dt);
            }
        }
    }

    fn step_smsp(&mut self, sp: usize, c: u64, mem: &mut MemSystem, done: &mut Vec<(u32, u64)>) {
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
                let (st, until) = self.evaluate(ws, sp, c);
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
                Policy::Gto => {
                    let g = self.smsps[sp].greedy;
                    if g.map(|g| eligible.contains(&g)).unwrap_or(false) {
                        g
                    } else {
                        eligible
                            .iter()
                            .copied()
                            .min_by_key(|&ws| self.warps[ws].as_ref().unwrap().age)
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
            self.issue(ws, sp, c, mem, done);
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
    fn evaluate(&mut self, ws: usize, sp: usize, c: u64) -> (Stall, u64) {
        let spec = self.spec.clone();
        let use_ctrl = self.use_ctrl;
        let mio_free = self.mio.free_at();
        let tensor_tail = self.tensor_tail;
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
        let cta_idx = w.cta;
        let Some(op) = Self::peek(w) else {
            // Implicit exit.
            return (Stall::Selected, c);
        };
        let inst = op.inst.clone();
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
            OpClass::DepBar if inst.base_opcode() == "DEPBAR" => {
                let n = inst.meta.imm_last.unwrap_or(0) as usize;
                if use_ctrl && inst.ctrl.is_some() {
                    let b = inst.meta.sb.unwrap_or(0) as usize;
                    let mut pend: Vec<u64> =
                        w.sb[b].iter().map(|x| x.0).filter(|&t| t > c).collect();
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
            OpClass::AsyncBarrier if inst.meta.flags & flags::MBAR_WAIT != 0 => {
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
        let t = timing(&spec, &inst);
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
            p => {
                let f = sps.pipe_free[p as usize];
                if f > c {
                    return (Stall::MathPipeThrottle, f);
                }
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
        let c = self.ctas[cta].as_mut().unwrap();
        let m = c.mbars.entry(id).or_insert(MBar {
            count: 1,
            arrivals: 0,
            max_t: 0,
            phases: vec![],
        });
        m.arrivals += 1;
        m.max_t = m.max_t.max(t);
        if m.arrivals >= m.count {
            m.phases.push((m.max_t, producer));
            m.arrivals = 0;
            m.max_t = 0;
            let at = m.phases.last().unwrap().0;
            self.poke_cta(cta, at);
        }
    }

    fn issue(
        &mut self,
        ws: usize,
        sp: usize,
        c: u64,
        mem: &mut MemSystem,
        done: &mut Vec<(u32, u64)>,
    ) {
        let spec = self.spec.clone();
        let w = self.warps[ws].as_mut().unwrap();
        let Some(op) = w.next.take() else {
            self.exit_warp(ws, sp, c, done);
            return;
        };
        let inst = op.inst.clone();
        let cta = w.cta;
        let kid = w.kid;
        let role = w.role;
        let t = timing(&spec, &inst);
        self.issued_total += 1;
        {
            let st = self.acc.get(kid);
            st.insts_issued += 1;
            st.class_counts[inst.class as usize] += 1;
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
        if self.use_ctrl {
            if let Some(ctrl) = inst.ctrl {
                next_ready = c + (ctrl.stall.max(1) as u64);
            }
        }
        // Result time and scoreboard kind for destination registers.
        let mut result: Option<(u64, Sb)> = None;
        let mut read_done = c + spec.latency.store_read as u64;

        if t.pipe != Pipe::Mio && t.pipe != Pipe::Tensor {
            self.smsps[sp].pipe_free[t.pipe as usize] = c + t.ii as u64;
        }

        match inst.class {
            // A guarded EXIT that the stream resolved as taken ends the
            // stream instead (next fetch returns `None`).
            OpClass::Exit if inst.guard.is_none() => {
                self.exit_warp(ws, sp, c, done);
                return;
            }
            OpClass::Mem | OpClass::AsyncCopy | OpClass::Ldsm | OpClass::Tmem => {
                let r = self.mem_op(&op, c, mem, kid);
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
                    spec.tensor.flops_per_cycle(mma_dtype(&inst)) / spec.sub_partitions as f64;
                let flops = mma_flops(&inst, op.payload);
                let ii = (flops / rate).ceil().max(1.0) as u64;
                let sps = &mut self.smsps[sp];
                sps.pipe_free[Pipe::Tensor as usize] = c + ii;
                let hold = spec.tensor.hmma_fma_pipe_cycles as u64;
                if hold > 0 {
                    let f = &mut sps.pipe_free[Pipe::Fma as usize];
                    *f = (*f).max(c + hold);
                }
                result = Some((c + ii + t.lat as u64, Sb::Short));
                self.tensor_busy_total += ii;
                let st = self.acc.get(kid);
                st.tensor_busy_cycles += ii;
                st.tensor_flops += flops;
            }
            OpClass::MmaWarpgroup => {
                let rate =
                    spec.tensor.flops_per_cycle(mma_dtype(&inst)) / spec.sub_partitions as f64;
                let flops = mma_flops(&inst, op.payload);
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
                let k_ext = mma_shape(&inst).map(|s| s.2).unwrap_or(0);
                let rate = spec.tensor.mma_rate(mma_dtype(&inst), k_ext);
                let flops = mma_flops(&inst, op.payload);
                let dur = (flops / rate).ceil().max(1.0) as u64;
                let start = self.tensor_tail.max(c + t.lat as u64);
                self.tensor_tail = start + dur;
                // Visible to tcgen05.commit after the pipeline latency.
                let fin = start + dur + spec.tensor.tcgen05_latency as u64;
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
                let r = mem.bulk(self.id, start, addr, bytes, frac, store);
                let st = self.acc.get(kid);
                st.l2_hits += r.l2_hit_sectors as u64;
                st.l2_misses += r.l2_miss_sectors as u64;
                if store {
                    st.dram_write_bytes += bytes;
                } else {
                    st.dram_read_bytes += r.dram_bytes;
                }
                let fin = r.done;
                let cta_ref = self.ctas[cta].as_mut().unwrap();
                cta_ref.async_tail = cta_ref.async_tail.max(fin);
                if let Some(BarOp::Arrive(id)) = self.bar_op(cta, role, inst.pc) {
                    self.mbar_arrive(cta, id, fin, Stall::TmaWait);
                }
            }
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
                if inst.meta.flags & flags::ARV != 0 {
                    // Arrive-only: no wait.
                } else {
                    let (id, expected) = (
                        inst.meta.bar_id,
                        inst.meta.bar_threads.map(|t| t.div_ceil(32)),
                    );
                    let cta_ref = self.ctas[cta].as_mut().unwrap();
                    let exp = expected.unwrap_or(cta_ref.live);
                    let b = cta_ref.bars.entry(id).or_insert(CtaBar {
                        expected: exp,
                        arrived: 0,
                    });
                    b.expected = expected.unwrap_or(cta_ref.live);
                    b.arrived += 1;
                    self.warps[ws].as_mut().unwrap().at_bar = Some(id);
                    self.try_release_bar(cta, id, c);
                }
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
    }

    fn try_release_bar(&mut self, cta: usize, id: u32, c: u64) {
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
        let ids: Vec<u32> = cta_ref.bars.keys().copied().collect();
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
    fn mem_op(&mut self, op: &DynOp, c: u64, mem: &mut MemSystem, kid: u32) -> u64 {
        let inst = &op.inst;
        let m = inst.mem.unwrap_or(rsim_sass::MemInfo {
            space: MemSpace::Global,
            store: false,
            atomic: false,
            width: 4,
        });
        let active = op.active.count_ones().clamp(1, 32);
        let spec = self.spec.clone();
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
                t1 + spec.latency.lds as u64
            }
            MemSpace::Const => c + spec.latency.ldc as u64,
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

/// Shared-memory wavefronts for a warp access (32 banks x 4 B).
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
