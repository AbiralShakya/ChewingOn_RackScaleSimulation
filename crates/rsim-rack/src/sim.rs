//! The rack simulator: one logical process per GPU plus one for the fabric,
//! advanced in parallel by the conservative PDES kernel.

use crate::report::{build_report, Report};
use crate::workload::{resolve_group, KernelSpec, Op, Workload};
use rsim_core::pdes::{self, Envelope, LogicalProcess, PdesStats};
use rsim_core::queue::EventQueue;
use rsim_core::trace::TraceSink;
use rsim_core::{Clock, Time};
use rsim_gpu::{synth, Gpu, GpuConfig, KernelDone, KernelLaunch};
use rsim_hw::{GpuSpec, System};
use rsim_net::{CollKind, Collective, NetConfig, NetEngine};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub gpu: GpuConfig,
    pub net: NetConfig,
    /// Override the per-kernel launch gap (ns); `None` = GPU preset value.
    /// Use a small value to model CUDA-graph replay.
    pub launch_gap_ns: Option<f64>,
    /// Write a Perfetto trace.
    pub trace: bool,
    /// Stop after this much simulated time (ns).
    pub horizon_ns: Option<f64>,
    /// Directory relative paths in the workload resolve against.
    pub base_dir: Option<PathBuf>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            gpu: GpuConfig::default(),
            net: NetConfig::default(),
            launch_gap_ns: None,
            trace: true,
            horizon_ns: None,
            base_dir: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error("workload has {0} ranks but the system has {1} GPUs")]
    WorldMismatch(usize, usize),
    #[error("workload: {0}")]
    Workload(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// A flattened op, ready to execute.
#[derive(Clone, Debug)]
pub enum FlatOp {
    Kernel {
        label: String,
        k: Arc<KernelLaunch>,
    },
    Fixed {
        label: String,
        ns: f64,
        sms: u32,
    },
    Coll {
        label: String,
        uid: u64,
        desc: Arc<Collective>,
        idx: usize,
        sms: u32,
    },
    Record(u32),
    Wait(u32),
    Delay(f64),
}

#[derive(Clone, Debug)]
pub enum Msg {
    Post {
        uid: u64,
        desc: Arc<Collective>,
        idx: usize,
    },
    Done {
        uid: u64,
    },
}

/// Everything a rank did, for the report.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Span {
    pub stream: usize,
    pub label: String,
    pub kind: &'static str,
    pub start: Time,
    pub end: Time,
}

#[derive(Clone, Copy, Debug)]
enum Local {
    StreamNext(usize),
    FixedDone(usize),
}

struct StreamState {
    ops: Vec<FlatOp>,
    pc: usize,
    busy: bool,
    blocked_on: Option<u32>,
    op_start: Time,
}

pub struct RankLp {
    pub rank: usize,
    pub gpu: Gpu,
    clock: Clock,
    streams: Vec<StreamState>,
    q: EventQueue<Local>,
    events: HashMap<u32, Vec<Time>>,
    waits: HashMap<(usize, u32), usize>,
    coll_stream: HashMap<u64, usize>,
    net_lp: usize,
    lookahead: Time,
    launch_gap: Time,
    pub spans: Vec<Span>,
    pub kernels: Vec<KernelDone>,
    pub finish: Time,
}

impl RankLp {
    fn ps(&self, cycle: u64) -> Time {
        self.clock.cycle_to_time(cycle)
    }
    fn cyc(&self, t: Time) -> u64 {
        self.clock.time_to_cycle_ceil(t)
    }

    fn all_done(&self) -> bool {
        self.streams.iter().all(|s| s.pc >= s.ops.len() && !s.busy)
    }

    /// Start the next op of stream `s` at time `t` (if possible).
    fn kick(&mut self, s: usize, t: Time, out: &mut Vec<Envelope<Msg>>) {
        loop {
            let st = &mut self.streams[s];
            if st.busy || st.pc >= st.ops.len() {
                return;
            }
            let op = st.ops[st.pc].clone();
            match op {
                FlatOp::Kernel { k, .. } => {
                    let start = t + self.launch_gap;
                    let c = self.cyc(start);
                    let st = &mut self.streams[s];
                    st.busy = true;
                    st.op_start = start;
                    self.gpu.submit(k, c, ((s as u64) << 32) | st.pc as u64);
                    return;
                }
                FlatOp::Fixed { ns, sms, .. } => {
                    let sms = sms.min(self.gpu.spec.sm_count);
                    self.gpu.reserve_sms(sms);
                    let st = &mut self.streams[s];
                    st.busy = true;
                    st.op_start = t;
                    self.q.push(t + (ns * 1000.0) as Time, Local::FixedDone(s));
                    return;
                }
                FlatOp::Coll {
                    uid,
                    desc,
                    idx,
                    sms,
                    ..
                } => {
                    self.gpu.reserve_sms(sms);
                    let st = &mut self.streams[s];
                    st.busy = true;
                    st.op_start = t;
                    self.coll_stream.insert(uid, s);
                    out.push(Envelope {
                        dst: self.net_lp,
                        at: t + self.lookahead,
                        msg: Msg::Post { uid, desc, idx },
                    });
                    return;
                }
                FlatOp::Record(e) => {
                    self.events.entry(e).or_default().push(t);
                    st.pc += 1;
                    // Wake streams blocked on this event.
                    for o in 0..self.streams.len() {
                        if self.streams[o].blocked_on == Some(e) {
                            self.streams[o].blocked_on = None;
                            self.q.push(t, Local::StreamNext(o));
                        }
                    }
                }
                FlatOp::Wait(e) => {
                    let n = *self.waits.get(&(s, e)).unwrap_or(&0);
                    match self.events.get(&e).and_then(|v| v.get(n)).copied() {
                        Some(rt) => {
                            self.waits.insert((s, e), n + 1);
                            self.streams[s].pc += 1;
                            if rt > t {
                                self.q.push(rt, Local::StreamNext(s));
                                return;
                            }
                        }
                        None => {
                            self.streams[s].blocked_on = Some(e);
                            return;
                        }
                    }
                }
                FlatOp::Delay(ns) => {
                    // A busy period on this stream ending at t + ns.
                    st.busy = true;
                    st.op_start = t;
                    self.q.push(t + (ns * 1000.0) as Time, Local::FixedDone(s));
                    return;
                }
            }
        }
    }

    fn finish_op(&mut self, s: usize, end: Time, kind: &'static str) {
        let st = &mut self.streams[s];
        let label = match &st.ops[st.pc] {
            FlatOp::Kernel { label, .. }
            | FlatOp::Fixed { label, .. }
            | FlatOp::Coll { label, .. } => label.clone(),
            FlatOp::Delay(_) => "delay".into(),
            _ => String::new(),
        };
        self.spans.push(Span {
            stream: s,
            label,
            kind,
            start: st.op_start,
            end,
        });
        st.pc += 1;
        st.busy = false;
        self.finish = self.finish.max(end);
        self.q.push(end, Local::StreamNext(s));
    }
}

impl LogicalProcess for RankLp {
    type Msg = Msg;

    fn advance(&mut self, until: Time, out: &mut Vec<Envelope<Msg>>) {
        let mut done = Vec::new();
        loop {
            let tl = self.q.peek_time().unwrap_or(Time::MAX);
            let target = tl.min(until);
            let tc = self.cyc(target);
            self.gpu.advance_to_completion(tc, &mut done);
            if !done.is_empty() {
                for d in done.drain(..) {
                    let s = (d.tag >> 32) as usize;
                    let end = self.ps(d.end_cycle);
                    self.finish_op(s, end, "kernel");
                    self.kernels.push(d);
                }
                continue;
            }
            if tl >= until {
                break;
            }
            let (t, ev) = self.q.pop().unwrap();
            match ev {
                Local::StreamNext(s) => self.kick(s, t, out),
                Local::FixedDone(s) => {
                    let kind = match self.streams[s].ops[self.streams[s].pc] {
                        FlatOp::Fixed { sms, .. } => {
                            self.gpu.release_sms(sms.min(self.gpu.spec.sm_count));
                            "fixed"
                        }
                        _ => "delay",
                    };
                    self.finish_op(s, t, kind);
                }
            }
        }
    }

    fn deliver(&mut self, at: Time, msg: Msg) {
        if let Msg::Done { uid } = msg {
            let s = self
                .coll_stream
                .remove(&uid)
                .expect("collective completion for unknown op");
            if let FlatOp::Coll { sms, .. } = self.streams[s].ops[self.streams[s].pc] {
                self.gpu.release_sms(sms);
            }
            self.finish_op(s, at, "collective");
        }
    }

    fn next_event_time(&self) -> Option<Time> {
        let l = self.q.peek_time();
        let g = self.gpu.next_event().map(|c| self.ps(c));
        match (l, g) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn done(&self) -> bool {
        self.all_done() && self.q.is_empty() && self.gpu.idle()
    }
}

pub struct NetLp {
    pub engine: NetEngine,
    pending: Vec<(Time, u64, Arc<Collective>, usize)>,
}

impl LogicalProcess for NetLp {
    type Msg = Msg;
    fn advance(&mut self, until: Time, out: &mut Vec<Envelope<Msg>>) {
        // Posts are applied in timestamp order before advancing.
        self.pending.sort_by_key(|p| (p.0, p.1, p.3));
        for (at, uid, desc, idx) in self.pending.drain(..) {
            self.engine.post(uid, &desc, idx, at);
        }
        let mut evs = Vec::new();
        self.engine.advance(until, &mut evs);
        for e in evs {
            out.push(Envelope {
                dst: e.gpu,
                at: e.at.max(until),
                msg: Msg::Done { uid: e.coll },
            });
        }
    }
    fn deliver(&mut self, at: Time, msg: Msg) {
        if let Msg::Post { uid, desc, idx } = msg {
            self.pending.push((at, uid, desc, idx));
        }
    }
    fn next_event_time(&self) -> Option<Time> {
        let p = self.pending.iter().map(|x| x.0).min();
        match (p, self.engine.next_event_time()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }
    fn done(&self) -> bool {
        self.pending.is_empty() && self.engine.idle()
    }
}

pub enum Lp {
    Rank(Box<RankLp>),
    Net(Box<NetLp>),
}

impl LogicalProcess for Lp {
    type Msg = Msg;
    fn advance(&mut self, until: Time, out: &mut Vec<Envelope<Msg>>) {
        match self {
            Lp::Rank(r) => r.advance(until, out),
            Lp::Net(n) => n.advance(until, out),
        }
    }
    fn deliver(&mut self, at: Time, msg: Msg) {
        match self {
            Lp::Rank(r) => r.deliver(at, msg),
            Lp::Net(n) => n.deliver(at, msg),
        }
    }
    fn next_event_time(&self) -> Option<Time> {
        match self {
            Lp::Rank(r) => r.next_event_time(),
            Lp::Net(n) => n.next_event_time(),
        }
    }
    fn done(&self) -> bool {
        match self {
            Lp::Rank(r) => r.done(),
            Lp::Net(n) => n.done(),
        }
    }
}

/// Builds kernels once per (GPU type, spec) and shares them across ranks.
struct KernelCache {
    map: Mutex<HashMap<(String, String), Arc<KernelLaunch>>>,
    base: PathBuf,
}

impl KernelCache {
    fn get(
        &self,
        g: &GpuSpec,
        spec: &KernelSpec,
        lib: &HashMap<String, KernelSpec>,
    ) -> Result<Arc<KernelLaunch>, SimError> {
        let spec = match spec {
            KernelSpec::Ref { name } => lib
                .get(name)
                .ok_or_else(|| SimError::Workload(format!("unknown kernel ref `{name}`")))?,
            s => s,
        };
        let key = (g.name.clone(), serde_json::to_string(spec).unwrap());
        if let Some(k) = self.map.lock().unwrap().get(&key) {
            return Ok(k.clone());
        }
        let k = Arc::new(self.build(g, spec)?);
        self.map.lock().unwrap().insert(key, k.clone());
        Ok(k)
    }

    fn path(&self, p: &str) -> PathBuf {
        let p = Path::new(p);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.base.join(p)
        }
    }

    fn build(&self, g: &GpuSpec, spec: &KernelSpec) -> Result<KernelLaunch, SimError> {
        Ok(match spec {
            KernelSpec::Gemm { m, n, k, dtype } => synth::gemm(g, *m, *n, *k, dtype),
            KernelSpec::GroupedGemm { ms, n, k, dtype } => {
                synth::grouped_gemm(g, ms, *n, *k, dtype)
            }
            KernelSpec::Attention {
                batch,
                heads_q,
                heads_kv,
                sq,
                skv,
                d,
                causal,
            } => synth::attention(g, *batch, *heads_q, *heads_kv, *sq, *skv, *d, *causal),
            KernelSpec::Elementwise {
                n,
                inputs,
                elem_bytes,
                flops_per_elem,
            } => synth::elementwise("elementwise", *n, *inputs, *elem_bytes, *flops_per_elem),
            KernelSpec::RowNorm {
                rows,
                cols,
                elem_bytes,
            } => synth::row_norm("row_norm", *rows, *cols, *elem_bytes),
            KernelSpec::Sass {
                path,
                kernel,
                grid,
                block,
                regs,
                smem,
                branch_profile,
            } => {
                let src = std::fs::read_to_string(self.path(path))?;
                let progs = rsim_sass::parse_listing(&src)
                    .map_err(|e| SimError::Workload(format!("{path}: {e}")))?;
                let prog = match kernel {
                    Some(name) => progs
                        .into_iter()
                        .find(|p| p.name.contains(name.as_str()))
                        .ok_or_else(|| {
                            SimError::Workload(format!("kernel `{name}` not in {path}"))
                        })?,
                    None => progs.into_iter().next().unwrap(),
                };
                let name = prog.name.clone();
                KernelLaunch::from_program(
                    &name,
                    prog,
                    *grid,
                    *block,
                    *regs,
                    *smem,
                    branch_profile.clone().unwrap_or_default(),
                )
            }
            KernelSpec::Trace { path } => {
                let src = std::fs::read_to_string(self.path(path))?;
                let t = rsim_sass::accelsim::parse_trace(&src)
                    .map_err(|e| SimError::Workload(format!("{path}: {e}")))?;
                KernelLaunch::from_trace(t)
            }
            KernelSpec::Fixed { .. } | KernelSpec::Ref { .. } => unreachable!(),
        })
    }
}

struct Flattener<'a> {
    world: usize,
    comms: HashMap<Vec<usize>, u32>,
    seq: HashMap<(usize, u32), u32>,
    events: HashMap<String, u32>,
    cache: &'a KernelCache,
    lib: &'a HashMap<String, KernelSpec>,
}

impl Flattener<'_> {
    fn comm_id(&mut self, key: Vec<usize>) -> u32 {
        let n = self.comms.len() as u32;
        *self.comms.entry(key).or_insert(n)
    }

    fn next_uid(&mut self, rank: usize, comm: u32) -> u64 {
        let s = self.seq.entry((rank, comm)).or_insert(0);
        let uid = ((comm as u64) << 32) | *s as u64;
        *s += 1;
        uid
    }

    fn flatten(
        &mut self,
        rank: usize,
        g: &GpuSpec,
        ops: &[Op],
        out: &mut Vec<FlatOp>,
    ) -> Result<(), SimError> {
        for op in ops {
            match op {
                Op::Loop { times, body } => {
                    for _ in 0..*times {
                        self.flatten(rank, g, body, out)?;
                    }
                }
                Op::Kernel { name, kernel } => match kernel {
                    KernelSpec::Fixed { ns, sms } => out.push(FlatOp::Fixed {
                        label: name.clone().unwrap_or_else(|| "fixed".into()),
                        ns: *ns,
                        sms: *sms,
                    }),
                    spec => {
                        let k = self.cache.get(g, spec, self.lib)?;
                        out.push(FlatOp::Kernel {
                            label: name.clone().unwrap_or_else(|| k.name.clone()),
                            k,
                        });
                    }
                },
                Op::Collective {
                    name,
                    kind,
                    bytes,
                    group,
                    algo,
                    sms,
                } => {
                    let ranks =
                        resolve_group(group, rank, self.world).map_err(SimError::Workload)?;
                    let idx = ranks.iter().position(|&r| r == rank).ok_or_else(|| {
                        SimError::Workload(format!("rank {rank} not in its group {ranks:?}"))
                    })?;
                    let mut key = ranks.clone();
                    key.push(usize::MAX - *kind as usize);
                    let comm = self.comm_id(key);
                    let uid = self.next_uid(rank, comm);
                    let desc = Arc::new(Collective {
                        kind: *kind,
                        bytes: *bytes,
                        ranks,
                        algo: *algo,
                    });
                    out.push(FlatOp::Coll {
                        label: name.clone().unwrap_or_else(|| format!("{kind:?} {bytes}B")),
                        uid,
                        desc,
                        idx,
                        sms: *sms,
                    });
                }
                Op::Send {
                    bytes,
                    peer,
                    relative,
                    sms,
                }
                | Op::Recv {
                    bytes,
                    peer,
                    relative,
                    sms,
                } => {
                    let p = if *relative {
                        (rank as i64 + peer).rem_euclid(self.world as i64) as usize
                    } else {
                        *peer as usize
                    };
                    let is_send = matches!(op, Op::Send { .. });
                    let (src, dst) = if is_send { (rank, p) } else { (p, rank) };
                    let ranks = vec![src, dst];
                    let mut key = ranks.clone();
                    key.push(usize::MAX - 100);
                    let comm = self.comm_id(key);
                    let uid = self.next_uid(rank, comm);
                    out.push(FlatOp::Coll {
                        label: format!(
                            "{} {} {bytes}B",
                            if is_send { "send to" } else { "recv from" },
                            p
                        ),
                        uid,
                        desc: Arc::new(Collective {
                            kind: CollKind::SendRecv,
                            bytes: *bytes,
                            ranks,
                            algo: rsim_net::Algo::Direct,
                        }),
                        idx: if is_send { 0 } else { 1 },
                        sms: *sms,
                    });
                }
                Op::Record { event } => {
                    let n = self.events.len() as u32;
                    out.push(FlatOp::Record(
                        *self.events.entry(event.clone()).or_insert(n),
                    ));
                }
                Op::Wait { event } => {
                    let n = self.events.len() as u32;
                    out.push(FlatOp::Wait(*self.events.entry(event.clone()).or_insert(n)));
                }
                Op::Delay { ns } => out.push(FlatOp::Delay(*ns)),
            }
        }
        Ok(())
    }
}

pub struct Simulation {
    pub system: System,
    pub workload: Workload,
    pub cfg: RunConfig,
}

pub struct Outcome {
    pub report: Report,
    pub trace: TraceSink,
}

impl Simulation {
    pub fn new(system: System, workload: Workload, cfg: RunConfig) -> Self {
        Self {
            system,
            workload,
            cfg,
        }
    }

    pub fn run(&self) -> Result<Outcome, SimError> {
        let wall = std::time::Instant::now();
        let world = self.workload.world();
        if world > self.system.num_gpus() || world == 0 {
            return Err(SimError::WorldMismatch(world, self.system.num_gpus()));
        }
        let cache = KernelCache {
            map: Mutex::new(HashMap::new()),
            base: self.cfg.base_dir.clone().unwrap_or_else(|| ".".into()),
        };
        let mut fl = Flattener {
            world,
            comms: HashMap::new(),
            seq: HashMap::new(),
            events: HashMap::new(),
            cache: &cache,
            lib: &self.workload.kernels,
        };
        let net = NetEngine::new(&self.system, self.cfg.net.clone(), self.cfg.trace);
        let lookahead = net.lookahead();
        let mut lps: Vec<Lp> = Vec::with_capacity(world + 1);
        for rank in 0..world {
            let spec = self.system.gpus[rank].clone();
            let prog = self.workload.program(rank);
            // Event names are scoped to the rank (shared by its streams).
            fl.events.clear();
            let mut final_streams = Vec::new();
            for ops in &prog.streams {
                let mut flat = Vec::new();
                fl.flatten(rank, &spec, ops, &mut flat)?;
                final_streams.push(flat);
            }
            let clock = Clock::from_mhz(spec.sm_clock_mhz);
            let gap = self.cfg.launch_gap_ns.unwrap_or(spec.launch_overhead_ns);
            let mut gpu = Gpu::new(rank, spec, self.cfg.gpu.clone(), self.cfg.trace);
            for s in 0..final_streams.len() {
                gpu.trace
                    .name_thread(rank as u64 + 1, s as u64 + 1, &format!("stream {s}"));
            }
            let mut lp = RankLp {
                rank,
                gpu,
                clock,
                streams: final_streams
                    .into_iter()
                    .map(|ops| StreamState {
                        ops,
                        pc: 0,
                        busy: false,
                        blocked_on: None,
                        op_start: 0,
                    })
                    .collect(),
                q: EventQueue::new(),
                events: HashMap::new(),
                waits: HashMap::new(),
                coll_stream: HashMap::new(),
                net_lp: world,
                lookahead,
                launch_gap: (gap * 1000.0) as Time,
                spans: vec![],
                kernels: vec![],
                finish: 0,
            };
            for s in 0..lp.streams.len() {
                lp.q.push(0, Local::StreamNext(s));
            }
            lps.push(Lp::Rank(Box::new(lp)));
        }
        lps.push(Lp::Net(Box::new(NetLp {
            engine: net,
            pending: vec![],
        })));
        let horizon = self
            .cfg
            .horizon_ns
            .map(|h| (h * 1000.0) as Time)
            .unwrap_or(Time::MAX);
        let mut st = PdesStats::default();
        let end = pdes::run(&mut lps, lookahead, horizon, &mut st);
        let wall_s = wall.elapsed().as_secs_f64();
        Ok(build_report(
            &self.system,
            &self.workload,
            lps,
            end,
            st,
            lookahead,
            wall_s,
            self.cfg.trace,
        ))
    }
}
