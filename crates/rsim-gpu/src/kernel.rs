//! Kernel launches and where their instruction streams come from.

use rsim_sass::accelsim::TraceKernel;
use rsim_sass::path::PathStream;
use rsim_sass::walker::{Annotations, DynOp};
use rsim_sass::{BranchProfile, Inst, PathKernel, Program, Walker};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

/// mbarrier semantics attached to an instruction PC.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum BarOp {
    /// Arrive on barrier `id` when this instruction *completes* (TMA load
    /// complete_tx, tcgen05.commit, cp.async.mbarrier.arrive) or, for plain
    /// SYNCS.ARRIVE, when it issues.
    Arrive(u32),
    /// Block until the next phase of barrier `id` (per-warp phase counter).
    Wait(u32),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BarInit {
    /// Arrivals per phase.
    pub count: u32,
    /// Phases already complete at kernel start (e.g. "empty" slots of a
    /// multi-stage pipeline start out free).
    pub initial_phases: u32,
}

/// One warp-specialised role: a program run by a contiguous range of warps
/// of every CTA (producer warp, MMA warp, consumer warpgroups, ...).
#[derive(Clone, Debug)]
pub struct WarpRole {
    pub name: String,
    pub warps: Range<u32>,
    pub program: Arc<Program>,
    pub insts: Arc<Vec<Arc<Inst>>>,
    pub profile: Arc<BranchProfile>,
    pub ann: Arc<Annotations>,
    pub bar_ops: Arc<HashMap<u32, BarOp>>,
    /// pc -> L2 fetch fraction for multicast TMA (1/cluster_size).
    pub multicast: Arc<HashMap<u32, f64>>,
}

impl WarpRole {
    pub fn new(name: impl Into<String>, warps: Range<u32>, program: Program) -> Self {
        let insts = Arc::new(Walker::share(&program));
        Self {
            name: name.into(),
            warps,
            program: Arc::new(program),
            insts,
            profile: Arc::new(BranchProfile::default()),
            ann: Arc::new(Annotations::default()),
            bar_ops: Arc::new(HashMap::new()),
            multicast: Arc::new(HashMap::new()),
        }
    }
}

#[derive(Clone, Debug)]
pub enum KernelSource {
    /// Static SASS (parsed or synthesised) with per-role control flow.
    Static {
        roles: Vec<WarpRole>,
        barriers: HashMap<u32, BarInit>,
    },
    /// Dynamic per-warp trace (exact control flow and addresses).
    Trace(Arc<TraceKernel>),
    /// Per-warp paths recorded with NVBit, with mbarrier operands.
    Path(Arc<PathKernel>),
}

#[derive(Clone, Debug)]
pub struct KernelLaunch {
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub regs_per_thread: u32,
    pub smem_per_cta: u32,
    /// Cluster size (CTAs co-scheduled on a GPC); 1 = no clusters.
    pub cluster: u32,
    pub source: KernelSource,
    /// Analytical metadata (for reports): FLOPs and bytes the kernel represents.
    pub flops: f64,
    pub bytes: f64,
}

impl KernelLaunch {
    pub fn num_ctas(&self) -> u64 {
        self.grid.iter().map(|&x| x as u64).product()
    }
    pub fn threads_per_cta(&self) -> u32 {
        self.block.iter().product()
    }
    pub fn warps_per_cta(&self) -> u32 {
        self.threads_per_cta().div_ceil(32)
    }

    /// Whether the kernel allocates tensor memory (`tcgen05.alloc` =
    /// `UTCATOMSWS`): an FA4/CUTLASS-style prologue.
    pub fn allocs_tmem(&self) -> bool {
        let has =
            |insts: &[Arc<rsim_sass::Inst>]| insts.iter().any(|i| i.base_opcode() == "UTCATOMSWS");
        match &self.source {
            KernelSource::Static { roles, .. } => roles.iter().any(|r| has(&r.insts)),
            KernelSource::Path(p) => has(&p.insts),
            KernelSource::Trace(_) => false,
        }
    }

    pub fn from_trace(t: TraceKernel) -> Self {
        let h = t.header.clone();
        Self {
            name: h.name.clone(),
            grid: h.grid,
            block: h.block,
            regs_per_thread: h.nregs.max(16),
            smem_per_cta: h.shmem,
            cluster: 1,
            source: KernelSource::Trace(Arc::new(t)),
            flops: 0.0,
            bytes: 0.0,
        }
    }

    pub fn from_path(p: PathKernel) -> Self {
        let h = &p.header;
        Self {
            name: h.kernel.clone(),
            grid: h.grid,
            block: h.block,
            regs_per_thread: h.regs.max(16),
            smem_per_cta: h.smem,
            cluster: h.cluster.iter().product::<u32>().max(1),
            flops: h.flops,
            bytes: h.bytes,
            source: KernelSource::Path(Arc::new(p)),
        }
    }

    /// A single-role kernel from a parsed SASS program.
    pub fn from_program(
        name: &str,
        program: Program,
        grid: [u32; 3],
        block: [u32; 3],
        regs: u32,
        smem: u32,
        profile: BranchProfile,
    ) -> Self {
        let warps = block.iter().product::<u32>().div_ceil(32);
        let mut role = WarpRole::new("all", 0..warps, program);
        role.profile = Arc::new(profile);
        Self {
            name: name.into(),
            grid,
            block,
            regs_per_thread: regs,
            smem_per_cta: smem,
            cluster: 1,
            source: KernelSource::Static {
                roles: vec![role],
                barriers: HashMap::new(),
            },
            flops: 0.0,
            bytes: 0.0,
        }
    }

    pub fn stream_for(&self, cta: u64, warp: u32) -> (Box<dyn InstStream>, Option<usize>) {
        match &self.source {
            KernelSource::Static { roles, .. } => {
                let (ri, r) = roles
                    .iter()
                    .enumerate()
                    .find(|(_, r)| r.warps.contains(&warp))
                    .unwrap_or((0, &roles[0]));
                let w = Walker::new(
                    r.program.clone(),
                    r.insts.clone(),
                    r.profile.clone(),
                    r.ann.clone(),
                    cta,
                    (warp - r.warps.start.min(warp)) as u64,
                );
                (Box::new(w), Some(ri))
            }
            KernelSource::Path(p) => (Box::new(p.stream(cta, warp)), None),
            KernelSource::Trace(t) => {
                let ops = t
                    .warps
                    .get(&(cta as u32, warp))
                    .cloned()
                    .unwrap_or_default();
                (
                    Box::new(TraceStream {
                        ops: ops.into_iter(),
                    }),
                    None,
                )
            }
        }
    }
}

pub trait InstStream: Send {
    fn next_op(&mut self) -> Option<DynOp>;
}

impl InstStream for Walker {
    fn next_op(&mut self) -> Option<DynOp> {
        Walker::next_op(self)
    }
}

impl InstStream for PathStream {
    fn next_op(&mut self) -> Option<DynOp> {
        PathStream::next_op(self)
    }
}

struct TraceStream {
    ops: std::vec::IntoIter<rsim_sass::accelsim::DynInst>,
}

impl InstStream for TraceStream {
    fn next_op(&mut self) -> Option<DynOp> {
        self.ops.next().map(|d| DynOp {
            inst: d.inst,
            active: d.mask,
            addrs: d.addrs,
            payload: 0,
            mbar: None,
            run_bytes: 0,
        })
    }
}
