//! Synthetic SASS for state-space-model kernels and FlashAttention-2.
//!
//! The kernels mirror the open-source implementations, one launch each:
//!
//! * **Mamba-2 decode**: `selective_state_update` (state-spaces/mamba, Triton).
//!   One CTA per (`BLOCK_SIZE_M` rows of headdim, sequence, head); every
//!   thread loads its slice of the recurrent state, applies
//!   `h = exp(A dt) h + dt B x`, stores it back and reduces `y = h C` with
//!   warp shuffles. With `d_state = 128` a CTA moves only 2 KB of state, so
//!   the kernel is limited by memory-level parallelism, not HBM bandwidth.
//! * **Mamba-2 prefill (SSD)**: the five Triton kernels behind
//!   `mamba_chunk_scan_combined`: `chunk_cumsum`, `chunk_state` (per-chunk
//!   `x^T (B * decay)` GEMM), `state_passing` (a sequential scan over chunks),
//!   `bmm_chunk` (`C B^T` per chunk) and `chunk_scan` (the intra-chunk
//!   masked, decayed `(C B^T) x` GEMM plus the inter-chunk `C h` term).
//!   `tl.dot` is lowered to the architecture's MMA path ([`dot_path`]); the
//!   per-element decay `exp(dA_i - dA_j)` is explicit `MUFU.EX2` work.
//! * **causal-conv1d** (Dao-AILab/causal-conv1d, CUDA): the decode
//!   `causal_conv1d_update` kernel and the channel-last prefill kernel.
//! * **FlashAttention-2** (Ampere): `mma.sync` (HMMA) forward with `cp.async`
//!   K/V double buffering, used for both prefill and split-KV decode.
//!
//! Tile sizes and warp counts follow the kernels' own heuristics where they
//! are fixed (state update, conv1d, FA2) and the autotuned configurations
//! Nsight Compute recorded on B300 and A100 otherwise (see
//! `research/ssm_vs_attention`).

use crate::kernel::{BarInit, BarOp, KernelLaunch, KernelSource};
use crate::synth::{dtype, mma_opcode, region, Asm, Dtype};
use rsim_hw::{Arch, GpuSpec};
use rsim_sass::walker::AddrGen;
use std::collections::HashMap;

const ACC_BAR: u32 = 7;

/// How Triton lowers `tl.dot` on an architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DotPath {
    /// `mma.sync` (HMMA) from registers loaded with `ldmatrix`.
    Sync,
    /// Hopper `wgmma` (HGMMA) with operands in shared memory.
    Warpgroup,
    /// Blackwell `tcgen05.mma` (UTCHMMA) with operands in shared memory and
    /// the accumulator in tensor memory.
    Tcgen05,
}

pub fn dot_path(g: &GpuSpec) -> DotPath {
    match g.arch {
        Arch::Ampere => DotPath::Sync,
        Arch::Hopper => DotPath::Warpgroup,
        Arch::Blackwell => DotPath::Tcgen05,
    }
}

/// Bytes moved per CTA by one instruction stream, and where they live.
#[derive(Clone, Copy, Debug)]
pub struct Stream {
    pub bytes: u64,
    pub addr: AddrGen,
}

fn stream(bytes: u64, base: u64, cta_stride: i64, iter_stride: i64) -> Stream {
    Stream {
        bytes,
        addr: AddrGen {
            base,
            cta_stride,
            iter_stride,
            warp_stride: 512,
            lane_stride: 16,
            ..Default::default()
        },
    }
}

/// Bound a stream's addresses to the tensor it reads (`footprint` bytes), so
/// tiles shared between CTAs (e.g. one chunk's `C B^T` read by every head of
/// its group) are re-read from L2 rather than counted as new DRAM traffic.
fn within(mut st: Stream, footprint: u64) -> Stream {
    st.addr.footprint = footprint.max(st.bytes);
    st
}

/// A per-(chunk, head)-block operand shared by the `share` consecutive CTAs
/// of the output-tile grid dimension (`cta_div`), optionally repeating every
/// `period` blocks (`cta_mod`, e.g. one chunk's `C B^T` for every head).
fn blocked(
    bytes: u64,
    base: u64,
    block_bytes: u64,
    share: u64,
    period: u64,
    iter_stride: i64,
) -> Stream {
    Stream {
        bytes,
        addr: AddrGen {
            base,
            cta_stride: block_bytes as i64,
            cta_div: share,
            cta_mod: period,
            iter_stride,
            warp_stride: 512,
            lane_stride: 16,
            ..Default::default()
        },
    }
}

/// A Triton-style (non warp-specialised) tiled kernel: every warp runs the
/// same program, operands stream through a `stages`-deep `cp.async` ring,
/// and each k-iteration does `iter_alu` per thread plus one CTA-tile MMA.
#[derive(Clone, Debug)]
pub struct TileKernel {
    pub name: String,
    pub ctas: u64,
    pub warps: u32,
    pub regs: u32,
    pub stages: u32,
    pub k_iters: u64,
    /// Fraction of CTAs that exit immediately (causal tiles above the diagonal).
    pub early_exit: f64,
    /// Operands per k-iteration. With `direct` they are loaded straight into
    /// registers (`LDG`), and the iteration's ALU work waits for them;
    /// otherwise they go through the `cp.async` ring.
    pub iter_loads: Vec<Stream>,
    pub direct: bool,
    /// CTA tile `m x n x k` per k-iteration.
    pub mma: Option<(u64, u64, u64)>,
    pub path: DotPath,
    /// Per-thread ALU work per k-iteration: (SASS text, count).
    pub iter_alu: Vec<(&'static str, usize)>,
    /// Register-resident operand bytes per CTA written back to shared memory
    /// before an MMA that reads smem (rescaled/masked operands).
    pub restage_bytes: u64,
    pub pre_loads: Vec<Stream>,
    pub pre_alu: Vec<(&'static str, usize)>,
    pub epi_loads: Vec<Stream>,
    pub epi_alu: Vec<(&'static str, usize)>,
    pub out: Vec<Stream>,
    pub dt: Dtype,
    pub flops: f64,
    pub bytes: f64,
}

impl TileKernel {
    fn new(name: String, ctas: u64, warps: u32, path: DotPath) -> Self {
        Self {
            name,
            ctas,
            warps,
            regs: 128,
            stages: 3,
            k_iters: 1,
            early_exit: 0.0,
            iter_loads: vec![],
            direct: false,
            mma: None,
            path,
            iter_alu: vec![],
            restage_bytes: 0,
            pre_loads: vec![],
            pre_alu: vec![],
            epi_loads: vec![],
            epi_alu: vec![],
            out: vec![],
            dt: dtype("bf16"),
            flops: 0.0,
            bytes: 0.0,
        }
    }

    fn threads(&self) -> u64 {
        32 * self.warps as u64
    }

    /// Per-thread 16-byte instructions needed to move `bytes` per CTA.
    fn vec_insts(&self, bytes: u64) -> u64 {
        bytes.div_ceil(16 * self.threads()).max(1)
    }

    fn loads(&self, a: &mut Asm, s: &[Stream], op: &str, first_reg: u32) {
        let mut r = first_reg;
        for st in s {
            for i in 0..self.vec_insts(st.bytes) {
                let pc = a.op(&format!("{op} R{r}, desc[UR4][R2.64]"));
                a.addr(
                    pc,
                    AddrGen {
                        base: st.addr.base + i * 16 * self.threads(),
                        ..st.addr
                    },
                );
                r = if r >= first_reg + 12 {
                    first_reg
                } else {
                    r + 4
                };
            }
        }
    }

    fn async_loads(&self, a: &mut Asm) {
        for st in &self.iter_loads {
            for i in 0..self.vec_insts(st.bytes) {
                let pc = a.op("LDGSTS.E.BYPASS.128 [R5+0x0], desc[UR4][R2.64]");
                a.addr(
                    pc,
                    AddrGen {
                        base: st.addr.base + i * 16 * self.threads(),
                        ..st.addr
                    },
                );
            }
        }
        a.op("LDGDEPBAR");
    }

    fn alu(a: &mut Asm, ops: &[(&str, usize)]) {
        for (text, n) in ops {
            match *text {
                // Independent-accumulator HMMA runs (S = QK^T, O += PV).
                "@HMMA_S" => a.hmma(*n, 8, 12, "BF16"),
                "@HMMA_O" => a.hmma(*n, 60, 8, "BF16"),
                _ => a.ops(text, *n),
            }
        }
    }

    fn mma_body(&self, a: &mut Asm, m: u64, n: u64, k: u64) {
        let w = self.warps as u64;
        match self.path {
            DotPath::Sync => {
                // Warps tile the CTA tile as wm x wn; each loads its A rows and
                // B columns from smem with ldmatrix (512 B per LDSM.x4).
                let wm = (m / 16).clamp(1, w);
                let wn = (w / wm).max(1);
                let ldsm = ((m * k * wn + k * n * wm) * 2).div_ceil(512 * w).max(1);
                let hmma = (m * n * k).div_ceil(16 * 8 * 16 * w).max(1);
                a.ops("LDSM.16.M88.4 R8, [R6]", ldsm as usize);
                a.hmma(hmma as usize, 8, 12, self.dt.sass);
            }
            DotPath::Warpgroup => {
                let wgs = (w / 4).max(1);
                let rows = (m / wgs).max(64);
                let reps = (rows / 64) * (k / 16).max(1);
                a.op("WARPGROUP.ARRIVE");
                a.ops(
                    &mma_opcode(Arch::Hopper, 64, n.min(256), 16, self.dt),
                    reps as usize,
                );
                a.op("WARPGROUP.DEPBAR.LE gsb0, 0x0");
            }
            DotPath::Tcgen05 => {
                if self.restage_bytes > 0 {
                    a.ops(
                        "STS.128 [R7], R24",
                        self.vec_insts(self.restage_bytes) as usize,
                    );
                    a.op("FENCE.VIEW.ASYNC.S");
                    a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
                }
                let mm = if m >= 128 { 128 } else { 64 };
                let reps = (m / mm).max(1) * (k / 16).max(1);
                let nn = n.min(256);
                let reps = reps * n.div_ceil(nn);
                for _ in 0..reps {
                    let pc = a.op(&mma_opcode(Arch::Blackwell, mm, nn, 16, self.dt));
                    a.single_thread(pc);
                }
                let c = a.op("UTCBAR [UR24], URZ");
                a.single_thread(c);
                a.bar(c, BarOp::Arrive(ACC_BAR));
                let wt = a.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR24], R3");
                a.bar(wt, BarOp::Wait(ACC_BAR));
            }
        }
    }

    pub fn build(&self) -> KernelLaunch {
        let mut a = Asm::new();
        a.op("S2R R0, SR_TID.X");
        a.op("S2R R1, SR_CTAID.X");
        a.op("IMAD R2, R1, UR6, R0");
        a.op("ISETP.GE.AND P0, PT, R2, UR5, PT");
        a.exit_if(self.early_exit);
        self.loads(&mut a, &self.pre_loads, "LDG.E.128", 40);
        Self::alu(&mut a, &self.pre_alu);
        if !self.direct && !self.iter_loads.is_empty() {
            for _ in 0..(self.stages.saturating_sub(1)).min(self.k_iters as u32) {
                self.async_loads(&mut a);
            }
        }
        let top = a.here();
        if self.direct {
            self.loads(&mut a, &self.iter_loads, "LDG.E.128", 24);
        } else if !self.iter_loads.is_empty() {
            a.op(&format!(
                "DEPBAR.LE SB0, 0x{:x}",
                self.stages.saturating_sub(2)
            ));
            a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
            self.async_loads(&mut a);
        }
        Self::alu(&mut a, &self.iter_alu);
        if let Some((m, n, k)) = self.mma {
            self.mma_body(&mut a, m, n, k);
        }
        if !self.direct && !self.iter_loads.is_empty() {
            a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
        }
        if self.k_iters > 1 {
            a.loop_to(top, self.k_iters - 1);
        }
        if let (Some((m, n, _)), DotPath::Tcgen05) = (self.mma, self.path) {
            let cols_per_thread = (m * n).div_ceil(self.threads());
            a.ops(
                "LDTM.32x32b.x32 R24, tmem[UR12]",
                cols_per_thread.div_ceil(32).max(1) as usize,
            );
        }
        self.loads(&mut a, &self.epi_loads, "LDG.E.128", 44);
        Self::alu(&mut a, &self.epi_alu);
        for st in &self.out {
            let n_st = self.vec_insts(st.bytes);
            for i in 0..n_st {
                let pc = a.op("STG.E.128 desc[UR4][R2.64], R24");
                a.addr(
                    pc,
                    AddrGen {
                        base: st.addr.base + i * 16 * self.threads(),
                        ..st.addr
                    },
                );
            }
        }
        let mut barriers = HashMap::new();
        let mut smem = 1024u32;
        if !self.direct {
            let per_stage: u64 = self.iter_loads.iter().map(|s| s.bytes).sum();
            smem += (per_stage * self.stages as u64) as u32;
        }
        if self.path == DotPath::Tcgen05 && self.mma.is_some() {
            barriers.insert(
                ACC_BAR,
                BarInit {
                    count: 1,
                    initial_phases: 0,
                },
            );
            smem += self.restage_bytes as u32;
        }
        KernelLaunch {
            name: self.name.clone(),
            grid: [self.ctas as u32, 1, 1],
            block: [self.threads() as u32, 1, 1],
            regs_per_thread: self.regs,
            smem_per_cta: smem,
            cluster: 1,
            source: KernelSource::Static {
                roles: vec![a.role("all", 0..self.warps)],
                barriers,
            },
            flops: self.flops,
            bytes: self.bytes,
        }
    }
}

// --------------------------------------------------------------------- Mamba-2 decode

/// Triton-compiled SASS of `selective_state_update` (state-spaces/mamba
/// 2.3.2, `d_state` 128, `BLOCK_SIZE_M` 4, 4 warps), dumped with `cuobjdump`
/// from the Triton cache on B300 (sm_103a, Triton 3.8) and A100 (sm_80,
/// Triton 3.4). The kernel has no loops, so the simulator can run the real
/// instruction stream, control bits included; only addresses are attached.
const SSU_SM103: &str = include_str!("../sass/selective_state_update_sm103.sass");
const SSU_SM80: &str = include_str!("../sass/selective_state_update_sm80.sass");

fn state_update_real_sass(
    g: &GpuSpec,
    batch: u64,
    heads: u64,
    headdim: u64,
    d_state: u64,
) -> Option<KernelLaunch> {
    if d_state != 128 {
        return None;
    }
    let src = match g.arch {
        Arch::Blackwell => SSU_SM103,
        // Hopper has no dump of its own; this kernel uses no sm_90 features.
        Arch::Ampere | Arch::Hopper => SSU_SM80,
    };
    let prog = rsim_sass::parse_listing(src).ok()?.into_iter().next()?;
    let (block_m, threads) = (4u64, 128u64);
    let m_blocks = headdim.div_ceil(block_m);
    let ctas = m_blocks * batch * heads;
    let state_cta = (block_m * d_state * 2) as i64;
    let mut ann = rsim_sass::walker::Annotations::default();
    let mut small = 0u64;
    for inst in &prog.insts {
        let Some(m) = inst.mem else { continue };
        if m.space != rsim_sass::MemSpace::Global {
            continue;
        }
        let g = if m.width == 8 {
            // The recurrent state: 8 B (4 x bf16) per thread, 1 KB per CTA.
            AddrGen {
                base: region(40),
                cta_stride: state_cta,
                lane_stride: 8,
                warp_stride: 256,
                ..Default::default()
            }
        } else {
            // dt, dt_bias, A, B, C, D, x, z, out: small per-(sequence, head) vectors.
            small += 1;
            AddrGen {
                base: region(41 + small),
                cta_stride: 2 * block_m as i64,
                lane_stride: m.width as i64,
                footprint: (batch * heads * headdim * 4).max(4096),
                ..Default::default()
            }
        };
        ann.addr.insert(inst.pc, g);
    }
    let mut role = crate::kernel::WarpRole::new("all", 0..4, prog);
    role.ann = std::sync::Arc::new(ann);
    Some(KernelLaunch {
        name: format!("selective_state_update_b{batch}_h{heads}_p{headdim}_n{d_state}"),
        grid: [ctas as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: if g.arch == Arch::Blackwell { 27 } else { 32 },
        smem_per_cta: 512,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![role],
            barriers: HashMap::new(),
        },
        flops: (batch * heads * headdim * d_state) as f64 * 6.0,
        bytes: 2.0 * (batch * heads * headdim * d_state * 2) as f64,
    })
}

/// `selective_state_update` (Triton) for one decode token of `batch`
/// sequences. Uses the real compiled SASS when one matches the shape
/// ([`state_update_real_sass`]); otherwise a synthetic equivalent. State layout `(batch, heads, headdim, d_state)` in
/// `state_bytes` (2 = bf16, the Mamba2 default; 4 = fp32).
pub fn ssm_state_update(
    _g: &GpuSpec,
    batch: u64,
    heads: u64,
    headdim: u64,
    d_state: u64,
    groups: u64,
    state_bytes: f64,
) -> KernelLaunch {
    ssm_state_update_tiled(
        _g,
        batch,
        heads,
        headdim,
        d_state,
        groups,
        state_bytes,
        None,
    )
}

/// [`ssm_state_update`] with an optional `BLOCK_SIZE_M` override: `Some(m)`
/// skips the real SASS and builds the synthetic kernel with `m` headdim rows
/// per CTA (a what-if for a state update that moves more bytes per CTA).
#[allow(clippy::too_many_arguments)]
pub fn ssm_state_update_tiled(
    _g: &GpuSpec,
    batch: u64,
    heads: u64,
    headdim: u64,
    d_state: u64,
    groups: u64,
    state_bytes: f64,
    block_m_override: Option<u64>,
) -> KernelLaunch {
    if state_bytes == 2.0 && block_m_override.is_none() {
        if let Some(k) = state_update_real_sass(_g, batch, heads, headdim, d_state) {
            return k;
        }
    }
    // The kernel's hand-tuned (BLOCK_SIZE_M, num_warps) table.
    let (block_m, warps) = match d_state {
        0..=16 => (32u64, 4u32),
        17..=32 => (16, 4),
        33..=64 => (8, 4),
        65..=128 => (4, 4),
        _ => (4, 8),
    };
    let block_m = block_m_override.unwrap_or(block_m);
    let ds = d_state.next_power_of_two();
    let threads = 32 * warps as u64;
    let elems = (block_m * ds).div_ceil(threads).max(1); // state elements per thread
    let m_blocks = headdim.div_ceil(block_m);
    let ctas = m_blocks * batch * heads;
    let st_vec = (elems as f64 * state_bytes) as u64; // bytes per thread per access
    let st_op = match st_vec {
        0..=2 => "U16",
        3..=4 => "32",
        5..=8 => "64",
        _ => "128",
    };
    let n_st = st_vec.div_ceil(16).max(1);
    let state_cta = (block_m * d_state) as f64 * state_bytes;
    let state = AddrGen {
        base: region(40),
        cta_stride: state_cta as i64,
        lane_stride: st_vec.min(16) as i64,
        warp_stride: (32 * st_vec.min(16)) as i64,
        ..Default::default()
    };
    // B and C: (batch, groups, d_state); one row per sequence, shared by the
    // heads of a group and by every headdim block.
    let bc = |base: u64| AddrGen {
        base,
        cta_stride: (groups * d_state * 2) as i64,
        cta_div: m_blocks,
        cta_mod: batch,
        lane_stride: (elems * 2).min(16) as i64,
        ..Default::default()
    };
    let per_row = |base: u64, b: i64| AddrGen {
        base,
        cta_stride: block_m as i64 * b,
        warp_stride: b,
        ..Default::default()
    };
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("S2UR UR4, SR_CTAID.X");
    a.op("S2UR UR5, SR_CTAID.Y");
    a.op("S2UR UR6, SR_CTAID.Z");
    a.ops("IMAD R2, R0, UR7, RZ", 6);
    a.ops("IADD3 R3, P0, R2, UR8, RZ", 6);
    // All loads issue up front (Triton hoists them); consumers follow.
    for i in 0..n_st {
        let pc = a.op(&format!("LDG.E.{} R{}, desc[UR4][R2.64]", st_op, 8 + 4 * i));
        a.addr(
            pc,
            AddrGen {
                base: state.base + i * 16 * threads,
                ..state
            },
        );
    }
    for (op, g) in [
        (
            "LDG.E R20, desc[UR4][R4.64]",
            AddrGen {
                base: region(41),
                ..Default::default()
            },
        ), // dt (tied)
        (
            "LDG.E R21, desc[UR4][R4.64]",
            AddrGen {
                base: region(42),
                ..Default::default()
            },
        ), // dt_bias
        (
            "LDG.E R22, desc[UR4][R4.64]",
            AddrGen {
                base: region(43),
                ..Default::default()
            },
        ), // A
        ("LDG.E.64 R24, desc[UR4][R6.64]", bc(region(44))), // B
        ("LDG.E.64 R26, desc[UR4][R6.64]", bc(region(45))), // C
        ("LDG.E.U16 R28, desc[UR4][R4.64]", per_row(region(46), 2)), // x
        ("LDG.E R29, desc[UR4][R4.64]", per_row(region(47), 4)), // D
        ("LDG.E.U16 R30, desc[UR4][R4.64]", per_row(region(48), 2)), // z
    ] {
        let pc = a.op(op);
        a.addr(pc, g);
    }
    // softplus(dt + dt_bias), dA = exp(A dt)
    a.op("FADD R20, R20, R21");
    a.op("FSETP.GT.AND P1, PT, R20, 20, PT");
    a.op("FMUL R31, R20, 1.4426950216293334961");
    a.op("MUFU.EX2 R31, R31");
    a.op("FADD R31, R31, 1");
    a.op("MUFU.LG2 R31, R31");
    a.op("FMUL R31, R31, 0.69314718246459960938");
    a.op("FSEL R20, R20, R31, P1");
    a.op("FMUL R32, R22, R20");
    a.op("FMUL R32, R32, 1.4426950216293334961");
    a.op("MUFU.EX2 R32, R32");
    // Unpack bf16 B, C and x; dB = B dt.
    let e = elems as usize;
    a.ops("IMAD.U32 R33, R24, 0x10000, RZ", e);
    a.ops("IMAD.U32 R34, R26, 0x10000, RZ", e);
    a.op("IMAD.U32 R28, R28, 0x10000, RZ");
    a.ops("FMUL R33, R33, R20", e);
    // h = h * dA + dB * x
    a.ops("IMAD.U32 R8, R8, 0x10000, RZ", e);
    a.ops("FMUL R35, R33, R28", e);
    a.ops("FFMA R8, R8, R32, R35", e);
    if state_bytes < 4.0 {
        a.ops("F2FP.BF16.F32.PACK_AB R9, R8, R8", e.div_ceil(2));
    }
    for i in 0..n_st {
        let pc = a.op(&format!("STG.E.{} desc[UR4][R2.64], R9", st_op));
        a.addr(
            pc,
            AddrGen {
                base: state.base + i * 16 * threads,
                ..state
            },
        );
    }
    // y = sum_n h C over d_state: per-thread partial, then the warp.
    a.ops("FFMA R36, R8, R34, R36", e);
    let lanes_per_row = (ds / elems).clamp(1, 32);
    let mut s = lanes_per_row / 2;
    while s >= 1 {
        a.op(&format!("SHFL.BFLY PT, R37, R36, 0x{s:x}, 0x1f"));
        a.op("FADD R36, R36, R37");
        s /= 2;
    }
    // y += D x; y *= silu(z)
    a.op("FFMA R36, R28, R29, R36");
    a.op("IMAD.U32 R30, R30, 0x10000, RZ");
    a.op("FMUL R38, R30, -1.4426950216293334961");
    a.op("MUFU.EX2 R38, R38");
    a.op("FADD R38, R38, 1");
    a.op("MUFU.RCP R38, R38");
    a.op("FMUL R38, R30, R38");
    a.op("FMUL R36, R36, R38");
    a.op("F2FP.BF16.F32.PACK_AB R36, RZ, R36");
    let pc = a.op("STG.E.U16 desc[UR4][R4.64], R36");
    a.addr(pc, per_row(region(49), 2));
    let state_total = (batch * heads * headdim * d_state) as f64 * state_bytes;
    KernelLaunch {
        name: format!("selective_state_update_b{batch}_h{heads}_p{headdim}_n{d_state}"),
        grid: [ctas as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: 40,
        smem_per_cta: 0,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..warps)],
            barriers: HashMap::new(),
        },
        flops: (batch * heads * headdim * d_state) as f64 * 6.0,
        bytes: 2.0 * state_total + (batch * heads * headdim * 3 * 2) as f64,
    }
}

// --------------------------------------------------------------------- causal conv1d

/// `causal_conv1d_update` (decode, `seqlen == 1`) or the channel-last
/// `causal_conv1d_fwd` (prefill) with SiLU, over `dim` channels.
pub fn conv1d(_g: &GpuSpec, batch: u64, dim: u64, seqlen: u64, width: u64) -> KernelLaunch {
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("S2R R1, SR_CTAID.X");
    a.op("S2R R2, SR_CTAID.Y");
    a.ops("IMAD R3, R1, UR6, R0", 4);
    let w = width as usize;
    if seqlen <= 1 {
        // One thread per channel, 64 threads per CTA: grid (batch, dim / 64).
        let threads = 64u64;
        let ctas = batch * dim.div_ceil(threads);
        let ch = |base: u64, b: i64| AddrGen {
            base,
            cta_stride: threads as i64 * b,
            warp_stride: 32 * b,
            lane_stride: b,
            ..Default::default()
        };
        let pc = a.op("LDG.E.64 R8, desc[UR4][R4.64]"); // weight row
        a.addr(
            pc,
            AddrGen {
                cta_mod: dim.div_ceil(threads),
                ..ch(region(50), (width * 2) as i64)
            },
        );
        let pc = a.op("LDG.E.U16 R12, desc[UR4][R4.64]"); // bias
        a.addr(
            pc,
            AddrGen {
                cta_mod: dim.div_ceil(threads),
                ..ch(region(51), 2)
            },
        );
        for i in 0..(w - 1) {
            let pc = a.op(&format!("LDG.E.U16 R{}, desc[UR4][R4.64]", 13 + i));
            a.addr(pc, ch(region(52) + 2 * i as u64, ((width - 1) * 2) as i64));
        }
        let pc = a.op("LDG.E.U16 R20, desc[UR4][R4.64]"); // x
        a.addr(pc, ch(region(53), 2));
        a.ops("IMAD.U32 R13, R13, 0x10000, RZ", w + 1);
        a.ops("FFMA R21, R13, R8, R21", w);
        a.op("FMUL R22, R21, -1.4426950216293334961");
        a.op("MUFU.EX2 R22, R22");
        a.op("FADD R22, R22, 1");
        a.op("MUFU.RCP R22, R22");
        a.op("FMUL R21, R21, R22");
        for i in 0..(w - 1) {
            let pc = a.op(&format!("STG.E.U16 desc[UR4][R4.64], R{}", 14 + i));
            a.addr(pc, ch(region(52) + 2 * i as u64, ((width - 1) * 2) as i64));
        }
        a.op("F2FP.BF16.F32.PACK_AB R21, RZ, R21");
        let pc = a.op("STG.E.U16 desc[UR4][R4.64], R21");
        a.addr(pc, ch(region(54), 2));
        return KernelLaunch {
            name: format!("causal_conv1d_update_b{batch}_d{dim}_w{width}"),
            grid: [ctas as u32, 1, 1],
            block: [threads as u32, 1, 1],
            regs_per_thread: 32,
            smem_per_cta: 0,
            cluster: 1,
            source: KernelSource::Static {
                roles: vec![a.role("all", 0..2)],
                barriers: HashMap::new(),
            },
            flops: (batch * dim * (2 * width + 4)) as f64,
            bytes: (batch * dim * (2 * (width - 1) + 4)) as f64 * 2.0 + (dim * width * 2) as f64,
        };
    }
    // Channel-last prefill: CTA tile = 64 positions x 64 channels, 128
    // threads; x (+ width-1 halo rows) -> smem with 16 B vectors, each thread
    // then produces 32 outputs (4 x 8 channels) from smem and stores 16 B.
    let (tl, tc, threads) = (64u64, 64u64, 128u64);
    let ctas = batch * seqlen.div_ceil(tl) * dim.div_ceil(tc);
    let tile_in = (tl + width - 1) * tc * 2;
    let tile_out = tl * tc * 2;
    let n_ld = tile_in.div_ceil(16 * threads);
    let n_st = tile_out.div_ceil(16 * threads);
    let vec = |base: u64, cta_bytes: i64| AddrGen {
        base,
        cta_stride: cta_bytes,
        lane_stride: 16,
        warp_stride: 512,
        ..Default::default()
    };
    for i in 0..n_ld {
        let pc = a.op("LDG.E.128 R8, desc[UR4][R4.64]");
        a.addr(pc, vec(region(55) + i * 16 * threads, tile_out as i64));
    }
    let pc = a.op("LDG.E.128 R16, desc[UR4][R6.64]"); // weights for my channels
    a.addr(
        pc,
        AddrGen {
            footprint: dim * width * 2,
            ..vec(region(56), 0)
        },
    );
    a.ops("STS.128 [R5], R8", n_ld as usize);
    a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
    let outs = (tl * tc / threads) as usize; // outputs per thread
    a.ops("LDS.128 R8, [R5]", (outs / 8) * w);
    a.ops("IMAD.U32 {r}, R8, 0x10000, RZ", outs);
    a.ops("FFMA {r}, {r}, R16, {r}", outs * w);
    a.ops("FMUL {r}, {r}, -1.4426950216293334961", outs);
    a.ops("MUFU.EX2 {r}, {r}", outs);
    a.ops("FADD {r}, {r}, 1", outs);
    a.ops("MUFU.RCP {r}, {r}", outs);
    a.ops("FMUL {r}, {r}, {r}", outs);
    a.ops("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", outs / 2);
    for i in 0..n_st {
        let pc = a.op("STG.E.128 desc[UR4][R4.64], R12");
        a.addr(pc, vec(region(57) + i * 16 * threads, tile_out as i64));
    }
    KernelLaunch {
        name: format!("causal_conv1d_fwd_b{batch}_l{seqlen}_d{dim}_w{width}"),
        grid: [ctas as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: 64,
        smem_per_cta: tile_in as u32,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..(threads / 32) as u32)],
            barriers: HashMap::new(),
        },
        flops: (batch * seqlen * dim * (2 * width + 4)) as f64,
        bytes: (batch * seqlen * dim * 4) as f64,
    }
}

// --------------------------------------------------------------------- Mamba-2 SSD prefill

/// SSD tile configurations per stage (autotuned in the Triton source; these
/// are the ones Nsight Compute recorded for d_state 128, headdim 64).
#[derive(Clone, Copy, Debug)]
pub struct SsdTiles {
    pub bm: u64,
    pub bn: u64,
    pub bk: u64,
    pub warps: u32,
    pub stages: u32,
}

/// Shapes shared by every SSD stage.
#[derive(Clone, Copy, Debug)]
pub struct SsdShape {
    pub batch: u64,
    pub seqlen: u64,
    pub heads: u64,
    pub headdim: u64,
    pub d_state: u64,
    pub groups: u64,
    pub chunk: u64,
}

impl SsdShape {
    fn nchunks(&self) -> u64 {
        self.seqlen.div_ceil(self.chunk)
    }
    fn bc(&self) -> u64 {
        self.batch * self.nchunks()
    }
}

/// The autotuned configurations Nsight Compute recorded (grid, block and
/// shared-memory size identify the Triton config) for d_state 128, headdim
/// 64, chunk 256: on sm_103 every `tl.dot` stage picks a 2-warp 64x32 tile;
/// on sm_80 `chunk_state` takes 64x64 and `bmm_chunk` the 8-warp 128x256.
fn ssd_tiles(g: &GpuSpec, stage: &str) -> SsdTiles {
    let t = |bm, bn, bk, warps, stages| SsdTiles {
        bm,
        bn,
        bk,
        warps,
        stages,
    };
    match (g.arch, stage) {
        (Arch::Ampere, "chunk_state") => t(64, 64, 32, 2, 4),
        (Arch::Ampere, "bmm_chunk") => t(128, 256, 64, 8, 3),
        (_, "chunk_scan") => t(64, 32, 32, 2, 4),
        _ => t(64, 32, 32, 2, 4),
    }
}

/// One of the five kernels of `mamba_chunk_scan_combined`.
pub fn ssd(g: &GpuSpec, stage: &str, s: SsdShape, tiles: Option<SsdTiles>) -> KernelLaunch {
    let path = dot_path(g);
    let tiles = tiles.or_else(|| Some(ssd_tiles(g, stage)));
    let (p, n, h, cs) = (s.headdim, s.d_state, s.heads, s.chunk);
    let tag = format!("b{}_l{}_h{h}_p{p}_n{n}_g{}", s.batch, s.seqlen, s.groups);
    match stage {
        "chunk_cumsum" => {
            // grid (batch, nchunks, heads / BLOCK_H), tile [BLOCK_H, chunk];
            // the autotuner picks BLOCK_H = 64 on both GPUs.
            let bh = 64u64.min(h);
            let mut k = TileKernel::new(
                format!("ssd_chunk_cumsum_{tag}"),
                s.bc() * h.div_ceil(bh),
                4,
                path,
            );
            let elems = (bh * cs).div_ceil(k.threads()) as usize;
            k.pre_loads = vec![stream(bh * cs * 2, region(60), (bh * cs * 2) as i64, 0)];
            k.pre_alu = vec![
                ("FADD {r}, {r}, R41", elems),
                ("MUFU.EX2 {r}, {r}", elems),
                ("MUFU.LG2 {r}, {r}", elems),
                ("FSEL {r}, {r}, R42, P1", elems),
                ("FMUL {r}, {r}, R44", elems),
                // tl.cumsum along the chunk, fully unrolled by Triton: ncu
                // counts ~7600 instructions per warp (sm_103) for 128
                // elements per thread, issued by 1 warp per SMSP.
                ("SHFL.UP PT, R45, R43, 0x1, RZ", 5 * elems / 4),
                ("FADD R43, R43, R45", 5 + elems),
                ("FSEL R43, R43, R45, P2", 12 * elems),
                ("IMAD R46, R46, UR8, R43", 34 * elems),
                ("STS [R5], R43", 1),
                ("BAR.SYNC.DEFER_BLOCKING 0x0", 1),
                ("LDS R46, [R5]", 1),
            ];
            k.out = vec![
                stream(bh * cs * 4, region(61), (bh * cs * 4) as i64, 0),
                stream(bh * cs * 4, region(62), (bh * cs * 4) as i64, 0),
            ];
            k.flops = (s.batch * s.seqlen * h * 8) as f64;
            k.bytes = (s.batch * s.seqlen * h * 10) as f64;
            k.build()
        }
        "chunk_state" => {
            // states[b, c, h] (P x N, fp32) = x^T (B * exp(dA_last - dA) * dt), K = chunk.
            let t = tiles.unwrap_or(SsdTiles {
                bm: 64,
                bn: 128,
                bk: 32,
                warps: 4,
                stages: 4,
            });
            let tiles_mn = p.div_ceil(t.bm) * n.div_ceil(t.bn);
            let mut k = TileKernel::new(
                format!("ssd_chunk_state_{tag}"),
                tiles_mn * s.bc() * h,
                t.warps,
                path,
            );
            k.stages = t.stages;
            k.k_iters = cs.div_ceil(t.bk);
            let x_it = t.bm * t.bk * 2;
            let b_it = t.bk * t.bn * 2;
            let (x_fp, bc_fp, da_fp) = (
                s.batch * s.seqlen * h * p * 2,
                s.batch * s.seqlen * s.groups * n * 2,
                s.batch * s.seqlen * h * 8,
            );
            k.iter_loads = vec![
                // CTA = tile + tiles_mn * (b*c + BC * head): x per (chunk, head),
                // B per chunk (shared by the heads of a group).
                within(
                    blocked(x_it, region(63), cs * t.bm * 2, tiles_mn, 0, x_it as i64),
                    x_fp,
                ),
                within(
                    blocked(b_it, region(64), cs * n * 2, tiles_mn, s.bc(), b_it as i64),
                    bc_fp,
                ),
                within(
                    blocked(t.bk * 8, region(65), cs * 8, tiles_mn, 0, (t.bk * 8) as i64),
                    da_fp,
                ),
            ];
            let per_thr = (t.bk * t.bn).div_ceil(k.threads()) as usize;
            k.iter_alu = vec![
                ("LDS R30, [R5]", 2),
                ("FADD R31, R30, -R32", 2),
                ("MUFU.EX2 R31, R31", 2),
                ("FMUL R31, R31, R33", 2),
                ("IMAD.U32 {r}, R24, 0x10000, RZ", per_thr),
                ("FMUL {r}, {r}, R31", per_thr),
                ("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", per_thr.div_ceil(2)),
            ];
            // Index/mask arithmetic: ncu counts ~2000 instructions per warp
            // on sm_103 and ~3300 on sm_80 for 8 k-iterations.
            k.iter_alu.push(("IMAD {r}, {r}, UR8, R2", 150));
            k.iter_alu.push(("ISETP.GE.AND P3, PT, {r}, UR9, PT", 20));
            k.restage_bytes = b_it;
            k.mma = Some((t.bm, t.bn, t.bk));
            k.pre_loads = vec![stream(64, region(66), 0, 0)];
            k.out = vec![stream(
                t.bm * t.bn * 4,
                region(67),
                (t.bm * t.bn * 4) as i64,
                0,
            )];
            k.flops = 2.0 * (s.bc() * h * p * n * cs) as f64;
            k.bytes = (s.batch * s.seqlen * (h * p + s.groups * n) * 2) as f64
                + (s.bc() * h * p * n * 4) as f64;
            k.build()
        }
        "state_passing" => {
            // grid (P*N / BLOCK, batch, heads); sequential over chunks.
            let dim = p * n;
            let block = 1024u64.min(dim);
            let mut k = TileKernel::new(
                format!("ssd_state_passing_{tag}"),
                dim.div_ceil(block) * s.batch * h,
                4,
                path,
            );
            k.direct = true;
            k.k_iters = s.nchunks();
            let it = block * 4;
            let chunk_stride = (h * dim * 4) as i64;
            k.iter_loads = vec![
                stream(it, region(68), it as i64, chunk_stride),
                stream(16, region(69), 0, 4),
            ];
            let e = block.div_ceil(k.threads()) as usize;
            k.iter_alu = vec![
                ("FMUL R30, R28, 1.4426950216293334961", 1),
                ("MUFU.EX2 R30, R30", 1),
                ("FFMA {r}, {r}, R30, R24", e),
                ("STG.E.128 desc[UR4][R2.64], R44", e.div_ceil(4)),
            ];
            k.out = vec![stream(it, region(70), it as i64, 0)];
            k.regs = 64;
            k.flops = (2 * s.bc() * h * dim) as f64;
            k.bytes = (2 * s.bc() * h * dim * 4) as f64;
            k.build()
        }
        "bmm_chunk" => {
            // CB[b, c, g] (chunk x chunk, fp32) = C B^T, K = d_state; causal tiles above the diagonal exit.
            let t = tiles.unwrap_or(SsdTiles {
                bm: 64,
                bn: 64,
                bk: 32,
                warps: 4,
                stages: 4,
            });
            let tm = cs.div_ceil(t.bm);
            let tn = cs.div_ceil(t.bn);
            let mut k = TileKernel::new(
                format!("ssd_bmm_chunk_{tag}"),
                tm * tn * s.bc() * s.groups,
                t.warps,
                path,
            );
            k.stages = t.stages;
            k.k_iters = n.div_ceil(t.bk);
            let a_it = t.bm * t.bk * 2;
            let b_it = t.bn * t.bk * 2;
            k.iter_loads = vec![
                within(
                    blocked(a_it, region(71), cs * n * 2, tm * tn, 0, a_it as i64),
                    s.batch * s.seqlen * s.groups * n * 2,
                ),
                within(
                    blocked(b_it, region(72), cs * n * 2, tm * tn, 0, b_it as i64),
                    s.batch * s.seqlen * s.groups * n * 2,
                ),
            ];
            k.mma = Some((t.bm, t.bn, t.bk));
            // ncu: ~950 (sm_103) to ~1500 (sm_80) instructions per warp.
            k.iter_alu = vec![
                ("IMAD {r}, {r}, UR8, R2", 160),
                ("ISETP.GE.AND P3, PT, {r}, UR9, PT", 20),
            ];
            let above = (0..tm)
                .map(|i| (0..tn).filter(|&j| j * t.bn >= (i + 1) * t.bm).count() as u64)
                .sum::<u64>();
            k.early_exit = above as f64 / (tm * tn) as f64;
            k.out = vec![stream(
                t.bm * t.bn * 4,
                region(73),
                (t.bm * t.bn * 4) as i64,
                0,
            )];
            k.flops = (s.bc() * s.groups * cs * (cs + 1)) as f64 * n as f64;
            k.bytes = (s.batch * s.seqlen * s.groups * n * 4) as f64
                + (s.bc() * s.groups * cs * cs * 4) as f64 / 2.0;
            k.build()
        }
        "chunk_scan" => {
            // out[b, c, h] (chunk x P) = (CB * exp(dA_i - dA_j) * dt_j, masked) x + C states^T * exp(dA_i).
            let t = tiles.unwrap_or(SsdTiles {
                bm: 64,
                bn: 64,
                bk: 32,
                warps: 4,
                stages: 2,
            });
            let tm = cs.div_ceil(t.bm);
            let tiles_mn = tm * p.div_ceil(t.bn);
            let mut k = TileKernel::new(
                format!("ssd_chunk_scan_{tag}"),
                tiles_mn * s.bc() * h,
                t.warps,
                path,
            );
            k.stages = t.stages;
            // Inter-chunk term (K = d_state) + causal intra-chunk term: the
            // m-th row block runs (m + 1) * bm / bk iterations; use the mean.
            let intra = ((tm + 1) * t.bm).div_ceil(2 * t.bk);
            k.k_iters = n.div_ceil(t.bk) + intra;
            let cb_it = t.bm * t.bk * 4;
            let x_it = t.bk * t.bn * 2;
            k.iter_loads = vec![
                // CTA = tile + tiles_mn * (b*c + BC * head): C B^T per chunk
                // (re-read by every head), x and dA per (chunk, head).
                within(
                    blocked(
                        cb_it,
                        region(74),
                        cs * cs * 4,
                        tiles_mn,
                        s.bc(),
                        cb_it as i64,
                    ),
                    s.bc() * s.groups * cs * cs * 4,
                ),
                within(
                    blocked(x_it, region(75), cs * p * 2, tiles_mn, 0, x_it as i64),
                    s.batch * s.seqlen * h * p * 2,
                ),
                within(
                    blocked(t.bk * 8, region(76), cs * 8, tiles_mn, 0, (t.bk * 8) as i64),
                    s.batch * s.seqlen * h * 8,
                ),
            ];
            let per_thr = (t.bm * t.bk).div_ceil(k.threads()) as usize;
            k.iter_alu = vec![
                ("LDS.128 R28, [R5]", per_thr.div_ceil(4)),
                ("FADD {r}, R30, -R31", per_thr),
                ("FMUL {r}, {r}, 1.4426950216293334961", per_thr),
                ("MUFU.EX2 {r}, {r}", per_thr),
                ("FMUL {r}, {r}, R29", per_thr),
                ("FMUL {r}, {r}, R32", per_thr),
                ("FSEL {r}, {r}, RZ, P2", per_thr),
                ("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", per_thr.div_ceil(2)),
            ];
            k.restage_bytes = t.bm * t.bk * 2;
            k.mma = Some((t.bm, t.bn, t.bk));
            // Previous chunk's state (P x N fp32) for the inter-chunk term.
            k.pre_loads = vec![within(
                blocked(t.bn * n * 4, region(77), p * n * 4, tiles_mn, 0, 0),
                s.bc() * h * p * n * 4,
            )];
            let out_e = (t.bm * t.bn).div_ceil(k.threads()) as usize;
            k.epi_loads = vec![
                stream(t.bm * t.bn * 2, region(78), (t.bm * t.bn * 2) as i64, 0),
                stream(t.bm * t.bn * 2, region(79), (t.bm * t.bn * 2) as i64, 0),
            ];
            k.epi_alu = vec![
                ("FMUL {r}, R24, R51", out_e),
                ("FFMA {r}, R44, R52, {r}", out_e),
                ("FMUL {r}, R45, -1.4426950216293334961", out_e),
                ("MUFU.EX2 {r}, {r}", out_e),
                ("FADD {r}, {r}, 1", out_e),
                ("MUFU.RCP {r}, {r}", out_e),
                ("FMUL {r}, {r}, {r}", out_e),
                ("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", out_e.div_ceil(2)),
            ];
            k.out = vec![stream(
                t.bm * t.bn * 2,
                region(80),
                (t.bm * t.bn * 2) as i64,
                0,
            )];
            k.flops = 2.0 * (s.bc() * h * p) as f64 * (cs * (cs + 1) / 2 + cs * n) as f64;
            k.bytes = (s.batch * s.seqlen * h * p * 2 * 3) as f64
                + (s.bc() * s.groups * cs * cs * 4) as f64 / 2.0
                + (s.bc() * h * p * n * 4) as f64;
            k.build()
        }
        other => panic!("unknown SSD stage `{other}`"),
    }
}

// --------------------------------------------------------------------- FlashAttention-2 (Ampere)

/// FlashAttention-2 forward on `mma.sync` hardware: 4 warps, Q tile in
/// registers, K/V tiles double-buffered with `cp.async`, S = QK^T and O += PV
/// with HMMA, online softmax in between. Decode (`sq == 1`) packs the GQA
/// group into the M dimension and splits the KV sequence across CTAs.
#[allow(clippy::too_many_arguments)]
pub fn attention_fa2(
    g: &GpuSpec,
    batch: u64,
    heads_q: u64,
    heads_kv: u64,
    sq: u64,
    skv: u64,
    d: u64,
    causal: bool,
) -> KernelLaunch {
    let dt = dtype("bf16");
    let group = (heads_q / heads_kv.max(1)).max(1);
    let decode = sq * group <= 64 && sq <= 16;
    let warps = 4u64;
    let (bm, bn) = if decode {
        (64u64, 128u64)
    } else if d <= 64 {
        (128, 128)
    } else {
        (128, 64)
    };
    let rows_total = if decode {
        batch * heads_kv * (sq * group).div_ceil(bm)
    } else {
        batch * heads_q * sq.div_ceil(bm)
    };
    let eff_kv = if causal && !decode {
        skv.div_ceil(2)
    } else {
        skv
    };
    let kv_iters = eff_kv.div_ceil(bn).max(1);
    let splits = if decode {
        ((g.sm_count as u64 * 2) / rows_total.max(1)).clamp(1, kv_iters.min(128))
    } else {
        1
    };
    let k_iters = kv_iters.div_ceil(splits);
    let kv_bytes = bn * d * 2;
    let mut k = TileKernel::new(
        format!(
            "fa2_fwd_{}b{batch}_hq{heads_q}_hkv{heads_kv}_sq{sq}_skv{skv}_d{d}",
            if decode { "splitkv_" } else { "" }
        ),
        rows_total * splits,
        warps as u32,
        DotPath::Sync,
    );
    k.stages = 2;
    k.k_iters = k_iters;
    let kv_fp = (batch * heads_kv * skv * d * 2).max(1);
    let gen = |base: u64| Stream {
        bytes: kv_bytes,
        addr: AddrGen {
            base,
            cta_stride: (kv_bytes * k_iters) as i64,
            iter_stride: kv_bytes as i64,
            lane_stride: 16,
            warp_stride: 512,
            footprint: kv_fp,
            ..Default::default()
        },
    };
    k.iter_loads = vec![gen(region(3)), gen(region(4))];
    // S = Q K^T (bm x bn x d) and O += P V (bm x d x bn), Q from registers.
    let s_elems = (bm * bn).div_ceil(k.threads()) as usize;
    let mut alu: Vec<(&'static str, usize)> = vec![];
    let hmma_s = (bm * bn * d).div_ceil(16 * 8 * 16 * warps) as usize;
    let hmma_o = (bm * d * bn).div_ceil(16 * 8 * 16 * warps) as usize;
    let ldsm = ((bn * d * 2) * 2).div_ceil(512 * warps) as usize; // K and V, each warp reads all of them
    alu.push(("LDSM.16.M88.4 R8, [R6]", ldsm / 2));
    alu.push(("@HMMA_S", hmma_s));
    alu.push(("FMNMX {r}, {r}, R40, !PT", s_elems / 2));
    alu.push(("SHFL.BFLY PT, R41, R40, 0x1, 0x1f", 1));
    alu.push(("FMNMX R40, R40, R41, !PT", 1));
    alu.push(("SHFL.BFLY PT, R41, R40, 0x2, 0x1f", 1));
    alu.push(("FMNMX R40, R40, R41, !PT", 1));
    alu.push(("FFMA {r}, {r}, R42, -R40", s_elems));
    alu.push(("MUFU.EX2 {r}, {r}", s_elems));
    alu.push(("FADD {r}, {r}, R43", s_elems / 2));
    alu.push((
        "FMUL {r}, {r}, R45",
        (bm * d).div_ceil(k.threads()) as usize,
    ));
    alu.push(("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", s_elems / 2));
    alu.push(("LDSM.16.M88.4.TRANS R8, [R6]", ldsm / 2));
    alu.push(("@HMMA_O", hmma_o));
    k.iter_alu = alu;
    k.pre_loads = vec![stream(bm * d * 2, region(81), (bm * d * 2) as i64, 0)];
    k.epi_alu = vec![
        ("MUFU.RCP R46, R43", 1),
        (
            "FMUL {r}, {r}, R46",
            (bm * d).div_ceil(k.threads()) as usize,
        ),
    ];
    let out_bytes = if decode && splits > 1 { 4 } else { 2 };
    k.out = vec![stream(
        bm * d * out_bytes,
        region(5),
        (bm * d * out_bytes) as i64,
        0,
    )];
    k.regs = 255;
    k.dt = dt;
    k.flops = 4.0 * (batch * heads_q * sq * eff_kv * d) as f64;
    k.bytes = (2 * batch * heads_kv * skv * d * 2 + 2 * batch * heads_q * sq * d * 2) as f64;
    let mut kl = k.build();
    // FA2 (Flash_fwd_kernel_traits): Q tile + one K and one V tile in smem.
    kl.smem_per_cta = ((bm * d + 2 * bn * d) * 2) as u32;
    kl
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsim_hw::presets;

    #[test]
    fn ssm_programs_build_and_fit() {
        let s = SsdShape {
            batch: 1,
            seqlen: 4096,
            heads: 128,
            headdim: 64,
            d_state: 128,
            groups: 8,
            chunk: 256,
        };
        for g in [presets::a100_pcie(), presets::h100_sxm(), presets::b300()] {
            let mut ks = vec![
                ssm_state_update(&g, 8, 128, 64, 128, 8, 2.0),
                conv1d(&g, 8, 10240, 1, 4),
                conv1d(&g, 1, 10240, 4096, 4),
                attention_fa2(&g, 8, 32, 8, 1, 4096, 128, false),
                attention_fa2(&g, 1, 32, 8, 4096, 4096, 128, true),
            ];
            for st in [
                "chunk_cumsum",
                "chunk_state",
                "state_passing",
                "bmm_chunk",
                "chunk_scan",
            ] {
                ks.push(ssd(&g, st, s, None));
            }
            for k in ks {
                assert!(k.num_ctas() > 0, "{}", k.name);
                assert!(
                    crate::sm::Sm::occupancy(&g, &k) >= 1,
                    "{} does not fit on {}",
                    k.name,
                    g.name
                );
            }
        }
    }
}
