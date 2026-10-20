//! Synthetic SASS kernels.
//!
//! When no binary is available (e.g. predicting a model that has not been
//! compiled yet), workloads are lowered to SASS programs that mirror the
//! structure of the production kernels:
//!
//! * **Hopper GEMM** — CUTLASS 3.x "TMA warp-specialized cooperative": one
//!   producer warp issues `UTMALDG` into a multi-stage smem ring guarded by
//!   full/empty mbarriers; two consumer warpgroups issue `HGMMA` and release
//!   stages one iteration late (`WARPGROUP.DEPBAR.LE gsb0, 0x1`).
//! * **Blackwell GEMM** — tcgen05 style: TMA warp, single MMA-issuer warp
//!   (`UTCHMMA` into TMEM, `UTCBAR` commits to mbarriers), epilogue warpgroup
//!   (`LDTM` → `F2FP` → `STG`).
//! * **Attention** — FlashAttention-3 (Hopper) / FlashAttention-4-like
//!   (Blackwell) pipelines with explicit softmax (`MUFU.EX2`) work, so the
//!   exp-vs-MMA balance and its bubbles appear in the model.
//! * **Norm / elementwise / reduction** kernels for memory-bound ops.
//!
//! The programs are written as SASS text and run through the same parser as
//! real `cuobjdump` output, so every model path is exercised identically.

use crate::kernel::{BarInit, BarOp, KernelLaunch, KernelSource, WarpRole};
use rsim_hw::{Arch, GpuSpec};
use rsim_sass::parse::parse_inst;
use rsim_sass::walker::{AddrGen, Annotations};
use rsim_sass::{BranchProfile, Inst, Program};
use std::collections::HashMap;
use std::sync::Arc;

const FULL: u32 = 1;
const EMPTY: u32 = 2;
const ACC: u32 = 3;
const S_READY: u32 = 4;
const P_READY: u32 = 5;

/// Tiny assembler: SASS text in, program + side tables out.
#[derive(Default)]
pub struct Asm {
    insts: Vec<Inst>,
    pc: u32,
    profile: BranchProfile,
    ann: Annotations,
    bars: HashMap<u32, BarOp>,
    multicast: HashMap<u32, f64>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }
    /// Append one instruction; returns its PC.
    pub fn op(&mut self, text: &str) -> u32 {
        let (inst, _) = parse_inst(self.pc, text)
            .unwrap_or_else(|e| panic!("bad synthetic SASS `{text}`: {e}"));
        let pc = self.pc;
        self.insts.push(inst);
        self.pc += 16;
        pc
    }
    pub fn ops(&mut self, text: &str, n: usize) {
        for _ in 0..n {
            self.op(text);
        }
    }
    pub fn here(&self) -> u32 {
        self.pc
    }
    /// Conditional backward branch to `target`, taken `trips` times.
    pub fn loop_to(&mut self, target: u32, trips: u64) {
        let pc = self.op(&format!("@P1 BRA 0x{target:x}"));
        self.profile.taken.insert(pc, trips);
    }
    pub fn addr(&mut self, pc: u32, g: AddrGen) {
        self.ann.addr.insert(pc, g);
    }
    pub fn payload(&mut self, pc: u32, n: u64) {
        self.ann.payload.insert(pc, n);
    }
    pub fn single_thread(&mut self, pc: u32) {
        self.ann.active.insert(pc, 1);
    }
    pub fn bar(&mut self, pc: u32, b: BarOp) {
        self.bars.insert(pc, b);
    }
    pub fn multicast(&mut self, pc: u32, cluster: u32) {
        self.multicast.insert(pc, 1.0 / cluster.max(1) as f64);
    }
    pub fn exit_if(&mut self, frac: f64) {
        let pc = self.op("@P0 EXIT");
        if frac > 0.0 {
            self.profile.exit_prob.insert(pc, frac);
        }
    }
    pub fn role(mut self, name: &str, warps: std::ops::Range<u32>) -> WarpRole {
        if !matches!(
            self.insts
                .last()
                .map(|i| i.base_opcode().to_string())
                .as_deref(),
            Some("EXIT")
        ) {
            self.op("EXIT");
        }
        let mut r = WarpRole::new(name, warps, Program::new(name, self.insts));
        r.profile = Arc::new(self.profile);
        r.ann = Arc::new(self.ann);
        r.bar_ops = Arc::new(self.bars);
        r.multicast = Arc::new(self.multicast);
        r
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dtype {
    pub name: &'static str,
    pub bytes: f64,
    pub sass: &'static str,
}

pub fn dtype(name: &str) -> Dtype {
    match name {
        "fp8" | "e4m3" => Dtype {
            name: "fp8",
            bytes: 1.0,
            sass: "E4M3",
        },
        "fp4" | "nvfp4" => Dtype {
            name: "fp4",
            bytes: 0.5,
            sass: "E2M1",
        },
        "fp16" => Dtype {
            name: "fp16",
            bytes: 2.0,
            sass: "F16",
        },
        "tf32" | "fp32" => Dtype {
            name: "tf32",
            bytes: 4.0,
            sass: "TF32",
        },
        _ => Dtype {
            name: "bf16",
            bytes: 2.0,
            sass: "BF16",
        },
    }
}

fn mma_opcode(arch: Arch, m: u64, n: u64, k: u64, dt: Dtype) -> String {
    match arch {
        Arch::Hopper => {
            let base = if dt.name == "fp8" { "QGMMA" } else { "HGMMA" };
            format!("{base}.{m}x{n}x{k}.F32.{} R24, gdesc[UR8], R24", dt.sass)
        }
        Arch::Blackwell => {
            let base = if dt.name == "fp8" || dt.name == "fp4" {
                "UTCQMMA"
            } else {
                "UTCHMMA"
            };
            format!(
                "{base}.{m}x{n}x{k}.{} gdesc[UR8], gdesc[UR10], tmem[UR12], UR14",
                dt.sass
            )
        }
        Arch::Ampere => format!("HMMA.16816.F32.{} R24, R8, R12, R24", dt.sass),
    }
}

/// Unique address region per tensor so caches see realistic reuse.
fn region(i: u64) -> u64 {
    (i + 1) << 40
}

/// Parameters of a TMA-fed, multi-stage MMA mainloop.
struct Mainloop {
    tile_m: u64,
    tile_n: u64,
    k_blk: u64,
    k_iters: u64,
    stages: u32,
    /// A and B stage bytes; B can be multicast across a cluster.
    a_bytes: u64,
    b_bytes: u64,
    cluster: u32,
    a_base: AddrGen,
    b_base: AddrGen,
    dt: Dtype,
}

fn tma_load(asm: &mut Asm, bytes: u64, g: AddrGen, mc: u32) {
    let pc = asm.op("UTMALDG.2D [UR16], [UR18], desc[UR20]");
    asm.payload(pc, bytes);
    asm.addr(pc, g);
    asm.single_thread(pc);
    asm.bar(pc, BarOp::Arrive(FULL));
    if mc > 1 {
        asm.multicast(pc, mc);
    }
}

fn producer(ml: &Mainloop, warps: std::ops::Range<u32>) -> WarpRole {
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("ELECT P0, URZ, PT");
    let top = a.here();
    let w = a.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR4+0x8], R3");
    a.bar(w, BarOp::Wait(EMPTY));
    let pc = a.op("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR4], R5");
    let _ = pc;
    tma_load(&mut a, ml.a_bytes, ml.a_base, 1);
    tma_load(&mut a, ml.b_bytes, ml.b_base, ml.cluster);
    a.op("UIADD3 UR22, UR22, 0x1, URZ");
    if ml.k_iters > 1 {
        a.loop_to(top, ml.k_iters - 1);
    }
    a.role("tma_producer", warps)
}

fn epilogue_store(a: &mut Asm, elems_per_thread: u64, out_bytes: f64, out: AddrGen) {
    let n_cvt = (elems_per_thread / 2).max(1) as usize;
    a.ops("F2FP.BF16.F32.PACK_AB R24, R25, R24", n_cvt);
    let bytes = (elems_per_thread as f64 * out_bytes) as u64;
    let n_st = bytes.div_ceil(16).max(1);
    for i in 0..n_st {
        let pc = a.op("STG.E.128 desc[UR4][R2.64], R24");
        a.addr(
            pc,
            AddrGen {
                iter_stride: 0,
                base: out.base + i * 16 * 128,
                ..out
            },
        );
    }
}

/// Warpgroup-MMA consumer (Hopper). `mma_per_iter` = HGMMA shapes issued per
/// k-block; `softmax_elems` > 0 inserts FA3-style softmax between them.
fn hopper_consumer(
    ml: &Mainloop,
    warps: std::ops::Range<u32>,
    wg_m: u64,
    mmas: &[(u64, u64, u64, usize)],
    softmax_elems: u64,
    out: AddrGen,
    out_elems_per_thread: u64,
) -> WarpRole {
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("IMAD R2, R0, 0x10, RZ");
    let body = |a: &mut Asm, release: bool| {
        let w = a.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR4], R3");
        a.bar(w, BarOp::Wait(FULL));
        for (i, &(m, n, k, reps)) in mmas.iter().enumerate() {
            a.op("WARPGROUP.ARRIVE");
            for _ in 0..reps {
                a.op(&mma_opcode(Arch::Hopper, m, n, k, ml.dt));
            }
            if softmax_elems > 0 && i == 0 {
                // S = QK^T must land before softmax reads it.
                a.op("WARPGROUP.DEPBAR.LE gsb0, 0x0");
                softmax_body(a, softmax_elems);
            }
        }
        a.op("WARPGROUP.DEPBAR.LE gsb0, 0x1");
        if release {
            let r = a.op("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR6], RZ");
            a.bar(r, BarOp::Arrive(EMPTY));
        }
    };
    let _ = wg_m;
    // Peeled first iteration: stage 0 is released one iteration late.
    body(&mut a, false);
    if ml.k_iters > 1 {
        let top = a.here();
        body(&mut a, true);
        if ml.k_iters > 2 {
            a.loop_to(top, ml.k_iters - 2);
        }
    }
    a.op("WARPGROUP.DEPBAR.LE gsb0, 0x0");
    let r = a.op("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR6], RZ");
    a.bar(r, BarOp::Arrive(EMPTY));
    epilogue_store(&mut a, out_elems_per_thread, 2.0, out);
    a.role("mma_consumer", warps)
}

/// Online-softmax work for `elems` scores per thread.
fn softmax_body(a: &mut Asm, elems: u64) {
    let e = elems as usize;
    a.ops("FMNMX R40, R24, R40, !PT", e / 2);
    a.op("SHFL.BFLY PT, R41, R40, 0x1, 0x1f");
    a.op("FMNMX R40, R40, R41, !PT");
    a.op("SHFL.BFLY PT, R41, R40, 0x2, 0x1f");
    a.op("FMNMX R40, R40, R41, !PT");
    a.ops("FFMA R24, R24, R42, -R40", e);
    a.ops("MUFU.EX2 R24, R24", e);
    a.ops("FADD R43, R24, R43", e / 2);
    a.op("FMUL R44, R44, R45");
    a.ops("F2FP.BF16.F32.PACK_AB R60, R25, R24", e / 2);
}

/// Hopper TMA warp-specialised GEMM.
fn gemm_hopper(m: u64, n: u64, k: u64, dt: Dtype) -> KernelLaunch {
    let (tm, tn, kb) = (128u64, if n >= 256 { 256 } else { 128 }, 64u64);
    let k_iters = k.div_ceil(kb).max(1);
    let tiles_m = m.div_ceil(tm);
    let tiles_n = n.div_ceil(tn);
    let a_bytes = (tm * kb) as f64 * dt.bytes;
    let b_bytes = (tn * kb) as f64 * dt.bytes;
    let cluster = if tiles_m >= 2 { 2 } else { 1 };
    let ml = Mainloop {
        tile_m: tm,
        tile_n: tn,
        k_blk: kb,
        k_iters,
        stages: 4,
        a_bytes: a_bytes as u64,
        b_bytes: b_bytes as u64,
        cluster,
        // CTA i computes tile (i / tiles_n, i % tiles_n): A rows are shared
        // by CTAs of the same tile row, B columns by the same tile column.
        a_base: AddrGen {
            base: region(0),
            cta_stride: ((tm * k) as f64 * dt.bytes) as i64,
            cta_div: tiles_n,
            iter_stride: a_bytes as i64,
            ..Default::default()
        },
        b_base: AddrGen {
            base: region(1),
            cta_stride: ((tn * k) as f64 * dt.bytes) as i64,
            cta_mod: tiles_n,
            iter_stride: b_bytes as i64,
            ..Default::default()
        },
        dt,
    };
    let _ = (ml.tile_m, ml.tile_n, ml.k_blk);
    let mmas = [(64u64, tn, 16u64, (kb / 16) as usize)];
    let out = AddrGen {
        base: region(2),
        cta_stride: (tm * tn * 2) as i64,
        warp_stride: (16 * tn * 2) as i64,
        lane_stride: 16,
        ..Default::default()
    };
    let elems_per_thread = 64 * tn / 128;
    let roles = vec![
        producer(&ml, 0..1),
        Asm::new().role("producer_idle", 1..4),
        hopper_consumer(&ml, 4..12, 64, &mmas, 0, out, elems_per_thread),
    ];
    let mut barriers = HashMap::new();
    barriers.insert(
        FULL,
        BarInit {
            count: 2,
            initial_phases: 0,
        },
    );
    barriers.insert(
        EMPTY,
        BarInit {
            count: 8,
            initial_phases: ml.stages,
        },
    );
    let smem = ((a_bytes + b_bytes) as u32) * ml.stages;
    KernelLaunch {
        name: format!("sm90_gemm_{}_{m}x{n}x{k}_tile{tm}x{tn}", dt.name),
        grid: [(tiles_m * tiles_n) as u32, 1, 1],
        block: [384, 1, 1],
        regs_per_thread: 168,
        smem_per_cta: smem,
        cluster,
        source: KernelSource::Static { roles, barriers },
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        bytes: (m * k + n * k) as f64 * dt.bytes + (m * n) as f64 * 2.0,
    }
}

/// Blackwell tcgen05 GEMM.
fn gemm_blackwell(m: u64, n: u64, k: u64, dt: Dtype) -> KernelLaunch {
    let (tm, tn, kb) = (
        128u64,
        if n >= 256 { 256 } else { 128 },
        if dt.bytes <= 1.0 { 128 } else { 64u64 },
    );
    let k_iters = k.div_ceil(kb).max(1);
    let tiles_m = m.div_ceil(tm);
    let tiles_n = n.div_ceil(tn);
    let a_bytes = (tm * kb) as f64 * dt.bytes;
    let b_bytes = (tn * kb) as f64 * dt.bytes;
    let cluster = if tiles_m >= 2 { 2 } else { 1 };
    let ml = Mainloop {
        tile_m: tm,
        tile_n: tn,
        k_blk: kb,
        k_iters,
        stages: 4,
        a_bytes: a_bytes as u64,
        b_bytes: b_bytes as u64,
        cluster,
        a_base: AddrGen {
            base: region(0),
            cta_stride: ((tm * k) as f64 * dt.bytes) as i64,
            cta_div: tiles_n,
            iter_stride: a_bytes as i64,
            ..Default::default()
        },
        b_base: AddrGen {
            base: region(1),
            cta_stride: ((tn * k) as f64 * dt.bytes) as i64,
            cta_mod: tiles_n,
            iter_stride: b_bytes as i64,
            ..Default::default()
        },
        dt,
    };
    let _ = (ml.tile_m, ml.tile_n);
    // MMA issuer warp.
    let mut mma = Asm::new();
    mma.op("ELECT P0, URZ, PT");
    let top = mma.here();
    let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR4], R3");
    mma.bar(w, BarOp::Wait(FULL));
    let ukk = if dt.bytes <= 1.0 { 32 } else { 16 };
    for _ in 0..(kb / ukk) {
        let pc = mma.op(&mma_opcode(Arch::Blackwell, tm, tn, ukk, dt));
        mma.single_thread(pc);
    }
    let c = mma.op("UTCBAR [UR6], URZ");
    mma.bar(c, BarOp::Arrive(EMPTY));
    if k_iters > 1 {
        mma.loop_to(top, k_iters - 1);
    }
    let c = mma.op("UTCBAR [UR24], URZ");
    mma.bar(c, BarOp::Arrive(ACC));
    // Epilogue warpgroup: TMEM -> RF -> global.
    let mut epi = Asm::new();
    epi.op("S2R R0, SR_TID.X");
    let w = epi.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR24], R3");
    epi.bar(w, BarOp::Wait(ACC));
    let cols = tn; // each thread owns one row of the 128 x tn fp32 accumulator
    for _ in 0..(cols / 32) {
        epi.op("LDTM.32x32b.x32 R24, tmem[UR12]");
    }
    let out = AddrGen {
        base: region(2),
        cta_stride: (tm * tn * 2) as i64,
        warp_stride: (32 * tn * 2) as i64,
        lane_stride: (tn * 2) as i64,
        ..Default::default()
    };
    epilogue_store(&mut epi, cols, 2.0, out);
    let roles = vec![
        producer(&ml, 0..1),
        mma.role("mma_issuer", 1..2),
        Asm::new().role("idle", 2..4),
        epi.role("epilogue", 4..8),
    ];
    let mut barriers = HashMap::new();
    barriers.insert(
        FULL,
        BarInit {
            count: 2,
            initial_phases: 0,
        },
    );
    barriers.insert(
        EMPTY,
        BarInit {
            count: 1,
            initial_phases: ml.stages,
        },
    );
    barriers.insert(
        ACC,
        BarInit {
            count: 1,
            initial_phases: 0,
        },
    );
    KernelLaunch {
        name: format!("sm100_gemm_{}_{m}x{n}x{k}_tile{tm}x{tn}", dt.name),
        grid: [(tiles_m * tiles_n) as u32, 1, 1],
        block: [256, 1, 1],
        regs_per_thread: 128,
        smem_per_cta: ((a_bytes + b_bytes) as u32) * ml.stages,
        cluster,
        source: KernelSource::Static { roles, barriers },
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        bytes: (m * k + n * k) as f64 * dt.bytes + (m * n) as f64 * 2.0,
    }
}

/// Ampere-style mma.sync GEMM (fallback / small shapes).
fn gemm_mma_sync(m: u64, n: u64, k: u64, dt: Dtype) -> KernelLaunch {
    let (tm, tn, kb) = (128u64, 128u64, 32u64);
    let k_iters = k.div_ceil(kb).max(1);
    let tiles = m.div_ceil(tm) * n.div_ceil(tn);
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("IMAD R2, R0, 0x10, RZ");
    let top = a.here();
    // cp.async the next stage (each thread 16 B x 4), commit, wait for N-2.
    for i in 0..4 {
        let pc = a.op("LDGSTS.E.BYPASS.128 [R5+0x0], desc[UR4][R2.64]");
        a.addr(
            pc,
            AddrGen {
                base: region(i % 2),
                lane_stride: 16,
                warp_stride: 512,
                cta_stride: 0,
                iter_stride: 8192,
                footprint: (m.max(n) * k * 2).max(1),
                ..Default::default()
            },
        );
    }
    a.op("LDGDEPBAR");
    a.op("DEPBAR.LE SB0, 0x2");
    a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
    // Each warp: 64x32 warp tile => per k16 step 4x4 HMMA.16816; kb=32 => 2 steps.
    for _ in 0..2 {
        a.ops("LDSM.16.M88.4 R8, [R6]", 4);
        a.ops(&mma_opcode(Arch::Ampere, 16, 8, 16, dt), 16);
    }
    a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
    if k_iters > 1 {
        a.loop_to(top, k_iters - 1);
    }
    let out = AddrGen {
        base: region(2),
        cta_stride: (tm * tn * 2) as i64,
        warp_stride: 4096,
        lane_stride: 16,
        ..Default::default()
    };
    epilogue_store(&mut a, 64, 2.0, out);
    KernelLaunch {
        name: format!("mma_sync_gemm_{}_{m}x{n}x{k}", dt.name),
        grid: [tiles as u32, 1, 1],
        block: [256, 1, 1],
        regs_per_thread: 128,
        smem_per_cta: 3 * (tm + tn) as u32 * kb as u32 * 2,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..8)],
            barriers: HashMap::new(),
        },
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        bytes: (m * k + n * k) as f64 * dt.bytes + (m * n) as f64 * 2.0,
    }
}

/// GEMM `C[m,n] = A[m,k] @ B[k,n]` lowered for the GPU's architecture.
pub fn gemm(g: &GpuSpec, m: u64, n: u64, k: u64, dtype_name: &str) -> KernelLaunch {
    let dt = dtype(dtype_name);
    match g.arch {
        Arch::Hopper => gemm_hopper(m, n, k, dt),
        Arch::Blackwell => gemm_blackwell(m, n, k, dt),
        Arch::Ampere => gemm_mma_sync(m, n, k, dt),
    }
}

/// Grouped GEMM (MoE experts): one launch, tiles from every group.
pub fn grouped_gemm(g: &GpuSpec, ms: &[u64], n: u64, k: u64, dtype_name: &str) -> KernelLaunch {
    let total_tiles: u64 = ms.iter().map(|&m| m.div_ceil(128)).sum();
    let mut kl = gemm(g, 128 * total_tiles.max(1), n, k, dtype_name);
    kl.name = format!(
        "grouped_gemm_{}x[{}]x{n}x{k}",
        ms.len(),
        ms.iter().sum::<u64>()
    );
    kl.flops = ms
        .iter()
        .map(|&m| 2.0 * m as f64 * n as f64 * k as f64)
        .sum();
    // Experts with zero tokens produce no tiles; otherwise all work is real.
    if total_tiles == 0 {
        kl.grid = [0, 1, 1];
    }
    kl
}

/// Attention forward. `sq`/`skv` per sequence; GQA via `heads_kv`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
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
    let decode = sq * group <= 128 && sq <= 16;
    // Rows per CTA tile, KV block.
    let (mut tile_q, mut kv_blk) = match g.arch {
        Arch::Blackwell => (128u64, 128u64),
        _ => (if decode { 64 } else { 128 }, 128u64),
    };
    // MLA-style heads (d > 256, e.g. DeepSeek's 576-wide latent) share one
    // smem buffer between K and V, as FlashMLA does.
    let kv_shared = d > 256;
    let kv_mult = if kv_shared { 1 } else { 2 };
    let fits =
        |tq: u64, kb: u64| (kv_mult * 2 * kb * d + tq * d) * 2 + 1024 <= g.smem_per_sm_bytes as u64;
    while !fits(tile_q, kv_blk) && kv_blk > 32 {
        kv_blk /= 2;
    }
    while !fits(tile_q, kv_blk) && tile_q > 64 {
        tile_q /= 2;
    }
    let rows_total = if decode {
        batch * heads_kv * (sq * group).div_ceil(tile_q)
    } else {
        batch * heads_q * sq.div_ceil(tile_q)
    };
    let eff_kv = if causal && !decode {
        skv.div_ceil(2)
    } else {
        skv
    };
    let kv_iters_total = eff_kv.div_ceil(kv_blk).max(1);
    // Split-KV for decode to fill the machine.
    let splits = if decode {
        ((2 * g.sm_count as u64) / rows_total.max(1)).clamp(1, kv_iters_total)
    } else {
        1
    };
    let k_iters = kv_iters_total.div_ceil(splits);
    let ctas = rows_total * splits;
    let kv_bytes = (kv_blk * d) as f64 * dt.bytes;
    let kv_fp = (batch * heads_kv * skv * d) as f64 * dt.bytes;
    let ml = Mainloop {
        tile_m: tile_q,
        tile_n: kv_blk,
        k_blk: d,
        k_iters,
        stages: 2,
        a_bytes: if kv_shared {
            kv_bytes as u64 / 2
        } else {
            kv_bytes as u64
        },
        b_bytes: if kv_shared {
            kv_bytes as u64 / 2
        } else {
            kv_bytes as u64
        },
        cluster: 1,
        a_base: AddrGen {
            base: region(3),
            cta_stride: (kv_bytes as i64) * k_iters as i64,
            iter_stride: kv_bytes as i64,
            footprint: kv_fp as u64,
            ..Default::default()
        },
        b_base: AddrGen {
            base: region(4),
            cta_stride: (kv_bytes as i64) * k_iters as i64,
            iter_stride: kv_bytes as i64,
            footprint: kv_fp as u64,
            ..Default::default()
        },
        dt,
    };
    let _ = (ml.tile_m, ml.tile_n, ml.k_blk);
    let flops_real = 4.0 * batch as f64 * heads_q as f64 * sq as f64 * eff_kv as f64 * d as f64;
    let bytes = (kv_mult * batch * heads_kv * skv * d) as f64 * dt.bytes
        + (2 * batch * heads_q * sq * d) as f64 * dt.bytes;
    let mut barriers = HashMap::new();
    barriers.insert(
        FULL,
        BarInit {
            count: 2,
            initial_phases: 0,
        },
    );
    let out = AddrGen {
        base: region(5),
        cta_stride: (tile_q * d * 2) as i64,
        warp_stride: (16 * d * 2) as i64,
        lane_stride: 16,
        ..Default::default()
    };
    let (roles, block, name) = match g.arch {
        Arch::Blackwell => {
            barriers.insert(
                EMPTY,
                BarInit {
                    count: 1,
                    initial_phases: ml.stages,
                },
            );
            barriers.insert(
                S_READY,
                BarInit {
                    count: 1,
                    initial_phases: 0,
                },
            );
            barriers.insert(
                P_READY,
                BarInit {
                    count: 4,
                    initial_phases: 0,
                },
            );
            barriers.insert(
                ACC,
                BarInit {
                    count: 1,
                    initial_phases: 0,
                },
            );
            let mut mma = Asm::new();
            let top = mma.here();
            let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR4], R3");
            mma.bar(w, BarOp::Wait(FULL));
            for _ in 0..(d / 16) {
                let pc = mma.op(&mma_opcode(Arch::Blackwell, tile_q, kv_blk, 16, dt));
                mma.single_thread(pc);
            }
            let c = mma.op("UTCBAR [UR26], URZ");
            mma.bar(c, BarOp::Arrive(S_READY));
            let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR28], R3");
            mma.bar(w, BarOp::Wait(P_READY));
            for _ in 0..(kv_blk / 16) {
                let pc = mma.op(&mma_opcode(Arch::Blackwell, tile_q, d, 16, dt));
                mma.single_thread(pc);
            }
            let c = mma.op("UTCBAR [UR6], URZ");
            mma.bar(c, BarOp::Arrive(EMPTY));
            if k_iters > 1 {
                mma.loop_to(top, k_iters - 1);
            }
            let c = mma.op("UTCBAR [UR24], URZ");
            mma.bar(c, BarOp::Arrive(ACC));
            let mut sm = Asm::new();
            let top = sm.here();
            let w = sm.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR26], R3");
            sm.bar(w, BarOp::Wait(S_READY));
            for _ in 0..(kv_blk / 32) {
                sm.op("LDTM.32x32b.x32 R24, tmem[UR12]");
            }
            softmax_body(&mut sm, kv_blk);
            for _ in 0..(kv_blk / 64) {
                sm.op("STTM.32x32b.x32 tmem[UR13], R60");
            }
            let r = sm.op("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR28], RZ");
            sm.bar(r, BarOp::Arrive(P_READY));
            if k_iters > 1 {
                sm.loop_to(top, k_iters - 1);
            }
            let w = sm.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR24], R3");
            sm.bar(w, BarOp::Wait(ACC));
            for _ in 0..(d / 32) {
                sm.op("LDTM.32x32b.x32 R24, tmem[UR12]");
            }
            epilogue_store(&mut sm, d, 2.0, out);
            (
                vec![
                    producer(&ml, 0..1),
                    mma.role("mma_issuer", 1..2),
                    Asm::new().role("idle", 2..4),
                    sm.role("softmax", 4..8),
                ],
                256u32,
                format!(
                    "sm100_fmha_{}b{batch}_hq{heads_q}_hkv{heads_kv}_sq{sq}_skv{skv}_d{d}",
                    if decode { "decode_" } else { "" }
                ),
            )
        }
        _ => {
            let wgs = if decode { 1u32 } else { 2 };
            barriers.insert(
                EMPTY,
                BarInit {
                    count: 4 * wgs,
                    initial_phases: ml.stages,
                },
            );
            let mmas = [
                (64u64, kv_blk, 16u64, (d / 16) as usize),
                (64u64, d, 16u64, (kv_blk / 16) as usize),
            ];
            let elems = 64 * kv_blk / 128;
            let cons = hopper_consumer(&ml, 4..(4 + 4 * wgs), 64, &mmas, elems, out, 64 * d / 128);
            (
                vec![
                    producer(&ml, 0..1),
                    Asm::new().role("producer_idle", 1..4),
                    cons,
                ],
                (128 + 128 * wgs) as u32,
                format!(
                    "sm90_fmha_{}b{batch}_hq{heads_q}_hkv{heads_kv}_sq{sq}_skv{skv}_d{d}",
                    if decode { "decode_" } else { "" }
                ),
            )
        }
    };
    KernelLaunch {
        name,
        grid: [ctas as u32, 1, 1],
        block: [block, 1, 1],
        regs_per_thread: 160,
        smem_per_cta: ((kv_mult as f64 * kv_bytes) as u32) * ml.stages + (tile_q * d * 2) as u32,
        cluster: 1,
        source: KernelSource::Static { roles, barriers },
        flops: flops_real,
        bytes,
    }
}

/// Memory-bound elementwise kernel over `n` elements: `n_in` input
/// tensors, one output, `flops_per_elem` FP32 ops each.
pub fn elementwise(
    name: &str,
    n: u64,
    n_in: u32,
    elem_bytes: f64,
    flops_per_elem: u32,
) -> KernelLaunch {
    let threads = 256u64;
    let vec_elems = (16.0 / elem_bytes).max(1.0) as u64;
    let ctas = n.div_ceil(threads * vec_elems).max(1);
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("S2R R1, SR_CTAID.X");
    a.op("IMAD R2, R1, 0x100, R0");
    a.op("ISETP.GE.AND P0, PT, R2, UR5, PT");
    for i in 0..n_in {
        let pc = a.op(&format!("LDG.E.128 R{}, desc[UR4][R2.64]", 8 + 4 * i));
        a.addr(
            pc,
            AddrGen {
                base: region(10 + i as u64),
                cta_stride: (threads * 16) as i64,
                warp_stride: 512,
                lane_stride: 16,
                ..Default::default()
            },
        );
    }
    let fl = (flops_per_elem as u64 * vec_elems).max(1) as usize;
    a.ops("FFMA R8, R8, R9, R10", fl);
    let pc = a.op("STG.E.128 desc[UR4][R2.64], R8");
    a.addr(
        pc,
        AddrGen {
            base: region(20),
            cta_stride: (threads * 16) as i64,
            warp_stride: 512,
            lane_stride: 16,
            ..Default::default()
        },
    );
    KernelLaunch {
        name: name.into(),
        grid: [ctas as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: 32,
        smem_per_cta: 0,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..8)],
            barriers: HashMap::new(),
        },
        flops: n as f64 * flops_per_elem as f64,
        bytes: n as f64 * elem_bytes * (n_in as f64 + 1.0),
    }
}

/// Row-wise normalisation / softmax (RMSNorm, LayerNorm, softmax): one CTA
/// per row, two-level (warp shuffle + smem) reduction.
pub fn row_norm(name: &str, rows: u64, cols: u64, elem_bytes: f64) -> KernelLaunch {
    let threads = 256u64.min((cols / 8).max(32)).next_power_of_two();
    let per_thread = cols.div_ceil(threads);
    let vec_elems = (16.0 / elem_bytes) as u64;
    let n_ld = per_thread.div_ceil(vec_elems).max(1);
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("S2R R1, SR_CTAID.X");
    a.op("IMAD R2, R1, UR6, R0");
    for i in 0..n_ld {
        let pc = a.op("LDG.E.128 R8, desc[UR4][R2.64]");
        a.addr(
            pc,
            AddrGen {
                base: region(30) + i * threads * 16,
                cta_stride: (cols as f64 * elem_bytes) as i64,
                warp_stride: 512,
                lane_stride: 16,
                ..Default::default()
            },
        );
    }
    a.ops("FFMA R12, R8, R8, R12", per_thread as usize);
    for s in [16, 8, 4, 2, 1] {
        a.op(&format!("SHFL.BFLY PT, R13, R12, 0x{s:x}, 0x1f"));
        a.op("FADD R12, R12, R13");
    }
    a.op("STS [R4], R12");
    a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
    a.op("LDS R12, [R4]");
    for s in [4, 2, 1] {
        a.op(&format!("SHFL.BFLY PT, R13, R12, 0x{s:x}, 0x1f"));
        a.op("FADD R12, R12, R13");
    }
    a.op("MUFU.RSQ R14, R12");
    let pc = a.op("LDG.E.128.CONSTANT R16, desc[UR4][R6.64]");
    a.addr(
        pc,
        AddrGen {
            base: region(31),
            lane_stride: 16,
            warp_stride: 512,
            ..Default::default()
        },
    );
    a.ops("FMUL R8, R8, R14", per_thread as usize);
    a.ops(
        "F2FP.BF16.F32.PACK_AB R8, R9, R8",
        (per_thread / 2).max(1) as usize,
    );
    for i in 0..n_ld {
        let pc = a.op("STG.E.128 desc[UR4][R2.64], R8");
        a.addr(
            pc,
            AddrGen {
                base: region(32) + i * threads * 16,
                cta_stride: (cols as f64 * elem_bytes) as i64,
                warp_stride: 512,
                lane_stride: 16,
                ..Default::default()
            },
        );
    }
    let warps = (threads / 32) as u32;
    KernelLaunch {
        name: name.into(),
        grid: [rows as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: 40,
        smem_per_cta: 256,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..warps)],
            barriers: HashMap::new(),
        },
        flops: (rows * cols * 4) as f64,
        bytes: (rows * cols) as f64 * elem_bytes * 2.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsim_hw::presets;

    #[test]
    fn programs_build() {
        let h = presets::h100_sxm();
        let b = presets::b200();
        for k in [
            gemm(&h, 4096, 4096, 4096, "bf16"),
            gemm(&b, 4096, 4096, 4096, "bf16"),
            gemm(&presets::h100_sxm(), 256, 256, 256, "fp8"),
            attention(&h, 4, 32, 8, 1, 4096, 128, false),
            attention(&b, 1, 32, 8, 4096, 4096, 128, true),
            elementwise("silu_mul", 1 << 24, 2, 2.0, 4),
            row_norm("rmsnorm", 4096, 8192, 2.0),
        ] {
            assert!(k.num_ctas() > 0, "{}", k.name);
            assert!(
                crate::sm::Sm::occupancy(if k.name.starts_with("sm100") { &b } else { &h }, &k)
                    >= 1,
                "{} does not fit",
                k.name
            );
        }
    }
}
