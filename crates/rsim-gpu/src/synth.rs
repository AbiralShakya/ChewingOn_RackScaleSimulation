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
    /// `n` copies of `text`. A `{r}` placeholder becomes a rotating register
    /// (R100, R101, ... R131) so the copies are independent per-element work
    /// (as in real kernels) rather than one dependent chain.
    pub fn ops(&mut self, text: &str, n: usize) {
        if text.contains("{r}") {
            for i in 0..n {
                self.op(&text.replace("{r}", &format!("R{}", 100 + i % 32)));
            }
        } else {
            for _ in 0..n {
                self.op(text);
            }
        }
    }
    pub fn here(&self) -> u32 {
        self.pc
    }
    /// `n` warp-level HMMA m16n8k16 instructions over the independent
    /// accumulator fragments of a warp tile. Real kernels accumulate into
    /// distinct registers per (m16, n8) fragment, so consecutive HMMAs are
    /// independent and pipeline at the tensor-pipe rate; only the same
    /// fragment in the next k-step depends on the previous result.
    pub fn hmma(&mut self, n: usize, a_reg: u32, b_reg: u32, dt_sass: &str) {
        for i in 0..n {
            let acc = 64 + 4 * (i % 16) as u32;
            self.op(&format!(
                "HMMA.16816.F32.{dt_sass} R{acc}, R{a_reg}, R{b_reg}, R{acc}"
            ));
        }
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
        "fp4" | "nvfp4" | "fp4_k64" => Dtype {
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

pub(crate) fn mma_opcode(arch: Arch, m: u64, n: u64, k: u64, dt: Dtype) -> String {
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
pub(crate) fn region(i: u64) -> u64 {
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

/// Online-softmax work for `elems` scores per thread: row max, scale and
/// exponentiate every score (independent per element), row sum, rescale and
/// pack P to bf16. Reductions use four partial accumulators.
fn softmax_body(a: &mut Asm, elems: u64) {
    let e = elems as usize;
    for i in 0..(e / 2) {
        a.op(&format!(
            "FMNMX R{}, R{}, R{}, !PT",
            40 + i % 4,
            100 + (2 * i) % 32,
            40 + i % 4
        ));
    }
    a.op("FMNMX R40, R40, R41, !PT");
    a.op("FMNMX R42, R42, R43, !PT");
    a.op("FMNMX R40, R40, R42, !PT");
    a.op("SHFL.BFLY PT, R41, R40, 0x1, 0x1f");
    a.op("FMNMX R40, R40, R41, !PT");
    a.op("SHFL.BFLY PT, R41, R40, 0x2, 0x1f");
    a.op("FMNMX R40, R40, R41, !PT");
    a.ops("FFMA {r}, {r}, R44, -R40", e);
    a.ops("MUFU.EX2 {r}, {r}", e);
    for i in 0..(e / 2) {
        a.op(&format!(
            "FADD R{}, R{}, R{}",
            48 + i % 4,
            100 + (2 * i) % 32,
            48 + i % 4
        ));
    }
    a.op("FMUL R45, R45, R46");
    a.ops("F2FP.BF16.F32.PACK_AB {r}, {r}, {r}", e / 2);
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
fn gemm_blackwell(m: u64, n: u64, k: u64, dt: Dtype, sms: u64, ultra_fp4: bool) -> KernelLaunch {
    let kb = if ultra_fp4 {
        192
    } else if dt.bytes <= 1.0 {
        128
    } else {
        64u64
    };
    let tm = 128u64;
    let tn = if n >= 256 && m.div_ceil(tm) * n.div_ceil(256) * 2 > sms {
        256
    } else {
        128
    };
    let k_total = k.div_ceil(kb).max(1);
    let tiles_m = m.div_ceil(tm);
    let tiles_n = n.div_ceil(tn);
    let split = split_k(sms, tiles_m * tiles_n, k_total);
    let k_iters = k_total.div_ceil(split);
    let a_bytes = (tm * kb) as f64 * dt.bytes;
    let b_bytes = (tn * kb) as f64 * dt.bytes;
    let cluster = if tiles_m >= 2 && split == 1 { 2 } else { 1 };
    // cuBLAS/CUTLASS sm100 GEMMs are persistent: with cluster launch control
    // a CTA that finishes a tile's mainloop starts the next tile while its
    // epilogue drains the other half of a double-buffered TMEM accumulator.
    // Here that overlap comes from two co-resident CTAs per SM, each with
    // half of the four-stage smem ring, whenever there is more than one
    // wave of tiles. One wave keeps a single CTA with the full ring.
    // Against Nsight Compute cycles on B300 (research/ssm_vs_attention),
    // one CTA per SM was 8-30% slow on multi-wave GEMMs; two are within
    // about 10%.
    let multi_wave = tiles_m * tiles_n * split > sms;
    let stages = if multi_wave { 2 } else { 4 };
    let ml = Mainloop {
        tile_m: tm,
        tile_n: tn,
        k_blk: kb,
        k_iters,
        stages,
        a_bytes: a_bytes as u64,
        b_bytes: b_bytes as u64,
        cluster,
        a_base: operand_a(tm, k, k_iters, kb, tiles_n, split, dt),
        b_base: operand_b(tn, k, k_iters, kb, tiles_n, split, dt),
        dt,
    };
    let _ = (ml.tile_m, ml.tile_n);
    // MMA issuer warp.
    let mut mma = Asm::new();
    mma.op("ELECT P0, URZ, PT");
    let top = mma.here();
    let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR4], R3");
    mma.bar(w, BarOp::Wait(FULL));
    let ukk = if ultra_fp4 {
        96
    } else if dt.name == "fp4" {
        64
    } else if dt.bytes <= 1.0 {
        32
    } else {
        16
    };
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
    // Multi-wave tiles are written as whole lines (CUTLASS stages the tile
    // through smem and stores it with TMA). Single-wave and split-K shapes
    // keep one row per lane: that pattern's extra traffic stands in for the
    // split-K reduction and partial-tile costs the model does not have, and
    // it matches CUDA-graph timing of decode GEMMs within about 5%.
    let out = AddrGen {
        base: region(2),
        cta_stride: (tm * tn * 2) as i64,
        warp_stride: (32 * tn * 2) as i64,
        lane_stride: if multi_wave { 16 } else { (tn * 2) as i64 },
        ..Default::default()
    };
    epilogue_store(&mut epi, cols, if split > 1 { 4.0 } else { 2.0 }, out);
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
        name: format!(
            "sm100_gemm_{}_{m}x{n}x{k}_tile{tm}x{tn}{}",
            dt.name,
            if split > 1 {
                format!("_splitk{split}")
            } else {
                String::new()
            }
        ),
        grid: [(tiles_m * tiles_n * split) as u32, 1, 1],
        block: [256, 1, 1],
        regs_per_thread: 128,
        smem_per_cta: ((a_bytes + b_bytes) as u32) * ml.stages,
        cluster,
        source: KernelSource::Static { roles, barriers },
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        bytes: (m * k + n * k) as f64 * dt.bytes + (m * n) as f64 * 2.0 * split as f64,
    }
}

/// Split-K factor for GEMMs whose output tiles cannot fill the GPU (decode
/// shapes, M <= 128): cuBLAS/CUTLASS then split the K loop across CTAs and
/// reduce fp32 partials, so the weight stream uses every SM.
fn split_k(sms: u64, tiles: u64, k_iters: u64) -> u64 {
    if tiles * 2 > sms {
        return 1;
    }
    (sms / tiles.max(1)).min(k_iters / 4).max(1)
}

/// A operand (activations, `m x k`) addresses: CTA `i` works on tile
/// `(i / split) / tiles_n` rows and k-slice `i % split`.
fn operand_a(
    tm: u64,
    k: u64,
    k_iters: u64,
    kb: u64,
    tiles_n: u64,
    split: u64,
    dt: Dtype,
) -> AddrGen {
    let a_bytes = (tm * kb) as f64 * dt.bytes;
    if split > 1 {
        // Few rows, shared by every CTA: one m x k slab per k-slice.
        AddrGen {
            base: region(0),
            cta_stride: (a_bytes * k_iters as f64) as i64,
            cta_mod: split,
            iter_stride: a_bytes as i64,
            ..Default::default()
        }
    } else {
        AddrGen {
            base: region(0),
            cta_stride: ((tm * k) as f64 * dt.bytes) as i64,
            cta_div: tiles_n,
            iter_stride: a_bytes as i64,
            ..Default::default()
        }
    }
}

/// B operand (weights, `n x k`): with split-K every CTA streams its own
/// `tn x (k / split)` slice exactly once.
fn operand_b(
    tn: u64,
    k: u64,
    k_iters: u64,
    kb: u64,
    tiles_n: u64,
    split: u64,
    dt: Dtype,
) -> AddrGen {
    let b_bytes = (tn * kb) as f64 * dt.bytes;
    if split > 1 {
        AddrGen {
            base: region(1),
            cta_stride: (b_bytes * k_iters as f64) as i64,
            iter_stride: b_bytes as i64,
            ..Default::default()
        }
    } else {
        AddrGen {
            base: region(1),
            cta_stride: ((tn * k) as f64 * dt.bytes) as i64,
            cta_mod: tiles_n,
            iter_stride: b_bytes as i64,
            ..Default::default()
        }
    }
}

/// Ampere-style mma.sync GEMM (fallback / small shapes).
fn gemm_mma_sync(m: u64, n: u64, k: u64, dt: Dtype, sms: u64) -> KernelLaunch {
    // cuBLAS/CUTLASS sm80 tile shapes: 64x64 (4 warps, 32x32 warp tiles) for
    // decode-sized M, else 128x256 (8 warps, 64x64 warp tiles); 3-stage
    // cp.async ring, k-block 32.
    let small = m <= 64;
    // (tile_m, tile_n, k_block, warps, stages): cuBLAS picks a deep 6-stage
    // 64x64x64 ring for skinny decode GEMMs (enough bytes in flight per SM
    // to stream weights with a few dozen CTAs) and 128x256x32, 3 stages,
    // for large ones.
    let (tm, tn, kb, warps, stages) = if small {
        (64u64, 64u64, 64u64, 4u64, 6u64)
    } else {
        (128, 256, 32, 8, 3)
    };
    let (wm, wn) = if small { (32u64, 32u64) } else { (64, 64) };
    let k_total = k.div_ceil(kb).max(1);
    let tiles_n = n.div_ceil(tn);
    let tiles = m.div_ceil(tm) * tiles_n;
    let split = split_k(sms, tiles, k_total);
    let k_iters = k_total.div_ceil(split);
    let threads = 32 * warps;
    let a_st = operand_a(tm, k, k_iters, kb, tiles_n, split, dt);
    let b_st = operand_b(tn, k, k_iters, kb, tiles_n, split, dt);
    let ld = |a: &mut Asm, bytes: u64, g: AddrGen| {
        for i in 0..bytes.div_ceil(16 * threads) {
            let pc = a.op("LDGSTS.E.BYPASS.128 [R5+0x0], desc[UR4][R2.64]");
            a.addr(
                pc,
                AddrGen {
                    base: g.base + i * 16 * threads,
                    lane_stride: 16,
                    warp_stride: 512,
                    ..g
                },
            );
        }
    };
    let mut a = Asm::new();
    a.op("S2R R0, SR_TID.X");
    a.op("IMAD R2, R0, 0x10, RZ");
    let a_tile = (tm.min(m.next_power_of_two().max(16)) * kb) as f64 * dt.bytes;
    for _ in 0..(stages - 1).min(k_iters) {
        ld(&mut a, a_tile as u64, a_st);
        ld(&mut a, (tn * kb) * dt.bytes as u64, b_st);
        a.op("LDGDEPBAR");
    }
    // CUTLASS sm80 multistage mainloop: fragments are double-buffered, and
    // the last k16 step of a tile issues the next tile's cp.async, waits for
    // the oldest stage, syncs and loads the next tile's first fragments
    // *before* its own HMMAs, so the barrier is hidden behind tensor work.
    let steps = (kb / 16) as usize;
    let ldsm = ((wm + wn) / 16) as usize;
    let frag = |j: usize| (8 + 8 * (j % 2) as u32, 12 + 8 * (j % 2) as u32);
    let wait_stage = |a: &mut Asm| {
        a.op(&format!("DEPBAR.LE SB0, 0x{:x}", stages - 2));
        a.op("BAR.SYNC.DEFER_BLOCKING 0x0");
    };
    wait_stage(&mut a);
    a.ops(&format!("LDSM.16.M88.4 R{}, [R6]", frag(0).0), ldsm);
    let top = a.here();
    for j in 0..steps {
        if j + 1 < steps {
            a.ops(&format!("LDSM.16.M88.4 R{}, [R6]", frag(j + 1).0), ldsm);
        } else {
            ld(&mut a, a_tile as u64, a_st);
            ld(&mut a, (tn * kb) * dt.bytes as u64, b_st);
            a.op("LDGDEPBAR");
            wait_stage(&mut a);
            a.ops(&format!("LDSM.16.M88.4 R{}, [R6]", frag(j + 1).0), ldsm);
        }
        let (ra, rb) = frag(j);
        a.hmma(((wm / 16) * (wn / 8)) as usize, ra, rb, dt.sass);
    }
    if k_iters > 1 {
        a.loop_to(top, k_iters - 1);
    }
    let out = AddrGen {
        base: region(2),
        cta_stride: (tm * tn * 2) as i64,
        warp_stride: (wm * wn * 2) as i64,
        lane_stride: 16,
        ..Default::default()
    };
    epilogue_store(&mut a, wm * wn / 32, if split > 1 { 4.0 } else { 2.0 }, out);
    KernelLaunch {
        name: format!(
            "mma_sync_gemm_{}_{m}x{n}x{k}_tile{tm}x{tn}{}",
            dt.name,
            if split > 1 {
                format!("_splitk{split}")
            } else {
                String::new()
            }
        ),
        grid: [(tiles * split) as u32, 1, 1],
        block: [threads as u32, 1, 1],
        regs_per_thread: if small { 128 } else { 232 },
        smem_per_cta: stages as u32 * (tm + tn) as u32 * kb as u32 * 2,
        cluster: 1,
        source: KernelSource::Static {
            roles: vec![a.role("all", 0..warps as u32)],
            barriers: HashMap::new(),
        },
        flops: 2.0 * m as f64 * n as f64 * k as f64,
        bytes: (m * k + n * k) as f64 * dt.bytes + (m * n) as f64 * 2.0 * split as f64,
    }
}

/// GEMM `C[m,n] = A[m,k] @ B[k,n]` lowered for the GPU's architecture.
pub fn gemm(g: &GpuSpec, m: u64, n: u64, k: u64, dtype_name: &str) -> KernelLaunch {
    let dt = dtype(dtype_name);
    match g.arch {
        Arch::Hopper => gemm_hopper(m, n, k, dt),
        Arch::Blackwell => {
            // Blackwell Ultra: the best FP4 kernels use the K=96 UMMA
            // (`fp4_k64` forces the K=64 form).
            let ultra = dt.name == "fp4"
                && dtype_name != "fp4_k64"
                && g.tensor.flops_per_cycle_per_sm.contains_key("fp4_k96");
            gemm_blackwell(m, n, k, dt, g.sm_count as u64, ultra)
        }
        Arch::Ampere => gemm_mma_sync(m, n, k, dt, g.sm_count as u64),
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
    attention_with(
        g,
        batch,
        heads_q,
        heads_kv,
        sq,
        skv,
        d,
        causal,
        AttnOpts::default(),
    )
}

/// Layout and schedule options of [`attention_with`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AttnOpts {
    /// `Some(true)` streams one cache tensor per KV head that serves as both
    /// K and V (MLA's latent, Grouped-Tied Attention's tied state, GLA's
    /// latent heads); `None` keeps the default, tied only for heads wider
    /// than 256.
    pub kv_tied: Option<bool>,
    /// Blackwell decode only: softmax warpgroups that split each K/V block
    /// (FlashMLA's schedule). With 2, the MMA warp issues one half's
    /// `S = Q K^T` while the other half's softmax runs, instead of the
    /// single-warpgroup chain QK -> softmax -> PV per block. 0 or 1 = one.
    pub kv_warpgroups: u32,
}

/// [`attention`] with explicit [`AttnOpts`].
#[allow(clippy::too_many_arguments)]
pub fn attention_with(
    g: &GpuSpec,
    batch: u64,
    heads_q: u64,
    heads_kv: u64,
    sq: u64,
    skv: u64,
    d: u64,
    causal: bool,
    opts: AttnOpts,
) -> KernelLaunch {
    let kv_tied = opts.kv_tied;
    if g.arch == Arch::Ampere {
        // No TMA / wgmma / tcgen05: FlashAttention-2 on mma.sync.
        return crate::synth_ssm::attention_fa2(g, batch, heads_q, heads_kv, sq, skv, d, causal);
    }
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
    let kv_shared = kv_tied.unwrap_or(d > 256);
    let kv_mult = if kv_shared { 1 } else { 2 };
    let fits =
        |tq: u64, kb: u64| (kv_mult * 2 * kb * d + tq * d) * 2 + 1024 <= g.smem_per_sm_bytes as u64;
    while !fits(tile_q, kv_blk) && kv_blk > 32 {
        kv_blk /= 2;
    }
    while !fits(tile_q, kv_blk) && tile_q > 64 {
        tile_q /= 2;
    }
    // Blackwell prefill packs two Q tiles per CTA (FA4 `q_stage = 2`).
    // Split-block decode runs the same two-stage pipeline on the two halves
    // of each K/V block of a single Q tile.
    let split_kv = g.arch == Arch::Blackwell && decode && opts.kv_warpgroups >= 2;
    let q_stage: u32 = if (g.arch == Arch::Blackwell && !decode && sq > tile_q) || split_kv {
        2
    } else {
        1
    };
    // K/V columns each stage handles per block, and Q tiles held in smem.
    let kv_w = if split_kv { kv_blk / 2 } else { kv_blk };
    let q_tiles = if split_kv { 1 } else { q_stage as u64 };
    let rows_total = if decode {
        batch * heads_kv * (sq * group).div_ceil(tile_q)
    } else {
        batch * heads_q * sq.div_ceil(tile_q * q_stage as u64)
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
            // FlashAttention-4 schedule: `q_stage` 128-row Q tiles per CTA
            // share every K/V tile; one warp issues S_i = Q_i K^T for all
            // tiles, then O_i += P_i V as each softmax warpgroup hands back
            // P_i, so softmax of one tile overlaps the MMAs of the other.
            let mut mma = Asm::new();
            let top = mma.here();
            let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR4], R3");
            mma.bar(w, BarOp::Wait(FULL));
            for st in 0..q_stage {
                for _ in 0..(d / 16) {
                    let pc = mma.op(&mma_opcode(Arch::Blackwell, tile_q, kv_w, 16, dt));
                    mma.single_thread(pc);
                }
                let c = mma.op("UTCBAR [UR26], URZ");
                mma.bar(c, BarOp::Arrive(S_READY + 10 * st));
            }
            for st in 0..q_stage {
                let w = mma.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P2, [UR28], R3");
                mma.bar(w, BarOp::Wait(P_READY + 10 * st));
                for _ in 0..(kv_w / 16) {
                    // P comes from tensor memory (TS form): only V is read from smem.
                    let ts = mma_opcode(Arch::Blackwell, tile_q, d, 16, dt)
                        .replace("gdesc[UR8], gdesc", "tmem[UR8], gdesc");
                    let pc = mma.op(&ts);
                    mma.single_thread(pc);
                }
            }
            let c = mma.op("UTCBAR [UR6], URZ");
            mma.bar(c, BarOp::Arrive(EMPTY));
            if k_iters > 1 {
                mma.loop_to(top, k_iters - 1);
            }
            let c = mma.op("UTCBAR [UR24], URZ");
            mma.bar(c, BarOp::Arrive(ACC));
            let mut roles = vec![
                producer(&ml, 0..1),
                mma.role("mma_issuer", 1..2),
                Asm::new().role("idle", 2..4),
            ];
            for st in 0..q_stage {
                barriers.insert(
                    S_READY + 10 * st,
                    BarInit {
                        count: 1,
                        initial_phases: 0,
                    },
                );
                barriers.insert(
                    P_READY + 10 * st,
                    BarInit {
                        count: 4,
                        initial_phases: 0,
                    },
                );
                let mut sm = Asm::new();
                let top = sm.here();
                let w = sm.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR26], R3");
                sm.bar(w, BarOp::Wait(S_READY + 10 * st));
                for _ in 0..(kv_w / 32) {
                    sm.op("LDTM.32x32b.x32 R24, tmem[UR12]");
                }
                softmax_body(&mut sm, kv_w);
                for _ in 0..(kv_w / 64) {
                    sm.op("STTM.32x32b.x32 tmem[UR13], R60");
                }
                let r = sm.op("SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR28], RZ");
                sm.bar(r, BarOp::Arrive(P_READY + 10 * st));
                if k_iters > 1 {
                    sm.loop_to(top, k_iters - 1);
                }
                let w = sm.op("SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR24], R3");
                sm.bar(w, BarOp::Wait(ACC));
                for _ in 0..(d / 32) {
                    sm.op("LDTM.32x32b.x32 R24, tmem[UR12]");
                }
                epilogue_store(&mut sm, d, 2.0, out);
                let w0 = 4 + 4 * st;
                roles.push(sm.role("softmax", w0..w0 + 4));
            }
            (
                roles,
                128 * (1 + q_stage),
                format!(
                    "sm100_fmha_{}b{batch}_hq{heads_q}_hkv{heads_kv}_sq{sq}_skv{skv}_d{d}{}",
                    if decode { "decode_" } else { "" },
                    if split_kv { "_kvwg2" } else { "" }
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
                (128 + 128 * wgs),
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
        smem_per_cta: ((kv_mult as f64 * kv_bytes) as u32) * ml.stages
            + (q_tiles * tile_q * d * 2) as u32,
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
