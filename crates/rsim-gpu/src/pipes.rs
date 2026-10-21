//! Opcode class → execution resource, initiation interval and latency.

use rsim_hw::{Arch, GpuSpec};
use rsim_sass::{Inst, OpClass};

/// Per-SMSP pipes (indices into `Smsp::pipe_free`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipe {
    Fma = 0,
    Alu = 1,
    Fp64 = 2,
    Xu = 3,
    Uniform = 4,
    Tensor = 5,
    Cbu = 6,
    /// Goes to the SM-wide MIO/LSU path (shared resource, not per-SMSP).
    Mio = 7,
}
pub const NUM_PIPES: usize = 8;

/// Which scoreboard a result is tracked on (for stall attribution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sb {
    Fixed,
    Short,
    Long,
}

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub pipe: Pipe,
    /// Cycles the pipe is occupied (issue → next issue to the same pipe).
    pub ii: u32,
    /// Result latency for fixed-latency ops (or nominal for MIO ops).
    pub lat: u32,
    pub sb: Sb,
    /// Fractional part of a measured issue interval (e.g. 0.033 for an
    /// FFMA measured at 1.033 cycles); accumulated per SMSP pipe.
    pub ii_frac: f32,
}

pub fn mma_shape(inst: &Inst) -> Option<(u64, u64, u64)> {
    inst.meta
        .mma
        .map(|(m, n, k)| (m as u64, n as u64, k as u64))
}

pub fn mma_dtype(inst: &Inst) -> &'static str {
    if inst.meta.dtype.is_empty() {
        "bf16"
    } else {
        inst.meta.dtype
    }
}

/// FLOPs of one dynamic MMA instruction *per issuing warp*.
pub fn mma_flops(inst: &Inst, payload: u64) -> f64 {
    if payload > 0 {
        return payload as f64;
    }
    match (inst.class, mma_shape(inst)) {
        (OpClass::MmaSync, Some((m, n, k))) => (2 * m * n * k) as f64,
        // wgmma is issued by all 4 warps of the warpgroup.
        (OpClass::MmaWarpgroup, Some((m, n, k))) => (2 * m * n * k) as f64 / 4.0,
        (OpClass::MmaTcgen05, Some((m, n, k))) => (2 * m * n * k) as f64,
        (OpClass::MmaSync, None) => 4096.0,
        (OpClass::MmaWarpgroup, None) => 2.0 * 64.0 * 128.0 * 16.0 / 4.0,
        (OpClass::MmaTcgen05, None) => 2.0 * 128.0 * 256.0 * 16.0,
        _ => 0.0,
    }
}

pub fn timing(g: &GpuSpec, inst: &Inst) -> Timing {
    use OpClass::*;
    let l = &g.latency;
    let p = &g.pipes;
    let ii = |lanes: u32| (32 / lanes.max(1)).max(1);
    let t = |pipe, ii, lat, sb| Timing {
        pipe,
        ii,
        lat,
        sb,
        ii_frac: 0.0,
    };
    let mut out = class_timing(g, inst, l, p, ii, t);
    // Measured per-opcode timing (calibration.json) for fixed/short-latency ops.
    if !g.opcode_timing.is_empty()
        && matches!(
            inst.class,
            IntAlu | Imad | Fp32 | Fp16 | Fp64 | Sfu | Conv | Shfl | Misc
        )
    {
        let op = &*inst.opcode;
        let m = g
            .opcode_timing
            .get(op)
            .or_else(|| g.opcode_timing.get(op.split('.').next().unwrap_or(op)));
        if let Some(m) = m {
            out.lat = m.latency;
            out.ii = m.issue_interval;
            if m.issue_interval_exact >= 1.0 {
                out.ii = m.issue_interval_exact.floor() as u32;
                out.ii_frac = (m.issue_interval_exact - m.issue_interval_exact.floor()) as f32;
            }
        }
    }
    out
}

fn class_timing(
    g: &GpuSpec,
    inst: &Inst,
    l: &rsim_hw::LatencyTable,
    p: &rsim_hw::PipeSpec,
    ii: impl Fn(u32) -> u32,
    t: impl Fn(Pipe, u32, u32, Sb) -> Timing,
) -> Timing {
    use OpClass::*;
    match inst.class {
        IntAlu => t(Pipe::Alu, ii(p.int_lanes), l.int_alu, Sb::Fixed),
        Imad => t(Pipe::Fma, ii(p.fp32_lanes / 2), l.imad, Sb::Fixed),
        Uniform => t(Pipe::Uniform, 1, l.int_alu, Sb::Fixed),
        Fp32 => t(Pipe::Fma, ii(p.fp32_lanes), l.fp32, Sb::Fixed),
        Fp16 => t(Pipe::Fma, ii(p.fp16_lanes), l.fp16, Sb::Fixed),
        Fp64 => t(
            Pipe::Fp64,
            ii(p.fp64_lanes),
            l.fp64,
            if g.arch == Arch::Ampere {
                Sb::Fixed
            } else {
                Sb::Short
            },
        ),
        Sfu => t(Pipe::Xu, ii(p.sfu_lanes), l.sfu, Sb::Short),
        Conv => t(Pipe::Xu, ii(p.sfu_lanes), l.conv, Sb::Short),
        Shfl => t(Pipe::Mio, 2, l.shfl, Sb::Short),
        S2R => t(Pipe::Mio, 1, l.s2r, Sb::Short),
        Ldc => t(Pipe::Mio, 1, l.ldc, Sb::Short),
        Mem => {
            let long = !matches!(inst.mem.map(|m| m.space), Some(rsim_sass::MemSpace::Shared));
            t(
                Pipe::Mio,
                1,
                if long { g.l1.hit_latency } else { l.lds },
                if long { Sb::Long } else { Sb::Short },
            )
        }
        AsyncCopy => t(Pipe::Mio, 1, g.l1.hit_latency, Sb::Long),
        Ldsm => t(Pipe::Mio, 1, l.ldsm, Sb::Short),
        MmaSync => t(Pipe::Tensor, 1, g.tensor.hmma_latency, Sb::Short),
        MmaWarpgroup | MmaTcgen05 => t(Pipe::Tensor, 1, g.tensor.async_mma_overhead, Sb::Fixed),
        Tma => t(Pipe::Uniform, 1, 4, Sb::Fixed),
        Tmem => t(Pipe::Mio, 1, l.lds, Sb::Short),
        AsyncBarrier => t(Pipe::Mio, 1, 8, Sb::Short),
        Warpgroup => t(Pipe::Cbu, 1, 2, Sb::Fixed),
        Barrier => t(Pipe::Cbu, 1, l.bar, Sb::Fixed),
        DepBar => t(Pipe::Cbu, 1, 2, Sb::Fixed),
        Fence => t(Pipe::Mio, 1, 40, Sb::Fixed),
        Branch => t(Pipe::Cbu, 1, l.branch, Sb::Fixed),
        Exit => t(Pipe::Cbu, 1, 1, Sb::Fixed),
        Nop => t(Pipe::Alu, 1, 1, Sb::Fixed),
        Misc => t(Pipe::Alu, 1, l.misc, Sb::Fixed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsim_sass::parse::parse_inst;
    #[test]
    fn shapes() {
        let (i, _) = parse_inst(0, "HMMA.16816.F32.BF16 R4, R8, R12, R4").unwrap();
        assert_eq!(mma_shape(&i), Some((16, 8, 16)));
        assert_eq!(mma_flops(&i, 0), 4096.0);
        let (i, _) = parse_inst(0, "HGMMA.64x256x16.F32.BF16 R24, gdesc[UR4], R24").unwrap();
        assert_eq!(mma_shape(&i), Some((64, 256, 16)));
        let (i, _) = parse_inst(0, "HMMA.1688.F32.TF32 R4, R8, R12, R4").unwrap();
        assert_eq!(mma_shape(&i), Some((16, 8, 8)));
        assert_eq!(mma_dtype(&i), "tf32");
    }
}
