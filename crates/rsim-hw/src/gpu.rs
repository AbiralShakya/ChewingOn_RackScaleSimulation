use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Arch {
    /// sm_80 / sm_86
    Ampere,
    /// sm_90 / sm_90a: wgmma (HGMMA), TMA, thread-block clusters, mbarrier.
    Hopper,
    /// sm_100 / sm_100a: tcgen05 (UTC*MMA), tensor memory (TMEM), CTA pairs.
    Blackwell,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CacheSpec {
    pub size_bytes: u64,
    pub line_bytes: u32,
    pub sector_bytes: u32,
    pub assoc: u32,
    /// Load-to-use latency on hit, in SM cycles.
    pub hit_latency: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HbmSpec {
    pub capacity_gb: f64,
    /// Peak bandwidth, GB/s.
    pub bw_gbps: f64,
    /// Fraction of peak that is achievable by streaming kernels (~0.85-0.92).
    pub efficiency: f64,
    /// Unloaded latency beyond L2 miss, ns.
    pub latency_ns: f64,
    /// Extra miss latency under load: `[utilization, ns]` points, utilization
    /// being the fraction of achievable read bandwidth in use. Taken from the
    /// calibration's latency-bound read sweep over SM counts (per-SM
    /// bandwidth falls as total load rises). Empty = load-independent.
    #[serde(default)]
    pub loaded_latency: Vec<[f64; 2]>,
}

impl HbmSpec {
    /// Extra latency (ns) at `util`, interpolated; flat past the last point.
    pub fn loaded_extra_ns(&self, util: f64) -> f64 {
        let p = &self.loaded_latency;
        match p.iter().position(|q| q[0] >= util) {
            None => p.last().map_or(0.0, |q| q[1]),
            Some(0) => p[0][1] * (util / p[0][0].max(1e-9)).clamp(0.0, 1.0),
            Some(i) => {
                let (a, b) = (p[i - 1], p[i]);
                a[1] + (b[1] - a[1]) * (util - a[0]) / (b[0] - a[0]).max(1e-9)
            }
        }
    }
}

fn one() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DieLink {
    /// Per-direction die-to-die bandwidth, GB/s.
    pub gbps_per_dir: f64,
    /// Extra latency for a far-die access, SM cycles.
    pub extra_latency: u32,
    /// Measured fraction of L2 accesses an SM sees at far latency. `None`
    /// = derive from address interleave (`(dies-1)/dies`). When set, a
    /// deterministic hash of (line, SM die) picks the far accesses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub far_fraction: Option<f64>,
}

/// `sm__cycles_elapsed` of an (almost) empty one-wave kernel of this shape.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WindowPoint {
    /// CTAs launched; 0 = one per SM (a full wave, the original configs).
    #[serde(default)]
    pub ctas: u32,
    pub threads: u32,
    pub smem_bytes: u32,
    pub cluster: u32,
    /// FA4-style prologue: TMEM alloc/dealloc, mbarrier init, CTA barrier.
    pub tmem_prologue: bool,
    pub cycles: f64,
}

/// TMA (bulk copy) engine of one SM.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct TmaSpec {
    /// Engine throughput, bytes per SM cycle.
    pub bytes_per_cycle: f64,
    /// Fixed issue latency on top of the L2/HBM access, cycles.
    pub issue_latency: u32,
}

impl Default for TmaSpec {
    fn default() -> Self {
        Self {
            bytes_per_cycle: 256.0, // est
            issue_latency: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NvlinkSpec {
    pub generation: u32,
    pub ports: u32,
    /// Per-port, per-direction bandwidth in GB/s.
    pub gbps_per_port_per_dir: f64,
    /// GPU-side PHY + controller latency, ns (one side).
    pub port_latency_ns: f64,
}

/// Execution pipe widths, in lanes per SM sub-partition (SMSP). A warp
/// instruction occupies a pipe for `32 / lanes` cycles.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PipeSpec {
    pub fp32_lanes: u32,
    pub int_lanes: u32,
    pub fp64_lanes: u32,
    pub sfu_lanes: u32,
    pub fp16_lanes: u32,
    /// LSU / MIO dispatch: warp memory instructions accepted per cycle per SM.
    pub lsu_per_sm_per_cycle: u32,
    /// L1/shared bandwidth, bytes per cycle per SM.
    pub l1_bytes_per_cycle: u32,
}

/// Dense tensor-core throughput per SM per cycle, keyed by datatype
/// (`fp16`, `bf16`, `tf32`, `fp8`, `fp6`, `fp4`, `int8`, `fp64`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorSpec {
    pub flops_per_cycle_per_sm: BTreeMap<String, f64>,
    /// Latency of a synchronous warp-level MMA (HMMA m16n8k16), cycles.
    pub hmma_latency: u32,
    /// Cycles each HMMA also holds the SMSP's FP32 (FMA) pipe. On sm_80
    /// this is ~4, so FFMA/FMUL cannot overlap HMMA while MUFU and
    /// LDSM can (structural hazard behind softmax bubbles). 0 = none.
    #[serde(default)]
    pub hmma_fma_pipe_cycles: u32,
    /// Fixed issue/setup overhead for an async (wgmma/tcgen05) MMA, cycles.
    pub async_mma_overhead: u32,
    /// Tensor memory per SM (Blackwell TMEM), bytes. 0 if absent.
    pub tmem_bytes: u64,
    /// tcgen05 pipeline latency: cycles from an MMA leaving the engine to
    /// its result being visible to `tcgen05.commit` (measured round trip
    /// minus the MMA's throughput time).
    #[serde(default)]
    pub tcgen05_latency: u32,
    /// `tcgen05.alloc` (SASS `UTCATOMSWS.FIND_AND_SET`): cycles until the
    /// granted TMEM column is back in the warp's register. Fitted to the
    /// ncu kernel-window points with and without FA4's alloc prologue
    /// (calibration/src/bench_window.cu, replayed from recorded paths).
    /// 0 = an ordinary misc op.
    #[serde(default)]
    pub tmem_alloc_latency: u32,
    /// The same for `cta_group::2` (`UTCATOMSWS.2CTA.FIND_AND_SET`), which
    /// allocates in both CTAs of the pair. 0 = `tmem_alloc_latency`.
    #[serde(default)]
    pub tmem_alloc_2cta_latency: u32,
    /// Synchronous warp MMA (`mma.sync` / HMMA) rate per SM per cycle, when it
    /// differs from the async tensor-core rate above (Hopper and Blackwell keep
    /// the legacy path at roughly Ampere speed: B300 measures 2037 FLOP/clk/SM
    /// for BF16 HMMA against 8192 for tcgen05).
    #[serde(default)]
    pub sync_flops_per_cycle_per_sm: BTreeMap<String, f64>,
}

impl TensorSpec {
    /// FLOP per SM-cycle of a synchronous warp MMA.
    pub fn sync_flops_per_cycle(&self, dtype: &str) -> f64 {
        self.sync_flops_per_cycle_per_sm
            .get(dtype)
            .or_else(|| self.sync_flops_per_cycle_per_sm.get("bf16"))
            .copied()
            .unwrap_or_else(|| self.flops_per_cycle(dtype))
    }
}

impl TensorSpec {
    /// Rate for one MMA instruction of K-extent `k`. Blackwell Ultra's
    /// K=96 FP4 UMMA (`fp4_k96`) runs faster than the K=64 form.
    pub fn mma_rate(&self, dtype: &str, k: u64) -> f64 {
        if dtype == "fp4" && k > 0 && k.is_multiple_of(96) {
            if let Some(r) = self.flops_per_cycle_per_sm.get("fp4_k96") {
                return *r;
            }
        }
        self.flops_per_cycle(dtype)
    }

    pub fn flops_per_cycle(&self, dtype: &str) -> f64 {
        self.flops_per_cycle_per_sm
            .get(dtype)
            .or_else(|| self.flops_per_cycle_per_sm.get("bf16"))
            .copied()
            .unwrap_or(1024.0)
    }
}

/// Fixed (dependent-issue) and nominal latencies, in SM cycles unless noted.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LatencyTable {
    pub int_alu: u32,
    pub imad: u32,
    pub fp32: u32,
    pub fp16: u32,
    pub fp64: u32,
    pub sfu: u32,
    pub conv: u32,
    pub shfl: u32,
    pub s2r: u32,
    pub ldc: u32,
    pub lds: u32,
    pub ldsm: u32,
    pub sts: u32,
    pub branch: u32,
    pub bar: u32,
    /// Generic fallback for unclassified opcodes.
    pub misc: u32,
    /// Extra cycles after the last operand is read for stores/atomics.
    pub store_read: u32,
    /// Remote (distributed shared memory) access within a cluster, e.g. an
    /// mbarrier arrive on the peer CTA of a 2-CTA pair.
    #[serde(default = "default_dsmem")]
    pub dsmem: u32,
}

fn default_dsmem() -> u32 {
    190
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuSpec {
    pub name: String,
    pub arch: Arch,
    pub sm_count: u32,
    pub sm_clock_mhz: f64,
    pub sub_partitions: u32,
    pub max_warps_per_sm: u32,
    pub max_ctas_per_sm: u32,
    pub regs_per_sm: u32,
    pub smem_per_sm_bytes: u32,
    pub l1: CacheSpec,
    pub l2: CacheSpec,
    /// Aggregate L2 bandwidth in bytes per SM-clock cycle (whole GPU).
    pub l2_bytes_per_cycle: f64,
    pub hbm: HbmSpec,
    /// Number of compute dies (B200 = 2, joined by NV-HBI). SMs and memory
    /// channels are split evenly; a far-die L2/HBM access pays `d2d`.
    #[serde(default = "one")]
    pub dies: u32,
    #[serde(default)]
    pub d2d: Option<DieLink>,
    pub nvlink: NvlinkSpec,
    pub pipes: PipeSpec,
    pub tensor: TensorSpec,
    pub latency: LatencyTable,
    /// Host-side kernel launch overhead (eager launch), ns.
    pub launch_overhead_ns: f64,
    /// Per-CTA dispatch latency from the work distributor, cycles.
    pub cta_launch_cycles: u32,
    /// Measured fixed cost of a kernel inside the SM-cycle window, by launch
    /// shape (calibration `bench_window` + ncu). Empty = not modelled.
    #[serde(default)]
    pub kernel_window: Vec<WindowPoint>,
    /// Free-form provenance notes.
    #[serde(default)]
    pub notes: Vec<String>,
    /// Measured per-opcode timing (from `calibration.json`), keyed by the
    /// full SASS opcode (`MUFU.EX2`) or its base (`FFMA`). Overrides the
    /// class-level latency table for fixed-latency ALU/FMA/XU/shuffle ops.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub opcode_timing: BTreeMap<String, OpTiming>,
    #[serde(default)]
    pub tma: TmaSpec,
    /// Board power model for power-capped clocks (DVFS). `None` = the SM
    /// clock is always `sm_clock_mhz`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub power: Option<PowerSpec>,
    /// Issue-stage details measured by `calibration/dissect` (register
    /// banks, operand reuse, multi-cycle dispatch, yield, tcgen05 queue).
    /// The default turns all of it off.
    #[serde(default)]
    pub issue: IssueSpec,
}

/// How one SM sub-partition issues instructions, beyond pipe rates and
/// latencies (docs/BLACKWELL.md, "Issue"). All fields default to "not
/// modelled", so presets without a calibration behave as before.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct IssueSpec {
    /// Register-file banks per sub-partition; a register's bank is its
    /// index mod `rf_banks`. Each bank delivers one 32-bit-per-lane operand
    /// per cycle, so an instruction whose register sources share a bank
    /// holds the operand-read stage for more than one cycle. 0 = off.
    pub rf_banks: u32,
    /// Honour the operand reuse flags: a source read in the same slot by the
    /// warp's previous instruction, with that slot's reuse flag set, comes
    /// from the reuse cache instead of the register file.
    pub reuse_cache: bool,
    /// Reuse-cache entries outlive the next instruction: an entry stays
    /// until its slot reads a different register (rather than only serving
    /// the immediately following instruction, the Volta rule).
    pub reuse_sticky: bool,
    /// Cycles an instruction keeps its warp in dispatch before the warp's
    /// stall count starts (warp-instruction operand collection: HMMA reads
    /// 10 registers, DFMA runs at 1/64 rate, ...), keyed by full opcode or
    /// base opcode. Missing = 0.
    pub dispatch_cycles: BTreeMap<String, f64>,
    /// Minimum cycles between two issues of the same warp (0 = 1).
    pub warp_min_interval: u32,
    /// A yielding instruction (yield bit clear) waits at least this many
    /// cycles before its warp issues again (0 = no extra rule).
    pub yield_min_stall: u32,
    /// tcgen05 MMAs an SM accepts before the issuing thread blocks (0 = use
    /// the cycle-based backlog limit).
    pub tcgen05_queue_depth: u32,
    /// Tensor-memory load/store timing (`LDTM`/`STTM`, i.e. tcgen05.ld/st).
    pub tmem: TmemSpec,
}

/// `tcgen05.ld/st` timing (calibration/src/bench_tc05x.cu). Each warp
/// reaches only its lane quadrant of TMEM, which is the sub-partition's own
/// port: an access of B bytes per warp occupies that port for B / rate cycles
/// and completes `base + B / rate` cycles after issue. The rate depends on
/// the shape: 32x32b and 16x64b move `rate_32x32b` resp. half of it, 16x128b
/// half, 16x256b a quarter. 0 = the old estimate (smem latency + B / 512).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TmemSpec {
    pub ld_base: f64,
    pub st_base: f64,
    /// Bytes per cycle of one sub-partition's TMEM port for 32x32b.
    pub ld_rate_32x32b: f64,
    pub st_rate_32x32b: f64,
}

/// Power-capped clock model fitted to measured steady-state runs.
///
/// Dynamic power is taken as `P_dyn(f) = D * (f / f_max)^3` (frequency
/// times voltage squared, with voltage tracking frequency). Each measured
/// run gives the `D` its kernel needs at `f_max`; a kernel's sustained
/// clock is the largest `f <= f_max` with `idle + P_dyn(f) <= cap`, with `D`
/// interpolated in log(FLOPs) between the runs of its dtype.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PowerSpec {
    pub cap_w: f64,
    pub idle_w: f64,
    pub max_mhz: f64,
    pub runs: Vec<PowerRun>,
    /// SM clock whenever tcgen05 MMAs run, independent of power (B300: the
    /// clock drops from 2032 to 1905 MHz within 10 us of the first UMMA and
    /// stays there, measured by calibration/src/bench_dvfs.cu). 0 = none.
    #[serde(default)]
    pub tcgen05_mhz: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PowerRun {
    pub dtype: String,
    /// FLOPs of the measured kernel (2*m*n*k for a GEMM).
    pub flops: f64,
    pub sm_mhz: f64,
    pub board_w: f64,
}

impl PowerSpec {
    /// Sustained SM clock for a tensor-bound kernel, or `None` when no
    /// measured run covers `dtype`.
    pub fn sustained_mhz(&self, dtype: &str, flops: f64) -> Option<f64> {
        let mut pts: Vec<(f64, f64)> = self
            .runs
            .iter()
            .filter(|r| r.dtype == dtype && r.flops > 0.0)
            .map(|r| {
                let d = (r.board_w - self.idle_w).max(1.0) / (r.sm_mhz / self.max_mhz).powi(3);
                (r.flops.ln(), d)
            })
            .collect();
        if pts.is_empty() || flops <= 0.0 {
            return None;
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let x = flops.ln();
        let d = if x <= pts[0].0 {
            pts[0].1
        } else if x >= pts[pts.len() - 1].0 {
            pts[pts.len() - 1].1
        } else {
            let i = pts.iter().position(|p| p.0 >= x).unwrap();
            let (a, b) = (pts[i - 1], pts[i]);
            a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0)
        };
        let budget = (self.cap_w - self.idle_w).max(1.0);
        Some(self.max_mhz * (budget / d).cbrt().min(1.0))
    }
}

/// Measured timing of one SASS opcode on one SMSP.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct OpTiming {
    /// Dependent-issue latency, cycles.
    pub latency: u32,
    /// Issue interval of back-to-back independent instructions, cycles.
    pub issue_interval: u32,
    /// The measured interval unrounded (0 = use `issue_interval`).
    #[serde(default)]
    pub issue_interval_exact: f64,
}

impl GpuSpec {
    /// The measured kernel-window point nearest to a launch shape: same
    /// cluster size, then the nearest CTA count (log scale), shared-memory
    /// size and thread count. Points with the FA4 TMEM prologue are
    /// diagnostics only: the simulator executes a kernel's own prologue, so
    /// the fixed cost comes from the prologue-free shapes.
    pub fn kernel_window_point(
        &self,
        ctas: u64,
        threads: u32,
        smem_bytes: u32,
        cluster: u32,
    ) -> Option<&WindowPoint> {
        let d = |p: &WindowPoint| {
            let c = if p.ctas == 0 { self.sm_count } else { p.ctas } as f64;
            (
                (p.cluster != cluster.max(1)) as u32,
                ((c.ln() - (ctas.max(1) as f64).ln()).abs() * 4.0) as u32,
                p.smem_bytes.abs_diff(smem_bytes),
                p.threads.abs_diff(threads),
            )
        };
        self.kernel_window
            .iter()
            .filter(|p| !p.tmem_prologue)
            .min_by_key(|p| d(p))
    }

    pub fn warps_per_smsp(&self) -> u32 {
        self.max_warps_per_sm / self.sub_partitions
    }

    /// Peak dense tensor TFLOPS for a dtype (sanity checks / roofline).
    pub fn peak_tensor_tflops(&self, dtype: &str) -> f64 {
        self.tensor.flops_per_cycle(dtype) * self.sm_count as f64 * self.sm_clock_mhz * 1e6 / 1e12
    }
}
