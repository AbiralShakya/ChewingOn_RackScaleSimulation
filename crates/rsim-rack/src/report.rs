//! Reports: from rack makespan down to per-kernel scheduler bubbles.

use crate::sim::{LinkUse, Lp, Outcome, RankOutcome, Span};
use crate::workload::Workload;
use rsim_core::pdes::PdesStats;
use rsim_core::trace::TraceSink;
use rsim_core::Time;
use rsim_gpu::stats::{StallCounts, ALL_STALLS};
use rsim_gpu::Stall;
use rsim_hw::System;
use rsim_net::engine::CollRecord;
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize)]
pub struct RankReport {
    pub rank: usize,
    pub gpu: String,
    pub finish_us: f64,
    /// Time with at least one kernel resident on the GPU.
    pub gpu_busy_us: f64,
    pub kernel_us: f64,
    pub collective_us: f64,
    pub kernels: usize,
    /// When each of the rank's streams finished. Streams on one GPU can
    /// belong to different jobs (colocated models, a background transfer),
    /// and each job's time is its own streams' finish.
    pub stream_finish_us: Vec<f64>,
    /// GPU board energy over the run and its average power (when the GPU
    /// has a power model; see [`PowerReport`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_j: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_power_w: Option<f64>,
}

/// Board power of the GPUs over time, summed over ranks (chapter 17 of the
/// book). A kernel draws what [`rsim_gpu::kernel_power_w`] gives it; a
/// collective draws idle plus [`COLLECTIVE_DYNAMIC_FRACTION`] of the
/// dynamic range (an estimate: copy engines, NVLink SerDes and HBM traffic);
/// a GPU with nothing resident draws idle. Overlapping work on one GPU adds,
/// up to its cap. GPU boards only: no CPUs, switches, fans or conversion.
#[derive(Clone, Debug, Serialize)]
pub struct PowerReport {
    pub bin_us: f64,
    /// Sum over ranks of the average board power in each bin.
    pub gpus_w: Vec<f64>,
    pub avg_w: f64,
    pub peak_w: f64,
    pub min_w: f64,
    pub energy_j: f64,
    /// Largest change between adjacent bins, W per µs.
    pub max_step_w_per_us: f64,
}

/// Dynamic power of a GPU running only a collective, as a fraction of
/// `cap - idle` (estimate).
pub const COLLECTIVE_DYNAMIC_FRACTION: f64 = 0.15;

/// Per-bin average board power of one rank (W) over `nbins` bins.
fn rank_power(
    spans: &[Span],
    kernel_w: &[f64],
    idle: f64,
    cap: f64,
    bin_ps: f64,
    nbins: usize,
) -> Vec<f64> {
    let mut dynamic = vec![0.0; nbins];
    for s in spans {
        let w = match (s.kind, s.kernel) {
            ("kernel", Some(i)) => kernel_w.get(i).copied().unwrap_or(f64::NAN) - idle,
            ("collective", _) => COLLECTIVE_DYNAMIC_FRACTION * (cap - idle),
            _ => continue,
        };
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let (a, b) = (s.start as f64, s.end as f64);
        let first = (a / bin_ps) as usize;
        let last = ((b / bin_ps).ceil() as usize).min(nbins);
        for (i, d) in dynamic.iter_mut().enumerate().take(last).skip(first) {
            let lo = a.max(i as f64 * bin_ps);
            let hi = b.min((i + 1) as f64 * bin_ps);
            if hi > lo {
                *d += w * (hi - lo) / bin_ps;
            }
        }
    }
    dynamic
        .into_iter()
        .map(|d| idle + d.min(cap - idle))
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub struct KernelAgg {
    pub name: String,
    pub count: u64,
    pub total_us: f64,
    pub mean_us: f64,
    pub tflops: f64,
    pub gbps: f64,
    /// Fraction of SMSP-cycles in which a tensor core was busy.
    pub tensor_util: f64,
    /// Fraction of SMSP-cycles that issued an instruction.
    pub issue_util: f64,
    pub ctas_per_sm: u32,
    pub l2_hit_rate: f64,
    /// Scheduler-cycle breakdown (fractions): the bubble anatomy.
    pub sched: BTreeMap<String, f64>,
    /// Warp-state breakdown (fractions of resident warp-cycles).
    pub warp_states: BTreeMap<String, f64>,
    pub unknown_opcodes: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Bubbles {
    /// Rack-wide SMSP-cycle breakdown over all kernels (fractions).
    pub sched: BTreeMap<String, f64>,
    /// Fraction of (GPU x makespan) with no kernel resident: launch gaps,
    /// waiting on collectives / events, pipeline bubbles, stragglers.
    pub gpu_idle_fraction: f64,
    /// Fraction of SMSP-cycles (over kernel time) with an idle tensor core.
    pub tensor_idle_fraction: f64,
    /// Kernels ranked by total non-issuing scheduler cycles.
    pub top_bubble_kernels: Vec<(String, f64)>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Report {
    pub system: String,
    pub workload: String,
    pub world: usize,
    pub makespan_us: f64,
    pub mfu: Option<f64>,
    pub ranks: Vec<RankReport>,
    pub kernels: Vec<KernelAgg>,
    pub collectives: Vec<CollRecord>,
    pub bubbles: Bubbles,
    /// Device memory of the fullest GPU, when the workload states it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory: Option<MemoryReport>,
    /// GPU board power over time, when the GPUs have a power model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power: Option<PowerReport>,
    /// Things that make the result suspect (e.g. a rank that cannot fit in
    /// its GPU's HBM). The simulation still runs.
    pub warnings: Vec<String>,
    pub sim: SimMeta,
}

/// Device memory needed on one GPU against its HBM capacity.
#[derive(Clone, Debug, Serialize)]
pub struct MemoryReport {
    pub rank: usize,
    pub gpu: String,
    pub total_gb: f64,
    pub capacity_gb: f64,
    pub by_category_gb: BTreeMap<String, f64>,
}

/// The fullest rank's memory, and a warning per GPU type that overflows.
fn memory_check(sys: &System, wl: &Workload) -> (Option<MemoryReport>, Vec<String>) {
    let mut worst: Option<MemoryReport> = None;
    let mut over: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    for rank in 0..wl.world() {
        let mem = wl.memory_of(rank);
        if mem.is_empty() {
            continue;
        }
        let spec = &sys.gpus[rank];
        let total: f64 = mem.values().sum::<f64>() / 1e9;
        let cap = spec.hbm.capacity_gb;
        if total > cap {
            let e = over.entry(spec.name.clone()).or_insert((0, total, cap));
            e.0 += 1;
            e.1 = e.1.max(total);
        }
        if worst
            .as_ref()
            .is_none_or(|w| total / cap > w.total_gb / w.capacity_gb)
        {
            worst = Some(MemoryReport {
                rank,
                gpu: spec.name.clone(),
                total_gb: total,
                capacity_gb: cap,
                by_category_gb: mem.iter().map(|(k, v)| (k.clone(), v / 1e9)).collect(),
            });
        }
    }
    let warnings = over
        .into_iter()
        .map(|(gpu, (n, need, cap))| {
            format!(
                "{n} rank(s) need up to {need:.1} GB but {gpu} has {cap:.0} GB of HBM: \
                 this configuration would not fit (shard further, quantise, or shrink the batch)"
            )
        })
        .collect();
    (worst, warnings)
}

#[derive(Clone, Debug, Serialize)]
pub struct SimMeta {
    pub wallclock_s: f64,
    pub pdes_windows: u64,
    pub pdes_messages: u64,
    pub lookahead_ns: f64,
    pub warp_instructions: u64,
    pub sim_minstr_per_s: f64,
    /// Ranks that stayed in lockstep with an identical lower rank for the
    /// whole run, so were reported from it instead of simulated again, and
    /// ranks that left lockstep part-way and were then simulated themselves.
    pub folded_ranks: u64,
    pub forked_ranks: u64,
}

fn fractions(c: &StallCounts) -> BTreeMap<String, f64> {
    let tot = c.total().max(1) as f64;
    c.to_map()
        .into_iter()
        .map(|(k, v)| (k, v as f64 / tot))
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn build_report(
    sys: &System,
    wl: &Workload,
    lps: Vec<Lp>,
    end: Time,
    st: PdesStats,
    lookahead: Time,
    wall_s: f64,
    trace_on: bool,
) -> Outcome {
    let mut trace = TraceSink::new(trace_on);
    let mut ranks = Vec::new();
    let mut agg: BTreeMap<String, (KernelAgg, StallCounts, StallCounts, f64, f64, u64, u64, u64)> =
        BTreeMap::new();
    let mut total_sched = StallCounts::default();
    let mut collectives = Vec::new();
    let mut makespan: Time = 0;
    let mut idle_sum = 0.0;
    let mut tensor_busy = 0u64;
    let mut tensor_slots = 0u64;
    let mut insts = 0u64;
    let mut peak_flops = 0.0;
    let mut rank_lps = Vec::new();
    let mut links = Vec::new();
    for lp in lps {
        match lp {
            Lp::Rank(r) => rank_lps.push(r),
            Lp::Net(n) => {
                collectives = n.engine.records.clone();
                trace.extend(n.engine.trace.clone());
                links = n
                    .engine
                    .fabric
                    .channels
                    .iter()
                    .map(|c| LinkUse {
                        from: c.from,
                        to: c.to,
                        bytes: c.bytes,
                        busy_ps: c.busy_ps,
                        bins: c
                            .busy_bins
                            .iter()
                            .map(|b| (*b as f64 / c.bin_ps) as f32)
                            .collect(),
                    })
                    .collect();
            }
        }
    }
    let mut outs = Vec::new();
    for r in &rank_lps {
        makespan = makespan.max(r.finish);
    }
    let makespan = makespan.max(1).min(end.max(1));
    let bin_ps = (makespan as f64 / 1000.0).max(1e6);
    let nbins = (makespan as f64 / bin_ps).ceil() as usize;
    let mut gpus_w = vec![0.0; nbins];
    let mut any_power = false;
    for mut r in rank_lps {
        let spec = r.gpu.spec.clone();
        let ghz = spec.sm_clock_mhz / 1000.0;
        let busy_us = r.gpu.busy_cycles as f64 / ghz / 1000.0;
        let kernel_us: f64 = r
            .spans
            .iter()
            .filter(|s| s.kind == "kernel")
            .map(|s| (s.end - s.start) as f64 / 1e6)
            .sum();
        let coll_us: f64 = r
            .spans
            .iter()
            .filter(|s| s.kind == "collective")
            .map(|s| (s.end - s.start) as f64 / 1e6)
            .sum();
        idle_sum += 1.0 - (busy_us * 1e6 / makespan as f64).min(1.0);
        let (energy_j, avg_power_w) = match &spec.power {
            Some(p) => {
                any_power = true;
                let w = rank_power(&r.spans, &r.kernel_w, p.idle_w, p.cap_w, bin_ps, nbins);
                for (g, x) in gpus_w.iter_mut().zip(&w) {
                    *g += x;
                }
                let e: f64 = w.iter().sum::<f64>() * bin_ps * 1e-12;
                (Some(e), Some(e / (nbins as f64 * bin_ps * 1e-12)))
            }
            None => (None, None),
        };
        peak_flops += spec.peak_tensor_tflops("bf16") * 1e12;
        ranks.push(RankReport {
            rank: r.rank,
            gpu: spec.name.clone(),
            finish_us: r.finish as f64 / 1e6,
            gpu_busy_us: busy_us,
            kernel_us,
            collective_us: coll_us,
            kernels: r.kernels.len(),
            stream_finish_us: r.stream_finish().iter().map(|&t| t as f64 / 1e6).collect(),
            energy_j,
            avg_power_w,
        });
        let slots_per_cycle = (spec.sm_count * spec.sub_partitions) as u64;
        for k in &r.kernels {
            let s = &k.stats;
            let cycles = s.cycles().max(1);
            let us = cycles as f64 / ghz / 1000.0;
            let e = agg.entry(s.name.clone()).or_insert_with(|| {
                (
                    KernelAgg {
                        name: s.name.clone(),
                        count: 0,
                        total_us: 0.0,
                        mean_us: 0.0,
                        tflops: 0.0,
                        gbps: 0.0,
                        tensor_util: 0.0,
                        issue_util: 0.0,
                        ctas_per_sm: s.ctas_per_sm,
                        l2_hit_rate: 0.0,
                        sched: BTreeMap::new(),
                        warp_states: BTreeMap::new(),
                        unknown_opcodes: BTreeMap::new(),
                    },
                    StallCounts::default(),
                    StallCounts::default(),
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                )
            });
            e.0.count += 1;
            e.0.total_us += us;
            e.1.merge(&s.sched);
            e.2.merge(&s.warp);
            e.3 += s.tensor_flops;
            e.4 += (s.dram_read_bytes + s.dram_write_bytes) as f64;
            e.5 += s.tensor_busy_cycles;
            e.6 += cycles * slots_per_cycle;
            e.7 += s.l2_hits;
            e.0.l2_hit_rate += s.l2_misses as f64; // temp: misses
            for (op, n) in &s.unknown_opcodes {
                *e.0.unknown_opcodes.entry(op.clone()).or_default() += n;
            }
            total_sched.merge(&s.sched);
            tensor_busy += s.tensor_busy_cycles;
            tensor_slots += cycles * slots_per_cycle;
            insts += s.insts_issued;
        }
        // Trace: kernel / collective spans per stream.
        let pid = r.rank as u64 + 1;
        for sp in &r.spans {
            let Span {
                stream,
                label,
                kind,
                start,
                end,
                ..
            } = sp;
            trace.span(
                label.clone(),
                kind,
                pid,
                *stream as u64 + 1,
                *start,
                *end,
                serde_json::Value::Null,
            );
        }
        trace.extend(r.gpu.trace.clone());
        let keep = r.gpu.cfg.detail.enabled();
        outs.push(RankOutcome {
            rank: r.rank,
            spans: std::mem::take(&mut r.spans),
            kernels: std::mem::take(&mut r.kernels)
                .into_iter()
                .map(|mut k| {
                    if !keep {
                        k.detail = None;
                        k.stats.pc.clear();
                    }
                    k
                })
                .collect(),
            busy_us,
        });
    }
    let mut kernels: Vec<KernelAgg> = agg
        .into_values()
        .map(|(mut a, sched, warp, flops, bytes, tbusy, slots, l2h)| {
            let secs = a.total_us * 1e-6;
            a.mean_us = a.total_us / a.count as f64;
            a.tflops = flops / secs.max(1e-12) / 1e12;
            a.gbps = bytes / secs.max(1e-12) / 1e9;
            a.tensor_util = tbusy as f64 / slots.max(1) as f64;
            a.issue_util = sched.get(Stall::Selected) as f64 / sched.total().max(1) as f64;
            let l2m = a.l2_hit_rate;
            a.l2_hit_rate = l2h as f64 / (l2h as f64 + l2m).max(1.0);
            a.sched = fractions(&sched);
            a.warp_states = fractions(&warp);
            a
        })
        .collect();
    kernels.sort_by(|a, b| b.total_us.partial_cmp(&a.total_us).unwrap());
    let mut top: Vec<(String, f64)> = kernels
        .iter()
        .map(|k| (k.name.clone(), k.total_us * (1.0 - k.issue_util)))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    top.truncate(10);
    let world = ranks.len();
    let makespan_us = makespan as f64 / 1e6;
    let mfu =
        (wl.model_flops > 0.0).then(|| wl.model_flops / (peak_flops * makespan as f64 * 1e-12));
    let _ = ALL_STALLS;
    let (memory, warnings) = memory_check(sys, wl);
    let power = any_power.then(|| {
        let bin_us = bin_ps / 1e6;
        let peak = gpus_w.iter().cloned().fold(0.0, f64::max);
        let min = gpus_w.iter().cloned().fold(f64::INFINITY, f64::min);
        let energy_j = gpus_w.iter().sum::<f64>() * bin_ps * 1e-12;
        let max_step = gpus_w
            .windows(2)
            .map(|w| (w[1] - w[0]).abs() / bin_us)
            .fold(0.0, f64::max);
        PowerReport {
            bin_us,
            avg_w: energy_j / (nbins as f64 * bin_ps * 1e-12),
            peak_w: peak,
            min_w: min,
            energy_j,
            max_step_w_per_us: max_step,
            gpus_w,
        }
    });
    let report = Report {
        system: sys.name.clone(),
        workload: wl.name.clone(),
        world,
        makespan_us,
        mfu,
        ranks,
        kernels,
        collectives,
        bubbles: Bubbles {
            sched: fractions(&total_sched),
            gpu_idle_fraction: idle_sum / world.max(1) as f64,
            tensor_idle_fraction: 1.0 - tensor_busy as f64 / tensor_slots.max(1) as f64,
            top_bubble_kernels: top,
        },
        memory,
        power,
        warnings,
        sim: SimMeta {
            wallclock_s: wall_s,
            pdes_windows: st.windows,
            pdes_messages: st.messages,
            lookahead_ns: lookahead as f64 / 1000.0,
            warp_instructions: insts,
            sim_minstr_per_s: insts as f64 / wall_s.max(1e-9) / 1e6,
            folded_ranks: st.folded,
            forked_ranks: st.forked,
        },
    };
    Outcome {
        report,
        trace,
        ranks: outs,
        links,
    }
}

impl Report {
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        use std::fmt::Write;
        let _ = writeln!(
            s,
            "== {} on {} ({} GPUs) ==",
            self.workload, self.system, self.world
        );
        let _ = writeln!(
            s,
            "makespan: {:.1} us{}",
            self.makespan_us,
            self.mfu
                .map(|m| format!("   MFU: {:.1}%", 100.0 * m))
                .unwrap_or_default()
        );
        let _ = writeln!(
            s,
            "GPU idle (no kernel resident): {:.1}%   tensor-core idle during kernels: {:.1}%",
            100.0 * self.bubbles.gpu_idle_fraction,
            100.0 * self.bubbles.tensor_idle_fraction
        );
        if let Some(m) = &self.memory {
            let parts: Vec<String> = m
                .by_category_gb
                .iter()
                .map(|(k, v)| format!("{k} {v:.1}"))
                .collect();
            let _ = writeln!(
                s,
                "memory (rank {}): {:.1} of {:.0} GB  [{}]",
                m.rank,
                m.total_gb,
                m.capacity_gb,
                parts.join(", ")
            );
        }
        for w in &self.warnings {
            let _ = writeln!(s, "WARNING: {w}");
        }
        let _ = writeln!(s, "\nscheduler-cycle anatomy (all SMSPs, all kernels):");
        let mut v: Vec<_> = self.bubbles.sched.iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (k, f) in v {
            let _ = writeln!(
                s,
                "  {:<20} {:>6.2}%  {}",
                k,
                100.0 * f,
                "#".repeat((f * 50.0).round() as usize)
            );
        }
        let _ = writeln!(s, "\ntop kernels:");
        let _ = writeln!(
            s,
            "  {:<52} {:>5} {:>10} {:>8} {:>8} {:>6} {:>6}",
            "kernel", "n", "total_us", "TFLOPS", "GB/s", "tc%", "issue%"
        );
        for k in self.kernels.iter().take(12) {
            let name: String = k.name.chars().take(52).collect();
            let _ = writeln!(
                s,
                "  {:<52} {:>5} {:>10.1} {:>8.1} {:>8.0} {:>6.1} {:>6.1}",
                name,
                k.count,
                k.total_us,
                k.tflops,
                k.gbps,
                100.0 * k.tensor_util,
                100.0 * k.issue_util
            );
        }
        if !self.collectives.is_empty() {
            let _ = writeln!(s, "\ncollectives (first 8):");
            for c in self.collectives.iter().take(8) {
                let _ = writeln!(
                    s,
                    "  {:<14} {:<6} {:>12} B  ranks={:<3} {:>9.1} us  busbw {:>7.1} GB/s",
                    c.kind,
                    c.algo,
                    c.bytes,
                    c.ranks,
                    (c.done_ps - c.last_post_ps) as f64 / 1e6,
                    c.busbw_gbps
                );
            }
        }
        let _ = writeln!(
            s,
            "\nsim: {:.2} s wall, {} PDES windows (lookahead {:.0} ns), {:.1} M warp-instr/s",
            self.sim.wallclock_s,
            self.sim.pdes_windows,
            self.sim.lookahead_ns,
            self.sim.sim_minstr_per_s
        );
        if self.sim.folded_ranks + self.sim.forked_ranks > 0 {
            let _ = writeln!(
                s,
                "     {} of {} ranks ran in lockstep with an identical rank and were folded onto it ({} forked part-way)",
                self.sim.folded_ranks + self.sim.forked_ranks,
                self.world,
                self.sim.forked_ranks
            );
        }
        s
    }
}
