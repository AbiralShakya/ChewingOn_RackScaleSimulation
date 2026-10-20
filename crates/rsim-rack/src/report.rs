//! Reports: from rack makespan down to per-kernel scheduler bubbles.

use crate::sim::{Lp, Outcome, Span};
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
    pub sim: SimMeta,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimMeta {
    pub wallclock_s: f64,
    pub pdes_windows: u64,
    pub pdes_messages: u64,
    pub lookahead_ns: f64,
    pub warp_instructions: u64,
    pub sim_minstr_per_s: f64,
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
    for lp in lps {
        match lp {
            Lp::Rank(r) => rank_lps.push(r),
            Lp::Net(n) => {
                collectives = n.engine.records.clone();
                trace.extend(n.engine.trace.clone());
            }
        }
    }
    for r in &rank_lps {
        makespan = makespan.max(r.finish);
    }
    let makespan = makespan.max(1).min(end.max(1));
    for r in rank_lps {
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
        peak_flops += spec.peak_tensor_tflops("bf16") * 1e12;
        ranks.push(RankReport {
            rank: r.rank,
            gpu: spec.name.clone(),
            finish_us: r.finish as f64 / 1e6,
            gpu_busy_us: busy_us,
            kernel_us,
            collective_us: coll_us,
            kernels: r.kernels.len(),
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
        sim: SimMeta {
            wallclock_s: wall_s,
            pdes_windows: st.windows,
            pdes_messages: st.messages,
            lookahead_ns: lookahead as f64 / 1000.0,
            warp_instructions: insts,
            sim_minstr_per_s: insts as f64 / wall_s.max(1e-9) / 1e6,
        },
    };
    Outcome { report, trace }
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
        s
    }
}
