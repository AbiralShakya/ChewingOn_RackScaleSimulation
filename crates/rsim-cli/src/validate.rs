//! `rsim validate`: replay the SASS of measured reference kernels
//! (`ncu_reference.json` from `calibration/`) and compare simulated SM
//! cycles with Nsight Compute's `sm__cycles_elapsed.avg`.
//!
//! The SASS walker has no functional execution, so loop trip counts are an
//! *input*: every backward branch gets the same per-warp trip count, solved
//! so the dynamic warp-instruction count matches `smsp__inst_executed.sum`.
//! Instruction counts are workload facts, not timing, so this does not fit
//! the model to the cycle counts it is validated against. Entries with a
//! `trace_file` (Accel-Sim/NVBit `.traceg`) are replayed exactly instead.
//! Each kernel is simulated at the SM clock its reference run ran at.

use anyhow::{Context, Result};
use rsim_gpu::{Gpu, GpuConfig, KernelDone, KernelLaunch};
use rsim_sass::{BranchProfile, Program};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;

#[derive(Serialize)]
pub struct Row {
    pub label: String,
    pub measured_cycles: f64,
    pub simulated_cycles: f64,
    pub error_pct: f64,
    pub trip_count: u64,
}

fn num(v: &Value, k: &str) -> Option<f64> {
    v.get("ncu")?.get(k)?.as_f64()
}

/// Put the whole dynamic-instruction budget on the loop with the largest
/// body (the measured kernel's main loop); smaller backward branches (spin
/// waits, clock-sync loops) fall through. Trips are solved so that
/// `warps * (static + trips * body) ~= inst_executed`.
fn solve_profile(p: &Program, warps: f64, inst_executed: f64) -> (BranchProfile, u64) {
    let main = p
        .insts
        .iter()
        .filter_map(|i| i.branch_target.filter(|&t| t <= i.pc).map(|t| (i.pc, t)))
        .map(|(b, t)| {
            let body = p.insts.iter().filter(|i| i.pc >= t && i.pc <= b).count();
            (body, b)
        })
        .max();
    let mut prof = BranchProfile::default();
    let per_warp = inst_executed / warps.max(1.0);
    // Warps that run no further than the first guarded EXIT (e.g. bench_window
    // with a null sink) take it: fire it for every CTA.
    if let Some((i, e)) = p
        .insts
        .iter()
        .enumerate()
        .find(|(_, i)| i.class == rsim_sass::OpClass::Exit && i.guard.is_some())
    {
        if per_warp > 0.0 && per_warp <= (i + 1) as f64 + 0.5 {
            prof.exit_prob.insert(e.pc, 1.0);
            return (prof, 0);
        }
    }
    let Some((body, pc)) = main else {
        return (prof, 0);
    };
    let trips = ((per_warp - p.insts.len() as f64) / body as f64)
        .round()
        .max(0.0) as u64;
    prof.taken.insert(pc, trips);
    (prof, trips)
}

fn simulate_done(gpu: &rsim_hw::GpuSpec, launch: KernelLaunch) -> Option<KernelDone> {
    let mut g = Gpu::new(
        0,
        gpu.clone(),
        GpuConfig {
            kernel_window: true,
            ..Default::default()
        },
        false,
    );
    g.submit(Arc::new(launch), 0, 0);
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    out.pop()
}

fn simulate(gpu: &rsim_hw::GpuSpec, launch: KernelLaunch) -> Option<f64> {
    let d = simulate_done(gpu, launch)?;
    // RSIM_VALIDATE_STATS=1: dump each kernel's simulated stats (stall
    // counts, cache hits, class mix) to stderr for digging into a miss.
    if std::env::var_os("RSIM_VALIDATE_STATS").is_some() {
        eprintln!("{}", serde_json::to_string(&d.stats).unwrap_or_default());
    }
    Some((d.end_cycle - d.start_cycle) as f64)
}

/// The SM clock the reference run actually ran at. `ncu --clock-control
/// none` leaves the clock to the driver, which can sit well away from the
/// calibration run's clock (A100 PCIe: 1065 MHz under ncu vs 1395 MHz);
/// cycle counts of memory-bound kernels only compare at the same clock.
/// Short kernels are skipped: their duration is mostly launch overhead.
fn run_clock_mhz(k: &Value) -> Option<f64> {
    let cycles = num(k, "sm__cycles_elapsed.avg")?;
    let ns = num(k, "gpu__time_duration.sum")?;
    (ns > 100e3).then(|| cycles / ns * 1e3)
}

pub fn run(reference: &Path, base: &rsim_hw::GpuSpec, only: &[String]) -> Result<Vec<Row>> {
    use rayon::prelude::*;
    let doc: Value = serde_json::from_str(&super::read(reference)?)?;
    let dir = reference.parent().unwrap_or(Path::new("."));
    let kernels: Vec<&Value> = doc["kernels"]
        .as_array()
        .context("no `kernels` array")?
        .iter()
        .filter(|k| {
            let label = k["label"].as_str().unwrap_or("?");
            only.is_empty() || only.iter().any(|o| o == label)
        })
        .collect();
    // Every reference kernel is an independent single-GPU simulation: run
    // them in parallel and keep the reference file's order.
    let rows = kernels
        .par_iter()
        .map(|k| row(k, dir, base))
        .collect::<Result<Vec<_>>>()?;
    Ok(rows.into_iter().flatten().collect())
}

fn row(k: &Value, dir: &Path, base: &rsim_hw::GpuSpec) -> Result<Option<Row>> {
    let label = k["label"].as_str().unwrap_or("?").to_string();
    let mut gpu = base.clone();
    if let Some(mhz) = run_clock_mhz(k) {
        gpu.sm_clock_mhz = mhz;
    }
    let gpu = &gpu;
    // An NVBit warp-path recording (`.rpath`, research/flash_attention/
    // capture/nvbit/warp_path): exact per-warp paths and barrier operands.
    if let (Some(pf), Some(meas)) = (k["path_file"].as_str(), num(k, "sm__cycles_elapsed.avg")) {
        let p = rsim_sass::PathKernel::from_file(&dir.join(pf).to_string_lossy())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut launch = KernelLaunch::from_path(p);
        if let Some(r) = num(k, "launch__registers_per_thread") {
            launch.regs_per_thread = r as u32;
        }
        if let Some(sim) = simulate(gpu, launch) {
            eprintln!("  {label:<26} ncu {meas:>12.0}  sim {sim:>12.0}  (path)");
            return Ok(Some(Row {
                error_pct: 100.0 * (sim - meas) / meas,
                label: format!("{label} (path)"),
                measured_cycles: meas,
                simulated_cycles: sim,
                trip_count: 0,
            }));
        }
        return Ok(None);
    }
    // An Accel-Sim/NVBit trace (exact control flow and addresses) wins
    // over replaying bare SASS with solved trip counts.
    if let (Some(tf), Some(meas)) = (k["trace_file"].as_str(), num(k, "sm__cycles_elapsed.avg")) {
        let src = super::read(&dir.join(tf))?;
        let t = rsim_sass::accelsim::parse_trace(&src).map_err(anyhow::Error::msg)?;
        let launch = KernelLaunch::from_trace(t);
        if let Some(sim) = simulate(gpu, launch) {
            return Ok(Some(Row {
                error_pct: 100.0 * (sim - meas) / meas,
                label: format!("{label} (trace)"),
                measured_cycles: meas,
                simulated_cycles: sim,
                trip_count: 0,
            }));
        }
        return Ok(None);
    }
    let (Some(sass), Some(mangled)) = (k["sass_file"].as_str(), k["kernel_mangled"].as_str())
    else {
        return Ok(None);
    };
    let (Some(meas), Some(grid), Some(block)) = (
        num(k, "sm__cycles_elapsed.avg"),
        num(k, "launch__grid_size"),
        num(k, "launch__block_size"),
    ) else {
        return Ok(None);
    };
    let src = super::read(&dir.join(sass))?;
    let progs = rsim_sass::parse_listing(&src)?;
    let Some(prog) = progs
        .into_iter()
        .find(|p| p.name == mangled || p.name.contains(mangled))
    else {
        eprintln!("{label}: function {mangled} not in {sass}, skipped");
        return Ok(None);
    };
    let regs = num(k, "launch__registers_per_thread").unwrap_or(32.0) as u32;
    let smem = (num(k, "launch__shared_mem_per_block_dynamic").unwrap_or(0.0)
        + num(k, "launch__shared_mem_per_block_static").unwrap_or(0.0)) as u32;
    let warps = grid * (block / 32.0).ceil();
    let (prof, trips) = solve_profile(
        &prog,
        warps,
        num(k, "smsp__inst_executed.sum").unwrap_or(0.0),
    );
    eprintln!(
        "  {label}: {} static insts, main-loop trips/warp {trips}",
        prog.insts.len()
    );
    let build = |prof: BranchProfile| {
        KernelLaunch::from_program(
            &label,
            prog.clone(),
            [grid as u32, 1, 1],
            [block as u32, 1, 1],
            regs,
            smem,
            prof,
        )
    };
    let t0 = std::time::Instant::now();
    let Some(mut d) = simulate_done(gpu, build(prof.clone())) else {
        eprintln!("{label}: did not complete");
        return Ok(None);
    };
    // Streaming kernels: the main loop's trip count is better pinned by
    // the measured DRAM traffic (a workload fact, like the instruction
    // count) than by instructions spread over every warp.
    let mut trips = trips;
    let ncu_bytes = num(k, "dram__bytes_read.sum").unwrap_or(0.0)
        + num(k, "dram__bytes_write.sum").unwrap_or(0.0);
    let sim_bytes = (d.stats.dram_read_bytes + d.stats.dram_write_bytes) as f64;
    if ncu_bytes > 64e6 && sim_bytes > 0.0 && trips > 0 && (ncu_bytes / sim_bytes - 1.0).abs() > 0.1
    {
        let t2 = ((trips as f64) * ncu_bytes / sim_bytes).round().max(1.0) as u64;
        let mut p2 = prof.clone();
        for v in p2.taken.values_mut() {
            *v = t2;
        }
        if let Some(d2) = simulate_done(gpu, build(p2)) {
            eprintln!(
                "  {label}: trips {trips} -> {t2} to match {:.2} GB DRAM traffic",
                ncu_bytes / 1e9
            );
            trips = t2;
            d = d2;
        }
    }
    let sim = (d.end_cycle - d.start_cycle) as f64;
    eprintln!(
        "  {label:<26} ncu {meas:>12.0}  sim {sim:>12.0}  ({:+.1}%, {:.1} s, DRAM ncu {:.3} GB sim {:.3} GB)",
        100.0 * (sim - meas) / meas,
        t0.elapsed().as_secs_f64(),
        ncu_bytes / 1e9,
        (d.stats.dram_read_bytes + d.stats.dram_write_bytes) as f64 / 1e9
    );
    Ok(Some(Row {
        error_pct: 100.0 * (sim - meas) / meas,
        label,
        measured_cycles: meas,
        simulated_cycles: sim,
        trip_count: trips,
    }))
}
