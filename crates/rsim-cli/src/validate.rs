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
    let Some((body, pc)) = main else {
        return (prof, 0);
    };
    let per_warp = inst_executed / warps.max(1.0);
    let trips = ((per_warp - p.insts.len() as f64) / body as f64)
        .round()
        .max(0.0) as u64;
    prof.taken.insert(pc, trips);
    (prof, trips)
}

fn simulate_done(gpu: &rsim_hw::GpuSpec, launch: KernelLaunch) -> Option<KernelDone> {
    let mut g = Gpu::new(0, gpu.clone(), GpuConfig::default(), false);
    g.submit(Arc::new(launch), 0, 0);
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    out.pop()
}

fn simulate(gpu: &rsim_hw::GpuSpec, launch: KernelLaunch) -> Option<f64> {
    simulate_done(gpu, launch).map(|d| (d.end_cycle - d.start_cycle) as f64)
}

pub fn run(reference: &Path, gpu: &rsim_hw::GpuSpec) -> Result<Vec<Row>> {
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let dir = reference.parent().unwrap_or(Path::new("."));
    let mut rows = Vec::new();
    for k in doc["kernels"].as_array().context("no `kernels` array")? {
        let label = k["label"].as_str().unwrap_or("?").to_string();
        // An Accel-Sim/NVBit trace (exact control flow and addresses) wins
        // over replaying bare SASS with solved trip counts.
        if let (Some(tf), Some(meas)) = (k["trace_file"].as_str(), num(k, "sm__cycles_elapsed.avg"))
        {
            let src = std::fs::read_to_string(dir.join(tf))?;
            let t = rsim_sass::accelsim::parse_trace(&src).map_err(anyhow::Error::msg)?;
            let launch = KernelLaunch::from_trace(t);
            if let Some(sim) = simulate(gpu, launch) {
                rows.push(Row {
                    error_pct: 100.0 * (sim - meas) / meas,
                    label: format!("{label} (trace)"),
                    measured_cycles: meas,
                    simulated_cycles: sim,
                    trip_count: 0,
                });
            }
            continue;
        }
        let (Some(sass), Some(mangled)) = (k["sass_file"].as_str(), k["kernel_mangled"].as_str())
        else {
            continue;
        };
        let (Some(meas), Some(grid), Some(block)) = (
            num(k, "sm__cycles_elapsed.avg"),
            num(k, "launch__grid_size"),
            num(k, "launch__block_size"),
        ) else {
            continue;
        };
        let src = std::fs::read_to_string(dir.join(sass))?;
        let progs = rsim_sass::parse_listing(&src)?;
        let Some(prog) = progs
            .into_iter()
            .find(|p| p.name == mangled || p.name.contains(mangled))
        else {
            eprintln!("{label}: function {mangled} not in {sass}, skipped");
            continue;
        };
        let regs = num(k, "launch__registers_per_thread").unwrap_or(32.0) as u32;
        let smem = (num(k, "launch__shared_mem_per_block_dynamic").unwrap_or(0.0)
            + num(k, "launch__shared_mem_per_block_static").unwrap_or(0.0))
            as u32;
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
            continue;
        };
        // Streaming kernels: the main loop's trip count is better pinned by
        // the measured DRAM traffic (a workload fact, like the instruction
        // count) than by instructions spread over every warp.
        let mut trips = trips;
        let ncu_bytes = num(k, "dram__bytes_read.sum").unwrap_or(0.0)
            + num(k, "dram__bytes_write.sum").unwrap_or(0.0);
        let sim_bytes = (d.stats.dram_read_bytes + d.stats.dram_write_bytes) as f64;
        if ncu_bytes > 64e6
            && sim_bytes > 0.0
            && trips > 0
            && (ncu_bytes / sim_bytes - 1.0).abs() > 0.1
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
            "  {label:<26} ncu {meas:>12.0}  sim {sim:>12.0}  ({:+.1}%, {:.1} s)",
            100.0 * (sim - meas) / meas,
            t0.elapsed().as_secs_f64()
        );
        rows.push(Row {
            error_pct: 100.0 * (sim - meas) / meas,
            label,
            measured_cycles: meas,
            simulated_cycles: sim,
            trip_count: trips,
        });
    }
    Ok(rows)
}
