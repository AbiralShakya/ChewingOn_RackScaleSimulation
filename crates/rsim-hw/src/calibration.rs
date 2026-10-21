//! Apply measured microbenchmark results (`rackscale-calibration/v1`, as
//! written by `calibration/` on real hardware) to a [`GpuSpec`].
//!
//! Everything is optional: a field missing from the JSON leaves the preset
//! value alone. Every value taken is recorded in `spec.notes` with a
//! `calibrated` tag so reports show provenance.

use crate::gpu::{DieLink, GpuSpec, OpTiming, PowerRun, PowerSpec};
use serde_json::Value;
use std::path::Path;

pub const SCHEMA: &str = "rackscale-calibration/v1";

#[derive(Debug, thiserror::Error)]
pub enum CalError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported schema `{0}` (expected {SCHEMA})")]
    Schema(String),
}

pub fn apply_file(spec: &mut GpuSpec, path: &Path) -> Result<Vec<String>, CalError> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    apply(spec, &v)
}

fn f(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for p in path {
        cur = cur.get(*p)?;
    }
    cur.as_f64()
}

fn cyc(x: f64) -> u32 {
    x.round().max(1.0) as u32
}

/// Apply a parsed calibration document. Returns the list of fields taken.
pub fn apply(spec: &mut GpuSpec, doc: &Value) -> Result<Vec<String>, CalError> {
    let schema = doc.get("schema").and_then(Value::as_str).unwrap_or("");
    if schema != SCHEMA {
        return Err(CalError::Schema(schema.into()));
    }
    let m = doc.get("model").unwrap_or(&Value::Null);
    let mut took = Vec::new();
    let set = |name: &str, took: &mut Vec<String>| took.push(name.to_string());

    if let Some(n) = m.pointer("/device/name").and_then(Value::as_str) {
        spec.name = n.to_string();
    }
    if let Some(n) = f(m, &["device", "sm_count"]) {
        spec.sm_count = n as u32;
        set("sm_count", &mut took);
    }
    if let Some(mhz) = f(m, &["sm_clock_mhz_measured"]) {
        spec.sm_clock_mhz = mhz;
        set("sm_clock_mhz", &mut took);
    }
    let ghz = spec.sm_clock_mhz / 1000.0;

    // Memory hierarchy.
    let mem = m.get("memory").unwrap_or(&Value::Null);
    if let Some(b) = f(mem, &["l2_bytes"]) {
        spec.l2.size_bytes = b as u64;
        set("l2.size_bytes", &mut took);
    }
    if let Some(b) = f(mem, &["smem_per_sm_bytes"]) {
        spec.smem_per_sm_bytes = b as u32;
        set("smem_per_sm_bytes", &mut took);
    }
    if let Some(l) = f(mem, &["latency_cycles", "l1_hit"]) {
        spec.l1.hit_latency = cyc(l);
        set("l1.hit_latency", &mut took);
    }
    if let Some(l) = f(mem, &["latency_cycles", "l2_hit_repeat"]) {
        spec.l2.hit_latency = cyc(l);
        set("l2.hit_latency", &mut took);
    }
    if let Some(l) = f(mem, &["latency_cycles", "dsmem_remote"]) {
        spec.latency.dsmem = cyc(l);
        set("latency.dsmem", &mut took);
    }
    if let Some(l) = f(mem, &["latency_cycles", "smem"]) {
        spec.latency.lds = cyc(l);
        set("latency.lds", &mut took);
    }
    if let (Some(dram), Some(l2)) = (
        f(mem, &["latency_cycles", "dram"]),
        f(mem, &["latency_cycles", "l2_hit_first_touch"]),
    ) {
        spec.hbm.latency_ns = ((dram - l2) / ghz).max(0.0);
        set("hbm.latency_ns", &mut took);
    }
    if let Some(peak) = f(mem, &["bandwidth", "hbm_peak_GBps_from_attrs"]) {
        spec.hbm.bw_gbps = peak;
        set("hbm.bw_gbps", &mut took);
        if let Some(rd) = f(mem, &["bandwidth", "hbm_read_GBps"]) {
            spec.hbm.efficiency = (rd / peak).clamp(0.3, 1.0);
            set("hbm.efficiency", &mut took);
        }
    }
    if let Some(b) = f(mem, &["bandwidth", "l2_read_bytes_per_clk_per_sm"]) {
        spec.l2_bytes_per_cycle = b * spec.sm_count as f64;
        set("l2_bytes_per_cycle", &mut took);
    }

    // L2 partitions: near/far latency and how often an SM sees far.
    if let (Some(near), Some(far), Some(frac)) = (
        f(mem, &["l2_partitions", "near_mean_cycles"]),
        f(mem, &["l2_partitions", "far_mean_cycles"]),
        f(mem, &["l2_partitions", "far_fraction"]),
    ) {
        spec.l2.hit_latency = cyc(near);
        let d = spec.d2d.get_or_insert(DieLink {
            // Single-die parts: the far partition is on-die; no BW limit.
            gbps_per_dir: 1e9,
            extra_latency: 0,
            far_fraction: None,
        });
        d.extra_latency = cyc(far - near);
        d.far_fraction = Some(frac.clamp(0.0, 1.0));
        set("l2_partitions", &mut took);
    }

    // TMA: fit latency = intercept + bytes / rate on the L2-resident curve.
    if let Some(pts) = m.pointer("/tma/bulk_g2s_latency").and_then(Value::as_array) {
        let xy: Vec<(f64, f64)> = pts
            .iter()
            .filter_map(|p| Some((f(p, &["bytes"])?, f(p, &["l2_cycles"])?)))
            .collect();
        if let Some((slope, icept)) = linear_fit(&xy) {
            if slope > 0.0 {
                spec.tma.bytes_per_cycle = 1.0 / slope;
            }
            spec.tma.issue_latency = (icept - spec.l2.hit_latency as f64).max(0.0).round() as u32;
            set("tma", &mut took);
        }
    }

    // Per-opcode ALU/FMA/XU timing.
    if let Some(ops) = m.get("sass").and_then(Value::as_object) {
        for (op, t) in ops {
            if op.contains('@') {
                continue; // `OP@bench` duplicates
            }
            let (Some(lat), Some(ii)) = (
                f(t, &["latency_cycles"]),
                f(t, &["issue_interval_cycles_per_smsp"]),
            ) else {
                continue;
            };
            spec.opcode_timing.insert(
                op.clone(),
                OpTiming {
                    latency: cyc(lat),
                    issue_interval: cyc(ii),
                },
            );
        }
        set("opcode_timing", &mut took);
    }

    // Tensor cores.
    let tc = m.get("tensor").unwrap_or(&Value::Null);
    if let Some(c) = f(tc, &["mma_sync_fp32_pipe_cycles"]) {
        spec.tensor.hmma_fma_pipe_cycles = c.round().max(0.0) as u32;
        set("tensor.hmma_fma_pipe_cycles", &mut took);
    }
    if let Some(l) = f(tc, &["mma_sync_m16n8k16_bf16_f32", "latency_cycles"]) {
        spec.tensor.hmma_latency = cyc(l);
        set("tensor.hmma_latency", &mut took);
    }
    // Synchronous warp MMA (mma.sync): `tensor.mma_sync_*` (B300 suite) or
    // `warp_mma.*` (Ampere suite).
    let wm = m.get("warp_mma").unwrap_or(&Value::Null);
    for (keys, d) in [
        (["mma_sync_m16n8k16_bf16_f32", "BF16_F32_m16n8k16"], "bf16"),
        (["mma_sync_m16n8k16_f16_f16", "F16_F32_m16n8k16"], "fp16"),
        (["mma_sync_m16n8k8_tf32_f32", "TF32_F32_m16n8k8"], "tf32"),
    ] {
        let fl = f(tc, &[keys[0], "flop_per_clk_per_sm"])
            .or_else(|| f(wm, &[keys[1], "flop_per_clk_per_sm"]));
        if let Some(fl) = fl {
            spec.tensor
                .sync_flops_per_cycle_per_sm
                .insert(d.to_string(), fl);
            set(&format!("tensor.sync_{d}"), &mut took);
        }
    }
    if let Some(l) = f(wm, &["BF16_F32_m16n8k16", "latency_cycles"]) {
        spec.tensor.hmma_latency = cyc(l);
        set("tensor.hmma_latency", &mut took);
    }
    // Without `tensor.mma_sync_fp32_pipe_cycles`, derive the FP32-pipe cycles per HMMA
    // from the Ampere co-issue test: 8 HMMA + 32 FFMA against the FFMA issue interval.
    if spec.tensor.hmma_fma_pipe_cycles == 0 {
        if let (Some(both), Some(ffma_ii)) = (
            f(m, &["hmma_coissue", "ffma", "h8_o32", "cycles"]),
            f(m, &["sass", "FFMA", "issue_interval_cycles_per_smsp"]),
        ) {
            let per = ((both - 32.0 * ffma_ii) / 8.0).max(0.0);
            spec.tensor.hmma_fma_pipe_cycles = per.round() as u32;
            set("tensor.hmma_fma_pipe_cycles", &mut took);
        }
    }
    for (key, dtypes) in [
        ("tcgen05_f16_bf16_m128n256k16", &["bf16", "fp16"][..]),
        ("tcgen05_tf32_m128n256k8", &["tf32"][..]),
        ("tcgen05_f8f6f4_e4m3_m128n256k32", &["fp8", "fp6"][..]),
        ("tcgen05_nvfp4_m128n256k64", &["fp4"][..]),
    ] {
        if let Some(fl) = f(tc, &[key, "flop_per_clk_per_sm_allsm"]) {
            for d in dtypes {
                spec.tensor.flops_per_cycle_per_sm.insert(d.to_string(), fl);
            }
            set(&format!("tensor.{}", dtypes[0]), &mut took);
        }
    }

    // Blackwell Ultra K=96 FP4 UMMA (2-SM, cta_group::2), per-SM rate.
    if let Some(fl) = [
        "tcgen05x2_nvfp4_k96_m256n256k96",
        "tcgen05_nvfp4_k96_m128n256k96",
    ]
    .iter()
    .find_map(|k| f(tc, &[k, "flop_per_clk_per_sm_allsm"]))
    {
        spec.tensor
            .flops_per_cycle_per_sm
            .insert("fp4_k96".into(), fl);
        set("tensor.fp4_k96", &mut took);
    }

    // Power-capped clocks from steady-state GEMM runs.
    if let (Some(cap), Some(idle), Some(runs)) = (
        f(m, &["power", "power_cap_w"]),
        f(m, &["power", "idle_w"]),
        m.pointer("/power/runs").and_then(Value::as_array),
    ) {
        let runs: Vec<PowerRun> = runs
            .iter()
            .filter_map(|r| {
                let kernel = r.get("kernel").and_then(Value::as_str).unwrap_or("");
                let dtype = r
                    .get("dtype")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| {
                        if kernel.to_ascii_lowercase().contains("fp4") {
                            "fp4".into()
                        } else {
                            "bf16".into()
                        }
                    });
                Some(PowerRun {
                    dtype,
                    flops: 2.0 * f(r, &["m"])? * f(r, &["n"])? * f(r, &["k"])?,
                    sm_mhz: f(r, &["sm_mhz_median"])?,
                    board_w: f(r, &["board_w_median"])?,
                })
            })
            .collect();
        if !runs.is_empty() {
            spec.power = Some(PowerSpec {
                cap_w: cap,
                idle_w: idle,
                max_mhz: f(m, &["power", "max_sm_mhz"]).unwrap_or(spec.sm_clock_mhz),
                runs,
            });
            set("power", &mut took);
        }
    }

    // tcgen05 pipeline latency: round trip minus throughput time, median
    // over the single-CTA shapes measured.
    if let Some(t) = tc.as_object() {
        let mut lat: Vec<f64> = t
            .iter()
            .filter(|(k, _)| k.starts_with("tcgen05_"))
            .filter_map(|(_, v)| {
                Some(f(v, &["roundtrip_latency_cycles"])? - f(v, &["cycles_per_mma_1sm"])?)
            })
            .collect();
        if !lat.is_empty() {
            lat.sort_by(|a, b| a.total_cmp(b));
            spec.tensor.tcgen05_latency = cyc(lat[lat.len() / 2]);
            set("tensor.tcgen05_latency", &mut took);
        }
    }

    if let Some(us) = f(m, &["launch", "back_to_back_us_per_kernel"]) {
        spec.launch_overhead_ns = us * 1000.0;
        set("launch_overhead_ns", &mut took);
    }
    if let Some(c) = f(m, &["syncthreads_cycles", "256"]) {
        spec.latency.bar = cyc(c);
        set("latency.bar", &mut took);
    }
    let gen = doc
        .get("generated_utc")
        .and_then(Value::as_str)
        .unwrap_or("?");
    spec.notes
        .push(format!("calibrated ({gen}): {}", took.join(", ")));
    Ok(took)
}

/// Least-squares line through `xy`: (slope, intercept).
fn linear_fit(xy: &[(f64, f64)]) -> Option<(f64, f64)> {
    if xy.len() < 2 {
        return None;
    }
    let n = xy.len() as f64;
    let (sx, sy) = xy.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (mx, my) = (sx / n, sy / n);
    let sxx: f64 = xy.iter().map(|p| (p.0 - mx).powi(2)).sum();
    if sxx == 0.0 {
        return None;
    }
    let sxy: f64 = xy.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let slope = sxy / sxx;
    Some((slope, my - slope * mx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_partial_document() {
        let mut g = crate::presets::b300_uncalibrated();
        let doc = serde_json::json!({
            "schema": SCHEMA,
            "model": {
                "sm_clock_mhz_measured": 2000.0,
                "memory": {"latency_cycles": {"l1_hit": 39.4, "l2_hit_repeat": 298.8}},
                "sass": {"MUFU.EX2": {"latency_cycles": 14.0, "issue_interval_cycles_per_smsp": 4.01},
                         "MUFU.EX2@OpEX2": {"latency_cycles": 99.0, "issue_interval_cycles_per_smsp": 9.0}},
                "tensor": {"tcgen05_nvfp4_m128n256k64": {"flop_per_clk_per_sm_allsm": 32768.0}}
            }
        });
        apply(&mut g, &doc).unwrap();
        assert_eq!(g.sm_clock_mhz, 2000.0);
        assert_eq!(g.l2.hit_latency, 299);
        assert_eq!(
            g.opcode_timing["MUFU.EX2"],
            OpTiming {
                latency: 14,
                issue_interval: 4
            }
        );
        assert_eq!(g.opcode_timing.len(), 1);
        assert_eq!(g.tensor.flops_per_cycle("fp4"), 32768.0);
    }

    #[test]
    fn partitions_tma_and_tcgen05() {
        let mut g = crate::presets::b300_uncalibrated();
        let doc = serde_json::json!({
            "schema": SCHEMA,
            "model": {
                "memory": {"l2_partitions": {"near_mean_cycles": 320.0, "far_mean_cycles": 706.0, "far_fraction": 0.25}},
                "tma": {"bulk_g2s_latency": [{"bytes": 0, "l2_cycles": 330}, {"bytes": 1000, "l2_cycles": 340}]},
                "tensor": {"tcgen05_a": {"roundtrip_latency_cycles": 428.0, "cycles_per_mma_1sm": 128.0},
                           "tcgen05_b": {"roundtrip_latency_cycles": 271.0, "cycles_per_mma_1sm": 48.0},
                           "tcgen05_c": {"roundtrip_latency_cycles": 426.0, "cycles_per_mma_1sm": 64.0}}
            }
        });
        apply(&mut g, &doc).unwrap();
        let d = g.d2d.unwrap();
        assert_eq!(
            (g.l2.hit_latency, d.extra_latency, d.far_fraction),
            (320, 386, Some(0.25))
        );
        assert!((g.tma.bytes_per_cycle - 100.0).abs() < 1e-6);
        assert_eq!(g.tma.issue_latency, 10);
        assert_eq!(g.tensor.tcgen05_latency, 300);
    }

    #[test]
    fn power_cap_reproduces_measured_clocks() {
        let g = crate::presets::b300();
        let p = g
            .power
            .as_ref()
            .expect("B300 calibration has a power model");
        for r in &p.runs {
            let f = p.sustained_mhz(&r.dtype, r.flops).unwrap();
            // Runs drawing less than the cap keep f_max; capped runs land
            // near their measured clock.
            assert!(f <= p.max_mhz + 1e-9);
            assert!(
                (f - r.sm_mhz).abs() / r.sm_mhz < 0.05,
                "{} vs {}",
                f,
                r.sm_mhz
            );
        }
        assert!(g.tensor.mma_rate("fp4", 96) > g.tensor.mma_rate("fp4", 64));
    }

    #[test]
    fn rejects_unknown_schema() {
        let mut g = crate::presets::b300_uncalibrated();
        assert!(apply(&mut g, &serde_json::json!({"schema": "v0"})).is_err());
    }
}
