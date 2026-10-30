//! Python bindings. Everything crosses the boundary as JSON/YAML text so the
//! Python side stays a thin, dependency-free layer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rsim_rack::RunConfig;
use std::sync::Arc;

/// simulate(system_yaml, workload, config_json="{}", trace_path=None) -> report_json
#[pyfunction]
#[pyo3(signature = (system_yaml, workload, config_json = "{}", trace_path = None))]
fn simulate(
    py: Python<'_>,
    system_yaml: &str,
    workload: &str,
    config_json: &str,
    trace_path: Option<String>,
) -> PyResult<String> {
    let mut cfg: RunConfig = if config_json.trim() == "{}" {
        RunConfig::default()
    } else {
        serde_json::from_str(config_json)
            .map_err(|e| PyValueError::new_err(format!("config: {e}")))?
    };
    cfg.trace = trace_path.is_some();
    let (sys, wl) = (system_yaml.to_string(), workload.to_string());
    let out = py
        .allow_threads(move || rsim_rack::simulate(&sys, &wl, cfg))
        .map_err(PyValueError::new_err)?;
    if let Some(p) = trace_path {
        let f = std::fs::File::create(&p).map_err(|e| PyValueError::new_err(e.to_string()))?;
        out.trace
            .write_json(std::io::BufWriter::new(f))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    serde_json::to_string(&out.report).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// viz(system_yaml, workload, config_json, name, out_path) -> report_json
///
/// Like `simulate`, with visualiser capture; writes a self-contained viewer
/// page (`.html`) or an rsviz/1 bundle (`.json`) to `out_path`.
#[pyfunction]
fn viz(
    py: Python<'_>,
    system_yaml: &str,
    workload: &str,
    config_json: &str,
    name: &str,
    out_path: &str,
) -> PyResult<String> {
    let cfg: RunConfig = serde_json::from_str(config_json)
        .map_err(|e| PyValueError::new_err(format!("config: {e}")))?;
    let bin = cfg.net.link_bin_ns;
    let sys = rsim_hw::SystemSpec::from_yaml(system_yaml)
        .and_then(|s| s.build())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let (sy, wl) = (system_yaml.to_string(), workload.to_string());
    let out = py
        .allow_threads(move || rsim_rack::simulate(&sy, &wl, cfg))
        .map_err(PyValueError::new_err)?;
    let b = rsim_rack::viz::bundle(name, vec![rsim_rack::viz::rack_run(name, &sys, &out, bin)]);
    let text = if out_path.ends_with(".html") {
        rsim_rack::viz::html(&b)
    } else {
        b.to_string()
    };
    std::fs::write(out_path, text).map_err(|e| PyValueError::new_err(e.to_string()))?;
    serde_json::to_string(&out.report).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Simulate a single synthetic kernel: kernel(gpu, kind, dims, dtype="bf16") -> json
#[pyfunction]
#[pyo3(signature = (gpu, kind, dims, dtype = "bf16"))]
fn kernel(py: Python<'_>, gpu: &str, kind: &str, dims: Vec<u64>, dtype: &str) -> PyResult<String> {
    let g = rsim_hw::presets::gpu(gpu)
        .ok_or_else(|| PyValueError::new_err(format!("unknown gpu {gpu}")))?;
    let d = |i: usize| {
        dims.get(i)
            .copied()
            .ok_or_else(|| PyValueError::new_err("not enough dims"))
    };
    let k = match kind {
        "gemm" => rsim_gpu::synth::gemm(&g, d(0)?, d(1)?, d(2)?, dtype),
        "attention" => rsim_gpu::synth::attention(
            &g,
            d(0)?,
            d(1)?,
            d(2)?,
            d(3)?,
            d(4)?,
            d(5)?,
            dims.get(6) == Some(&1),
        ),
        "elementwise" => rsim_gpu::synth::elementwise("elementwise", d(0)?, 2, 2.0, 1),
        "rownorm" => rsim_gpu::synth::row_norm("rmsnorm", d(0)?, d(1)?, 2.0),
        "embedding_bag" => rsim_gpu::synth::embedding_bag(
            "embedding_bag",
            d(0)?,
            d(1)?,
            d(2)?,
            rsim_gpu::synth::dtype(dtype).bytes,
            d(3)?,
            dims.get(4) == Some(&1),
        ),
        "ssm_update" => {
            rsim_gpu::synth_ssm::ssm_state_update(&g, d(0)?, d(1)?, d(2)?, d(3)?, d(4)?, 2.0)
        }
        "conv1d" => {
            rsim_gpu::synth_ssm::conv1d(&g, d(0)?, d(1)?, d(2)?, dims.get(3).copied().unwrap_or(4))
        }
        s if s.starts_with("ssd_") => rsim_gpu::synth_ssm::ssd(
            &g,
            &s[4..],
            rsim_gpu::synth_ssm::SsdShape {
                batch: d(0)?,
                seqlen: d(1)?,
                heads: d(2)?,
                headdim: d(3)?,
                d_state: d(4)?,
                groups: d(5)?,
                chunk: dims.get(6).copied().unwrap_or(256),
            },
            None,
        ),
        _ => return Err(PyValueError::new_err(format!("unknown kind {kind}"))),
    };
    let ghz = g.sm_clock_mhz / 1000.0;
    let (flops, bytes) = (k.flops, k.bytes);
    let done = py.allow_threads(move || {
        let mut gpu = rsim_gpu::Gpu::new(
            0,
            g,
            rsim_gpu::GpuConfig {
                kernel_window: true,
                ..Default::default()
            },
            false,
        );
        gpu.submit(Arc::new(k), 0, 0);
        let mut out = vec![];
        gpu.advance(u64::MAX / 2, &mut out);
        out.pop()
    });
    let d = done.ok_or_else(|| PyValueError::new_err("kernel did not complete"))?;
    let us = (d.end_cycle - d.start_cycle) as f64 / ghz / 1e3;
    let v = serde_json::json!({ "time_us": us, "tflops": flops / us / 1e6, "gbps": bytes / us / 1e3, "stats": d.stats, "sched": d.stats.sched.to_map() });
    Ok(v.to_string())
}

/// resolve_system(system_yaml) -> json
///
/// The system after preset expansion (`spec`, editable and re-submittable as
/// YAML/JSON) plus the resolved GPU spec of every group (`gpus`, overrides and
/// calibration applied) and a topology summary.
#[pyfunction]
fn resolve_system(system_yaml: &str) -> PyResult<String> {
    let err = |e: rsim_hw::HwError| PyValueError::new_err(e.to_string());
    let spec = rsim_hw::SystemSpec::from_yaml(system_yaml).map_err(err)?;
    let groups = spec
        .gpus
        .iter()
        .map(|g| g.resolve())
        .collect::<Result<Vec<_>, _>>()
        .map_err(err)?;
    let sys = spec.build().map_err(err)?;
    let switches = sys
        .topology
        .nodes
        .iter()
        .filter(|n| matches!(n, rsim_hw::NodeKind::Switch { .. }))
        .count();
    let v = serde_json::json!({
        "spec": spec,
        "gpus": groups,
        "num_gpus": sys.num_gpus(),
        "switches": switches,
        "links": sys.topology.links.len(),
        "min_link_latency_ns": sys.min_link_latency_ns(),
    });
    Ok(v.to_string())
}

/// kernel_detail(system_yaml, kernel_json, base_dir=".", rank=0, top_pcs=30) -> json
///
/// Simulate one workload kernel (`{"type": "gemm", ...}`, a `.rpath`, SASS,
/// ...) alone on a cold GPU of rank `rank`, as Nsight Compute times it, with
/// per-PC stall sampling on. Returns the time, the scheduler and warp-state
/// anatomy, the per-warp-role split and the `top_pcs` SASS instructions with
/// the most stalled warp-cycles.
#[pyfunction]
#[pyo3(signature = (system_yaml, kernel_json, base_dir = ".", rank = 0, top_pcs = 30))]
fn kernel_detail(
    py: Python<'_>,
    system_yaml: &str,
    kernel_json: &str,
    base_dir: &str,
    rank: usize,
    top_pcs: usize,
) -> PyResult<String> {
    let sys = rsim_hw::SystemSpec::from_yaml(system_yaml)
        .and_then(|s| s.build())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = sys
        .gpus
        .get(rank)
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("rank {rank} out of range")))?;
    let ks: rsim_rack::KernelSpec = serde_json::from_str(kernel_json)
        .map_err(|e| PyValueError::new_err(format!("kernel: {e}")))?;
    let k = rsim_rack::build_kernel(&g, &ks, std::path::Path::new(base_dir))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let spec = g.clone();
    let launch = Arc::new(k);
    let l2 = launch.clone();
    let cfg = rsim_gpu::GpuConfig {
        kernel_window: true,
        detail: rsim_gpu::DetailConfig {
            pc_stats: true,
            ..Default::default()
        },
        ..Default::default()
    };
    let cfg2 = cfg.clone();
    let done = py.allow_threads(move || {
        let mut gpu = rsim_gpu::Gpu::new(0, g, cfg2, false);
        gpu.submit(l2, 0, 0);
        let mut out = vec![];
        gpu.advance(u64::MAX / 2, &mut out);
        out.pop()
    });
    let d = done.ok_or_else(|| PyValueError::new_err("kernel did not complete"))?;
    let stretch = rsim_gpu::dvfs_stretch(&spec, &cfg, &launch, &d, 0).max(1.0);
    let cycles = (d.end_cycle - d.start_cycle) as f64;
    let us = cycles * stretch / (spec.sm_clock_mhz / 1000.0) / 1e3;
    let slots = cycles * (spec.sm_count * spec.sub_partitions) as f64;
    let st = &d.stats;
    let roles = d
        .detail
        .as_ref()
        .map(|x| x.roles.clone())
        .unwrap_or_default();
    let mut pcs: Vec<serde_json::Value> = st
        .pc
        .iter()
        .map(|(key, p)| {
            let role = (key >> 32) as usize;
            serde_json::json!({
                "pc": key & 0xffff_ffff,
                "role": roles.get(role).cloned().unwrap_or_else(|| role.to_string()),
                "text": p.text,
                "class": p.class,
                "issued": p.issued,
                "stall_cycles": p.stalls.total().saturating_sub(p.stalls.get(rsim_gpu::Stall::Selected)),
                "stalls": p.stalls.to_map(),
            })
        })
        .collect();
    // Per-role totals over every PC: the warp-specialised pipeline's view
    // (which role waits on what).
    let mut by_role: std::collections::BTreeMap<String, (u64, rsim_gpu::StallCounts)> =
        Default::default();
    for (key, p) in &st.pc {
        let role = (key >> 32) as usize;
        let name = roles.get(role).cloned().unwrap_or_else(|| role.to_string());
        let e = by_role.entry(name).or_default();
        e.0 += p.issued;
        e.1.merge(&p.stalls);
    }
    let by_role: serde_json::Map<String, serde_json::Value> = by_role
        .into_iter()
        .map(|(k, (issued, c))| {
            (
                k,
                serde_json::json!({ "issued": issued, "stalls": c.to_map() }),
            )
        })
        .collect();
    pcs.sort_by(|a, b| b["stall_cycles"].as_u64().cmp(&a["stall_cycles"].as_u64()));
    pcs.truncate(top_pcs);
    let by_index: Vec<_> = st.warp_by_index.iter().map(|c| c.to_map()).collect();
    let hits = st.l2_hits as f64;
    let v = serde_json::json!({
        "name": launch.name,
        "gpu": spec.name,
        "time_us": us,
        "cycles": cycles as u64,
        "clock_mhz": spec.sm_clock_mhz,
        "dvfs_stretch": stretch,
        "grid": launch.grid,
        "block": launch.block,
        "regs_per_thread": launch.regs_per_thread,
        "smem_per_cta": launch.smem_per_cta,
        "ctas": st.ctas,
        "ctas_per_sm": st.ctas_per_sm,
        "sm_count": spec.sm_count,
        "flops": launch.flops,
        "bytes": launch.bytes,
        "tflops": launch.flops / us / 1e6,
        "gbps": launch.bytes / us / 1e3,
        "dram_bytes": st.dram_read_bytes + st.dram_write_bytes,
        "tensor_util": st.tensor_busy_cycles as f64 / slots.max(1.0),
        "issue_util": st.insts_issued as f64 / slots.max(1.0),
        "l2_hit_rate": hits / (hits + st.l2_misses as f64).max(1.0),
        "sched": st.sched.to_map(),
        "warp_states": st.warp.to_map(),
        "roles": roles,
        "by_role": by_role,
        "warp_by_index": by_index,
        "issued_by_index": st.issued_by_index,
        "mbar_wait": st.mbar_wait,
        "unknown_opcodes": st.unknown_opcodes,
        "top_pcs": pcs,
    });
    Ok(v.to_string())
}

/// kernel_info(system_yaml, kernel_json, base_dir=".", rank=0) -> json
///
/// Build a workload kernel without simulating it: the name it reports under,
/// launch shape and the FLOPs/bytes it stands for. Cheap; used to map report
/// rows back to workload ops.
#[pyfunction]
#[pyo3(signature = (system_yaml, kernel_json, base_dir = ".", rank = 0))]
fn kernel_info(
    system_yaml: &str,
    kernel_json: &str,
    base_dir: &str,
    rank: usize,
) -> PyResult<String> {
    let sys = rsim_hw::SystemSpec::from_yaml(system_yaml)
        .and_then(|s| s.build())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let g = sys
        .gpus
        .get(rank)
        .ok_or_else(|| PyValueError::new_err(format!("rank {rank} out of range")))?;
    let ks: rsim_rack::KernelSpec = serde_json::from_str(kernel_json)
        .map_err(|e| PyValueError::new_err(format!("kernel: {e}")))?;
    let k = rsim_rack::build_kernel(g, &ks, std::path::Path::new(base_dir))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let v = serde_json::json!({
        "name": k.name,
        "grid": k.grid,
        "block": k.block,
        "regs_per_thread": k.regs_per_thread,
        "smem_per_cta": k.smem_per_cta,
        "cluster": k.cluster,
        "flops": k.flops,
        "bytes": k.bytes,
    });
    Ok(v.to_string())
}

/// Hardware preset names and resolved specs as JSON.
#[pyfunction]
fn presets() -> String {
    let gpus: serde_json::Map<String, serde_json::Value> = rsim_hw::presets::GPU_PRESETS
        .iter()
        .map(|g| {
            (
                g.to_string(),
                serde_json::to_value(rsim_hw::presets::gpu(g).unwrap()).unwrap(),
            )
        })
        .collect();
    serde_json::json!({ "gpus": gpus, "systems": rsim_hw::presets::SYSTEM_PRESETS }).to_string()
}

/// rsim_main(argv, prog="rsim") -> exit code. Runs the `rsim` command line
/// (argv without the program name), so `pip install rackscale` also gives
/// `rsim` and the `rackscale presets|kernel|validate|viz|...` commands.
#[pyfunction]
#[pyo3(signature = (argv, prog = "rsim"))]
fn rsim_main(py: Python<'_>, argv: Vec<String>, prog: &str) -> i32 {
    let args = std::iter::once(prog.to_string()).chain(argv);
    let code = py.allow_threads(move || rsim_cli::run_args(args));
    if code == std::process::ExitCode::SUCCESS {
        0
    } else {
        // ExitCode is opaque; clap's usage errors (2) and runtime errors (1) both map to 1 here.
        1
    }
}

/// Version of the simulator core (the workspace version).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(rsim_main, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(kernel, m)?)?;
    m.add_function(wrap_pyfunction!(viz, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_system, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_detail, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_info, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
