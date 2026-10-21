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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(kernel, m)?)?;
    m.add_function(wrap_pyfunction!(viz, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
