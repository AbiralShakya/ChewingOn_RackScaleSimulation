//! `rsviz/1` bundles: everything the visualiser (`viz/viewer.html`) draws,
//! from the rack's links down to per-PC stall samples.
//!
//! A bundle holds one or more *runs*. A run is either a rack run (many
//! ranks, a topology, op spans per stream) or a single kernel; both use the
//! same schema, so the viewer has one code path. Bulk arrays are flat
//! (`[a, b, c, a, b, c, ...]`) to keep files small; strings go through a
//! per-run string table. See `docs/VIZ.md` for the field reference.

use crate::sim::Outcome;
use rsim_gpu::stats::ALL_STALLS;
use rsim_gpu::{KernelDetail, KernelDone};
use rsim_hw::{GpuSpec, NodeKind, System};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

pub const FORMAT: &str = "rsviz/1";

/// The viewer page; bundles are embedded at [`DATA_MARKER`].
pub const VIEWER: &str = include_str!("../../../viz/viewer.html");
const DATA_MARKER: &str = "/*RSVIZ_DATA*/";

fn stall_names() -> Vec<String> {
    ALL_STALLS
        .iter()
        .map(|s| {
            serde_json::to_value(s)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect()
}

#[derive(Default)]
struct Strings {
    list: Vec<String>,
    idx: HashMap<String, usize>,
}

impl Strings {
    fn id(&mut self, s: &str) -> usize {
        if let Some(&i) = self.idx.get(s) {
            return i;
        }
        self.list.push(s.to_string());
        self.idx.insert(s.to_string(), self.list.len() - 1);
        self.list.len() - 1
    }
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn gpu_json(g: &GpuSpec) -> Value {
    json!({
        "name": g.name,
        "arch": format!("{:?}", g.arch),
        "sm_count": g.sm_count,
        "smsp": g.sub_partitions,
        "max_warps_per_sm": g.max_warps_per_sm,
        "clock_mhz": g.sm_clock_mhz,
        "bf16_tflops": g.peak_tensor_tflops("bf16"),
        "hbm_gbps": g.hbm.bw_gbps,
        "nvlink_gbps": g.nvlink.gbps_per_port_per_dir * g.nvlink.ports as f64,
    })
}

/// Distinct GPU specs and the index of each rank's spec.
fn gpu_table(gpus: &[GpuSpec]) -> (Vec<Value>, Vec<usize>) {
    let mut table: Vec<Value> = Vec::new();
    let mut of_rank = Vec::new();
    for g in gpus {
        let j = gpu_json(g);
        let i = table.iter().position(|t| *t == j).unwrap_or_else(|| {
            table.push(j);
            table.len() - 1
        });
        of_rank.push(i);
    }
    (table, of_rank)
}

/// Builds the kernel instance table and the deduplicated detail table.
struct KernelTable {
    kernels: Vec<Value>,
    details: Vec<Value>,
    by_ptr: HashMap<usize, usize>,
}

impl KernelTable {
    fn new() -> Self {
        Self {
            kernels: Vec::new(),
            details: Vec::new(),
            by_ptr: HashMap::new(),
        }
    }

    fn detail(&mut self, d: &Arc<KernelDetail>, k: &KernelDone, st: &mut Strings) -> usize {
        let key = Arc::as_ptr(d) as usize;
        if let Some(&i) = self.by_ptr.get(&key) {
            return i;
        }
        let s = &k.stats;
        let pc: Vec<Value> =
            s.pc.iter()
                .map(|(pc, p)| {
                    json!([
                        pc,
                        st.id(&p.text),
                        st.id(&p.class),
                        p.issued,
                        p.stalls.0.to_vec()
                    ])
                })
                .collect();
        let mut v = json!({
            "ctas": d.ctas,
            "roles": d.roles,
            "warps": d.warps,
            "pc": pc,
            "warp": s.warp.0.to_vec(),
            "warp_by_index": s.warp_by_index.iter().map(|c| c.0.to_vec()).collect::<Vec<_>>(),
            "issued_by_index": s.issued_by_index,
            "mbar_wait": s.mbar_wait,
            "class_mix": s.class_mix,
            "l1_hits": s.l1_hits, "l1_misses": s.l1_misses,
            "l2_hits": s.l2_hits, "l2_misses": s.l2_misses,
            "smem_wavefronts": s.smem_wavefronts,
        });
        if !d.probes.is_empty() {
            v["probes"] = json!(d.probes);
        }
        self.details.push(v);
        self.by_ptr.insert(key, self.details.len() - 1);
        self.details.len() - 1
    }

    fn push(&mut self, rank: usize, spec: &GpuSpec, k: &KernelDone, st: &mut Strings) -> usize {
        let s = &k.stats;
        let cycles = k.end_cycle.saturating_sub(k.start_cycle).max(1);
        let slots = cycles as f64 * (spec.sm_count * spec.sub_partitions) as f64;
        let detail = k
            .detail
            .as_ref()
            .map(|d| self.detail(d, k, st) as i64)
            .unwrap_or(-1);
        let v = json!({
            "name": st.id(&s.name),
            "rank": rank,
            "start": k.start_cycle,
            "cycles": cycles,
            "ctas": s.ctas,
            "warps": s.warps,
            "ctas_per_sm": s.ctas_per_sm,
            "insts": s.insts_issued,
            "tensor_util": r3(s.tensor_busy_cycles as f64 / slots),
            "issue_util": r3(s.sched.get(rsim_gpu::Stall::Selected) as f64 / s.sched.total().max(1) as f64),
            "tflops": r3(s.tensor_flops / (cycles as f64 / (spec.sm_clock_mhz * 1e6)) / 1e12),
            "dram_bytes": s.dram_read_bytes + s.dram_write_bytes,
            "sched": s.sched.0.to_vec(),
            "d": detail,
        });
        self.kernels.push(v);
        self.kernels.len() - 1
    }
}

fn kind_code(kind: &str) -> u8 {
    match kind {
        "kernel" => 0,
        "collective" => 1,
        "fixed" => 2,
        _ => 3,
    }
}

/// Mean of `bins` over channels, quantised to 1/1000.
fn mean_bins(chs: &[&Vec<f32>]) -> Vec<f64> {
    let n = chs.iter().map(|b| b.len()).max().unwrap_or(0);
    let mut out = vec![0.0; n];
    for b in chs {
        for (i, x) in b.iter().enumerate() {
            out[i] += *x as f64;
        }
    }
    let k = chs.len().max(1) as f64;
    out.iter().map(|x| r3(x / k)).collect()
}

/// A rack run from a finished simulation.
pub fn rack_run(name: &str, sys: &System, out: &Outcome, link_bin_ns: f64) -> Value {
    let mut st = Strings::default();
    let (gpus, rank_gpu) = gpu_table(&sys.gpus);
    let nodes: Vec<Value> = sys
        .topology
        .nodes
        .iter()
        .map(|n| match n {
            NodeKind::Gpu { rank } => json!({"t": "gpu", "rank": rank}),
            NodeKind::Switch { spec, label } => {
                json!({"t": "switch", "label": label, "name": spec.name, "ports": spec.ports})
            }
        })
        .collect();
    let links: Vec<Value> = sys
        .topology
        .links
        .iter()
        .map(|l| json!([l.a, l.b, l.count, l.gbps_per_dir, l.latency_ns]))
        .collect();
    let mut kt = KernelTable::new();
    let mut ranks = Vec::new();
    for r in &out.ranks {
        let spec = &sys.gpus[r.rank];
        let kidx: Vec<usize> = r
            .kernels
            .iter()
            .map(|k| kt.push(r.rank, spec, k, &mut st))
            .collect();
        let mut spans = Vec::with_capacity(r.spans.len() * 6);
        for s in &r.spans {
            spans.push(json!(s.stream));
            spans.push(json!(kind_code(s.kind)));
            spans.push(json!(st.id(&s.label)));
            spans.push(json!(r3(s.start as f64 / 1000.0)));
            spans.push(json!(r3(s.end as f64 / 1000.0)));
            spans.push(json!(s
                .kernel
                .and_then(|i| kidx.get(i).copied())
                .map(|i| i as i64)
                .unwrap_or(-1)));
        }
        let finish = r.spans.iter().map(|s| s.end).max().unwrap_or(0);
        ranks.push(json!({
            "rank": r.rank,
            "finish_us": r3(finish as f64 / 1e6),
            "busy_us": r3(r.busy_us),
            "spans": spans,
        }));
    }
    // Links: total use per direction, plus mean busy-over-time per node.
    let mk = out.report.makespan_us.max(1e-9) * 1e6;
    let link_use: Vec<Value> = out
        .links
        .iter()
        .map(|l| json!([l.from, l.to, l.bytes, r3(l.busy_ps / mk)]))
        .collect();
    let mut node_bins = Vec::new();
    if link_bin_ns > 0.0 {
        for n in 0..sys.topology.nodes.len() {
            let chs: Vec<&Vec<f32>> = out
                .links
                .iter()
                .filter(|l| l.from == n)
                .map(|l| &l.bins)
                .collect();
            node_bins.push(mean_bins(&chs));
        }
    }
    let rep = &out.report;
    json!({
        "kind": "rack",
        "name": name,
        "system": {"name": sys.name, "gpus": gpus, "rank_gpu": rank_gpu, "nodes": nodes, "links": links},
        "workload": rep.workload,
        "makespan_us": rep.makespan_us,
        "mfu": rep.mfu,
        "gpu_idle": rep.bubbles.gpu_idle_fraction,
        "tensor_idle": rep.bubbles.tensor_idle_fraction,
        "sched": rep.bubbles.sched,
        "collectives": rep.collectives,
        "stalls": stall_names(),
        "ranks": ranks,
        "kernels": kt.kernels,
        "details": kt.details,
        "link_use": link_use,
        "link_bin_ns": link_bin_ns,
        "node_bins": node_bins,
        "strings": st.list,
    })
}

/// A single-kernel run (`rsim viz path|kernel|sass`).
pub fn kernel_run(name: &str, spec: &GpuSpec, k: &KernelDone, note: &str) -> Value {
    let mut st = Strings::default();
    let mut kt = KernelTable::new();
    let ki = kt.push(0, spec, k, &mut st);
    let ghz = spec.sm_clock_mhz / 1000.0;
    let us = |c: u64| r3(c as f64 / ghz / 1000.0);
    let label = st.id(&k.stats.name);
    let mut sched = serde_json::Map::new();
    let tot = k.stats.sched.total().max(1) as f64;
    for (n, v) in stall_names().iter().zip(k.stats.sched.0.iter()) {
        if *v > 0 {
            sched.insert(n.clone(), json!(*v as f64 / tot));
        }
    }
    json!({
        "kind": "kernel",
        "name": name,
        "note": note,
        "system": {"name": spec.name, "gpus": [gpu_json(spec)], "rank_gpu": [0],
                   "nodes": [{"t": "gpu", "rank": 0}], "links": []},
        "workload": k.stats.name,
        "makespan_us": us(k.end_cycle),
        "sched": sched,
        "stalls": stall_names(),
        "ranks": [{"rank": 0, "finish_us": us(k.end_cycle), "busy_us": us(k.end_cycle - k.start_cycle),
                   "spans": [0, 0, label, us(k.start_cycle) * 1000.0, us(k.end_cycle) * 1000.0, ki]}],
        "kernels": kt.kernels,
        "details": kt.details,
        "link_use": [],
        "strings": st.list,
    })
}

/// Wrap runs into a bundle.
pub fn bundle(title: &str, runs: Vec<Value>) -> Value {
    json!({
        "format": FORMAT,
        "title": title,
        "generator": format!("rsim {}", env!("CARGO_PKG_VERSION")),
        "runs": runs,
    })
}

/// Runs of an existing bundle (to merge several into one page).
pub fn runs_of(bundle: &Value) -> Result<Vec<Value>, String> {
    if bundle.get("format").and_then(|f| f.as_str()) != Some(FORMAT) {
        return Err(format!("not an {FORMAT} bundle"));
    }
    Ok(bundle["runs"].as_array().cloned().unwrap_or_default())
}

/// The viewer with `bundle` embedded (gzip + base64, inflated by the
/// browser): one self-contained HTML file.
pub fn html(bundle: &Value) -> String {
    use std::io::Write;
    let json = serde_json::to_vec(bundle).unwrap();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    gz.write_all(&json).unwrap();
    let data = format!("gz:{}", base64_encode(&gz.finish().unwrap()));
    VIEWER.replacen(DATA_MARKER, &data, 1)
}

/// The bundle embedded in a page written by [`html`].
pub fn from_html(page: &str) -> Result<Value, String> {
    use std::io::Read;
    let open = r#"<script type="application/json" id="rsviz-data">"#;
    let a = page.find(open).ok_or("no embedded bundle")? + open.len();
    let b = a + page[a..].find("</script>").ok_or("unterminated bundle")?;
    let body = page[a..b].trim();
    let Some(b64) = body.strip_prefix("gz:") else {
        return serde_json::from_str(body).map_err(|e| e.to_string());
    };
    let mut json = Vec::new();
    flate2::read::GzDecoder::new(&base64_decode(b64)?[..])
        .read_to_end(&mut json)
        .map_err(|e| e.to_string())?;
    serde_json::from_slice(&json).map_err(|e| e.to_string())
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut val = [255u8; 256];
    for (i, &c) in B64.iter().enumerate() {
        val[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0);
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val[c as usize];
        if v == 255 {
            return Err("bad base64".into());
        }
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}
