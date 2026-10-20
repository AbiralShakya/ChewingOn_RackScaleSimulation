//! Simulator throughput benchmark: one GEMM on one GPU.
//! `cargo run --release -p rsim-gpu --example bench_gemm -- 4096`
use rsim_gpu::{synth, Gpu, GpuConfig};
use std::sync::Arc;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let arch = std::env::args().nth(2).unwrap_or_else(|| "h100_sxm".into());
    let spec = rsim_hw::presets::gpu(&arch).expect("gpu preset");
    let k = synth::gemm(&spec, n, n, n, "bf16");
    let flops = k.flops;
    let ghz = spec.sm_clock_mhz / 1000.0;
    let mut g = Gpu::new(0, spec.clone(), GpuConfig::default(), false);
    g.submit(Arc::new(k), 0, 0);
    let t0 = std::time::Instant::now();
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    let wall = t0.elapsed().as_secs_f64();
    let d = &out[0];
    let ns = (d.end_cycle - d.start_cycle) as f64 / ghz;
    println!(
        "{arch} gemm {n}^3: {:.1} us simulated, {:.0} TFLOPS ({:.0}% peak) | wall {:.2} s, {:.2} M warp-instr/s, {:.1} M SM-cycles/s",
        ns / 1e3,
        flops / ns / 1e3,
        100.0 * flops / ns / 1e3 / spec.peak_tensor_tflops("bf16"),
        wall,
        d.stats.insts_issued as f64 / wall / 1e6,
        (d.end_cycle - d.start_cycle) as f64 * spec.sm_count as f64 / wall / 1e6
    );
}
