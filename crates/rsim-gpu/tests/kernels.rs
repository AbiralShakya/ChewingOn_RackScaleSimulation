//! End-to-end single-GPU checks against first-principles expectations.

use rsim_gpu::{synth, Gpu, GpuConfig, KernelDone, KernelLaunch};
use rsim_hw::presets;
use std::sync::Arc;

fn run(spec: rsim_hw::GpuSpec, k: KernelLaunch) -> (KernelDone, f64) {
    let ghz = spec.sm_clock_mhz / 1000.0;
    let mut g = Gpu::new(0, spec, GpuConfig::default(), false);
    g.submit(Arc::new(k), 0, 7);
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    assert_eq!(out.len(), 1, "kernel did not complete");
    let d = out.pop().unwrap();
    let ns = (d.end_cycle - d.start_cycle) as f64 / ghz;
    (d, ns)
}

#[test]
fn elementwise_is_hbm_bound() {
    let spec = presets::h100_sxm();
    let n = 64u64 << 20; // 64 Mi bf16 elements, 2 inputs + 1 output = 384 MiB
    let k = synth::elementwise("add", n, 2, 2.0, 1);
    let bytes = k.bytes;
    let (d, ns) = run(spec.clone(), k);
    let gbps = bytes / ns;
    let peak = spec.hbm.bw_gbps;
    eprintln!("elementwise: {:.0} GB/s ({:.0}% of peak), {:.1} us dram_rd={} l2h={} l2m={} l1h={} insts={} stalls {:?}", gbps, 100.0 * gbps / peak, ns / 1e3, d.stats.dram_read_bytes, d.stats.l2_hits, d.stats.l2_misses, d.stats.l1_hits, d.stats.insts_issued, d.stats.sched.to_map());
    assert!(gbps > 0.6 * peak && gbps < 1.0 * peak, "got {gbps} GB/s");
    assert!(d.stats.sched.total() > 0);
}

#[test]
fn hopper_gemm_is_tensor_bound() {
    let spec = presets::h100_sxm();
    let k = synth::gemm(&spec, 4096, 4096, 4096, "bf16");
    let flops = k.flops;
    let (d, ns) = run(spec.clone(), k);
    let tflops = flops / ns / 1e3;
    let peak = spec.peak_tensor_tflops("bf16");
    eprintln!(
        "H100 gemm 4k: {:.0} TFLOPS ({:.0}% of peak) tensor_busy={} l2h={} l2m={} dram={} sched={:?}",
        tflops,
        100.0 * tflops / peak,
        d.stats.tensor_busy_cycles, 
        d.stats.l2_hits, d.stats.l2_misses, d.stats.dram_read_bytes, d.stats.sched.to_map()
    );
    assert!(
        tflops > 0.45 * peak && tflops <= 1.0 * peak,
        "got {tflops} TFLOPS"
    );
}

#[test]
fn blackwell_gemm_runs() {
    let spec = presets::b200();
    let k = synth::gemm(&spec, 8192, 8192, 4096, "bf16");
    let flops = k.flops;
    let (d, ns) = run(spec.clone(), k);
    let tflops = flops / ns / 1e3;
    let peak = spec.peak_tensor_tflops("bf16");
    eprintln!(
        "B200 gemm: {:.0} TFLOPS ({:.0}% of peak) sched={:?}",
        tflops,
        100.0 * tflops / peak,
        d.stats.sched.to_map()
    );
    assert!(
        tflops > 0.4 * peak && tflops <= 1.0 * peak,
        "got {tflops} TFLOPS"
    );
}

#[test]
fn decode_attention_is_memory_bound() {
    let spec = presets::h100_sxm();
    // 64 sequences x 4k context, GQA 32/8, d=128.
    let k = synth::attention(&spec, 64, 32, 8, 1, 4096, 128, false);
    let bytes = k.bytes;
    let (d, ns) = run(spec.clone(), k);
    let gbps = bytes / ns;
    eprintln!(
        "decode attn: {:.0} GB/s, {:.1} us, sched={:?}",
        gbps,
        ns / 1e3,
        d.stats.sched.to_map()
    );
    assert!(gbps > 0.3 * spec.hbm.bw_gbps && gbps < spec.hbm.bw_gbps);
}

#[test]
fn parsed_sass_with_control_bits_runs() {
    let src = include_str!("../../../examples/sass/vecadd_sm90.sass");
    let prog = rsim_sass::parse_listing(src).unwrap().remove(0);
    assert!(prog.has_control_bits());
    let k = KernelLaunch::from_program(
        "vecadd",
        prog,
        [1024, 1, 1],
        [256, 1, 1],
        16,
        0,
        Default::default(),
    );
    let (d, ns) = run(presets::h100_sxm(), k);
    assert!(d.stats.insts_issued > 0 && ns > 0.0);
    assert!(
        d.stats.sched.get(rsim_gpu::Stall::LongScoreboard) > 0,
        "{:?} {:?}",
        d.stats.sched.to_map(),
        d.stats.class_mix
    );
}
