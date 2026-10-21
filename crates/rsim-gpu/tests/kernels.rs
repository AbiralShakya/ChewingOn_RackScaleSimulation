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

/// sm_80-style structural hazard: HMMA holds the FP32 pipe, so FFMA can't
/// hide under it, while MUFU (XU pipe) still can.
#[test]
fn hmma_blocks_fp32_pipe_not_mufu() {
    fn listing(filler: &str) -> String {
        let mut s = String::from("\t\tFunction : hazard\n");
        let mut pc = 0;
        for i in 0..64 {
            let d = 4 + 4 * (i % 16);
            let (f1, f2) = (120 + (2 * i) % 16, 121 + (2 * i) % 16);
            for op in [
                format!("HMMA.16816.F32.BF16 R{d}, R200, R204, R{d}"),
                filler.replace("RD", &format!("R{f1}")),
                filler.replace("RD", &format!("R{f2}")),
            ] {
                s += &format!("        /*{pc:04x}*/                   {op} ;\n");
                pc += 16;
            }
        }
        s += &format!("        /*{pc:04x}*/                   EXIT ;\n");
        s
    }
    let time = |filler: &str, hold: u32| {
        let mut spec = presets::h100_sxm();
        spec.tensor.hmma_fma_pipe_cycles = hold;
        let prog = rsim_sass::parse_listing(&listing(filler))
            .unwrap()
            .remove(0);
        let k = KernelLaunch::from_program(
            "hazard",
            prog,
            [1, 1, 1],
            [32, 1, 1],
            64,
            0,
            Default::default(),
        );
        run(spec, k).0.end_cycle as f64
    };
    let ffma = "FFMA RD, R101, R102, R103";
    let mufu = "MUFU.EX2 RD, R111";
    let (f0, f4) = (time(ffma, 0), time(ffma, 4));
    let (m0, m4) = (time(mufu, 0), time(mufu, 4));
    eprintln!("ffma {f0} -> {f4}, mufu {m0} -> {m4}");
    assert!(
        f4 > 1.25 * f0,
        "FFMA should serialise behind HMMA: {f0} -> {f4}"
    );
    assert!(m4 < 1.05 * m0, "MUFU should still overlap: {m0} -> {m4}");
}

/// A measured kernel window is charged once per kernel, from the nearest
/// launch shape, before the first CTA starts.
#[test]
fn kernel_window_delays_first_cta() {
    let listing = "\t\tFunction : empty\n        /*0000*/                   EXIT ;\n";
    let time = |pts: Vec<rsim_hw::WindowPoint>, smem: u32| {
        let mut spec = presets::b200();
        spec.kernel_window = pts;
        let prog = rsim_sass::parse_listing(listing).unwrap().remove(0);
        let k = KernelLaunch::from_program(
            "empty",
            prog,
            [148, 1, 1],
            [512, 1, 1],
            32,
            smem,
            Default::default(),
        );
        let cfg = GpuConfig {
            kernel_window: true,
            ..Default::default()
        };
        let mut g = Gpu::new(0, spec, cfg, false);
        g.submit(Arc::new(k), 0, 7);
        let mut out = vec![];
        g.advance(u64::MAX / 2, &mut out);
        let d = out.pop().unwrap();
        d.end_cycle - d.start_cycle
    };
    let pt = |smem_bytes, cycles| rsim_hw::WindowPoint {
        ctas: 0,
        threads: 512,
        smem_bytes,
        cluster: 1,
        tmem_prologue: false,
        cycles,
    };
    let base = time(vec![], 0);
    let small = time(vec![pt(0, 2000.0), pt(200 << 10, 9000.0)], 0);
    let big = time(vec![pt(0, 2000.0), pt(200 << 10, 9000.0)], 190 << 10);
    eprintln!("empty kernel: {base} cycles, window 2000 -> {small}, window 9000 -> {big}");
    // The model's own time for the measured empty kernel is subtracted, so
    // the charge differs between the two points by exactly their difference.
    assert!(base < 1000);
    assert!(small > base + 1000, "{small}");
    assert!((big - small).abs_diff(7000) < 100, "{big} - {small}");
}
