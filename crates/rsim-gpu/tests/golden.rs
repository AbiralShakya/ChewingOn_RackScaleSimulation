//! Bit-exactness guard for the cycle-level core.
//!
//! Each case is pinned to its cycle count and to a digest of every counter
//! the kernel reports (stall attribution per warp index, class mix, cache
//! and DRAM traffic, mbarrier waits). The digests were taken before the
//! simulator-speed work (hot-path layout, memos, parallel runs) and must not
//! move: a change that is meant to alter timing updates them in the same
//! commit and says why (`RSIM_GOLDEN_PRINT=1` prints the new values
//! instead of checking them). Digests last moved for a format-only change:
//! a warp's issue cycle is now counted as `selected` (as Nsight does)
//! instead of being folded into the stall that follows it.

use rsim_gpu::{synth, Gpu, GpuConfig, KernelDone, KernelLaunch, SchedPolicy};
use std::sync::Arc;

/// FNV-1a over the JSON form of the result.
fn digest(d: &KernelDone) -> u64 {
    let s = serde_json::to_string(d).unwrap();
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

fn run(spec: rsim_hw::GpuSpec, cfg: GpuConfig, k: KernelLaunch) -> KernelDone {
    let mut g = Gpu::new(0, spec, cfg, false);
    g.submit(Arc::new(k), 0, 0);
    let mut out = vec![];
    g.advance(u64::MAX / 2, &mut out);
    out.pop().expect("kernel completes")
}

fn check(name: &str, d: &KernelDone, cycles: u64, want: u64) {
    let got = digest(d);
    eprintln!(
        "{name}: cycles {} digest {got:#018x}",
        d.end_cycle - d.start_cycle
    );
    if std::env::var_os("RSIM_GOLDEN_PRINT").is_some() {
        return;
    }
    assert_eq!(d.end_cycle - d.start_cycle, cycles, "{name}: cycles moved");
    assert_eq!(got, want, "{name}: counters moved (cycles still match)");
}

fn gpu(name: &str) -> rsim_hw::GpuSpec {
    rsim_hw::presets::gpu(name).unwrap()
}

#[test]
fn hopper_gemm_and_attention() {
    let g = gpu("h100_sxm");
    let d = run(
        g.clone(),
        GpuConfig::default(),
        synth::gemm(&g, 1024, 1024, 1024, "bf16"),
    );
    check("h100 gemm 1024^3", &d, 19767, 0x69fe_6ab7_77df_732c);
    let d = run(
        g.clone(),
        GpuConfig::default(),
        synth::attention(&g, 1, 16, 16, 1024, 1024, 128, true),
    );
    check("h100 attention", &d, 19112, 0x8d63_1e2d_e567_e1de);
}

#[test]
fn blackwell_gemm_and_elementwise() {
    let g = gpu("b300");
    let d = run(
        g.clone(),
        GpuConfig::default(),
        synth::gemm(&g, 2048, 2048, 1024, "bf16"),
    );
    check("b300 gemm", &d, 18145, 0x04cd_39ae_e0ab_036d);
    let d = run(
        g.clone(),
        GpuConfig::default(),
        synth::elementwise("add", 1 << 24, 2, 2.0, 1),
    );
    check("b300 elementwise", &d, 21847, 0xd8c6_1116_cff8_874f);
}

#[test]
fn ampere_gemm_round_robin() {
    let g = gpu("a100_sxm");
    let cfg = GpuConfig {
        policy: SchedPolicy::Lrr,
        ..Default::default()
    };
    let d = run(g.clone(), cfg, synth::gemm(&g, 1024, 1024, 2048, "bf16"));
    check("a100 gemm lrr", &d, 45429, 0x69fb_fd1e_c66b_55a2);
}

/// FA4 replayed from its NVBit recording on the calibrated B300: exact
/// per-warp paths, mbarriers, TMA and UMMA, checked per warp role.
#[test]
fn fa4_recording_on_calibrated_b300() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let mut g = gpu("b300");
    rsim_hw::calibration::apply_file(
        &mut g,
        std::path::Path::new(&format!(
            "{root}/calibration/results/b300-sxm6-ac-sm103/calibration.json"
        )),
    )
    .unwrap();
    let p = rsim_sass::PathKernel::from_file(&format!(
        "{root}/research/flash_attention/rpath/b300_fa4_cp_c1024_kv8.rpath.gz"
    ))
    .unwrap();
    let d = run(g, GpuConfig::default(), KernelLaunch::from_path(p));
    assert!(d.stats.warp_by_index.len() > 1, "per-role stalls recorded");
    check("b300 fa4 c1024 kv8", &d, 49398, 0x02f5_3268_3ec2_e9cd);
}
