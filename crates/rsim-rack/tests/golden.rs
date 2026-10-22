//! Bit-exactness guard for rack runs: exact and fast mode are pinned to a
//! digest of the whole report (wall-clock fields aside), and neither may
//! depend on how many host threads run the PDES windows or the memoized
//! kernels. The digests were taken before the simulator-speed work
//! (`RSIM_GOLDEN_PRINT=1` prints new values instead of checking them).

use rsim_rack::{simulate, RunConfig};

const HGX: &str = "preset: hgx_h100_8\n";

/// Expert-parallel MoE layer with compute/communication overlap: GEMMs,
/// an all-to-all on reserved SMs and an all-reduce, on two streams.
const WL: &str = r#"{
  "name": "golden",
  "spmd": {"world": 8, "program": {"streams": [
    [
      {"op": "kernel", "name": "qkv", "kernel": {"type": "gemm", "m": 512, "n": 3072, "k": 4096}},
      {"op": "kernel", "name": "act", "kernel": {"type": "elementwise", "n": 2097152}},
      {"op": "record", "event": "a"},
      {"op": "kernel", "name": "expert", "kernel": {"type": "gemm", "m": 256, "n": 4096, "k": 2048}},
      {"op": "wait", "event": "d"},
      {"op": "kernel", "name": "down", "kernel": {"type": "gemm", "m": 512, "n": 4096, "k": 1024}},
      {"op": "collective", "kind": "all_reduce", "bytes": 4194304, "group": "world"}
    ],
    [
      {"op": "wait", "event": "a"},
      {"op": "collective", "kind": "all_to_all", "bytes": 8388608, "group": "world", "sms": 16},
      {"op": "record", "event": "d"}
    ]
  ]}}
}"#;

fn report(fast: bool, threads: usize) -> serde_json::Value {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let out = pool
        .install(|| {
            simulate(
                HGX,
                WL,
                RunConfig {
                    trace: false,
                    memoize_kernels: fast,
                    ..Default::default()
                },
            )
        })
        .unwrap();
    let mut v = serde_json::to_value(&out.report).unwrap();
    for k in ["wallclock_s", "sim_minstr_per_s"] {
        v["sim"].as_object_mut().unwrap().remove(k);
    }
    v
}

fn digest(v: &serde_json::Value) -> u64 {
    v.to_string()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
        })
}

#[test]
fn exact_mode_is_pinned_and_thread_count_independent() {
    let one = report(false, 1);
    let many = report(false, 4);
    assert_eq!(one, many, "exact report depends on the thread count");
    eprintln!("exact digest {:#018x}", digest(&one));
    if std::env::var_os("RSIM_GOLDEN_PRINT").is_none() {
        assert_eq!(digest(&one), 0xf5fa_dd8a_e058_4c7e);
    }
}

#[test]
fn fast_mode_is_pinned_and_thread_count_independent() {
    let one = report(true, 1);
    let many = report(true, 4);
    assert_eq!(one, many, "fast report depends on the thread count");
    eprintln!("fast digest {:#018x}", digest(&one));
    if std::env::var_os("RSIM_GOLDEN_PRINT").is_none() {
        assert_eq!(digest(&one), 0x636b_ce19_91dc_7077);
    }
}
