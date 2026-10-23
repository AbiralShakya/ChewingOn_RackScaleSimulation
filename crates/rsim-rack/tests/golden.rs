//! Bit-exactness guard for rack runs: exact and fast mode are pinned to a
//! digest of the whole report (wall-clock fields aside), and neither may
//! depend on how many host threads run the PDES windows or the memoized
//! kernels. The digests are main's before the simulator-speed work, re-taken
//! on main after the NCCL network model (#13) changed collective timing
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
    run(HGX, WL, fast, true, threads).0
}

/// The report without its wall-clock fields, and the (folded, forked) rank counts.
fn run(
    system: &str,
    wl: &str,
    fast: bool,
    fold: bool,
    threads: usize,
) -> (serde_json::Value, (u64, u64)) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let out = pool
        .install(|| {
            simulate(
                system,
                wl,
                RunConfig {
                    trace: false,
                    memoize_kernels: fast,
                    fold_ranks: fold,
                    ..Default::default()
                },
            )
        })
        .unwrap();
    let folds = (out.report.sim.folded_ranks, out.report.sim.forked_ranks);
    let mut v = serde_json::to_value(&out.report).unwrap();
    for k in [
        "wallclock_s",
        "sim_minstr_per_s",
        "folded_ranks",
        "forked_ranks",
    ] {
        v["sim"].as_object_mut().unwrap().remove(k);
    }
    (v, folds)
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
        assert_eq!(digest(&one), 0x29f1_7b59_b1d4_9712);
    }
}

#[test]
fn fast_mode_is_pinned_and_thread_count_independent() {
    let one = report(true, 1);
    let many = report(true, 4);
    assert_eq!(one, many, "fast report depends on the thread count");
    eprintln!("fast digest {:#018x}", digest(&one));
    if std::env::var_os("RSIM_GOLDEN_PRINT").is_none() {
        assert_eq!(digest(&one), 0xaabf_763d_09c6_0736);
    }
}

/// Rank folding (exact mode) gives the same report as simulating every rank:
/// on a rack where the ranks stay in lockstep, and on two nodes over
/// InfiniBand where the all-to-all completes at different times on
/// different ranks, so folded ranks must fork part-way.
#[test]
fn folding_ranks_matches_simulating_every_rank() {
    let two_node = "preset: h100_2node_ib\n";
    let a2a = r#"{
      "name": "a2a16",
      "spmd": {"world": 16, "program": {"streams": [[
        {"op": "kernel", "name": "pre", "kernel": {"type": "gemm", "m": 256, "n": 2048, "k": 2048}},
        {"op": "collective", "kind": "all_to_all", "bytes": 16777216, "group": "world", "sms": 16},
        {"op": "kernel", "name": "post", "kernel": {"type": "gemm", "m": 256, "n": 2048, "k": 2048}},
        {"op": "collective", "kind": "all_reduce", "bytes": 8388608, "group": "world"},
        {"op": "kernel", "name": "tail", "kernel": {"type": "elementwise", "n": 1048576}}
      ]]}}
    }"#;
    let (folded, f) = run(HGX, WL, false, true, 4);
    let (every, _) = run(HGX, WL, false, false, 4);
    assert_eq!(folded, every, "folding changed the HGX report");
    assert_eq!(f, (7, 0), "8 identical ranks in lockstep fold onto one");
    let (folded, f) = run(two_node, a2a, false, true, 4);
    let (every, _) = run(two_node, a2a, false, false, 4);
    assert_eq!(folded, every, "folding changed the two-node report");
    eprintln!("two-node: {} folded, {} forked", f.0, f.1);
    assert!(f.1 > 0, "the two-node run exercises forking");
}
