use rsim_rack::{simulate, RunConfig};

const HGX: &str = "preset: hgx_h100_8\n";

#[test]
fn tp8_mlp_with_allreduce() {
    let wl = r#"{
      "name": "tp8-mlp",
      "spmd": {"world": 8, "program": {"streams": [[
        {"op": "kernel", "name": "up_proj", "kernel": {"type": "gemm", "m": 2048, "n": 3584, "k": 8192}},
        {"op": "kernel", "name": "act", "kernel": {"type": "elementwise", "n": 7340032}},
        {"op": "kernel", "name": "down_proj", "kernel": {"type": "gemm", "m": 2048, "n": 8192, "k": 3584}},
        {"op": "collective", "kind": "all_reduce", "bytes": 33554432, "group": "world"}
      ]]}}
    }"#;
    let out = simulate(
        HGX,
        wl,
        RunConfig {
            trace: true,
            ..Default::default()
        },
    )
    .unwrap();
    let r = &out.report;
    eprintln!("{}", r.summary());
    assert_eq!(r.world, 8);
    assert_eq!(r.collectives.len(), 1);
    assert!(r.makespan_us > 10.0 && r.makespan_us < 10_000.0);
    // All ranks finish at (nearly) the same time: they are symmetric.
    let f: Vec<f64> = r.ranks.iter().map(|x| x.finish_us).collect();
    let (lo, hi) = (
        f.iter().cloned().fold(f64::MAX, f64::min),
        f.iter().cloned().fold(0.0, f64::max),
    );
    assert!((hi - lo) / hi < 0.01, "{f:?}");
    assert!(!out.trace.events.is_empty());
}

#[test]
fn overlap_compute_and_comm_on_two_streams() {
    // Stream 1 runs an all-reduce while stream 0 computes: the GEMM loses
    // the SMs reserved by the communication kernel.
    let wl = r#"{
      "name": "overlap",
      "spmd": {"world": 8, "program": {"streams": [
        [{"op": "kernel", "kernel": {"type": "gemm", "m": 4096, "n": 4096, "k": 4096}}, {"op": "record", "event": "g"}],
        [{"op": "collective", "kind": "all_reduce", "bytes": 268435456, "group": "world", "sms": 24}, {"op": "wait", "event": "g"}]
      ]}}
    }"#;
    let out = simulate(
        HGX,
        wl,
        RunConfig {
            trace: false,
            ..Default::default()
        },
    )
    .unwrap();
    eprintln!("{}", out.report.summary());
    assert_eq!(out.report.ranks[0].kernels, 1);
}

#[test]
fn pipeline_send_recv() {
    // 2-stage pipeline: rank 0 computes then sends activations to rank 1.
    let wl = r#"{
      "name": "pp2",
      "ranks": [
        {"streams": [[{"op": "kernel", "kernel": {"type": "gemm", "m": 1024, "n": 1024, "k": 1024}},
                      {"op": "send", "bytes": 8388608, "peer": 1}]]},
        {"streams": [[{"op": "recv", "bytes": 8388608, "peer": 0},
                      {"op": "kernel", "kernel": {"type": "gemm", "m": 1024, "n": 1024, "k": 1024}}]]}
      ]
    }"#;
    let out = simulate(
        HGX,
        wl,
        RunConfig {
            trace: false,
            ..Default::default()
        },
    )
    .unwrap();
    let r = &out.report;
    eprintln!("{}", r.summary());
    // Rank 1 must idle while rank 0 computes: a pipeline bubble.
    assert!(r.ranks[1].finish_us > r.ranks[0].finish_us);
    assert!(r.bubbles.gpu_idle_fraction > 0.2);
}
