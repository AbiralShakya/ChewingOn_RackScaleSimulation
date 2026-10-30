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

#[test]
fn viz_bundle_round_trips() {
    let wl = r#"{
      "name": "tp8-mlp",
      "spmd": {"world": 8, "program": {"streams": [[
        {"op": "kernel", "name": "up_proj", "kernel": {"type": "gemm", "m": 1024, "n": 3584, "k": 8192}},
        {"op": "collective", "kind": "all_reduce", "bytes": 33554432, "group": "world"}
      ]]}}
    }"#;
    let mut cfg = RunConfig {
        trace: false,
        memoize_kernels: true,
        detail_ranks: vec![0],
        ..Default::default()
    };
    cfg.gpu.detail = rsim_gpu::DetailConfig::full(vec![0]);
    cfg.net.link_bin_ns = 1000.0;
    let sys = rsim_hw::SystemSpec::from_yaml(HGX)
        .unwrap()
        .build()
        .unwrap();
    let out = simulate(HGX, wl, cfg).unwrap();
    // Detail only on rank 0; every rank has its spans and kernels.
    assert_eq!(out.ranks.len(), 8);
    assert!(out.ranks[0].kernels[0].detail.is_some());
    assert!(out.ranks[1].kernels[0].detail.is_none());
    assert!(out.links.iter().any(|l| l.bytes > 0 && !l.bins.is_empty()));
    let run = rsim_rack::viz::rack_run("t", &sys, &out, 1000.0);
    let b = rsim_rack::viz::bundle("t", vec![run]);
    assert_eq!(rsim_rack::viz::runs_of(&b).unwrap().len(), 1);
    let r = &b["runs"][0];
    assert_eq!(r["ranks"].as_array().unwrap().len(), 8);
    assert_eq!(r["details"].as_array().unwrap().len(), 1);
    let spans = r["ranks"][0]["spans"].as_array().unwrap();
    assert_eq!(spans.len(), 2 * 6);
    assert!(
        spans[5].as_i64().unwrap() >= 0,
        "kernel span links to its kernel"
    );
    let page = rsim_rack::viz::html(&b);
    assert!(!page.contains("/*RSVIZ_DATA*/"));
    assert_eq!(rsim_rack::viz::from_html(&page).unwrap(), b);
}

/// A job's timing must not depend on unrelated jobs elsewhere in the system:
/// two TP4 groups of an NVL72 on disjoint GPUs and links (ring all-reduces,
/// no shared switch ALU) give the second group the same finish whether or
/// not the first group's communicators are created before its own.
#[test]
fn disjoint_jobs_do_not_perturb_each_other() {
    let job = |ranks: &str| {
        format!(
            r#"{{"streams": [[
              {{"op": "kernel", "kernel": {{"type": "gemm", "m": 1024, "n": 4096, "k": 4096}}}},
              {{"op": "collective", "kind": "all_reduce", "bytes": 3145728, "group": {ranks}, "algo": "ring"}},
              {{"op": "collective", "kind": "all_gather", "bytes": 1048576, "group": {ranks}}}
            ]]}}"#
        )
    };
    let idle = r#"{"streams": [[{"op": "delay", "ns": 0}]]}"#;
    let wl = |first: bool| {
        let ranks: Vec<String> = (0..72)
            .map(|r| {
                if (4..8).contains(&r) {
                    job("[4, 5, 6, 7]")
                } else if first && r < 4 {
                    job("[0, 1, 2, 3]")
                } else {
                    idle.to_string()
                }
            })
            .collect();
        format!(r#"{{"name": "two-jobs", "ranks": [{}]}}"#, ranks.join(","))
    };
    let finish = |w: String| -> Vec<f64> {
        let out = simulate(
            "preset: gb300_nvl72\n",
            &w,
            RunConfig {
                trace: false,
                memoize_kernels: true,
                ..Default::default()
            },
        )
        .unwrap();
        out.report.ranks[4..8].iter().map(|r| r.finish_us).collect()
    };
    assert_eq!(finish(wl(true)), finish(wl(false)));
}
