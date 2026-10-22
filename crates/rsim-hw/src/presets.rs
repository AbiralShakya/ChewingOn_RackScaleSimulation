//! Calibrated starting points.
//!
//! Provenance legend used in `notes`:
//! * `spec`   — NVIDIA datasheet / whitepaper value.
//! * `ubench` — public microbenchmark literature (Jia et al.; Luo et al.,
//!   "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture", 2024;
//!   Chips and Cheese). Approximate.
//! * `est`    — estimate; calibrate with `ubench/` on real hardware.
//!
//! Tensor throughput is expressed per SM per cycle and the preset clock is
//! the one NVIDIA uses for its headline dense-TFLOPS figure, so
//! `peak_tensor_tflops("bf16")` reproduces the datasheet.

use crate::fabric::{LinkSpec, SwitchSpec};
use crate::gpu::*;
use crate::system::{GpuGroup, SystemSpec, TopologySpec};
use std::collections::BTreeMap;

pub const GPU_PRESETS: &[&str] = &[
    "a100_sxm",
    "a100_pcie",
    "h100_sxm",
    "h200_sxm",
    "b200",
    "gb200",
    "b300",
    "gb300",
];
pub const SWITCH_PRESETS: &[&str] = &["nvswitch3", "nvswitch_nvl5", "ib_ndr_leaf"];
pub const SYSTEM_PRESETS: &[&str] = &[
    "hgx_h100_8",
    "hgx_b200_8",
    "hgx_b300_8",
    "gb200_nvl72",
    "gb300_nvl72",
    "h100_2node_ib",
];

fn flops(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

pub fn gpu(name: &str) -> Option<GpuSpec> {
    Some(match name {
        "a100_sxm" => a100_sxm(),
        "a100_pcie" => a100_pcie(),
        "h100_sxm" => h100_sxm(),
        "h200_sxm" => h200_sxm(),
        "b200" => b200(),
        "gb200" => gb200(),
        "b300" => b300(),
        "b300_uncalibrated" => b300_uncalibrated(),
        "gb300" => gb300(),
        _ => return None,
    })
}

pub fn switch(name: &str) -> Option<SwitchSpec> {
    Some(match name {
        "nvswitch3" => SwitchSpec {
            name: "NVSwitch3 (NVLink4)".into(),
            ports: 64,
            gbps_per_port_per_dir: 25.0,
            latency_ns: 100.0, // est
            in_network_reduction: true,
        },
        "nvswitch_nvl5" => SwitchSpec {
            name: "NVLink5 switch chip".into(),
            ports: 72,
            gbps_per_port_per_dir: 50.0,
            latency_ns: 100.0, // est
            in_network_reduction: true,
        },
        "ib_ndr_leaf" => SwitchSpec {
            name: "InfiniBand NDR leaf".into(),
            ports: 64,
            gbps_per_port_per_dir: 50.0,
            latency_ns: 130.0, // est (~100-150 ns cut-through)
            in_network_reduction: false,
        },
        _ => return None,
    })
}

pub fn h100_sxm() -> GpuSpec {
    GpuSpec {
        name: "H100 SXM5 80GB".into(),
        arch: Arch::Hopper,
        sm_count: 132,
        sm_clock_mhz: 1830.0,
        sub_partitions: 4,
        max_warps_per_sm: 64,
        max_ctas_per_sm: 32,
        regs_per_sm: 65536,
        smem_per_sm_bytes: 228 * 1024,
        l1: CacheSpec {
            size_bytes: 64 * 1024,
            line_bytes: 128,
            sector_bytes: 32,
            assoc: 4,
            hit_latency: 33,
        },
        l2: CacheSpec {
            size_bytes: 50 * 1024 * 1024,
            line_bytes: 128,
            sector_bytes: 32,
            assoc: 16,
            hit_latency: 260,
        },
        l2_bytes_per_cycle: 3000.0,
        hbm: HbmSpec {
            capacity_gb: 80.0,
            bw_gbps: 3350.0,
            efficiency: 0.9,
            latency_ns: 350.0,
            loaded_latency: Vec::new(),
        },
        dies: 1,
        d2d: None,
        nvlink: NvlinkSpec {
            generation: 4,
            ports: 18,
            gbps_per_port_per_dir: 25.0,
            port_latency_ns: 60.0,
        },
        pipes: PipeSpec {
            fp32_lanes: 32,
            int_lanes: 16,
            fp64_lanes: 16,
            sfu_lanes: 4,
            fp16_lanes: 32,
            lsu_per_sm_per_cycle: 1,
            l1_bytes_per_cycle: 128,
        },
        tensor: TensorSpec {
            flops_per_cycle_per_sm: flops(&[
                ("fp16", 4096.0),
                ("bf16", 4096.0),
                ("tf32", 2048.0),
                ("fp8", 8192.0),
                ("int8", 8192.0),
                ("fp64", 256.0),
            ]),
            hmma_latency: 32,
            async_mma_overhead: 16,
            tmem_bytes: 0,
            tcgen05_latency: 0,
            tmem_alloc_latency: 0,
            tmem_alloc_2cta_latency: 0,
            hmma_fma_pipe_cycles: 0,
            sync_flops_per_cycle_per_sm: BTreeMap::new(),
            umma_smem_bytes_per_clk: 0.0,
            umma_lds_interference: 0.0,
        },
        latency: LatencyTable {
            int_alu: 4,
            imad: 4,
            fp32: 4,
            fp16: 5,
            fp64: 8,
            sfu: 18,
            conv: 14,
            shfl: 24,
            s2r: 22,
            ldc: 8,
            lds: 29,
            ldsm: 33,
            sts: 20,
            branch: 6,
            bar: 20,
            misc: 6,
            store_read: 4,
            dsmem: 190, // est; B300 measured 193 (calibration)
        },
        launch_overhead_ns: 1500.0,
        cta_launch_cycles: 120,
        kernel_window: Vec::new(),
        notes: vec![
            "sm_count, clocks, HBM, NVLink, tensor peak: spec (989 TFLOPS dense BF16 @1830 MHz)"
                .into(),
            "pipe latencies, L1/L2 latency: ubench (approx)".into(),
            "L2 bandwidth, HBM latency, launch gap, CTA launch: est".into(),
        ],
        opcode_timing: BTreeMap::new(),
        tma: TmaSpec::default(),
        power: None,
        issue: Default::default(),
    }
}

/// A100 SXM4 80GB (sm_80): HMMA tensor cores issued per SMSP (mma.sync), cp.async,
/// no TMA, no clusters. Two L2 partitions joined by a crossbar.
pub fn a100_sxm() -> GpuSpec {
    let mut g = h100_sxm();
    g.name = "A100 SXM4 80GB".into();
    g.arch = Arch::Ampere;
    g.sm_count = 108;
    // 312 TFLOPS dense BF16 / (108 SM * 2048 FLOP/clk) => 1410 MHz.
    g.sm_clock_mhz = 1410.0;
    g.smem_per_sm_bytes = 164 * 1024;
    g.l1.size_bytes = 192 * 1024 - 164 * 1024;
    g.l2.size_bytes = 40 * 1024 * 1024;
    g.l2.hit_latency = 210;
    g.l2_bytes_per_cycle = 2300.0;
    g.hbm = HbmSpec {
        capacity_gb: 80.0,
        bw_gbps: 2039.0,
        efficiency: 0.88,
        latency_ns: 400.0,
        loaded_latency: Vec::new(),
    };
    g.nvlink = NvlinkSpec {
        generation: 3,
        ports: 12,
        gbps_per_port_per_dir: 25.0,
        port_latency_ns: 60.0,
    };
    g.pipes.int_lanes = 16;
    g.pipes.fp64_lanes = 8;
    g.tensor.flops_per_cycle_per_sm = flops(&[
        ("fp16", 2048.0),
        ("bf16", 2048.0),
        ("tf32", 1024.0),
        ("int8", 4096.0),
        ("int4", 8192.0),
        ("fp64", 128.0),
    ]);
    g.tensor.async_mma_overhead = 0;
    g.notes = vec![
        "sm_count, clocks, HBM, NVLink, tensor peak: spec (312 TFLOPS dense BF16 @1410 MHz)".into(),
        "L2 latency/bandwidth: calibration/ A100 80GB PCIe (load calibration.json for measured values)".into(),
    ];
    g
}

/// A100 80GB PCIe: same chip as SXM4 at 1.94 TB/s HBM and a lower boost clock under load.
pub fn a100_pcie() -> GpuSpec {
    let mut g = a100_sxm();
    g.name = "A100 80GB PCIe".into();
    g.hbm.bw_gbps = 1935.0;
    g.notes.push(
        "PCIe: HBM 1935 GB/s (spec); measured clock 1395 MHz comes from calibration.json".into(),
    );
    g
}

pub fn h200_sxm() -> GpuSpec {
    let mut g = h100_sxm();
    g.name = "H200 SXM 141GB".into();
    g.hbm = HbmSpec {
        capacity_gb: 141.0,
        bw_gbps: 4800.0,
        efficiency: 0.9,
        latency_ns: 350.0,
        loaded_latency: Vec::new(),
    };
    g.notes
        .push("H200 = H100 compute + HBM3e 141 GB / 4.8 TB/s (spec)".into());
    g
}

pub fn b200() -> GpuSpec {
    GpuSpec {
        name: "B200 (HGX, 1000 W)".into(),
        arch: Arch::Blackwell,
        sm_count: 148,
        // 2.25 PFLOPS dense BF16 / (148 SM * 8192 FLOP/clk) => ~1856 MHz.
        sm_clock_mhz: 1856.0,
        sub_partitions: 4,
        max_warps_per_sm: 64,
        max_ctas_per_sm: 32,
        regs_per_sm: 65536,
        smem_per_sm_bytes: 228 * 1024,
        l1: CacheSpec { size_bytes: 64 * 1024, line_bytes: 128, sector_bytes: 32, assoc: 4, hit_latency: 33 },
        l2: CacheSpec { size_bytes: 126 * 1024 * 1024, line_bytes: 128, sector_bytes: 32, assoc: 16, hit_latency: 300 },
        l2_bytes_per_cycle: 6000.0,
        hbm: HbmSpec { capacity_gb: 180.0, bw_gbps: 7700.0, efficiency: 0.9, latency_ns: 380.0, loaded_latency: Vec::new() },
        dies: 2,
        d2d: Some(DieLink {
            gbps_per_dir: 5000.0,
            extra_latency: 60,
            far_fraction: None,
        }),
        nvlink: NvlinkSpec { generation: 5, ports: 18, gbps_per_port_per_dir: 50.0, port_latency_ns: 60.0 },
        pipes: PipeSpec {
            fp32_lanes: 32,
            int_lanes: 32,
            fp64_lanes: 8,
            sfu_lanes: 4,
            fp16_lanes: 32,
            lsu_per_sm_per_cycle: 1,
            l1_bytes_per_cycle: 128,
        },
        tensor: TensorSpec {
            flops_per_cycle_per_sm: flops(&[
                ("fp16", 8192.0),
                ("bf16", 8192.0),
                ("tf32", 4096.0),
                ("fp8", 16384.0),
                ("fp6", 16384.0),
                ("int8", 16384.0),
                ("fp4", 32768.0),
                ("fp64", 128.0),
            ]),
            hmma_latency: 32,
            async_mma_overhead: 12,
            tmem_bytes: 256 * 1024,
            tcgen05_latency: 300, // est
            // Fitted to the B300 FA4-prologue window points (ncu, recorded paths).
            tmem_alloc_latency: 1430,
            tmem_alloc_2cta_latency: 1990,
            hmma_fma_pipe_cycles: 0,
            // Legacy mma.sync path: measured 2037 FLOP/clk/SM BF16 on B300.
            sync_flops_per_cycle_per_sm: flops(&[("bf16", 2048.0), ("fp16", 2048.0), ("tf32", 1024.0)]),
            // Measured on B300 (bench_tc05x); SemiAnalysis's Blackwell
            // microbenchmarks report the same 48-cycle M128N64K16.
            umma_smem_bytes_per_clk: 128.0,
            umma_lds_interference: 0.107,
        },
        latency: LatencyTable {
            int_alu: 4,
            imad: 4,
            fp32: 4,
            fp16: 5,
            fp64: 10,
            sfu: 18,
            conv: 14,
            shfl: 24,
            s2r: 22,
            ldc: 8,
            lds: 29,
            ldsm: 33,
            sts: 20,
            branch: 6,
            bar: 20,
            misc: 6,
            store_read: 4,
            dsmem: 190, // est; B300 measured 193 (calibration)
        },
        launch_overhead_ns: 1500.0,
        cta_launch_cycles: 120,
        kernel_window: Vec::new(),
        notes: vec![
            "sm_count, HBM, NVLink5, dense tensor peaks: spec (2.25 PF BF16, 4.5 PF FP8, 9 PF FP4)".into(),
            "dual-die split, D2D latency, pipe latencies, L2 latency/bandwidth: est (Blackwell ubench data is sparse)".into(),
            "tcgen05 (UTC*MMA) issue/complete timing: est — calibrate with ubench/tcgen05_mma.cu".into(),
        ],
        opcode_timing: BTreeMap::new(),
        tma: TmaSpec::default(),
        power: None,
        issue: Default::default(),
    }
}

/// B200 as configured in GB200 (higher power envelope, 186 GB HBM3e).
/// B300 (Blackwell Ultra, SM103). Starts from B200 and takes what the
/// calibration run measured (`calibration/results/b300-*/calibration.json`)
/// via [`crate::calibration`]; the values here are the uncalibrated defaults.
pub fn b300_uncalibrated() -> GpuSpec {
    let mut g = b200();
    g.name = "B300 SXM".into();
    g.sm_clock_mhz = 2032.0;
    g.hbm.capacity_gb = 288.0;
    g.tensor
        .flops_per_cycle_per_sm
        .insert("fp4".into(), 32768.0);
    g.notes
        .push("b300: structure derived from b200 (est)".into());
    g
}

/// Measured on a B300 SXM6 AC (sm_103) by `calibration/`; compiled in so
/// the `b300` preset is calibrated out of the box.
pub const B300_CALIBRATION: &str =
    include_str!("../../../calibration/results/b300-sxm6-ac-sm103/calibration.json");

/// B300 with every measured value from the B300 calibration run applied
/// (clock, memory latencies/bandwidth, L2 partitions, per-opcode SASS
/// timing, tcgen05 throughput/latency, TMA, launch overhead).
pub fn b300() -> GpuSpec {
    let mut g = b300_uncalibrated();
    let doc: serde_json::Value =
        serde_json::from_str(B300_CALIBRATION).expect("embedded B300 calibration is valid JSON");
    crate::calibration::apply(&mut g, &doc).expect("embedded B300 calibration applies");
    g
}

/// GB300 (Grace + B300) as used in GB300 NVL72: the calibrated B300 die.
/// Board power/clock differences from the measured HGX part are not modelled (est).
pub fn gb300() -> GpuSpec {
    let mut g = b300();
    g.name = format!("GB300 ({})", g.name);
    g
}

pub fn gb200() -> GpuSpec {
    let mut g = b200();
    g.name = "B200 in GB200 (1200 W)".into();
    // 2.5 PFLOPS dense BF16 per GPU => ~2062 MHz at 8192 FLOP/clk/SM.
    g.sm_clock_mhz = 2062.0;
    g.hbm.capacity_gb = 186.0;
    g.hbm.bw_gbps = 8000.0;
    g.notes
        .push("GB200 per-GPU: 2.5 PF dense BF16, 8 TB/s HBM3e (spec)".into());
    g
}

fn nvlink_link(gpu: &GpuSpec) -> LinkSpec {
    LinkSpec {
        gbps_per_dir: gpu.nvlink.gbps_per_port_per_dir,
        latency_ns: 20.0,
    }
}

pub fn system(name: &str) -> Option<SystemSpec> {
    Some(match name {
        // 8x H100 + 4 NVSwitch3 chips. Real HGX splits the 18 links 5/4/4/5;
        // we use 4-4-5-5 via an explicit per-switch link count.
        "hgx_h100_8" => {
            let g = h100_sxm();
            SystemSpec {
                name: "HGX H100 8-GPU".into(),
                gpus: vec![GpuGroup::preset("h100_sxm", 8)],
                topology: TopologySpec::Switched {
                    switches: 4,
                    switch: switch("nvswitch3").unwrap(),
                    links_per_gpu_per_switch: vec![5, 4, 4, 5],
                    link: nvlink_link(&g),
                },
            }
        }
        "hgx_b200_8" => {
            let g = b200();
            SystemSpec {
                name: "HGX B200 8-GPU".into(),
                gpus: vec![GpuGroup::preset("b200", 8)],
                topology: TopologySpec::Switched {
                    switches: 2,
                    switch: SwitchSpec {
                        ports: 144,
                        ..switch("nvswitch_nvl5").unwrap()
                    },
                    links_per_gpu_per_switch: vec![9, 9],
                    link: nvlink_link(&g),
                },
            }
        }
        "hgx_b300_8" => {
            let mut s = system("hgx_b200_8").unwrap();
            s.name = "HGX B300 8-GPU".into();
            s.gpus = vec![GpuGroup::preset("b300", 8)];
            s
        }
        "gb300_nvl72" => {
            let mut s = system("gb200_nvl72").unwrap();
            s.name = "GB300 NVL72".into();
            s.gpus = vec![GpuGroup::preset("gb300", 72)];
            s
        }
        // 72 GPUs, 18 NVLink5 switch chips (9 trays x 2). Every GPU has one
        // link to every switch chip: 18 x 50 GB/s/dir = 1.8 TB/s bidir.
        "gb200_nvl72" => {
            let g = gb200();
            SystemSpec {
                name: "GB200 NVL72".into(),
                gpus: vec![GpuGroup::preset("gb200", 72)],
                topology: TopologySpec::Switched {
                    switches: 18,
                    switch: switch("nvswitch_nvl5").unwrap(),
                    links_per_gpu_per_switch: vec![1; 18],
                    link: nvlink_link(&g),
                },
            }
        }
        // Two HGX H100 nodes, rail-optimised 400G InfiniBand (1 NIC / GPU).
        "h100_2node_ib" => {
            let base = system("hgx_h100_8").unwrap();
            SystemSpec {
                name: "2x HGX H100, 400G IB rail-optimised".into(),
                gpus: vec![GpuGroup::preset("h100_sxm", 16)],
                topology: TopologySpec::MultiNode {
                    nodes: 2,
                    intra: Box::new(base.topology),
                    nic: LinkSpec {
                        gbps_per_dir: 50.0,
                        latency_ns: 600.0,
                    },
                    leaf: switch("ib_ndr_leaf").unwrap(),
                },
            }
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn datasheet_peaks() {
        let close = |a: f64, b: f64| (a - b).abs() / b < 0.01;
        assert!(close(h100_sxm().peak_tensor_tflops("bf16"), 989.4));
        assert!(close(b200().peak_tensor_tflops("bf16"), 2250.0));
        assert!(close(b200().peak_tensor_tflops("fp4"), 9000.0));
        assert!(close(gb200().peak_tensor_tflops("bf16"), 2500.0));
    }
}
