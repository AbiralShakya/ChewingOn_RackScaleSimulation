//! `rsim` — rack-scale, cycle-level GPU simulator.
//!
//! ```text
//! rsim presets                                   # list hardware presets
//! rsim system  configs/gb200_nvl72.yaml          # print resolved topology summary
//! rsim kernel  --gpu h100_sxm gemm 8192 8192 8192
//! rsim sass    kernel.sass --grid 1024 --block 256 --gpu b200
//! rsim path    fa4.rpath --gpu b300 --calibration calibration.json   # NVBit-recorded kernel
//! rsim run     --system configs/gb200_nvl72.yaml --workload wl.json \
//!              --trace trace.json --report report.json
//! rsim viz path fa4.rpath --gpu b300 -o fa4.html  # rack -> GPU -> SM -> warp -> SASS viewer
//! rsim timeline k.sass --kernel t_alu --block 128 --gpu b300 -o t.json  # per-warp issue times
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rsim_gpu::{synth, Gpu, GpuConfig, KernelDone, KernelLaunch};
use rsim_hw::{presets, SystemSpec};
use rsim_rack::{RunConfig, Simulation, Workload};
use std::path::PathBuf;
use std::sync::Arc;

mod validate;
mod viz;

#[derive(Parser)]
#[command(
    name = "rsim",
    version,
    about = "Rack-scale cycle-level GPU simulator (Hopper / Blackwell / NVL72)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List built-in GPU, switch and system presets.
    Presets,
    /// Resolve a system YAML and print its topology.
    System { path: PathBuf },
    /// Simulate one synthetic kernel on one GPU.
    Kernel {
        #[arg(long, default_value = "h100_sxm")]
        gpu: String,
        /// gemm M N K | attention B HQ HKV SQ SKV D | elementwise N | rownorm ROWS COLS |
        /// ssm_update B H P N G | conv1d B DIM L [W] | ssd_<stage> B L H P N G [CHUNK]
        kind: String,
        dims: Vec<u64>,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        /// Apply a rackscale-calibration/v1 JSON on top of the preset.
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Replay measured reference kernels (ncu_reference.json) and report
    /// simulated vs measured SM cycles.
    Validate {
        reference: PathBuf,
        #[arg(long, default_value = "b300")]
        gpu: String,
        /// Apply a rackscale-calibration/v1 JSON on top of the preset.
        #[arg(long)]
        calibration: Option<PathBuf>,
        /// Only these reference labels (comma separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Fail if any kernel's |error| exceeds this many percent.
        #[arg(long)]
        max_error: Option<f64>,
        #[arg(long)]
        json: bool,
    },
    /// Parse a cuobjdump/nvdisasm listing and simulate one kernel from it.
    Sass {
        path: PathBuf,
        #[arg(long, default_value = "h100_sxm")]
        gpu: String,
        #[arg(long, default_value_t = 1)]
        grid: u32,
        #[arg(long, default_value_t = 128)]
        block: u32,
        #[arg(long, default_value_t = 32)]
        regs: u32,
        #[arg(long, default_value_t = 0)]
        smem: u32,
        /// Only print the parsed program (classes, control bits).
        #[arg(long)]
        inspect: bool,
    },
    /// Simulate one kernel from a SASS listing and dump every warp's issue
    /// timeline (cycle and PC of each issued instruction) on SM 0. Used to
    /// compare against clock64 probes read on hardware
    /// (calibration/dissect/timeline.py).
    Timeline {
        path: PathBuf,
        /// Function name in the listing (default: the first one).
        #[arg(long)]
        kernel: Option<String>,
        #[arg(long, default_value = "b300")]
        gpu: String,
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        grid: u32,
        #[arg(long, default_value_t = 32)]
        block: u32,
        #[arg(long, default_value_t = 64)]
        regs: u32,
        #[arg(long, default_value_t = 0)]
        smem: u32,
        /// Loose round-robin warp scheduling instead of greedy-then-oldest.
        #[arg(long)]
        lrr: bool,
        /// Greedy-then-youngest warp scheduling.
        #[arg(long)]
        gty: bool,
        /// Ignore the compiler's control bits (per-register scoreboarding).
        #[arg(long)]
        no_control_bits: bool,
        /// SMs whose warps are traced (default: all).
        #[arg(long, value_delimiter = ',')]
        sms: Vec<u32>,
        /// Run the kernel this many times first (warm caches, as a
        /// hardware measurement after warm-up launches); the timeline is the
        /// last run's.
        #[arg(long, default_value_t = 0)]
        warm: u32,
        #[arg(short, long)]
        out: PathBuf,
    },
    /// Simulate a kernel recorded on hardware with the NVBit warp_path tool
    /// (exact per-warp paths, mbarrier/TMA/UMMA operands; see research/flash_attention).
    Path {
        path: PathBuf,
        #[arg(long, default_value = "b300")]
        gpu: String,
        /// rackscale-calibration/v1 JSON measured on the target GPU.
        #[arg(long)]
        calibration: Option<PathBuf>,
        /// Override registers per thread (NVBit reports 0 for some JIT kernels).
        #[arg(long)]
        regs: Option<u32>,
        #[arg(long)]
        json: bool,
    },
    /// Run a workload on a system.
    Run {
        #[arg(long)]
        system: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        #[arg(long)]
        trace: Option<PathBuf>,
        #[arg(long)]
        report: Option<PathBuf>,
        /// Kernel launch gap override in ns (e.g. 0 for CUDA graphs).
        #[arg(long)]
        launch_gap_ns: Option<f64>,
        /// Use loose round-robin warp scheduling instead of GTO.
        #[arg(long)]
        lrr: bool,
        /// Ignore SASS control bits (use register scoreboarding).
        #[arg(long)]
        no_control_bits: bool,
        /// Fast mode: simulate each distinct kernel once (cold GPU) and reuse it across ranks.
        #[arg(long)]
        fast: bool,
    },
    /// Simulate with full capture and write a zoomable viewer: rack, GPU,
    /// SMs, warps and SASS (see docs/VIZ.md).
    Viz {
        #[command(subcommand)]
        cmd: viz::VizCmd,
    },
}

/// Simulate one kernel alone on a cold GPU (kernel window charged, as ncu times it).
pub(crate) fn sim_kernel(
    gpu: rsim_hw::GpuSpec,
    k: KernelLaunch,
    detail: rsim_gpu::DetailConfig,
) -> Result<(KernelDone, f64)> {
    let spec = gpu.clone();
    let launch = Arc::new(k);
    let mut g = Gpu::new(
        0,
        gpu,
        GpuConfig {
            kernel_window: true,
            detail,
            ..Default::default()
        },
        false,
    );
    g.submit(launch.clone(), 0, 0);
    let mut out = vec![];
    let t0 = std::time::Instant::now();
    g.advance(u64::MAX / 2, &mut out);
    let wall = t0.elapsed().as_secs_f64();
    let mut d = out.pop().context("kernel did not finish")?;
    let stretch = rsim_gpu::dvfs_stretch(&spec, &launch, &d);
    if stretch > 1.0 {
        d.end_cycle = d.start_cycle + ((d.end_cycle - d.start_cycle) as f64 * stretch) as u64;
        eprintln!(
            "  power cap: sustained SM clock {:.0} MHz (x{stretch:.3} on time)",
            spec.sm_clock_mhz / stretch
        );
    }
    Ok((d, wall))
}

/// Build one synthetic kernel (`rsim kernel` / `rsim viz kernel`).
pub(crate) fn synth_kernel(
    g: &rsim_hw::GpuSpec,
    kind: &str,
    dims: &[u64],
    dtype: &str,
) -> Result<KernelLaunch> {
    let need = |n: usize| -> Result<()> {
        if dims.len() < n {
            bail!("{kind} needs {n} dimensions");
        }
        Ok(())
    };
    let k = match kind {
        "gemm" => {
            need(3)?;
            synth::gemm(g, dims[0], dims[1], dims[2], dtype)
        }
        "attention" => {
            need(6)?;
            synth::attention(
                g,
                dims[0],
                dims[1],
                dims[2],
                dims[3],
                dims[4],
                dims[5],
                dims.get(6).copied().unwrap_or(0) == 1,
            )
        }
        "elementwise" => {
            need(1)?;
            synth::elementwise("elementwise", dims[0], 2, 2.0, 1)
        }
        "rownorm" => {
            need(2)?;
            synth::row_norm("rmsnorm", dims[0], dims[1], 2.0)
        }
        "ssm_update" => {
            need(5)?;
            rsim_gpu::synth_ssm::ssm_state_update(
                g, dims[0], dims[1], dims[2], dims[3], dims[4], 2.0,
            )
        }
        "conv1d" => {
            need(3)?;
            rsim_gpu::synth_ssm::conv1d(
                g,
                dims[0],
                dims[1],
                dims[2],
                dims.get(3).copied().unwrap_or(4),
            )
        }
        s if s.starts_with("ssd_") => {
            need(6)?;
            rsim_gpu::synth_ssm::ssd(
                g,
                &s[4..],
                rsim_gpu::synth_ssm::SsdShape {
                    batch: dims[0],
                    seqlen: dims[1],
                    heads: dims[2],
                    headdim: dims[3],
                    d_state: dims[4],
                    groups: dims[5],
                    chunk: dims.get(6).copied().unwrap_or(256),
                },
                None,
            )
        }
        _ => bail!("unknown kernel kind {kind}"),
    };
    Ok(k)
}

fn run_kernel(gpu: rsim_hw::GpuSpec, k: KernelLaunch, json: bool) -> Result<()> {
    let ghz = gpu.sm_clock_mhz / 1000.0;
    let peak = gpu.peak_tensor_tflops("bf16");
    let hbm = gpu.hbm.bw_gbps;
    let (flops, bytes, name) = (k.flops, k.bytes, k.name.clone());
    let (d, wall) = sim_kernel(gpu, k, Default::default())?;
    if json {
        println!("{}", serde_json::to_string_pretty(&d)?);
        return Ok(());
    }
    let us = (d.end_cycle - d.start_cycle) as f64 / ghz / 1e3;
    println!("{name}");
    println!(
        "  time        {us:.2} us   ({} cycles, {} CTAs, {} CTA/SM)",
        d.end_cycle - d.start_cycle,
        d.stats.ctas,
        d.stats.ctas_per_sm
    );
    if flops > 0.0 {
        println!(
            "  compute     {:.1} TFLOPS  ({:.1}% of dense BF16 peak)",
            flops / us / 1e6,
            100.0 * flops / us / 1e6 / peak
        );
    }
    if bytes > 0.0 {
        println!(
            "  memory      {:.0} GB/s model-bytes ({:.1}% of HBM peak)",
            bytes / us / 1e3,
            100.0 * bytes / us / 1e3 / hbm
        );
    }
    println!(
        "  dram        {:.1} MB read, {:.1} MB written; L2 hit {:.1}%",
        d.stats.dram_read_bytes as f64 / 1e6,
        d.stats.dram_write_bytes as f64 / 1e6,
        100.0 * d.stats.l2_hits as f64 / (d.stats.l2_hits + d.stats.l2_misses).max(1) as f64
    );
    println!("  scheduler-cycle anatomy:");
    let tot = d.stats.sched.total().max(1) as f64;
    let mut v: Vec<_> = d.stats.sched.to_map().into_iter().collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.1));
    for (k, n) in v {
        let f = n as f64 / tot;
        println!(
            "    {:<20} {:>6.2}%  {}",
            k,
            100.0 * f,
            "#".repeat((f * 50.0).round() as usize)
        );
    }
    if !d.stats.unknown_opcodes.is_empty() {
        println!(
            "  unclassified opcodes (timed as misc): {:?}",
            d.stats.unknown_opcodes
        );
    }
    println!(
        "  [sim {:.2} s wall, {:.2} M warp-instr/s]",
        wall,
        d.stats.insts_issued as f64 / wall / 1e6
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Viz { cmd } => viz::main(cmd)?,
        Cmd::Presets => {
            println!("GPUs:");
            for g in presets::GPU_PRESETS {
                let s = presets::gpu(g).unwrap();
                println!(
                    "  {:<10} {:<28} {:>3} SMs @ {:>4.0} MHz  {:>6.0} TFLOPS bf16  {:>5.0} GB/s HBM  NVLink{} {}x{} GB/s",
                    g,
                    s.name,
                    s.sm_count,
                    s.sm_clock_mhz,
                    s.peak_tensor_tflops("bf16"),
                    s.hbm.bw_gbps,
                    s.nvlink.generation,
                    s.nvlink.ports,
                    s.nvlink.gbps_per_port_per_dir
                );
            }
            println!("switches: {}", presets::SWITCH_PRESETS.join(", "));
            println!("systems:  {}", presets::SYSTEM_PRESETS.join(", "));
        }
        Cmd::System { path } => {
            let src = std::fs::read_to_string(&path)?;
            let sys = SystemSpec::from_yaml(&src)?.build()?;
            println!(
                "{}: {} GPUs, {} nodes, {} links",
                sys.name,
                sys.num_gpus(),
                sys.topology.nodes.len(),
                sys.topology.links.len()
            );
            let bw: f64 = sys
                .topology
                .links
                .iter()
                .filter(|l| l.a == 0 || l.b == 0)
                .map(|l| l.gbps_per_dir)
                .sum();
            println!(
                "GPU0 injection bandwidth: {bw:.0} GB/s per direction; min link latency {:.0} ns",
                sys.min_link_latency_ns()
            );
        }
        Cmd::Kernel {
            gpu,
            kind,
            dims,
            dtype,
            calibration,
            json,
        } => {
            let mut g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            if let Some(c) = &calibration {
                rsim_hw::calibration::apply_file(&mut g, c)?;
            }
            let g = g;
            let k = synth_kernel(&g, &kind, &dims, &dtype)?;
            run_kernel(g, k, json)?;
        }
        Cmd::Validate {
            reference,
            gpu,
            calibration,
            only,
            max_error,
            json,
        } => {
            let mut g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            if let Some(c) = &calibration {
                rsim_hw::calibration::apply_file(&mut g, c)?;
            }
            let rows = validate::run(&reference, &g, &only)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&rows)?);
            } else {
                println!(
                    "{:<26} {:>12} {:>12} {:>8} {:>8}",
                    "kernel", "ncu cycles", "sim cycles", "error", "trips"
                );
                for r in &rows {
                    println!(
                        "{:<26} {:>12.0} {:>12.0} {:>7.1}% {:>8}",
                        r.label, r.measured_cycles, r.simulated_cycles, r.error_pct, r.trip_count
                    );
                }
            }
            if !only.is_empty() && rows.len() < only.len() {
                bail!(
                    "{} of {} requested kernels did not run",
                    only.len() - rows.len(),
                    only.len()
                );
            }
            if let Some(tol) = max_error {
                let bad: Vec<_> = rows.iter().filter(|r| r.error_pct.abs() > tol).collect();
                if !bad.is_empty() {
                    let names: Vec<_> = bad
                        .iter()
                        .map(|r| format!("{} ({:+.1}%)", r.label, r.error_pct))
                        .collect();
                    bail!("outside ±{tol}%: {}", names.join(", "));
                }
            }
        }
        Cmd::Path {
            path,
            gpu,
            calibration,
            regs,
            json,
        } => {
            let mut g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            if let Some(c) = &calibration {
                let took = rsim_hw::calibration::apply_file(&mut g, c)
                    .map_err(|e| anyhow::anyhow!("calibration {}: {e}", c.display()))?;
                if !json {
                    eprintln!("calibration: {}", took.join(", "));
                }
            }
            let p = rsim_sass::PathKernel::from_file(&path.to_string_lossy())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut k = KernelLaunch::from_path(p);
            if let Some(r) = regs {
                k.regs_per_thread = r;
            }
            run_kernel(g, k, json)?;
        }
        Cmd::Timeline {
            path,
            kernel,
            gpu,
            calibration,
            grid,
            block,
            regs,
            smem,
            lrr,
            gty,
            no_control_bits,
            sms,
            warm,
            out,
        } => {
            let mut g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            if let Some(c) = &calibration {
                rsim_hw::calibration::apply_file(&mut g, c)?;
            }
            let ps = path.to_string_lossy();
            let k = if ps.ends_with(".rpath") || ps.ends_with(".rpath.gz") {
                // An NVBit warp-path recording: exact per-warp paths and
                // addresses (grid and block come from the recording).
                let p =
                    rsim_sass::PathKernel::from_file(&ps).map_err(|e| anyhow::anyhow!("{e}"))?;
                let mut k = KernelLaunch::from_path(p);
                k.regs_per_thread = regs;
                k
            } else {
                let src = std::fs::read_to_string(&path)?;
                let progs = rsim_sass::parse_listing(&src)?;
                let p = match &kernel {
                    Some(k) => progs
                        .into_iter()
                        .find(|p| &p.name == k)
                        .with_context(|| format!("no function {k} in {}", path.display()))?,
                    None => progs.into_iter().next().context("empty listing")?,
                };
                let name = p.name.clone();
                KernelLaunch::from_program(
                    &name,
                    p,
                    [grid, 1, 1],
                    [block, 1, 1],
                    regs,
                    smem,
                    Default::default(),
                )
            };
            let name = k.name.clone();
            let sms: Vec<u32> = if sms.is_empty() {
                (0..g.sm_count).collect()
            } else {
                sms
            };
            let mut sim = Gpu::new(
                0,
                g,
                GpuConfig {
                    kernel_window: false,
                    use_control_bits: !no_control_bits,
                    policy: if lrr {
                        rsim_gpu::SchedPolicy::Lrr
                    } else if gty {
                        rsim_gpu::SchedPolicy::Gty
                    } else {
                        rsim_gpu::SchedPolicy::Gto
                    },
                    detail: rsim_gpu::DetailConfig {
                        warp_sms: sms,
                        max_warp_events: 20_000_000,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                false,
            );
            let k = Arc::new(k);
            let mut done = vec![];
            let mut t = 0;
            for _ in 0..=warm {
                done.clear();
                sim.submit(k.clone(), t, 0);
                sim.advance_to_completion(u64::MAX / 2, &mut done);
                t = done.last().context("kernel did not finish")?.end_cycle + 1;
            }
            let d = done.pop().context("kernel did not finish")?;
            let mut detail = (*d.detail.context("no warp timeline captured")?).clone();
            // Only SMs that ran warps of this launch.
            detail.warps.retain(|t| !t.warps.is_empty());
            std::fs::write(&out, serde_json::to_string(&detail)?)?;
            eprintln!(
                "{name}: {} cycles, timeline -> {}",
                d.end_cycle - d.start_cycle,
                out.display()
            );
        }
        Cmd::Sass {
            path,
            gpu,
            grid,
            block,
            regs,
            smem,
            inspect,
        } => {
            let src = std::fs::read_to_string(&path)?;
            let progs = rsim_sass::parse_listing(&src)?;
            if inspect {
                for p in &progs {
                    println!(
                        "{}: {} instructions, control bits: {}",
                        p.name,
                        p.insts.len(),
                        p.has_control_bits()
                    );
                    for i in &p.insts {
                        let c = i
                            .ctrl
                            .map(|c| {
                                format!(
                                    "S{:02} {}W{} R{} wait{:06b}",
                                    c.stall,
                                    if c.yield_hint { "Y " } else { "  " },
                                    c.wbar.map(|x| x.to_string()).unwrap_or("-".into()),
                                    c.rbar.map(|x| x.to_string()).unwrap_or("-".into()),
                                    c.wait_mask
                                )
                            })
                            .unwrap_or_default();
                        println!("  {:04x}  {:<24} {:<16?} {}", i.pc, i.opcode, i.class, c);
                    }
                }
                return Ok(());
            }
            let g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            let p = progs.into_iter().next().unwrap();
            let name = p.name.clone();
            let k = KernelLaunch::from_program(
                &name,
                p,
                [grid, 1, 1],
                [block, 1, 1],
                regs,
                smem,
                Default::default(),
            );
            run_kernel(g, k, false)?;
        }
        Cmd::Run {
            system,
            workload,
            trace,
            report,
            launch_gap_ns,
            lrr,
            no_control_bits,
            fast,
        } => {
            let sys = SystemSpec::from_yaml(&std::fs::read_to_string(&system)?)?.build()?;
            let wsrc = std::fs::read_to_string(&workload)?;
            let wl = if workload.extension().map(|e| e == "json").unwrap_or(false) {
                Workload::from_json(&wsrc)?
            } else {
                Workload::from_yaml(&wsrc)?
            };
            let mut cfg = RunConfig {
                trace: trace.is_some(),
                launch_gap_ns,
                base_dir: workload.parent().map(|p| p.to_path_buf()),
                ..Default::default()
            };
            if lrr {
                cfg.gpu.policy = rsim_gpu::SchedPolicy::Lrr;
            }
            cfg.gpu.use_control_bits = !no_control_bits;
            cfg.memoize_kernels = fast;
            let out = Simulation::new(sys, wl, cfg).run()?;
            print!("{}", out.report.summary());
            if let Some(p) = report {
                std::fs::write(&p, serde_json::to_string_pretty(&out.report)?)?;
                eprintln!("report -> {}", p.display());
            }
            if let Some(p) = trace {
                out.trace
                    .write_json(std::io::BufWriter::new(std::fs::File::create(&p)?))?;
                eprintln!(
                    "trace  -> {} (open in https://ui.perfetto.dev)",
                    p.display()
                );
            }
        }
    }
    Ok(())
}
