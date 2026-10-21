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
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use rsim_gpu::{synth, Gpu, GpuConfig, KernelLaunch};
use rsim_hw::{presets, SystemSpec};
use rsim_rack::{RunConfig, Simulation, Workload};
use std::path::PathBuf;
use std::sync::Arc;

mod validate;

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
        /// gemm M N K | attention B HQ HKV SQ SKV D | elementwise N | rownorm ROWS COLS
        kind: String,
        dims: Vec<u64>,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long)]
        json: bool,
    },
    /// Replay measured reference kernels (ncu_reference.json) and report
    /// simulated vs measured SM cycles.
    Validate {
        reference: PathBuf,
        #[arg(long, default_value = "b300")]
        gpu: String,
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
}

fn run_kernel(gpu: rsim_hw::GpuSpec, k: KernelLaunch, json: bool) -> Result<()> {
    let ghz = gpu.sm_clock_mhz / 1000.0;
    let peak = gpu.peak_tensor_tflops("bf16");
    let hbm = gpu.hbm.bw_gbps;
    let (flops, bytes, name) = (k.flops, k.bytes, k.name.clone());
    let spec = gpu.clone();
    let launch = Arc::new(k);
    let mut g = Gpu::new(0, gpu, GpuConfig::default(), false);
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
            json,
        } => {
            let g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            let need = |n: usize| -> Result<()> {
                if dims.len() < n {
                    bail!("{kind} needs {n} dimensions");
                }
                Ok(())
            };
            let k = match kind.as_str() {
                "gemm" => {
                    need(3)?;
                    synth::gemm(&g, dims[0], dims[1], dims[2], &dtype)
                }
                "attention" => {
                    need(6)?;
                    synth::attention(
                        &g,
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
                _ => bail!("unknown kernel kind {kind}"),
            };
            run_kernel(g, k, json)?;
        }
        Cmd::Validate {
            reference,
            gpu,
            json,
        } => {
            let g = presets::gpu(&gpu).with_context(|| format!("unknown gpu preset {gpu}"))?;
            let rows = validate::run(&reference, &g)?;
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
