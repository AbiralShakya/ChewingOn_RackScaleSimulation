//! `rsim viz`: simulate with full capture and write a zoomable viewer page
//! (or an `rsviz/1` JSON bundle). See docs/VIZ.md.

use crate::{sim_kernel, synth_kernel};
use anyhow::{bail, Context, Result};
use clap::{Args, Subcommand};
use rsim_gpu::{DetailConfig, KernelLaunch};
use rsim_hw::{presets, SystemSpec};
use rsim_rack::{viz, RunConfig, Simulation, Workload};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Args)]
pub struct Out {
    /// Output file: `.html` (self-contained viewer) or `.json` (bundle).
    #[arg(short, long)]
    out: PathBuf,
    /// Run name shown in the viewer (default: derived from the input).
    #[arg(long)]
    name: Option<String>,
    /// Page title.
    #[arg(long)]
    title: Option<String>,
    /// SMs that get per-warp timelines (comma separated).
    #[arg(long, value_delimiter = ',', default_value = "0")]
    warp_sms: Vec<u32>,
    /// Cap on warp-timeline events per traced SM.
    #[arg(long, default_value_t = 400_000)]
    max_warp_events: usize,
}

impl Out {
    fn detail(&self) -> DetailConfig {
        DetailConfig {
            max_warp_events: self.max_warp_events,
            ..DetailConfig::full(self.warp_sms.clone())
        }
    }
}

#[derive(Subcommand)]
pub enum VizCmd {
    /// Run a workload on a system (like `rsim run`) and capture it.
    Run {
        #[arg(long)]
        system: PathBuf,
        #[arg(long)]
        workload: PathBuf,
        /// Fast mode (memoized kernels; see `rsim run --fast`).
        #[arg(long)]
        fast: bool,
        /// Ranks with per-SM, per-warp and per-PC capture.
        #[arg(long, value_delimiter = ',', default_value = "0")]
        detail_ranks: Vec<usize>,
        /// Link utilisation time-bin width, ns.
        #[arg(long, default_value_t = 1000.0)]
        link_bin_ns: f64,
        #[arg(long)]
        launch_gap_ns: Option<f64>,
        #[command(flatten)]
        out: Out,
    },
    /// A kernel recorded on hardware (`.rpath`, see `rsim path`).
    Path {
        path: PathBuf,
        #[arg(long, default_value = "b300")]
        gpu: String,
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[arg(long)]
        regs: Option<u32>,
        #[command(flatten)]
        out: Out,
    },
    /// A synthetic kernel (same kinds as `rsim kernel`).
    Kernel {
        #[arg(long, default_value = "b200")]
        gpu: String,
        kind: String,
        dims: Vec<u64>,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long)]
        calibration: Option<PathBuf>,
        #[command(flatten)]
        out: Out,
    },
    /// A cuobjdump/nvdisasm listing (same flags as `rsim sass`).
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
        #[command(flatten)]
        out: Out,
    },
    /// Merge bundles (`.json` or `.html` written by `rsim viz`) into one page.
    Merge {
        inputs: Vec<PathBuf>,
        #[arg(short, long)]
        out: PathBuf,
        #[arg(long)]
        title: Option<String>,
    },
}

fn stem(p: &Path) -> String {
    let s = p
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    s.split('.').next().unwrap_or(&s).to_string()
}

fn write(out: &Path, title: &str, runs: Vec<Value>) -> Result<()> {
    let b = viz::bundle(title, runs);
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let text = if out.extension().map(|e| e == "html").unwrap_or(false) {
        viz::html(&b)
    } else {
        serde_json::to_string(&b)?
    };
    std::fs::write(out, &text)?;
    eprintln!(
        "viz -> {} ({:.1} MB)",
        out.display(),
        text.len() as f64 / 1e6
    );
    Ok(())
}

/// A bundle from a `.json` bundle or from a page `rsim viz` wrote.
fn read_bundle(p: &Path) -> Result<Value> {
    let s = std::fs::read_to_string(p).with_context(|| p.display().to_string())?;
    if s.trim_start().starts_with('{') {
        return Ok(serde_json::from_str(&s)?);
    }
    viz::from_html(&s).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))
}

fn kernel_page(
    gpu: rsim_hw::GpuSpec,
    k: KernelLaunch,
    out: &Out,
    default_name: String,
    note: String,
) -> Result<()> {
    let spec = gpu.clone();
    let (d, wall) = sim_kernel(gpu, k, out.detail())?;
    let name = out.name.clone().unwrap_or(default_name);
    eprintln!(
        "{name}: {} cycles, {} warp-instr ({wall:.1} s)",
        d.end_cycle - d.start_cycle,
        d.stats.insts_issued
    );
    let run = viz::kernel_run(&name, &spec, &d, &note);
    write(&out.out, out.title.as_deref().unwrap_or(&name), vec![run])
}

fn gpu_with(name: &str, cal: &Option<PathBuf>) -> Result<(rsim_hw::GpuSpec, String)> {
    let mut g = presets::gpu(name).with_context(|| format!("unknown gpu preset {name}"))?;
    let mut note = format!("{} preset", g.name);
    if let Some(c) = cal {
        rsim_hw::calibration::apply_file(&mut g, c)
            .map_err(|e| anyhow::anyhow!("calibration {}: {e}", c.display()))?;
        note = format!("{} calibrated with {}", g.name, c.display());
    }
    Ok((g, note))
}

pub fn main(cmd: VizCmd) -> Result<()> {
    match cmd {
        VizCmd::Run {
            system,
            workload,
            fast,
            detail_ranks,
            link_bin_ns,
            launch_gap_ns,
            out,
        } => {
            let sys = SystemSpec::from_yaml(&std::fs::read_to_string(&system)?)?.build()?;
            let wsrc = std::fs::read_to_string(&workload)?;
            let wl = if workload.extension().map(|e| e == "json").unwrap_or(false) {
                Workload::from_json(&wsrc)?
            } else {
                Workload::from_yaml(&wsrc)?
            };
            let mut cfg = RunConfig {
                trace: false,
                launch_gap_ns,
                base_dir: workload.parent().map(|p| p.to_path_buf()),
                memoize_kernels: fast,
                detail_ranks,
                ..Default::default()
            };
            cfg.gpu.detail = out.detail();
            cfg.net.link_bin_ns = link_bin_ns;
            let name = out
                .name
                .clone()
                .unwrap_or_else(|| format!("{} on {}", wl.name, sys.name));
            let o = Simulation::new(sys.clone(), wl, cfg).run()?;
            eprint!("{}", o.report.summary());
            let run = viz::rack_run(&name, &sys, &o, link_bin_ns);
            write(&out.out, out.title.as_deref().unwrap_or(&name), vec![run])
        }
        VizCmd::Path {
            path,
            gpu,
            calibration,
            regs,
            out,
        } => {
            let (g, note) = gpu_with(&gpu, &calibration)?;
            let p = rsim_sass::PathKernel::from_file(&path.to_string_lossy())
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut k = KernelLaunch::from_path(p);
            if let Some(r) = regs {
                k.regs_per_thread = r;
            }
            let note = format!("recorded path {}; {note}", path.display());
            kernel_page(g, k, &out, stem(&path), note)
        }
        VizCmd::Kernel {
            gpu,
            kind,
            dims,
            dtype,
            calibration,
            out,
        } => {
            let (g, note) = gpu_with(&gpu, &calibration)?;
            let k = synth_kernel(&g, &kind, &dims, &dtype)?;
            let name = k.name.clone();
            kernel_page(g, k, &out, name, format!("synthetic kernel; {note}"))
        }
        VizCmd::Sass {
            path,
            gpu,
            grid,
            block,
            regs,
            smem,
            out,
        } => {
            let (g, note) = gpu_with(&gpu, &None)?;
            let src = std::fs::read_to_string(&path)?;
            let Some(p) = rsim_sass::parse_listing(&src)?.into_iter().next() else {
                bail!("no kernel in {}", path.display());
            };
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
            let note = format!(
                "SASS listing {} (grid {grid}, block {block}); {note}",
                path.display()
            );
            kernel_page(g, k, &out, name, note)
        }
        VizCmd::Merge { inputs, out, title } => {
            let mut runs = Vec::new();
            for p in &inputs {
                runs.extend(viz::runs_of(&read_bundle(p)?).map_err(anyhow::Error::msg)?);
            }
            let t = title.unwrap_or_else(|| "rackscale runs".into());
            write(&out, &t, runs)
        }
    }
}
