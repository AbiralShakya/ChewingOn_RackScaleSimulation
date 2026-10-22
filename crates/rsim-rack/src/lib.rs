//! `rsim-rack`: whole-system simulation.
//!
//! ```text
//!  Workload (SPMD / per-rank streams of kernels, collectives, events)
//!        │ flatten + kernel lowering (synthetic SASS / real SASS / NVBit trace)
//!        ▼
//!  RankLp[0..N]  ── Post ──▶  NetLp (fabric + collective engine)
//!   (cycle-level Gpu)  ◀── Done ──
//!        │   conservative PDES, window = min link latency, rayon-parallel
//!        ▼
//!  Report (makespan, MFU, per-kernel bubble anatomy, busbw) + Perfetto trace
//! ```

mod fold;
pub mod report;
pub mod sim;
pub mod viz;
pub mod workload;

pub use report::Report;
pub use sim::{Outcome, RunConfig, SimError, Simulation};
pub use workload::{Group, KernelSpec, Op, RankProgram, Spmd, Workload};

/// Convenience: build a system from YAML text, a workload from JSON/YAML
/// text, and run.
pub fn simulate(system_yaml: &str, workload_src: &str, cfg: RunConfig) -> Result<Outcome, String> {
    let sys = rsim_hw::SystemSpec::from_yaml(system_yaml)
        .map_err(|e| e.to_string())?
        .build()
        .map_err(|e| e.to_string())?;
    let wl = if workload_src.trim_start().starts_with('{') {
        Workload::from_json(workload_src)?
    } else {
        Workload::from_yaml(workload_src)?
    };
    Simulation::new(sys, wl, cfg)
        .run()
        .map_err(|e| e.to_string())
}
