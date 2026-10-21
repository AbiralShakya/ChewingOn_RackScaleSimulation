//! `rsim-hw`: everything that describes *hardware*, nothing that simulates it.
//!
//! * [`gpu`]      — per-GPU microarchitecture parameters (SM count, pipes,
//!   caches, HBM, tensor cores, latency tables).
//! * [`fabric`]   — switches and links.
//! * [`system`]   — a full system (rack or cluster): which GPUs exist and how
//!   they are wired, built from a YAML description.
//! * [`presets`]  — calibrated starting points: H100 SXM, H200, B200,
//!   GB200 NVL72, HGX H100 8-GPU.
//!
//! Every number carries a provenance tag in the presets (`spec` = vendor
//! datasheet, `ubench` = public microbenchmark literature, `est` = estimate
//! awaiting calibration with `ubench/`). See `docs/CALIBRATION.md`.

pub mod calibration;
pub mod fabric;
pub mod gpu;
pub mod merge;
pub mod presets;
pub mod system;

pub use fabric::{LinkSpec, SwitchSpec};
pub use gpu::{
    Arch, CacheSpec, DieLink, GpuSpec, HbmSpec, LatencyTable, NvlinkSpec, OpTiming, PipeSpec,
    TensorSpec, TmaSpec,
};
pub use system::{Link, NodeId, NodeKind, System, SystemSpec, Topology, TopologySpec};

#[derive(Debug, thiserror::Error)]
pub enum HwError {
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("unknown preset `{0}` (known: {1})")]
    UnknownPreset(String, String),
    #[error("invalid system: {0}")]
    Invalid(String),
}
