//! `rsim-gpu`: a cycle-level, SASS-driven model of one GPU.
//!
//! ```text
//!  Gpu ── CTA dispatcher (occupancy, concurrent kernels, SM reservation)
//!   ├── Sm[0..N]
//!   │    ├── Smsp[0..4]  warp scheduler (GTO/LRR), 1 issue/cycle, pipes:
//!   │    │               FMA, ALU, FP64, XU(SFU/conv), uniform, tensor, branch
//!   │    ├── warps       scoreboards: control-bit SB0..5 *or* per-register
//!   │    ├── MIO/LSU     shared-memory banks, L1 (sectored), throughput
//!   │    ├── tensor eng. tcgen05 single-issue queue (Blackwell), HGMMA groups
//!   │    ├── TMA unit    bulk copies → L2/HBM, completes mbarrier phases
//!   │    └── barriers    BAR.SYNC, mbarrier phase FIFOs, cp.async groups
//!   └── MemSystem        per-die L2 (sectored) + bandwidth servers, HBM, D2D
//! ```
//!
//! **Timing method.** SM pipelines are simulated at cycle granularity but
//! event-skipped: every stalled warp knows the exact cycle its blocking
//! condition clears (all variable-latency completions are computed at issue
//! by the memory/tensor models), so idle cycles are jumped over and accounted
//! in bulk. Results are identical to ticking every cycle.
//!
//! **Stall taxonomy** mirrors Nsight Compute's warp states, so a simulated
//! "bubble" can be compared 1:1 with `smsp__pcsamp_warps_issue_stalled_*`.

pub mod gpu;
pub mod kernel;
pub mod mem;
pub mod pipes;
pub mod sm;
pub mod stats;
pub mod synth;

pub use gpu::{dvfs_stretch, Gpu, GpuConfig, KernelDone, SchedPolicy};
pub use kernel::{BarInit, BarOp, KernelLaunch, KernelSource, WarpRole};
pub use stats::{KernelStats, Stall, StallCounts};
