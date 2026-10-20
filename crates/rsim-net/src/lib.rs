//! `rsim-net`: the interconnect.
//!
//! * [`fabric`] — directional channels built from the hardware topology,
//!   shortest-path ECMP routing that never transits a GPU, and a chunked,
//!   event-driven transport with FCFS per-channel serialization.
//! * [`collective`] — NCCL-style algorithms lowered to dependency DAGs of
//!   point-to-point transfers: ring all-reduce / all-gather / reduce-scatter,
//!   NVLink SHARP (NVLS) in-switch all-reduce, direct all-to-all, send/recv.
//! * [`engine`] — the network logical process: ranks post collectives, the
//!   engine schedules transfers as their dependencies and participants become
//!   ready, and reports per-rank completion times.

pub mod collective;
pub mod engine;
pub mod fabric;

pub use collective::{Algo, CollKind, Collective};
pub use engine::{NetConfig, NetEngine, NetEvent};
pub use fabric::Fabric;
