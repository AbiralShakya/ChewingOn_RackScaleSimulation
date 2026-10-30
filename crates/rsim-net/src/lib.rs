//! `rsim-net`: the interconnect.
//!
//! * [`phy`] — link technologies (NVLink 4/5, InfiniBand, RoCE) and their
//!   packet framing: how many wire bytes a payload costs.
//! * [`fabric`] — directional channels built from the hardware topology,
//!   shortest-path routing that never transits a GPU.
//! * [`nccl`] — NCCL's protocols (Simple, LL, LL128), channels, chunking, and
//!   a port of NCCL 2.32.3's tuning model, used to pick the algorithm and
//!   protocol exactly as NCCL would.
//! * [`collective`] — NCCL algorithms lowered to dependency DAGs of
//!   point-to-point transfers: ring, double binary tree, NVLS (NVLink SHARP
//!   `multimem.ld_reduce` + `multimem.st`), NVLSTree, CollNet (IB SHARP),
//!   all-to-all with PXN, broadcast, send/recv.
//! * [`engine`] — the network logical process; executes plans at transfer
//!   fidelity (chunked, FCFS links) or
//! * [`packet`] — packet fidelity: credit-based flow control, switch buffers,
//!   round-robin arbitration, adaptive routing / ECMP, packet-by-packet SHARP.

pub mod collective;
pub mod engine;
pub mod fabric;
pub mod nccl;
pub mod packet;
pub mod phy;

pub use collective::{Algo, CollKind, Collective};
pub use engine::{run_one, Fidelity, LinkFault, NetConfig, NetEngine, NetEvent};
pub use fabric::Fabric;
pub use nccl::Proto;
pub use packet::{PacketConfig, Routing};
