//! `rsim-core`: the simulation kernel shared by every rsim model.
//!
//! * [`time`]   — global simulated time (picoseconds) and clock domains.
//! * [`queue`]  — deterministic event queue (stable tie-breaking).
//! * [`pdes`]   — conservative, window-synchronous parallel discrete-event
//!   execution (YAWNS-style) used to run every GPU of a rack on its own core.
//! * [`trace`]  — Perfetto / Chrome-trace JSON writer for timelines.
//! * [`stats`]  — counters and histograms.

pub mod pdes;
pub mod queue;
pub mod stats;
pub mod time;
pub mod trace;

pub use time::{Clock, Time};
