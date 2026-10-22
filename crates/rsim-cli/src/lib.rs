//! The `rsim` command line as a library, so the Python package
//! (`rackscale._native.rsim_main`) ships the same commands as the binary.

#[allow(dead_code)]
#[path = "main.rs"]
mod app;

pub use app::run_args;
