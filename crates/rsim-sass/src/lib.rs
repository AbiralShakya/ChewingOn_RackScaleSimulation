//! `rsim-sass`: the instruction layer.
//!
//! * [`parse`]  — parse `cuobjdump -sass` / `nvdisasm` listings, including
//!   the 128-bit encodings, from which the compiler-scheduled **control
//!   bits** (stall count, yield, read/write dependency barriers, wait mask,
//!   operand-reuse) are decoded. These are what make SASS-level timing
//!   tractable: the compiler has already statically scheduled fixed-latency
//!   dependencies; only variable-latency ops need dynamic tracking.
//! * [`isa`]    — opcode → [`OpClass`] classification for sm_80/90/100,
//!   including Hopper `HGMMA`/`UTMALDG`/`SYNCS` and Blackwell tcgen05
//!   (`UTC*MMA`, `LDTM`/`STTM`, `UTCBAR`).
//! * [`accelsim`] — reader for Accel-Sim/NVBit dynamic per-warp traces.
//! * [`walker`] — turns a static program + branch profile into a dynamic
//!   per-warp instruction stream.

pub mod accelsim;
pub mod inst;
pub mod isa;
pub mod parse;
pub mod walker;

pub use inst::{Ctrl, Inst, MemInfo, MemSpace, Program, Reg, NO_REG};
pub use isa::OpClass;
pub use parse::{parse_listing, ParseError};
pub use walker::{AddrGen, BranchProfile, Walker};
