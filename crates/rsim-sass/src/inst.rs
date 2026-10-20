use crate::isa::OpClass;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Register id. Layout: `R0..R254` = 0..254, `P0..P6` = 256..262,
/// `UR0..UR62` = 264..326, `UP0..UP6` = 328..334. Zero/true registers
/// (`RZ`, `PT`, `URZ`, `UPT`) are never tracked.
pub type Reg = u16;
pub const NO_REG: Reg = u16::MAX;
pub const NUM_REGS: usize = 336;

pub fn reg_r(n: u16) -> Reg {
    n
}
pub fn reg_p(n: u16) -> Reg {
    256 + n
}
pub fn reg_ur(n: u16) -> Reg {
    264 + n
}
pub fn reg_up(n: u16) -> Reg {
    328 + n
}

pub fn reg_name(r: Reg) -> String {
    match r {
        0..=255 => format!("R{r}"),
        256..=263 => format!("P{}", r - 256),
        264..=327 => format!("UR{}", r - 264),
        328..=335 => format!("UP{}", r - 328),
        _ => "?".into(),
    }
}

/// Volta+ scheduling control information (bits 105..125 of the encoding).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ctrl {
    /// Cycles to wait before issuing the next instruction of this warp.
    pub stall: u8,
    pub yield_hint: bool,
    /// Scoreboard (dependency barrier, SB0..SB5) set when the result is written.
    pub wbar: Option<u8>,
    /// Scoreboard released when source operands have been read.
    pub rbar: Option<u8>,
    /// Bitmask of SB0..SB5 this instruction waits on before issuing.
    pub wait_mask: u8,
    pub reuse: u8,
}

impl Ctrl {
    /// Decode from the high 64-bit word of a 128-bit Volta+ instruction.
    pub fn from_hi_word(hi: u64) -> Self {
        let bits = hi >> 41;
        let stall = (bits & 0xF) as u8;
        let yield_hint = (bits >> 4) & 1 == 1;
        let wbar = ((bits >> 5) & 0x7) as u8;
        let rbar = ((bits >> 8) & 0x7) as u8;
        let wait_mask = ((bits >> 11) & 0x3F) as u8;
        let reuse = ((bits >> 17) & 0xF) as u8;
        Ctrl {
            stall,
            yield_hint,
            wbar: (wbar != 7).then_some(wbar),
            rbar: (rbar != 7).then_some(rbar),
            wait_mask,
            reuse,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemSpace {
    Global,
    Shared,
    Local,
    Generic,
    Const,
    /// Tensor memory (Blackwell TMEM).
    Tensor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemInfo {
    pub space: MemSpace,
    pub store: bool,
    pub atomic: bool,
    /// Bytes per thread.
    pub width: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inst {
    pub pc: u32,
    /// Full mnemonic with modifiers, e.g. `LDG.E.128.CONSTANT`.
    pub opcode: Arc<str>,
    pub class: OpClass,
    pub dsts: Vec<Reg>,
    pub srcs: Vec<Reg>,
    /// Guard predicate `(reg, negated)`; `None` = always executes.
    pub guard: Option<(Reg, bool)>,
    pub ctrl: Option<Ctrl>,
    pub mem: Option<MemInfo>,
    pub branch_target: Option<u32>,
    /// Raw operand text (debugging / reports).
    #[serde(default)]
    pub operands: String,
    /// Pre-decoded facts for the timing model's hot path.
    #[serde(skip)]
    pub meta: InstMeta,
}

pub mod flags {
    pub const MBAR_WAIT: u32 = 1;
    pub const DEPBAR: u32 = 2;
    pub const ARV: u32 = 4;
    pub const STRONG: u32 = 8;
    pub const BYPASS: u32 = 16;
    pub const BULK_STORE: u32 = 32;
    pub const LDSM_X4: u32 = 64;
    pub const LDSM_X2: u32 = 128;
}

/// Decoded once at parse time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InstMeta {
    pub flags: u32,
    /// Last operand as an immediate (`DEPBAR.LE SB0, 0x2` => 2).
    pub imm_last: Option<u64>,
    /// `SBx` index in the first operand (DEPBAR).
    pub sb: Option<u8>,
    /// Barrier id / expected thread count for `BAR.SYNC id, threads`.
    pub bar_id: u32,
    pub bar_threads: Option<u32>,
    /// MMA shape (m, n, k) from the modifiers, and operand dtype.
    pub mma: Option<(u32, u32, u32)>,
    pub dtype: &'static str,
}

impl InstMeta {
    pub fn compute(opcode: &str, class: OpClass, operands: &str) -> Self {
        let mods: Vec<&str> = opcode.split('.').skip(1).collect();
        let has = |m: &str| mods.iter().any(|x| *x == m);
        let base = opcode.split('.').next().unwrap_or("");
        let mut f = 0;
        if class == OpClass::AsyncBarrier
            && (opcode.contains("PHASECHK") || opcode.contains("WAIT"))
        {
            f |= flags::MBAR_WAIT;
        }
        if has("DEPBAR") {
            f |= flags::DEPBAR;
        }
        if has("ARV") {
            f |= flags::ARV;
        }
        if has("STRONG") {
            f |= flags::STRONG;
        }
        if has("BYPASS") {
            f |= flags::BYPASS;
        }
        if class == OpClass::Tma && (base.contains("STG") || base.contains("RED")) {
            f |= flags::BULK_STORE;
        }
        if class == OpClass::Ldsm {
            if has("4") {
                f |= flags::LDSM_X4;
            } else if has("2") {
                f |= flags::LDSM_X2;
            }
        }
        let parts: Vec<&str> = operands.split(',').map(|x| x.trim()).collect();
        let imm = |x: &str| {
            x.strip_prefix("0x")
                .and_then(|h| u64::from_str_radix(h, 16).ok())
        };
        let imm_last = parts.last().and_then(|x| imm(x));
        let sb = parts
            .first()
            .and_then(|x| x.strip_prefix("SB"))
            .and_then(|x| x.parse().ok());
        let (bar_id, bar_threads) = if class == OpClass::Barrier {
            (
                parts.first().and_then(|x| imm(x)).unwrap_or(0) as u32,
                parts.get(1).and_then(|x| imm(x)).map(|t| t as u32),
            )
        } else {
            (0, None)
        };
        let mut mma = None;
        let mut dtype = "";
        if class.is_tensor() {
            for m in &mods {
                if m.contains('x') {
                    let v: Vec<u32> = m.split('x').filter_map(|x| x.parse().ok()).collect();
                    if v.len() == 3 {
                        mma = Some((v[0], v[1], v[2]));
                        break;
                    }
                } else if m.len() >= 3 && m.chars().all(|c| c.is_ascii_digit()) {
                    let (mm, rest) = if m.starts_with("16") {
                        (16, &m[2..])
                    } else {
                        (8, &m[1..])
                    };
                    if let (Ok(n), Ok(k)) = (rest[..1].parse::<u32>(), rest[1..].parse::<u32>()) {
                        mma = Some((mm, n, k));
                        break;
                    }
                }
            }
            dtype = match base {
                "DMMA" => "fp64",
                "IMMA" => "int8",
                _ => mods
                    .iter()
                    .find_map(|m| match *m {
                        "F16" => Some("fp16"),
                        "BF16" => Some("bf16"),
                        "TF32" => Some("tf32"),
                        "E4M3" | "E5M2" | "F8" | "F8F6F4" => Some("fp8"),
                        "E2M1" | "MXF4" | "F4" | "NVF4" => Some("fp4"),
                        "S8" | "U8" => Some("int8"),
                        _ => None,
                    })
                    .unwrap_or("bf16"),
            };
        }
        Self {
            flags: f,
            imm_last,
            sb,
            bar_id,
            bar_threads,
            mma,
            dtype,
        }
    }
}

impl Inst {
    pub fn base_opcode(&self) -> &str {
        self.opcode.split('.').next().unwrap_or("")
    }
    pub fn has_modifier(&self, m: &str) -> bool {
        self.opcode.split('.').skip(1).any(|x| x == m)
    }
}

/// A static kernel program.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Program {
    pub name: String,
    pub insts: Vec<Inst>,
    #[serde(skip)]
    pc_index: HashMap<u32, usize>,
}

impl Program {
    pub fn new(name: impl Into<String>, insts: Vec<Inst>) -> Self {
        let pc_index = insts.iter().enumerate().map(|(i, x)| (x.pc, i)).collect();
        Self {
            name: name.into(),
            insts,
            pc_index,
        }
    }
    pub fn index_of_pc(&self, pc: u32) -> Option<usize> {
        if self.pc_index.is_empty() && !self.insts.is_empty() {
            return self.insts.iter().position(|i| i.pc == pc);
        }
        self.pc_index.get(&pc).copied()
    }
    pub fn has_control_bits(&self) -> bool {
        !self.insts.is_empty() && self.insts.iter().all(|i| i.ctrl.is_some())
    }
    /// Histogram of opcode classes (static).
    pub fn class_mix(&self) -> HashMap<OpClass, usize> {
        let mut m = HashMap::new();
        for i in &self.insts {
            *m.entry(i.class).or_default() += 1;
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ctrl_decode() {
        // `0x000fe40000000800`: canonical "stall 2, yield, no barriers" word.
        let c = Ctrl::from_hi_word(0x000fe40000000800);
        assert_eq!(c.stall, 2);
        assert!(c.yield_hint);
        assert_eq!(c.wbar, None);
        assert_eq!(c.rbar, None);
        assert_eq!(c.wait_mask, 0);
        // `0x000e220000002100`: stall 1, wbar SB0, no wait.
        let c = Ctrl::from_hi_word(0x000e220000002100);
        assert_eq!(c.stall, 1);
        assert_eq!(c.wbar, Some(0));
        assert_eq!(c.rbar, None);
        // `0x001fca00078e0203`: waits on SB0 (mask 0b000001) — e.g. consumer of an S2R.
        let c = Ctrl::from_hi_word(0x001fca00078e0203);
        assert_eq!(c.wait_mask, 0b000001);
        assert_eq!(c.stall, 5);
    }
}
