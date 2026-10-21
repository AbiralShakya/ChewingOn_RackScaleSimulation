//! Reader for Accel-Sim / NVBit dynamic SASS traces (`kernel-N.traceg`).
//!
//! Format (tracer v3/v4):
//! ```text
//! -kernel name = _Z...
//! -grid dim = (80,1,1)
//! -block dim = (256,1,1)
//! -shmem = 0
//! -nregs = 32
//! #BEGIN_TB
//! thread block = 0,0,0
//! warp = 0
//! insts = 22
//! 0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0
//! 0050 ffffffff 1 R8 LDG.E.128 1 R2 16 1 0x7f0000000000 16
//! #END_TB
//! ```
//! Memory address formats after `mem_width`: `0` = list of addresses for
//! every active thread, `1` = base + stride, `2` = base + per-thread deltas.

use crate::inst::{Inst, Reg};
use crate::isa::{classify, mem_info, OpClass};
use crate::parse::{parse_inst, parse_reg};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct KernelHeader {
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shmem: u32,
    pub nregs: u32,
    pub stream: u32,
}

#[derive(Debug, Clone)]
pub struct DynInst {
    pub inst: Arc<Inst>,
    pub mask: u32,
    /// Per-active-thread addresses (empty for non-memory instructions).
    pub addrs: Vec<u64>,
}

#[derive(Debug, Clone, Default)]
pub struct TraceKernel {
    pub header: KernelHeader,
    /// (cta linear id, warp id within CTA) -> dynamic instruction stream.
    pub warps: HashMap<(u32, u32), Vec<DynInst>>,
}

fn dims(s: &str) -> [u32; 3] {
    let v: Vec<u32> = s
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    [
        v.first().copied().unwrap_or(1),
        v.get(1).copied().unwrap_or(1),
        v.get(2).copied().unwrap_or(1),
    ]
}

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

pub fn parse_trace(src: &str) -> Result<TraceKernel, String> {
    let mut k = TraceKernel::default();
    let mut cta = 0u32;
    let mut warp = 0u32;
    let mut inst_cache: HashMap<(u32, String), Arc<Inst>> = HashMap::new();
    for (ln, line) in src.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('-') {
            if let Some((key, val)) = rest.split_once('=') {
                let (key, val) = (key.trim(), val.trim());
                match key {
                    "kernel name" => k.header.name = val.to_string(),
                    "grid dim" => k.header.grid = dims(val),
                    "block dim" => k.header.block = dims(val),
                    "shmem" => k.header.shmem = val.parse().unwrap_or(0),
                    "nregs" => k.header.nregs = val.parse().unwrap_or(0),
                    "cuda stream id" => k.header.stream = val.parse().unwrap_or(0),
                    _ => {}
                }
            }
            continue;
        }
        if let Some(v) = line.strip_prefix("thread block =") {
            let d = dims(v);
            let g = k.header.grid;
            cta = d[0] + d[1] * g[0] + d[2] * g[0] * g[1];
            continue;
        }
        if let Some(v) = line.strip_prefix("warp =") {
            warp = v
                .trim()
                .parse()
                .map_err(|_| format!("line {}: bad warp", ln + 1))?;
            continue;
        }
        // Tracer v6 adds `cluster id/cta/rank = ...` per thread block.
        if line.starts_with("insts =") || line.starts_with("cluster ") {
            continue;
        }
        let toks: Vec<&str> = line.split_whitespace().collect();
        let err = || format!("line {}: malformed instruction `{line}`", ln + 1);
        let mut i = 0;
        let pc = hex(toks.get(i).ok_or_else(err)?).ok_or_else(err)? as u32;
        i += 1;
        let mask = hex(toks.get(i).ok_or_else(err)?).ok_or_else(err)? as u32;
        i += 1;
        let ndst: usize = toks.get(i).ok_or_else(err)?.parse().map_err(|_| err())?;
        i += 1;
        let dsts: Vec<&str> = toks.get(i..i + ndst).ok_or_else(err)?.to_vec();
        i += ndst;
        let opcode = *toks.get(i).ok_or_else(err)?;
        i += 1;
        let nsrc: usize = toks.get(i).ok_or_else(err)?.parse().map_err(|_| err())?;
        i += 1;
        let srcs: Vec<&str> = toks.get(i..i + nsrc).ok_or_else(err)?.to_vec();
        i += nsrc;
        let width: u32 = toks.get(i).map(|t| t.parse().unwrap_or(0)).unwrap_or(0);
        i += 1;
        // Without memory operands the next field is the first immediate
        // (`DEPBAR.LE SB0, 0x3` is traced as `DEPBAR.LE 0 0 3 0`).
        let imm1: Option<u64> = (width == 0)
            .then(|| toks.get(i).and_then(|t| t.parse().ok()))
            .flatten();
        let mut addrs = vec![];
        if width > 0 {
            let active = mask.count_ones() as usize;
            let mode: u32 = toks.get(i).and_then(|t| t.parse().ok()).unwrap_or(0);
            i += 1;
            match mode {
                1 => {
                    let base = toks.get(i).and_then(|t| hex(t)).unwrap_or(0);
                    let stride: i64 = toks.get(i + 1).and_then(|t| t.parse().ok()).unwrap_or(0);
                    addrs = (0..active as i64)
                        .map(|k| (base as i64 + k * stride) as u64)
                        .collect();
                }
                2 => {
                    let mut a = toks.get(i).and_then(|t| hex(t)).unwrap_or(0) as i64;
                    addrs.push(a as u64);
                    for t in toks.iter().skip(i + 1).take(active.saturating_sub(1)) {
                        a += t.parse::<i64>().unwrap_or(0);
                        addrs.push(a as u64);
                    }
                }
                _ => {
                    addrs = toks
                        .iter()
                        .skip(i)
                        .take(active)
                        .filter_map(|t| hex(t))
                        .collect();
                }
            }
        }
        let key = (pc, opcode.to_string());
        let inst = inst_cache
            .entry(key)
            .or_insert_with(|| {
                // Re-use the listing parser for classification; registers come from the trace.
                let text = format!(
                    "{opcode} {}",
                    dsts.iter()
                        .chain(srcs.iter())
                        .copied()
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                let (mut inst, _) =
                    parse_inst(pc, &text).unwrap_or_else(|_| parse_inst(pc, opcode).unwrap());
                let class = classify(opcode);
                inst.class = class;
                inst.meta = crate::inst::InstMeta::compute(opcode, class, &inst.operands);
                if class == OpClass::DepBar && inst.meta.imm_last.is_none() {
                    inst.meta.imm_last = imm1;
                }
                inst.mem = mem_info(opcode, class).map(|mut m| {
                    if width > 0 {
                        m.width = width;
                    }
                    m
                });
                let regs = |v: &[&str]| -> Vec<Reg> {
                    v.iter()
                        .filter_map(|t| parse_reg(t))
                        .flat_map(|(r, w)| (0..w).map(move |k| r + k))
                        .collect()
                };
                inst.dsts = regs(&dsts);
                inst.srcs = regs(&srcs);
                Arc::new(inst)
            })
            .clone();
        k.warps
            .entry((cta, warp))
            .or_default()
            .push(DynInst { inst, mask, addrs });
    }
    if k.header.name.is_empty() {
        return Err("missing `-kernel name` header".into());
    }
    Ok(k)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_small_trace() {
        let t = r#"
-kernel name = _Z3addPf
-kernel id = 1
-grid dim = (2,1,1)
-block dim = (64,1,1)
-shmem = 0
-nregs = 16
#BEGIN_TB
thread block = 1,0,0
warp = 1
insts = 3
0000 ffffffff 1 R1 IMAD.MOV.U32 2 R255 R255 0
0050 ffffffff 1 R8 LDG.E 1 R2 4 1 0x7f0000001000 4
0060 0000ffff 0 STG.E 2 R4 R8 4 2 0x7f0000002000 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
#END_TB
"#;
        let k = parse_trace(t).unwrap();
        assert_eq!(k.header.grid, [2, 1, 1]);
        let w = &k.warps[&(1, 1)];
        assert_eq!(w.len(), 3);
        assert_eq!(w[1].addrs.len(), 32);
        assert_eq!(w[1].addrs[31], 0x7f0000001000 + 31 * 4);
        assert_eq!(w[2].addrs.len(), 16);
        assert_eq!(w[2].addrs[15], 0x7f0000002000 + 15 * 4);
        assert_eq!(w[1].inst.dsts, vec![8]);
        assert!(w[2].inst.srcs.contains(&4) && w[2].inst.srcs.contains(&8));
        assert!(w[0].inst.srcs.is_empty());
    }
}
