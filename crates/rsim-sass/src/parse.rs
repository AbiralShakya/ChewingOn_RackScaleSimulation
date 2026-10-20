//! Parser for `cuobjdump -sass` and `nvdisasm` listings.
//!
//! Handles:
//! * `Function : name` (cuobjdump) and `.text.name:` (nvdisasm) headers;
//! * `/*0a30*/  @!P0 BRA 0xb40 ;  /* 0x...lo */` with the high word on the
//!   following line (decoded into [`Ctrl`]);
//! * nvdisasm labels (`.L_x_12:`) and symbolic branch targets;
//! * register operands incl. vectors (`R4.64`), uniform (`UR6`), predicates,
//!   `.reuse` flags, and address operands (`desc[UR4][R2.64+0x10]`).

use crate::inst::*;
use crate::isa::{classify, first_operand_is_src, mem_info, writes_trailing_predicates, OpClass};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("line {line}: {msg}")]
    Syntax { line: usize, msg: String },
    #[error("no instructions found")]
    Empty,
}

struct Pending {
    name: String,
    insts: Vec<Inst>,
    labels: HashMap<String, u32>,
    label_targets: Vec<(usize, String)>,
    pending_labels: Vec<String>,
}

impl Pending {
    fn new(name: String) -> Self {
        Self { name, insts: vec![], labels: HashMap::new(), label_targets: vec![], pending_labels: vec![] }
    }
    fn finish(mut self) -> Program {
        for (idx, l) in self.label_targets.drain(..) {
            if let Some(&pc) = self.labels.get(&l) {
                self.insts[idx].branch_target = Some(pc);
            }
        }
        Program::new(self.name, self.insts)
    }
}

/// Parse a listing that may contain several kernels.
pub fn parse_listing(src: &str) -> Result<Vec<Program>, ParseError> {
    let mut out = Vec::new();
    let mut cur: Option<Pending> = None;
    for (ln, raw) in src.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("Function :").or_else(|| line.strip_prefix("Function:")) {
            if let Some(p) = cur.take() {
                out.push(p.finish());
            }
            cur = Some(Pending::new(rest.trim().to_string()));
            continue;
        }
        if let Some(name) = line.strip_prefix(".text.").and_then(|s| s.strip_suffix(':')) {
            if let Some(p) = cur.take() {
                out.push(p.finish());
            }
            cur = Some(Pending::new(name.to_string()));
            continue;
        }
        // Label line, e.g. `.L_x_3:` (may be followed by a comment).
        if line.starts_with(".L") {
            if let Some(l) = line.split(':').next() {
                cur.get_or_insert_with(|| Pending::new("kernel".into())).pending_labels.push(l.to_string());
            }
            continue;
        }
        // Continuation line carrying the high 64 bits.
        if line.starts_with("/* 0x") {
            if let (Some(p), Some(hi)) = (cur.as_mut(), parse_hex_comment(line)) {
                if let Some(last) = p.insts.last_mut() {
                    if last.ctrl.is_none() {
                        last.ctrl = Some(Ctrl::from_hi_word(hi));
                    }
                }
            }
            continue;
        }
        if !line.starts_with("/*") {
            continue; // directives, headers, etc.
        }
        let Some(end) = line.find("*/") else { continue };
        let Ok(pc) = u32::from_str_radix(line[2..end].trim(), 16) else { continue };
        let body = &line[end + 2..];
        let Some(semi) = body.find(';') else {
            return Err(ParseError::Syntax { line: ln + 1, msg: "missing ';'".into() });
        };
        let text = body[..semi].trim();
        let p = cur.get_or_insert_with(|| Pending::new("kernel".into()));
        for l in p.pending_labels.drain(..) {
            p.labels.insert(l, pc);
        }
        let (inst, label) = parse_inst(pc, text).map_err(|msg| ParseError::Syntax { line: ln + 1, msg })?;
        if let Some(l) = label {
            p.label_targets.push((p.insts.len(), l));
        }
        p.insts.push(inst);
        // cuobjdump: a second hex comment on the same line is the *low* word;
        // nvdisasm -hex can put both words on one line: `/* lo */ /* hi */`.
        let comments: Vec<u64> = body[semi + 1..].split("/*").filter_map(|c| parse_hex_comment(&format!("/*{c}"))).collect();
        if comments.len() >= 2 {
            p.insts.last_mut().unwrap().ctrl = Some(Ctrl::from_hi_word(comments[1]));
        }
    }
    if let Some(p) = cur.take() {
        out.push(p.finish());
    }
    out.retain(|p| !p.insts.is_empty());
    if out.is_empty() {
        return Err(ParseError::Empty);
    }
    Ok(out)
}

fn parse_hex_comment(s: &str) -> Option<u64> {
    let s = s.trim().strip_prefix("/*")?.trim();
    let s = s.split("*/").next()?.trim();
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}

/// Split operands on top-level commas (not inside `[...]` or `(...)`).
fn split_operands(s: &str) -> Vec<&str> {
    let mut out = vec![];
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, c) in s.char_indices() {
        match c {
            '[' | '(' | '{' => depth += 1,
            ']' | ')' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        out.push(last);
    }
    out
}

/// Parse one register token (without leading `!`/`-`/`|`), returning the
/// base id and the vector width suffix (`.64` => 2 regs).
pub(crate) fn parse_reg(tok: &str) -> Option<(Reg, u16)> {
    let tok = tok.trim_start_matches(['!', '-', '|', '~']).trim_end_matches('|');
    let mut parts = tok.split('.');
    let name = parts.next()?;
    let width = parts
        .map(|m| match m {
            "64" => 2,
            "128" => 4,
            _ => 1,
        })
        .max()
        .unwrap_or(1);
    let num = |p: &str| p.parse::<u16>().ok();
    let r = if let Some(n) = name.strip_prefix("UR") {
        if n == "Z" {
            return None;
        }
        reg_ur(num(n)?)
    } else if let Some(n) = name.strip_prefix("UP") {
        if n == "T" {
            return None;
        }
        reg_up(num(n)?)
    } else if let Some(n) = name.strip_prefix('R') {
        if n == "Z" {
            return None;
        }
        let n = num(n)?;
        if n >= 255 {
            return None;
        }
        reg_r(n)
    } else if let Some(n) = name.strip_prefix('P') {
        if n == "T" {
            return None;
        }
        reg_p(num(n)?)
    } else {
        return None;
    };
    Some((r, width))
}

fn push_reg(v: &mut Vec<Reg>, r: Reg, width: u16) {
    for k in 0..width {
        let x = r + k;
        if !v.contains(&x) {
            v.push(x);
        }
    }
}

/// All register tokens found inside an operand (address operands may hold several).
fn regs_in(op: &str) -> Vec<(Reg, u16)> {
    op.split(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '!'))
        .filter(|t| !t.is_empty())
        .filter_map(parse_reg)
        .collect()
}

fn is_pred_only(op: &str) -> bool {
    let t = op.trim_start_matches('!');
    t == "PT" || t == "UPT" || (t.starts_with('P') && t[1..].parse::<u16>().is_ok()) || (t.starts_with("UP") && t[2..].parse::<u16>().is_ok())
}

pub fn parse_inst(pc: u32, text: &str) -> Result<(Inst, Option<String>), String> {
    let mut text = text.trim();
    let mut guard = None;
    if let Some(rest) = text.strip_prefix('@') {
        let (g, r) = rest.split_once(char::is_whitespace).ok_or("guard without opcode")?;
        let neg = g.starts_with('!');
        if let Some((reg, _)) = parse_reg(g) {
            guard = Some((reg, neg));
        }
        text = r.trim();
    }
    let (opcode, ops) = match text.split_once(char::is_whitespace) {
        Some((o, rest)) => (o, rest.trim()),
        None => (text, ""),
    };
    if opcode.is_empty() {
        return Err("empty opcode".into());
    }
    let class = classify(opcode);
    let base = opcode.split('.').next().unwrap();
    let mem = mem_info(opcode, class);
    let operands = split_operands(ops);

    let mut dsts = vec![];
    let mut srcs = vec![];
    let mut branch_target = None;
    let mut label = None;

    let vec_w = mem.map(|m| (m.width / 4).max(1) as u16).unwrap_or(1);
    let mut first_src = 0;
    if !operands.is_empty() && !first_operand_is_src(base) {
        let op0 = operands[0];
        if !op0.contains('[') {
            for (r, w) in regs_in(op0) {
                let w = if mem.is_some() && !mem.unwrap().store { w.max(vec_w) } else { w };
                let w = if class == OpClass::MmaSync { w.max(4) } else { w };
                push_reg(&mut dsts, r, w);
            }
        }
        first_src = 1;
        if writes_trailing_predicates(base) {
            while first_src < operands.len() && first_src <= 2 && is_pred_only(operands[first_src]) {
                for (r, w) in regs_in(operands[first_src]) {
                    push_reg(&mut dsts, r, w);
                }
                first_src += 1;
            }
        }
    }
    for (i, op) in operands.iter().enumerate().skip(first_src) {
        let is_data = mem.map(|m| m.store).unwrap_or(false) && !op.contains('[');
        for (r, w) in regs_in(op) {
            let w = if is_data { w.max(vec_w) } else { w };
            let w = if class == OpClass::MmaSync && i <= 3 { w.max(if i == 2 { 2 } else { 4 }) } else { w };
            push_reg(&mut srcs, r, w);
        }
    }
    if class == OpClass::Branch && matches!(base, "BRA" | "JMP" | "CALL" | "BSSY" | "BREAK") {
        if let Some(t) = operands.last() {
            let t = t.trim().trim_start_matches('`').trim_start_matches('(').trim_end_matches(')');
            if let Some(h) = t.strip_prefix("0x") {
                branch_target = u32::from_str_radix(h, 16).ok();
            } else if t.starts_with(".L") {
                label = Some(t.to_string());
            }
        }
    }
    if let Some((g, _)) = guard {
        if !srcs.contains(&g) {
            srcs.push(g);
        }
    }
    Ok((
        Inst {
            pc,
            opcode: Arc::from(opcode),
            class,
            dsts,
            srcs,
            guard,
            ctrl: None,
            mem,
            branch_target,
            operands: ops.to_string(),
        },
        label,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUOBJDUMP: &str = r#"
	code for sm_90
		Function : _Z6vecAddPKfS0_Pfi
	.headerflags	@"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"
        /*0000*/                   LDC R1, c[0x0][0x28] ;                    /* 0x00000a00ff017b82 */
                                                                             /* 0x000fe40000000800 */
        /*0010*/                   S2R R6, SR_TID.X ;                        /* 0x0000000000067919 */
                                                                             /* 0x000e220000002100 */
        /*0020*/                   IMAD R6, R3, UR4, R6 ;                    /* 0x0000000403067c24 */
                                                                             /* 0x001fca000f8e0206 */
        /*0030*/                   ISETP.GE.AND P0, PT, R6, UR5, PT ;        /* 0x0000000506007c0c */
                                                                             /* 0x000fda000bf06270 */
        /*0040*/               @P0 EXIT ;                                    /* 0x000000000000094d */
                                                                             /* 0x000fea0003800000 */
        /*0050*/                   LDG.E.128 R8, desc[UR4][R2.64] ;          /* 0x0000000402087981 */
                                                                             /* 0x000ea8000c1e1d00 */
        /*0060*/                   STG.E.128 desc[UR4][R4.64], R8 ;          /* 0x0000000804007986 */
                                                                             /* 0x004fe2000c101d04 */
        /*0070*/                   BRA 0x70;                                 /* 0xfffffffc00fc7947 */
                                                                             /* 0x000fc0000383ffff */
"#;

    #[test]
    fn parse_cuobjdump() {
        let progs = parse_listing(CUOBJDUMP).unwrap();
        assert_eq!(progs.len(), 1);
        let p = &progs[0];
        assert_eq!(p.name, "_Z6vecAddPKfS0_Pfi");
        assert_eq!(p.insts.len(), 8);
        assert!(p.has_control_bits());
        let s2r = &p.insts[1];
        assert_eq!(s2r.class, OpClass::S2R);
        assert_eq!(s2r.ctrl.unwrap().wbar, Some(0));
        let imad = &p.insts[2];
        assert_eq!(imad.ctrl.unwrap().wait_mask & 1, 1);
        assert_eq!(imad.dsts, vec![6]);
        assert!(imad.srcs.contains(&3) && imad.srcs.contains(&6) && imad.srcs.contains(&reg_ur(4)));
        let isetp = &p.insts[3];
        assert_eq!(isetp.dsts, vec![reg_p(0)]);
        let exit = &p.insts[4];
        assert_eq!(exit.guard, Some((reg_p(0), false)));
        let ldg = &p.insts[5];
        assert_eq!(ldg.dsts, vec![8, 9, 10, 11]);
        assert_eq!(ldg.srcs, vec![reg_ur(4), 2, 3]);
        let stg = &p.insts[6];
        assert!(stg.dsts.is_empty());
        assert!(stg.srcs.contains(&11) && stg.srcs.contains(&4));
        assert_eq!(p.insts[7].branch_target, Some(0x70));
    }

    #[test]
    fn parse_nvdisasm_labels() {
        let src = r#"
.text.kern:
        /*0000*/                   MOV R0, 0x10 ;
.L_x_0:
        /*0010*/                   IADD3 R0, R0, -0x1, RZ ;
        /*0020*/                   ISETP.NE.AND P0, PT, R0, RZ, PT ;
        /*0030*/               @P0 BRA `(.L_x_0) ;
        /*0040*/                   EXIT ;
"#;
        let p = &parse_listing(src).unwrap()[0];
        assert_eq!(p.name, "kern");
        assert_eq!(p.insts[3].branch_target, Some(0x10));
        assert!(!p.has_control_bits());
    }
}
