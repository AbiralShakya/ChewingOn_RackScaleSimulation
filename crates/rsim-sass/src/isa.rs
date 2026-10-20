//! Opcode classification.
//!
//! Classes are architecture-neutral; the GPU model maps each class to an
//! execution pipe, an initiation interval and a latency for a given arch.
//! Unknown opcodes classify as [`OpClass::Misc`] and are counted in reports
//! so coverage gaps are visible rather than silently mis-timed.

use crate::inst::{MemInfo, MemSpace};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum OpClass {
    /// Integer ALU (IADD3, LOP3, SHF, ISETP, MOV, SEL, LEA, PRMT, ...).
    IntAlu,
    /// Integer multiply-add (IMAD) — shares the FMA-heavy pipe.
    Imad,
    /// Uniform datapath (U* ops: UIADD3, UMOV, ULOP3, ...), one per warp.
    Uniform,
    Fp32,
    Fp16,
    Fp64,
    /// Multi-function unit: MUFU.EX2/RSQ/RCP/SIN/... (variable latency).
    Sfu,
    /// Conversions (F2F, F2I, I2F, F2FP).
    Conv,
    /// Warp-wide data exchange (SHFL, VOTE, MATCH, REDUX).
    Shfl,
    /// Special-register read (S2R, S2UR) — variable latency.
    S2R,
    /// Constant-cache load (LDC, ULDC).
    Ldc,
    /// Load/store/atomic through the LSU/MIO path.
    Mem,
    /// `cp.async` global→shared (LDGSTS).
    AsyncCopy,
    /// ldmatrix / stmatrix.
    Ldsm,
    /// Synchronous warp MMA (HMMA, IMMA, DMMA, BMMA).
    MmaSync,
    /// Hopper warpgroup MMA (HGMMA, IGMMA, QGMMA, BGMMA) — asynchronous.
    MmaWarpgroup,
    /// Blackwell tcgen05 MMA issued to the tensor core (UTC*MMA) — asynchronous,
    /// single-thread issue, accumulator in TMEM.
    MmaTcgen05,
    /// TMA bulk tensor copies (UTMALDG / UTMASTG / UBLKCP / UTMAPF).
    Tma,
    /// Tensor-memory load/store and copies (LDTM, STTM, UTCCP).
    Tmem,
    /// mbarrier / transaction barrier ops (SYNCS.*), tcgen05.commit (UTCBAR).
    AsyncBarrier,
    /// WARPGROUP.ARRIVE / WARPGROUP.DEPBAR (wgmma fence / wait).
    Warpgroup,
    /// CTA barrier (BAR.SYNC / BAR.ARV / BAR.RED).
    Barrier,
    /// DEPBAR / LDGDEPBAR: wait on scoreboard counters.
    DepBar,
    /// Memory fences (MEMBAR, FENCE, ERRBAR, CCTL).
    Fence,
    /// Control flow (BRA, BSSY, BSYNC, CALL, RET, WARPSYNC, ...).
    Branch,
    Exit,
    Nop,
    Misc,
}

impl OpClass {
    pub const ALL: [OpClass; 28] = [
        OpClass::IntAlu,
        OpClass::Imad,
        OpClass::Uniform,
        OpClass::Fp32,
        OpClass::Fp16,
        OpClass::Fp64,
        OpClass::Sfu,
        OpClass::Conv,
        OpClass::Shfl,
        OpClass::S2R,
        OpClass::Ldc,
        OpClass::Mem,
        OpClass::AsyncCopy,
        OpClass::Ldsm,
        OpClass::MmaSync,
        OpClass::MmaWarpgroup,
        OpClass::MmaTcgen05,
        OpClass::Tma,
        OpClass::Tmem,
        OpClass::AsyncBarrier,
        OpClass::Warpgroup,
        OpClass::Barrier,
        OpClass::DepBar,
        OpClass::Fence,
        OpClass::Branch,
        OpClass::Exit,
        OpClass::Nop,
        OpClass::Misc,
    ];

    pub fn is_variable_latency(self) -> bool {
        use OpClass::*;
        matches!(
            self,
            Sfu | Conv
                | Shfl
                | S2R
                | Ldc
                | Mem
                | AsyncCopy
                | Ldsm
                | MmaSync
                | MmaWarpgroup
                | MmaTcgen05
                | Tma
                | Tmem
                | AsyncBarrier
        )
    }
    pub fn is_tensor(self) -> bool {
        matches!(
            self,
            OpClass::MmaSync | OpClass::MmaWarpgroup | OpClass::MmaTcgen05
        )
    }
}

pub fn classify(opcode: &str) -> OpClass {
    use OpClass::*;
    let base = opcode.split('.').next().unwrap_or(opcode);
    match base {
        "IADD3" | "IADD" | "IABS" | "IMNMX" | "VIMNMX" | "VIADD" | "LOP3" | "LOP" | "SHF"
        | "SHL" | "SHR" | "LEA" | "SEL" | "ISETP" | "PSETP" | "PLOP3" | "MOV" | "MOV32I"
        | "PRMT" | "POPC" | "FLO" | "BREV" | "BMSK" | "SGXT" | "P2R" | "R2P" | "CS2R" | "IDP"
        | "IDP4A" | "ICMP" | "LEPC" | "R2UR" => IntAlu,
        "IMAD" | "IMUL" | "IMADSP" | "IMUL32I" | "IMAD32I" => Imad,
        "FADD" | "FMUL" | "FFMA" | "FSETP" | "FSET" | "FMNMX" | "FSEL" | "FCHK" | "FSWZADD"
        | "FADD32I" | "FMUL32I" | "FFMA32I" | "FCMP" | "FRND" => Fp32,
        "HADD2" | "HMUL2" | "HFMA2" | "HSETP2" | "HSET2" | "HMNMX2" | "HADD2_32I" | "HFMA2_32I"
        | "HMUL2_32I" => Fp16,
        "DADD" | "DMUL" | "DFMA" | "DSETP" | "DMNMX" => Fp64,
        "MUFU" => Sfu,
        "F2F" | "F2I" | "I2F" | "I2I" | "F2FP" | "I2FP" | "F2IP" => Conv,
        "SHFL" | "VOTE" | "VOTEU" | "MATCH" | "REDUX" | "CREDUX" => Shfl,
        "S2R" | "S2UR" => S2R,
        "LDC" | "ULDC" => Ldc,
        "LDG" | "STG" | "LDS" | "STS" | "LDL" | "STL" | "LD" | "ST" | "ATOM" | "ATOMG"
        | "ATOMS" | "RED" | "REDG" | "SUST" | "SULD" | "TEX" | "TLD" | "TLD4" | "LDSTS" => Mem,
        "LDGSTS" => AsyncCopy,
        "LDSM" | "STSM" | "MOVM" => Ldsm,
        "HMMA" | "IMMA" | "DMMA" | "BMMA" | "QMMA" | "OMMA" => MmaSync,
        "HGMMA" | "IGMMA" | "QGMMA" | "BGMMA" => MmaWarpgroup,
        "UTCHMMA" | "UTCIMMA" | "UTCQMMA" | "UTCOMMA" | "UTCMMA" | "UTCBMMA" => MmaTcgen05,
        "UTMALDG" | "UTMASTG" | "UTMAPF" | "UTMAREDG" | "UTMACCTL" | "UBLKCP" | "UBLKRED"
        | "UBLKPF" => Tma,
        "LDTM" | "STTM" | "UTCCP" | "UTCSHIFT" => Tmem,
        "SYNCS" | "UTCBAR" | "ARRIVES" => AsyncBarrier,
        "WARPGROUP" => Warpgroup,
        "BAR" | "B2R" | "R2B" => Barrier,
        "DEPBAR" | "LDGDEPBAR" => DepBar,
        "MEMBAR" | "FENCE" | "ERRBAR" | "CCTL" | "CCTLL" => Fence,
        "BRA" | "BRX" | "JMP" | "JMX" | "CALL" | "RET" | "BSSY" | "BSYNC" | "BREAK" | "BPT"
        | "WARPSYNC" | "YIELD" | "KILL" | "NANOSLEEP" | "ELECT" | "ACQBULK" | "ENDCOLLECTIVE"
        | "UCGABAR_ARV" | "UCGABAR_WAIT" => Branch,
        "EXIT" => Exit,
        "NOP" => Nop,
        b if b.starts_with('U') && b.len() > 1 => match &b[1..] {
            "IADD3" | "MOV" | "LOP3" | "SHF" | "LEA" | "SEL" | "ISETP" | "PRMT" | "FLO"
            | "POPC" | "PLOP3" | "IMAD" | "SGXT" | "BMSK" | "BREV" | "P2UR" | "PSETP" | "IABS"
            | "IMNMX" | "VIADD" | "VIMNMX" | "STG" | "LDG" => Uniform,
            _ => Misc,
        },
        _ => Misc,
    }
}

/// Memory info for load/store-like opcodes. Width from `.64/.128/.U8/...`.
pub fn mem_info(opcode: &str, class: OpClass) -> Option<MemInfo> {
    let mut parts = opcode.split('.');
    let base = parts.next()?;
    let mods: Vec<&str> = parts.collect();
    let width = mods
        .iter()
        .find_map(|m| match *m {
            "U8" | "S8" => Some(1),
            "U16" | "S16" => Some(2),
            "32" => Some(4),
            "64" => Some(8),
            "128" => Some(16),
            _ => None,
        })
        .unwrap_or(4);
    let (space, store, atomic) = match (class, base) {
        (OpClass::Mem, "LDG") => (MemSpace::Global, false, false),
        (OpClass::Mem, "STG") => (MemSpace::Global, true, false),
        (OpClass::Mem, "LDS") => (MemSpace::Shared, false, false),
        (OpClass::Mem, "STS") => (MemSpace::Shared, true, false),
        (OpClass::Mem, "LDL") => (MemSpace::Local, false, false),
        (OpClass::Mem, "STL") => (MemSpace::Local, true, false),
        (OpClass::Mem, "LD")
        | (OpClass::Mem, "TEX")
        | (OpClass::Mem, "TLD")
        | (OpClass::Mem, "TLD4")
        | (OpClass::Mem, "SULD") => (MemSpace::Generic, false, false),
        (OpClass::Mem, "ST") | (OpClass::Mem, "SUST") => (MemSpace::Generic, true, false),
        (OpClass::Mem, "ATOMS") => (MemSpace::Shared, false, true),
        (OpClass::Mem, "ATOM") | (OpClass::Mem, "ATOMG") => (MemSpace::Global, false, true),
        (OpClass::Mem, "RED") | (OpClass::Mem, "REDG") => (MemSpace::Global, true, true),
        (OpClass::AsyncCopy, _) => (MemSpace::Global, false, false),
        (OpClass::Ldsm, "STSM") => (MemSpace::Shared, true, false),
        (OpClass::Ldsm, _) => (MemSpace::Shared, false, false),
        (OpClass::Ldc, _) => (MemSpace::Const, false, false),
        (OpClass::Tmem, "STTM") => (MemSpace::Tensor, true, false),
        (OpClass::Tmem, _) => (MemSpace::Tensor, false, false),
        _ => return None,
    };
    Some(MemInfo {
        space,
        store,
        atomic,
        width,
    })
}

/// Opcodes whose first operand is *not* a destination.
pub fn first_operand_is_src(base: &str) -> bool {
    matches!(
        base,
        "STG"
            | "STS"
            | "STL"
            | "ST"
            | "RED"
            | "REDG"
            | "SUST"
            | "STSM"
            | "STTM"
            | "BAR"
            | "BRA"
            | "BRX"
            | "JMP"
            | "JMX"
            | "CALL"
            | "RET"
            | "EXIT"
            | "NOP"
            | "BSSY"
            | "BSYNC"
            | "BREAK"
            | "WARPSYNC"
            | "YIELD"
            | "KILL"
            | "DEPBAR"
            | "LDGDEPBAR"
            | "MEMBAR"
            | "FENCE"
            | "ERRBAR"
            | "CCTL"
            | "UTMASTG"
            | "UTMAPF"
            | "UTMALDG"
            | "UBLKCP"
            | "UTCHMMA"
            | "UTCIMMA"
            | "UTCQMMA"
            | "UTCOMMA"
            | "UTCMMA"
            | "UTCBAR"
            | "UTCCP"
            | "WARPGROUP"
            | "NANOSLEEP"
            | "LDGSTS"
            | "ARRIVES"
            | "UTMAREDG"
            | "UBLKRED"
    )
}

/// Opcodes that may write predicate(s) right after their first destination.
pub fn writes_trailing_predicates(base: &str) -> bool {
    matches!(
        base,
        "ISETP"
            | "FSETP"
            | "DSETP"
            | "HSETP2"
            | "PSETP"
            | "PLOP3"
            | "IADD3"
            | "LEA"
            | "UISETP"
            | "UIADD3"
            | "ULEA"
            | "UPLOP3"
            | "UPSETP"
            | "VOTE"
            | "VOTEU"
            | "R2P"
            | "ELECT"
            | "SYNCS"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classes() {
        assert_eq!(classify("FFMA.FTZ"), OpClass::Fp32);
        assert_eq!(classify("HGMMA.64x128x16.F32.BF16"), OpClass::MmaWarpgroup);
        assert_eq!(classify("UTCHMMA.2CTA"), OpClass::MmaTcgen05);
        assert_eq!(classify("UTMALDG.2D"), OpClass::Tma);
        assert_eq!(
            classify("SYNCS.PHASECHK.TRANS64.TRYWAIT"),
            OpClass::AsyncBarrier
        );
        assert_eq!(classify("UIADD3"), OpClass::Uniform);
        assert_eq!(classify("FOOBAR"), OpClass::Misc);
        let m = mem_info("LDG.E.128.CONSTANT", OpClass::Mem).unwrap();
        assert_eq!((m.space, m.width, m.store), (MemSpace::Global, 16, false));
    }
}
