use serde::Serialize;
use std::collections::BTreeMap;

/// Warp / scheduler states, named after Nsight Compute's stall reasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stall {
    /// Issued an instruction this cycle.
    Selected,
    /// Eligible but another warp was chosen.
    NotSelected,
    /// Fixed-latency dependency or compiler stall count.
    Wait,
    /// Waiting on an L1TEX result (global/local memory).
    LongScoreboard,
    /// Waiting on an MIO result (shared memory, SFU, shuffles, S2R, LDC).
    ShortScoreboard,
    /// Target execution pipe busy.
    MathPipeThrottle,
    /// MIO/LSU instruction queue full (shared/special ops).
    MioThrottle,
    /// LSU queue full for global/local memory.
    LgThrottle,
    /// Waiting at a CTA barrier (BAR.SYNC).
    Barrier,
    /// Waiting on an asynchronous producer: mbarrier phase (TMA, tcgen05
    /// commit), wgmma group (WARPGROUP.DEPBAR) or cp.async group (DEPBAR).
    AsyncWait,
    /// Waiting on tensor-core resources (queue full).
    TensorThrottle,
    /// Branch target resolution.
    BranchResolving,
    /// Memory fence.
    Membar,
    /// Warp finished, waiting for CTA siblings.
    Drain,
    /// Scheduler cycle with no resident warps at all (SM idle).
    Idle,
}

pub const ALL_STALLS: &[Stall] = &[
    Stall::Selected,
    Stall::NotSelected,
    Stall::Wait,
    Stall::LongScoreboard,
    Stall::ShortScoreboard,
    Stall::MathPipeThrottle,
    Stall::MioThrottle,
    Stall::LgThrottle,
    Stall::Barrier,
    Stall::AsyncWait,
    Stall::TensorThrottle,
    Stall::BranchResolving,
    Stall::Membar,
    Stall::Drain,
    Stall::Idle,
];

#[derive(Clone, Debug, Default, Serialize)]
pub struct StallCounts(pub [u64; 15]);

impl StallCounts {
    #[inline]
    pub fn add(&mut self, s: Stall, n: u64) {
        self.0[s as usize] += n;
    }
    pub fn get(&self, s: Stall) -> u64 {
        self.0[s as usize]
    }
    pub fn total(&self) -> u64 {
        self.0.iter().sum()
    }
    pub fn merge(&mut self, o: &StallCounts) {
        for (a, b) in self.0.iter_mut().zip(o.0.iter()) {
            *a += b;
        }
    }
    pub fn to_map(&self) -> BTreeMap<String, u64> {
        ALL_STALLS
            .iter()
            .filter(|s| self.get(**s) > 0)
            .map(|s| {
                (
                    serde_json::to_value(s)
                        .unwrap()
                        .as_str()
                        .unwrap()
                        .to_string(),
                    self.get(*s),
                )
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct KernelStats {
    pub name: String,
    pub tag: u64,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub ctas: u64,
    pub warps: u64,
    pub ctas_per_sm: u32,
    pub insts_issued: u64,
    /// Opcode-class histogram of issued warp instructions.
    pub class_mix: BTreeMap<String, u64>,
    /// Scheduler-cycle attribution (one sample per SMSP per cycle while the
    /// kernel owned at least one CTA on that SM): this is the *bubble* view.
    pub sched: StallCounts,
    /// Warp-cycle attribution (one sample per resident warp per cycle): the
    /// Nsight "warp state statistics" view.
    pub warp: StallCounts,
    pub tensor_busy_cycles: u64,
    pub tensor_flops: f64,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
    pub l2_hits: u64,
    pub l2_misses: u64,
    pub l1_hits: u64,
    pub l1_misses: u64,
    pub smem_wavefronts: u64,
    pub unknown_opcodes: BTreeMap<String, u64>,
}

impl KernelStats {
    pub fn cycles(&self) -> u64 {
        self.end_cycle.saturating_sub(self.start_cycle)
    }
}
