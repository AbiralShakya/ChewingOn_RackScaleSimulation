//! Workload description: what every GPU runs.
//!
//! A workload is a set of *rank programs*. Each program has CUDA-like
//! streams; each stream is an ordered list of ops (kernels, collectives,
//! event record/wait, delays, loops). Programs can be given per rank or as a
//! single SPMD program with communicator groups expressed relative to the
//! rank (`block:8` = my tensor-parallel group of 8, `stride:8` = my
//! data-parallel group, `world`).
//!
//! This is the narrow waist between inference/training engines and the
//! simulator: the Python front-end lowers model configs, or captured
//! torch-profiler traces from vLLM / SGLang / any PyTorch engine, into it.

use rsim_net::{Algo, CollKind};
use rsim_sass::BranchProfile;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Workload {
    pub name: String,
    /// Explicit per-rank programs...
    #[serde(default)]
    pub ranks: Vec<RankProgram>,
    /// ...or one SPMD program replicated over `world` ranks.
    #[serde(default)]
    pub spmd: Option<Spmd>,
    /// Named kernels referenced by `{"type": "ref", "name": ...}`.
    #[serde(default)]
    pub kernels: HashMap<String, KernelSpec>,
    /// Model FLOPs the workload represents (for MFU in reports).
    #[serde(default)]
    pub model_flops: f64,
    /// Device memory each rank needs, by category in bytes (`weights`,
    /// `kv_cache`, `activations`, `grads`, `optimizer`, ...), as the
    /// lowering computed it. The report checks the total against each GPU's
    /// HBM capacity. With explicit `ranks` it can differ per rank: use
    /// `memory_per_rank` instead.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub memory: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_per_rank: Vec<BTreeMap<String, f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spmd {
    pub world: usize,
    pub program: RankProgram,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RankProgram {
    pub streams: Vec<Vec<Op>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Group {
    Ranks(Vec<usize>),
    Named(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    Kernel {
        #[serde(default)]
        name: Option<String>,
        kernel: KernelSpec,
    },
    Collective {
        #[serde(default)]
        name: Option<String>,
        kind: CollKind,
        bytes: u64,
        group: Group,
        #[serde(default)]
        algo: Algo,
        /// SMs occupied by the communication kernel (NCCL channels).
        #[serde(default = "default_comm_sms")]
        sms: u32,
    },
    /// Point-to-point send to `peer` (absolute, or relative with `relative`).
    Send {
        bytes: u64,
        peer: i64,
        #[serde(default)]
        relative: bool,
        #[serde(default = "default_p2p_sms")]
        sms: u32,
    },
    Recv {
        bytes: u64,
        peer: i64,
        #[serde(default)]
        relative: bool,
        #[serde(default = "default_p2p_sms")]
        sms: u32,
    },
    Record {
        event: String,
    },
    Wait {
        event: String,
    },
    Delay {
        ns: f64,
    },
    /// The host issues the stream's next op at `at_ns` after the start of the
    /// run: it cannot start earlier. Captured programs use it to replay the
    /// measured CPU launch timeline of a host-bound engine.
    Host {
        at_ns: f64,
    },
    Loop {
        times: u32,
        body: Vec<Op>,
    },
}

fn default_comm_sms() -> u32 {
    16
}
fn default_p2p_sms() -> u32 {
    2
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KernelSpec {
    Gemm {
        m: u64,
        n: u64,
        k: u64,
        #[serde(default = "bf16")]
        dtype: String,
        /// Tiling overrides (Blackwell only; `None` keeps the lowering's
        /// own choice): output tile width, shared-memory stages, cluster
        /// size for B multicast, and split-K factor. Chapter 15.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tile_n: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        stages: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cluster: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        split_k: Option<u64>,
    },
    GroupedGemm {
        ms: Vec<u64>,
        n: u64,
        k: u64,
        #[serde(default = "bf16")]
        dtype: String,
    },
    Attention {
        batch: u64,
        heads_q: u64,
        heads_kv: u64,
        sq: u64,
        skv: u64,
        d: u64,
        #[serde(default)]
        causal: bool,
        /// One cache tensor serves as both K and V (MLA latent, GTA, GLA).
        /// Default: tied only when `d > 256`.
        #[serde(default)]
        kv_tied: Option<bool>,
        /// Decode: softmax warpgroups splitting each K/V block (FlashMLA
        /// schedule); default 1, the FA4 single-warpgroup chain.
        #[serde(default)]
        kv_warpgroups: u32,
    },
    Elementwise {
        n: u64,
        #[serde(default = "two")]
        inputs: u32,
        #[serde(default = "two_f")]
        elem_bytes: f64,
        #[serde(default = "one")]
        flops_per_elem: u32,
    },
    RowNorm {
        rows: u64,
        cols: u64,
        #[serde(default = "two_f")]
        elem_bytes: f64,
    },
    /// Embedding-bag gather with sum pooling (recommenders): `bags` outputs,
    /// each the sum of `pooling` random rows of a `table_rows` x `dim` table;
    /// `update` = backward with the optimizer fused (read-modify-write rows).
    EmbeddingBag {
        bags: u64,
        pooling: u64,
        dim: u64,
        #[serde(default = "four_f")]
        elem_bytes: f64,
        table_rows: u64,
        #[serde(default)]
        update: bool,
    },
    /// Causal depthwise conv1d + SiLU (Dao-AILab/causal-conv1d): the decode
    /// update kernel when `seqlen == 1`, the channel-last forward otherwise.
    Conv1d {
        batch: u64,
        dim: u64,
        seqlen: u64,
        #[serde(default = "four")]
        width: u64,
    },
    /// Mamba-2 decode recurrence (`selective_state_update`).
    SsmStateUpdate {
        batch: u64,
        heads: u64,
        headdim: u64,
        d_state: u64,
        #[serde(default = "one_u64")]
        groups: u64,
        #[serde(default = "two_f")]
        state_bytes: f64,
        /// What-if: headdim rows per CTA (default: the kernel's own table).
        #[serde(default)]
        block_m: Option<u64>,
    },
    /// One kernel of the Mamba-2 SSD chunked scan (`mamba_chunk_scan_combined`):
    /// `stage` is `chunk_cumsum`, `chunk_state`, `state_passing`, `bmm_chunk`
    /// or `chunk_scan`.
    Ssd {
        stage: String,
        batch: u64,
        seqlen: u64,
        heads: u64,
        headdim: u64,
        d_state: u64,
        #[serde(default = "one_u64")]
        groups: u64,
        #[serde(default = "chunk")]
        chunk: u64,
    },
    /// Real SASS from `cuobjdump -sass` / `nvdisasm`.
    Sass {
        path: String,
        #[serde(default)]
        kernel: Option<String>,
        grid: [u32; 3],
        block: [u32; 3],
        #[serde(default = "regs")]
        regs: u32,
        #[serde(default)]
        smem: u32,
        #[serde(default)]
        branch_profile: Option<BranchProfile>,
    },
    /// Accel-Sim / NVBit dynamic trace.
    Trace {
        path: String,
    },
    /// Kernel recorded with the NVBit warp_path tool (`.rpath`): exact
    /// per-warp paths with mbarrier, TMA and UMMA operands.
    Path {
        path: String,
        /// Registers per thread, when the recording does not know them.
        #[serde(default)]
        regs: Option<u32>,
    },
    /// Opaque kernel with a known duration (e.g. measured), occupying `sms`.
    Fixed {
        ns: f64,
        #[serde(default = "all_sms")]
        sms: u32,
    },
    Ref {
        name: String,
    },
}

fn bf16() -> String {
    "bf16".into()
}
fn two() -> u32 {
    2
}
fn one() -> u32 {
    1
}
fn four_f() -> f64 {
    4.0
}
fn two_f() -> f64 {
    2.0
}
fn four() -> u64 {
    4
}
fn one_u64() -> u64 {
    1
}
fn chunk() -> u64 {
    256
}
fn regs() -> u32 {
    32
}
fn all_sms() -> u32 {
    u32::MAX
}

impl Workload {
    /// Parse a JSON workload. Keys no field takes are an error (serde
    /// would drop them, so `"dtpye": "fp8"` would silently run BF16).
    pub fn from_json(s: &str) -> Result<Self, String> {
        let v: serde_yaml::Value = serde_json::from_str(s).map_err(|e| format!("workload: {e}"))?;
        Self::from_value(v)
    }
    pub fn from_yaml(s: &str) -> Result<Self, String> {
        let v: serde_yaml::Value = serde_yaml::from_str(s).map_err(|e| format!("workload: {e}"))?;
        Self::from_value(v)
    }
    fn from_value(v: serde_yaml::Value) -> Result<Self, String> {
        let wl: Self = serde_yaml::from_value(v.clone()).map_err(|e| format!("workload: {e}"))?;
        let back = serde_yaml::to_value(&wl).map_err(|e| e.to_string())?;
        let errs = rsim_hw::merge::unknown_keys(&v, &back, &["memory", "memory_per_rank"]);
        if !errs.is_empty() {
            return Err(format!("workload: {}", errs.join("; ")));
        }
        Ok(wl)
    }
    /// Memory needed by `rank`, by category (empty when not given).
    pub fn memory_of(&self, rank: usize) -> &BTreeMap<String, f64> {
        self.memory_per_rank.get(rank).unwrap_or(&self.memory)
    }
    pub fn world(&self) -> usize {
        match &self.spmd {
            Some(s) => s.world,
            None => self.ranks.len(),
        }
    }
    pub fn program(&self, rank: usize) -> &RankProgram {
        match &self.spmd {
            Some(s) => &s.program,
            None => &self.ranks[rank],
        }
    }
}

/// Resolve a communicator group for `rank`.
pub fn resolve_group(g: &Group, rank: usize, world: usize) -> Result<Vec<usize>, String> {
    match g {
        Group::Ranks(v) => Ok(v.clone()),
        Group::Named(s) => {
            let s = s.trim();
            if s == "world" {
                return Ok((0..world).collect());
            }
            let (kind, n) = s
                .split_once(':')
                .ok_or_else(|| format!("bad group `{s}`"))?;
            let n: usize = n.parse().map_err(|_| format!("bad group size in `{s}`"))?;
            if n == 0 || !world.is_multiple_of(n) {
                return Err(format!("group `{s}` does not divide world {world}"));
            }
            match kind {
                "block" => {
                    let b = rank / n * n;
                    Ok((b..b + n).collect())
                }
                "stride" => {
                    let start = rank % n;
                    Ok((start..world).step_by(n).collect())
                }
                _ => Err(format!(
                    "unknown group kind `{kind}` (block, stride, world)"
                )),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups() {
        assert_eq!(
            resolve_group(&Group::Named("block:4".into()), 6, 8).unwrap(),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            resolve_group(&Group::Named("stride:4".into()), 6, 8).unwrap(),
            vec![2, 6]
        );
        assert_eq!(
            resolve_group(&Group::Named("world".into()), 1, 3).unwrap(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn parse_json() {
        let j = r#"{
          "name": "tp-mlp",
          "spmd": {"world": 8, "program": {"streams": [[
            {"op": "kernel", "kernel": {"type": "gemm", "m": 4096, "n": 3584, "k": 8192}},
            {"op": "collective", "kind": "all_reduce", "bytes": 67108864, "group": "block:8"},
            {"op": "loop", "times": 2, "body": [{"op": "delay", "ns": 10}]}
          ]]}}
        }"#;
        let w = Workload::from_json(j).unwrap();
        assert_eq!(w.world(), 8);
        assert_eq!(w.program(3).streams[0].len(), 3);
    }
}
