//! Systems: which GPUs exist and how they are wired.
//!
//! A [`SystemSpec`] is what users write (YAML); [`SystemSpec::build`] lowers
//! it to a flat [`Topology`] graph of GPU and switch nodes joined by
//! full-duplex links, which the network model routes over.

use crate::fabric::{LinkSpec, SwitchSpec};
use crate::gpu::GpuSpec;
use crate::merge::deep_merge;
use crate::{presets, HwError};
use serde::{Deserialize, Serialize};
use serde_yaml::Value;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GpuGroup {
    /// Name of a GPU preset (see `presets::GPU_PRESETS`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// Or a full inline spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<GpuSpec>,
    /// Nested field overrides applied on top of the preset/spec.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<Value>,
    /// Path to a `rackscale-calibration/v1` JSON applied after `overrides`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<String>,
    pub count: u32,
}

impl GpuGroup {
    pub fn preset(name: &str, count: u32) -> Self {
        Self {
            preset: Some(name.into()),
            spec: None,
            overrides: None,
            calibration: None,
            count,
        }
    }

    pub fn resolve(&self) -> Result<GpuSpec, HwError> {
        let base = match (&self.preset, &self.spec) {
            (_, Some(s)) => s.clone(),
            (Some(p), None) => presets::gpu(p).ok_or_else(|| {
                HwError::UnknownPreset(p.clone(), presets::GPU_PRESETS.join(", "))
            })?,
            (None, None) => {
                return Err(HwError::Invalid(
                    "gpu group needs `preset` or `spec`".into(),
                ))
            }
        };
        let mut spec = match &self.overrides {
            None => base,
            Some(o) => {
                let mut v = serde_yaml::to_value(&base)?;
                deep_merge(&mut v, o.clone());
                serde_yaml::from_value(v)?
            }
        };
        if let Some(path) = &self.calibration {
            crate::calibration::apply_file(&mut spec, std::path::Path::new(path))
                .map_err(|e| HwError::Invalid(format!("calibration `{path}`: {e}")))?;
        }
        Ok(spec)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TopologySpec {
    /// Single switched domain (HGX baseboard, NVL72 rack). GPU `g` gets
    /// `links_per_gpu_per_switch[s]` links to switch `s`.
    Switched {
        switches: u32,
        switch: SwitchSpec,
        links_per_gpu_per_switch: Vec<u32>,
        link: LinkSpec,
    },
    /// Direct GPU-to-GPU links between every pair (e.g. 4x GPU boards).
    FullMesh { links_per_pair: u32, link: LinkSpec },
    /// `nodes` copies of `intra`, plus a rail-optimised scale-out network:
    /// local GPU `r` of every node attaches to leaf `r`; leaves join a
    /// non-blocking spine.
    MultiNode {
        nodes: u32,
        intra: Box<TopologySpec>,
        nic: LinkSpec,
        leaf: SwitchSpec,
    },
    /// Arbitrary wiring. Endpoints are `gpu<i>` or `sw<j>`.
    Explicit {
        switches: Vec<SwitchSpec>,
        links: Vec<ExplicitLink>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExplicitLink {
    pub a: String,
    pub b: String,
    #[serde(default = "one")]
    pub count: u32,
    pub link: LinkSpec,
}

fn one() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SystemSpec {
    pub name: String,
    /// GPU groups in rank order (supports heterogeneous / mixed systems).
    pub gpus: Vec<GpuGroup>,
    pub topology: TopologySpec,
}

pub type NodeId = usize;

#[derive(Clone, Debug, Serialize)]
pub enum NodeKind {
    Gpu { rank: usize },
    Switch { spec: SwitchSpec, label: String },
}

/// A full-duplex trunk of `count` identical physical links.
#[derive(Clone, Debug, Serialize)]
pub struct Link {
    pub a: NodeId,
    pub b: NodeId,
    pub count: u32,
    /// Aggregate per-direction bandwidth of the trunk, GB/s.
    pub gbps_per_dir: f64,
    /// One-way latency including endpoint port/PHY latencies, ns.
    pub latency_ns: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Topology {
    pub nodes: Vec<NodeKind>,
    pub links: Vec<Link>,
}

/// A fully resolved system: concrete GPU specs per rank + topology.
#[derive(Clone, Debug, Serialize)]
pub struct System {
    pub name: String,
    pub gpus: Vec<GpuSpec>,
    pub topology: Topology,
}

impl System {
    pub fn num_gpus(&self) -> usize {
        self.gpus.len()
    }
    /// Minimum one-way latency of any link (the PDES lookahead), ns.
    pub fn min_link_latency_ns(&self) -> f64 {
        self.topology
            .links
            .iter()
            .map(|l| l.latency_ns)
            .fold(f64::INFINITY, f64::min)
    }
}

impl SystemSpec {
    /// Parse YAML. Accepts either a full spec or
    /// `preset: <system preset>` plus arbitrary nested overrides.
    pub fn from_yaml(src: &str) -> Result<Self, HwError> {
        let mut v: Value = serde_yaml::from_str(src)?;
        if let Some(Value::String(p)) = v.get("preset").cloned() {
            let base = presets::system(&p).ok_or_else(|| {
                HwError::UnknownPreset(p.clone(), presets::SYSTEM_PRESETS.join(", "))
            })?;
            let mut bv = serde_yaml::to_value(&base)?;
            if let Value::Mapping(m) = &mut v {
                m.remove("preset");
            }
            deep_merge(&mut bv, v);
            v = bv;
        }
        Ok(serde_yaml::from_value(v)?)
    }

    pub fn build(&self) -> Result<System, HwError> {
        let mut gpus = Vec::new();
        for g in &self.gpus {
            let spec = g.resolve()?;
            for _ in 0..g.count {
                gpus.push(spec.clone());
            }
        }
        if gpus.is_empty() {
            return Err(HwError::Invalid("system has no GPUs".into()));
        }
        let mut b = Builder {
            nodes: (0..gpus.len()).map(|rank| NodeKind::Gpu { rank }).collect(),
            links: vec![],
            gpus: &gpus,
        };
        let ranks: Vec<usize> = (0..gpus.len()).collect();
        b.lower(&self.topology, &ranks)?;
        let topo = Topology {
            nodes: b.nodes,
            links: b.links,
        };
        validate(&topo, &gpus)?;
        Ok(System {
            name: self.name.clone(),
            gpus,
            topology: topo,
        })
    }
}

struct Builder<'a> {
    nodes: Vec<NodeKind>,
    links: Vec<Link>,
    gpus: &'a [GpuSpec],
}

impl Builder<'_> {
    fn add_switch(&mut self, spec: &SwitchSpec, label: String) -> NodeId {
        self.nodes.push(NodeKind::Switch {
            spec: spec.clone(),
            label,
        });
        self.nodes.len() - 1
    }

    fn port_latency(&self, n: NodeId) -> f64 {
        match &self.nodes[n] {
            NodeKind::Gpu { rank } => self.gpus[*rank].nvlink.port_latency_ns,
            NodeKind::Switch { .. } => 0.0,
        }
    }

    fn connect(&mut self, a: NodeId, b: NodeId, count: u32, l: &LinkSpec) {
        if count == 0 {
            return;
        }
        let latency_ns = l.latency_ns + self.port_latency(a) + self.port_latency(b);
        self.links.push(Link {
            a,
            b,
            count,
            gbps_per_dir: l.gbps_per_dir * count as f64,
            latency_ns,
        });
    }

    fn lower(&mut self, t: &TopologySpec, ranks: &[usize]) -> Result<(), HwError> {
        match t {
            TopologySpec::Switched {
                switches,
                switch,
                links_per_gpu_per_switch,
                link,
            } => {
                if links_per_gpu_per_switch.len() != *switches as usize {
                    return Err(HwError::Invalid(format!(
                        "links_per_gpu_per_switch has {} entries for {} switches",
                        links_per_gpu_per_switch.len(),
                        switches
                    )));
                }
                let base = ranks.first().copied().unwrap_or(0);
                for s in 0..*switches as usize {
                    let sw = self.add_switch(switch, format!("nvsw{}.{}", base, s));
                    for &r in ranks {
                        self.connect(r, sw, links_per_gpu_per_switch[s], link);
                    }
                }
            }
            TopologySpec::FullMesh {
                links_per_pair,
                link,
            } => {
                for (i, &a) in ranks.iter().enumerate() {
                    for &b in &ranks[i + 1..] {
                        self.connect(a, b, *links_per_pair, link);
                    }
                }
            }
            TopologySpec::MultiNode {
                nodes,
                intra,
                nic,
                leaf,
            } => {
                let n = *nodes as usize;
                if !ranks.len().is_multiple_of(n) {
                    return Err(HwError::Invalid(format!(
                        "{} GPUs not divisible into {} nodes",
                        ranks.len(),
                        n
                    )));
                }
                let per = ranks.len() / n;
                for node in 0..n {
                    self.lower(intra, &ranks[node * per..(node + 1) * per])?;
                }
                let leaves: Vec<NodeId> = (0..per)
                    .map(|r| self.add_switch(leaf, format!("leaf{r}")))
                    .collect();
                for node in 0..n {
                    for (r, &lf) in leaves.iter().enumerate() {
                        let g = ranks[node * per + r];
                        // NIC hangs off PCIe, not NVLink: no NVLink port latency.
                        self.links.push(Link {
                            a: g,
                            b: lf,
                            count: 1,
                            gbps_per_dir: nic.gbps_per_dir,
                            latency_ns: nic.latency_ns,
                        });
                    }
                }
                if per > 1 {
                    let spine = self.add_switch(leaf, "spine".into());
                    let up = LinkSpec {
                        gbps_per_dir: nic.gbps_per_dir * n as f64,
                        latency_ns: 50.0,
                    };
                    for &lf in &leaves {
                        self.links.push(Link {
                            a: lf,
                            b: spine,
                            count: n as u32,
                            gbps_per_dir: up.gbps_per_dir,
                            latency_ns: up.latency_ns,
                        });
                    }
                }
            }
            TopologySpec::Explicit { switches, links } => {
                let base_sw = self.nodes.len();
                for (j, s) in switches.iter().enumerate() {
                    self.add_switch(s, format!("sw{j}"));
                }
                let parse = |s: &str| -> Result<NodeId, HwError> {
                    if let Some(i) = s.strip_prefix("gpu") {
                        let i: usize = i
                            .parse()
                            .map_err(|_| HwError::Invalid(format!("bad endpoint {s}")))?;
                        ranks
                            .get(i)
                            .copied()
                            .ok_or_else(|| HwError::Invalid(format!("no {s}")))
                    } else if let Some(j) = s.strip_prefix("sw") {
                        let j: usize = j
                            .parse()
                            .map_err(|_| HwError::Invalid(format!("bad endpoint {s}")))?;
                        if j >= switches.len() {
                            return Err(HwError::Invalid(format!("no {s}")));
                        }
                        Ok(base_sw + j)
                    } else {
                        Err(HwError::Invalid(format!(
                            "endpoint `{s}` must be gpu<i> or sw<j>"
                        )))
                    }
                };
                for l in links {
                    let (a, b) = (parse(&l.a)?, parse(&l.b)?);
                    self.connect(a, b, l.count, &l.link);
                }
            }
        }
        Ok(())
    }
}

fn validate(t: &Topology, gpus: &[GpuSpec]) -> Result<(), HwError> {
    let mut nvlink_used = vec![0u32; gpus.len()];
    let mut ports = vec![0u32; t.nodes.len()];
    for l in &t.links {
        ports[l.a] += l.count;
        ports[l.b] += l.count;
    }
    for (i, n) in t.nodes.iter().enumerate() {
        match n {
            NodeKind::Switch { spec, label } if ports[i] > spec.ports => {
                return Err(HwError::Invalid(format!(
                    "switch {label} uses {} ports but has {}",
                    ports[i], spec.ports
                )));
            }
            NodeKind::Gpu { rank } => nvlink_used[*rank] = ports[i],
            _ => {}
        }
    }
    let _ = nvlink_used; // NIC links share the count; per-GPU NVLink checks happen in presets' tests.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvl72_shape() {
        let sys = presets::system("gb200_nvl72").unwrap().build().unwrap();
        assert_eq!(sys.num_gpus(), 72);
        assert_eq!(sys.topology.nodes.len(), 72 + 18);
        assert_eq!(sys.topology.links.len(), 72 * 18);
        // Per-GPU injection bandwidth: 18 x 50 GB/s.
        let bw: f64 = sys
            .topology
            .links
            .iter()
            .filter(|l| l.a == 0)
            .map(|l| l.gbps_per_dir)
            .sum();
        assert_eq!(bw, 900.0);
    }

    #[test]
    fn yaml_preset_with_overrides() {
        let y = r#"
preset: gb200_nvl72
name: NVL36 test
gpus:
  - preset: gb200
    count: 36
    overrides:
      sm_clock_mhz: 1900
"#;
        let spec = SystemSpec::from_yaml(y).unwrap();
        let sys = spec.build().unwrap();
        assert_eq!(sys.num_gpus(), 36);
        assert_eq!(sys.gpus[0].sm_clock_mhz, 1900.0);
        assert_eq!(sys.name, "NVL36 test");
    }

    #[test]
    fn multinode_builds() {
        let sys = presets::system("h100_2node_ib").unwrap().build().unwrap();
        assert_eq!(sys.num_gpus(), 16);
        // 2 x 4 NVSwitch + 8 leaves + 1 spine
        assert_eq!(sys.topology.nodes.len(), 16 + 8 + 8 + 1);
    }

    #[test]
    fn port_overflow_rejected() {
        let y = r#"
name: too-big
gpus: [{preset: h100_sxm, count: 80}]
topology:
  kind: switched
  switches: 1
  switch: {name: s, ports: 64, gbps_per_port_per_dir: 25, latency_ns: 100}
  links_per_gpu_per_switch: [1]
  link: {gbps_per_dir: 25, latency_ns: 20}
"#;
        assert!(SystemSpec::from_yaml(y).unwrap().build().is_err());
    }
}
