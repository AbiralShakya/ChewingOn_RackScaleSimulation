use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SwitchSpec {
    pub name: String,
    pub ports: u32,
    pub gbps_per_port_per_dir: f64,
    /// Port-to-port cut-through latency, ns.
    pub latency_ns: f64,
    /// In-network reduction (NVLink SHARP / NVLS) support.
    #[serde(default)]
    pub in_network_reduction: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LinkSpec {
    /// Per-direction bandwidth, GB/s.
    pub gbps_per_dir: f64,
    /// Wire + SerDes latency, ns.
    pub latency_ns: f64,
}
