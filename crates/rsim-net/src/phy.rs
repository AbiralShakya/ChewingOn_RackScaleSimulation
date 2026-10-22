//! Link layer: how many wire bytes a payload costs, per link technology.
//!
//! Every link in the fabric is one of a few technologies. Each technology
//! carries data in packets with a fixed header and a maximum payload, and
//! loses a further fraction of the raw signalling rate to framing: FEC,
//! CRC, flow-control credits, acks and replay. The goodput of a link is
//! therefore
//!
//! ```text
//! goodput = raw × payload / (payload + header) × (1 − framing)
//! ```
//!
//! **NVLink.** The packet format is public up to NVLink 2: 16-byte flits, a
//! request is one header flit (CRC included) followed by up to 16 data
//! flits (256 B). SM traffic leaves the L2 as 128 B writes, so a store-based
//! copy pays one header flit per 8 data flits. NVIDIA has not published the
//! NVLink 4/5 link layer; we assume the flit format carried forward and fit
//! `framing` so that the 128 B goodput matches the per-link bandwidth NCCL
//! itself assumes when it builds its rings and trees (`SM90_NVLINK_BW` =
//! 20.6 GB/s of 25, `SM100_NVLINK_BW` = 40.1 GB/s of 50; NCCL 2.32.3,
//! `src/graph/topo.h`). Those constants are NVIDIA's measured achievable
//! per-link bandwidth.
//!
//! **InfiniBand / RoCE.** RDMA writes carry up to one MTU (4096 B) per
//! packet. IB adds LRH + BTH + RETH + ICRC + VCRC (42 B); RoCEv2 adds
//! Ethernet + IP + UDP + BTH + ICRC + FCS + preamble/IFG (82 B). 64b/66b
//! and RS-FEC overhead is already excluded from the 400 Gb/s data rate.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkTech {
    Nvlink4,
    Nvlink5,
    IbNdr,
    IbXdr,
    Roce400,
    /// Unknown wiring (explicit topologies): treated as an ideal link.
    Ideal,
}

/// Link-layer framing of one technology.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Framing {
    /// Header bytes per packet (on the wire, per packet).
    pub header_bytes: u32,
    /// Largest payload per packet.
    pub max_payload: u32,
    /// Fraction of the raw rate lost to FEC/CRC/credits/acks/replay.
    pub framing: f64,
    /// Payload granule the hardware actually emits for SM-issued traffic
    /// (NVLink: 128 B L2 writes; network: one MTU).
    pub sm_payload: u32,
}

impl LinkTech {
    pub fn framing(self) -> Framing {
        match self {
            // 128/(128+16) x (1-0.073) = 0.824 = 20.6/25 (NCCL SM90_NVLINK_BW).
            LinkTech::Nvlink4 => Framing {
                header_bytes: 16,
                max_payload: 256,
                framing: 0.073,
                sm_payload: 128,
            },
            // 128/(128+16) x (1-0.098) = 0.802 = 40.1/50 (NCCL SM100_NVLINK_BW).
            LinkTech::Nvlink5 => Framing {
                header_bytes: 16,
                max_payload: 256,
                framing: 0.098,
                sm_payload: 128,
            },
            LinkTech::IbNdr | LinkTech::IbXdr => Framing {
                header_bytes: 42,
                max_payload: 4096,
                framing: 0.0,
                sm_payload: 4096,
            },
            LinkTech::Roce400 => Framing {
                header_bytes: 82,
                max_payload: 4096,
                framing: 0.0,
                sm_payload: 4096,
            },
            LinkTech::Ideal => Framing {
                header_bytes: 0,
                max_payload: 4096,
                framing: 0.0,
                sm_payload: 4096,
            },
        }
    }

    pub fn is_nvlink(self) -> bool {
        matches!(self, LinkTech::Nvlink4 | LinkTech::Nvlink5)
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "nvlink4" => LinkTech::Nvlink4,
            "nvlink5" => LinkTech::Nvlink5,
            "ib_ndr" => LinkTech::IbNdr,
            "ib_xdr" => LinkTech::IbXdr,
            "roce400" => LinkTech::Roce400,
            "ideal" => LinkTech::Ideal,
            _ => return None,
        })
    }

    pub fn nvlink(generation: u32) -> Self {
        if generation >= 5 {
            LinkTech::Nvlink5
        } else {
            LinkTech::Nvlink4
        }
    }
}

impl Framing {
    /// Wire bytes needed to carry `payload` bytes as packets of `granule`.
    pub fn wire_bytes(&self, payload: u64, granule: u32) -> f64 {
        let g = granule.clamp(1, self.max_payload) as u64;
        let pkts = payload.div_ceil(g).max(1);
        (payload + pkts * self.header_bytes as u64) as f64 / (1.0 - self.framing)
    }

    /// Goodput fraction for back-to-back packets of `granule` payload bytes.
    pub fn efficiency(&self, granule: u32) -> f64 {
        let g = granule.clamp(1, self.max_payload) as f64;
        g / (g + self.header_bytes as f64) * (1.0 - self.framing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvlink_goodput_matches_nccl_constants() {
        let e4 = LinkTech::Nvlink4.framing().efficiency(128);
        let e5 = LinkTech::Nvlink5.framing().efficiency(128);
        assert!((e4 * 25.0 - 20.6).abs() < 0.05, "{}", e4 * 25.0);
        assert!((e5 * 50.0 - 40.1).abs() < 0.05, "{}", e5 * 50.0);
        // Bigger packets amortise the header flit.
        assert!(LinkTech::Nvlink5.framing().efficiency(256) > e5);
    }

    #[test]
    fn ib_mtu_overhead() {
        let e = LinkTech::IbNdr.framing().efficiency(4096);
        assert!(e > 0.98 && e < 0.995);
        let w = LinkTech::IbNdr.framing().wire_bytes(10_000, 4096);
        assert_eq!(w, (10_000 + 3 * 42) as f64);
    }
}
