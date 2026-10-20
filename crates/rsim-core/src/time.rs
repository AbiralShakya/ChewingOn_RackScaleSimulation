//! Simulated time.
//!
//! Global time is an integer number of **picoseconds**. Every component with a
//! clock (SM, NVLink PHY, switch core) owns a [`Clock`] that converts between
//! its local cycles and global time. Picosecond resolution keeps sub-cycle
//! ordering exact across clock domains (a 1.98 GHz SM cycle is ~505 ps).

use serde::{Deserialize, Serialize};

/// Global simulated time in picoseconds.
pub type Time = u64;

pub const PS: Time = 1;
pub const NS: Time = 1_000;
pub const US: Time = 1_000_000;
pub const MS: Time = 1_000_000_000;

/// A clock domain.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    pub mhz: f64,
}

impl Clock {
    pub fn from_mhz(mhz: f64) -> Self {
        assert!(mhz > 0.0, "clock frequency must be positive");
        Self { mhz }
    }

    /// Period of one cycle in (fractional) picoseconds.
    #[inline]
    pub fn period_ps(&self) -> f64 {
        1.0e6 / self.mhz
    }

    /// Global time at the *start* of local cycle `cycle`.
    #[inline]
    pub fn cycle_to_time(&self, cycle: u64) -> Time {
        (cycle as f64 * self.period_ps()).round() as Time
    }

    /// First local cycle that starts at or after global time `t`.
    #[inline]
    pub fn time_to_cycle_ceil(&self, t: Time) -> u64 {
        let mut c = (t as f64 / self.period_ps()).ceil() as u64;
        // Correct floating point round-off against the exact mapping.
        while c > 0 && self.cycle_to_time(c - 1) >= t {
            c -= 1;
        }
        while self.cycle_to_time(c) < t {
            c += 1;
        }
        c
    }

    /// Last local cycle that starts at or before global time `t`.
    #[inline]
    pub fn time_to_cycle_floor(&self, t: Time) -> u64 {
        let mut c = (t as f64 / self.period_ps()).floor() as u64;
        while self.cycle_to_time(c + 1) <= t {
            c += 1;
        }
        while c > 0 && self.cycle_to_time(c) > t {
            c -= 1;
        }
        c
    }
}

/// Time to serialize `bytes` on a link of `gbytes_per_s` (GB/s, 1e9 B/s).
#[inline]
pub fn serialize_time(bytes: u64, gbytes_per_s: f64) -> Time {
    // bytes / (GB/s) = ns ; *1000 = ps
    ((bytes as f64 / gbytes_per_s) * 1_000.0).ceil() as Time
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_roundtrip() {
        let c = Clock::from_mhz(1980.0);
        for cyc in [0u64, 1, 7, 1000, 123_456_789] {
            let t = c.cycle_to_time(cyc);
            assert_eq!(c.time_to_cycle_ceil(t), cyc);
            assert_eq!(c.time_to_cycle_floor(t), cyc);
        }
        assert_eq!(c.time_to_cycle_ceil(c.cycle_to_time(10) + 1), 11);
    }

    #[test]
    fn serialization() {
        // 50 GB/s => 1 byte = 20 ps
        assert_eq!(serialize_time(1, 50.0), 20);
        assert_eq!(serialize_time(1 << 20, 50.0), 20_971_520); // 1 MiB
    }
}
