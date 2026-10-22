//! Memory hierarchy timing.
//!
//! Caches are functional (tags, sectors, LRU) so hit rates come from real
//! address streams. Bandwidth is modelled with FCFS *servers*
//! (`free_at + bytes/rate`): because the GPU model processes requests in global time
//! order, a server's queueing delay is exact for its service discipline.

use rsim_hw::{CacheSpec, GpuSpec};

/// FCFS bandwidth server.
#[derive(Clone, Debug)]
pub struct Server {
    free_at: f64,
    /// Bytes per cycle.
    rate: f64,
    pub bytes: u64,
    pub busy: f64,
}

impl Server {
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0);
        Self {
            free_at: 0.0,
            rate,
            bytes: 0,
            busy: 0.0,
        }
    }
    /// Returns the cycle at which the last byte has been served.
    #[inline]
    pub fn serve(&mut self, arrive: u64, bytes: u64) -> u64 {
        let start = self.free_at.max(arrive as f64);
        let t = bytes as f64 / self.rate;
        self.free_at = start + t;
        self.bytes += bytes;
        self.busy += t;
        self.free_at.ceil() as u64
    }
    pub fn free_at(&self) -> u64 {
        self.free_at.ceil() as u64
    }
}

/// Sectored, set-associative, LRU cache (tags only).
///
/// Each set is one contiguous block of `3 * assoc` words: the tags, then the
/// LRU stamps, then `valid | dirty << 8` per way, so an access touches a few
/// adjacent cache lines of the host. The array is zero-initialised, so a
/// fresh cache costs no more than the pages its accesses touch (a 126 MB L2
/// has a million lines, and short kernels touch few of them). Within a set
/// the tags are distinct and every way's LRU stamp is distinct (one access
/// stamps one way), so lookup and victim choice do not depend on the order
/// of the ways.
#[derive(Clone, Debug)]
pub struct Cache {
    sets: Vec<u64>,
    /// Ways in use per set.
    len: Vec<u32>,
    nsets: u64,
    assoc: usize,
    line_shift: u32,
    sector_shift: u32,
    sectors_per_line: u32,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
}

pub struct Access {
    /// Sectors (of the requested mask) that missed.
    pub miss_mask: u8,
    /// A dirty line was evicted (write-back traffic, in bytes).
    pub writeback_bytes: u64,
}

const DIRTY: u64 = 1 << 8;

impl Cache {
    pub fn new(spec: &CacheSpec, size_bytes: u64) -> Self {
        let lines = (size_bytes / spec.line_bytes as u64).max(spec.assoc as u64);
        let nsets = (lines / spec.assoc as u64).max(1) as usize;
        let assoc = spec.assoc as usize;
        Self {
            sets: vec![0; nsets * assoc * 3],
            len: vec![0; nsets],
            nsets: nsets as u64,
            assoc,
            line_shift: spec.line_bytes.trailing_zeros(),
            sector_shift: spec.sector_bytes.trailing_zeros(),
            sectors_per_line: spec.line_bytes / spec.sector_bytes,
            clock: 0,
            hits: 0,
            misses: 0,
        }
    }

    pub fn line_shift(&self) -> u32 {
        self.line_shift
    }

    pub fn sector_mask_of(&self, addr: u64, bytes: u32) -> u8 {
        let line = addr >> self.line_shift << self.line_shift;
        let first = ((addr - line) >> self.sector_shift) as u32;
        let last = (((addr + bytes.max(1) as u64 - 1).min(line + (1 << self.line_shift) - 1)
            - line)
            >> self.sector_shift) as u32;
        let mut m = 0u8;
        for s in first..=last.min(self.sectors_per_line - 1) {
            m |= 1 << s;
        }
        m
    }

    /// Look up (and fill) `mask` sectors of the line containing `addr`.
    pub fn access(&mut self, addr: u64, mask: u8, write: bool) -> Access {
        self.clock += 1;
        let line = addr >> self.line_shift;
        // XOR-fold hash to avoid pathological power-of-two strides.
        let idx = ((line ^ (line >> 7) ^ (line >> 13)) % self.nsets) as usize;
        let tag = line;
        let a = self.assoc;
        let used = self.len[idx] as usize;
        let clock = self.clock;
        let set = &mut self.sets[idx * a * 3..(idx + 1) * a * 3];
        let (tags, rest) = set.split_at_mut(a);
        let (lru, meta) = rest.split_at_mut(a);
        if let Some(w) = tags[..used].iter().position(|&t| t == tag) {
            let valid = meta[w] as u8;
            let miss = mask & !valid;
            meta[w] |= mask as u64 | if write { DIRTY } else { 0 };
            lru[w] = clock;
            let n_miss = miss.count_ones() as u64;
            self.misses += n_miss;
            self.hits += mask.count_ones() as u64 - n_miss;
            return Access {
                miss_mask: miss,
                writeback_bytes: 0,
            };
        }
        self.misses += mask.count_ones() as u64;
        let mut wb = 0;
        let w = if used >= a {
            let mut v = 0;
            for i in 1..used {
                if lru[i] < lru[v] {
                    v = i;
                }
            }
            if meta[v] & DIRTY != 0 {
                wb = ((meta[v] as u8).count_ones() as u64) << self.sector_shift;
            }
            v
        } else {
            self.len[idx] += 1;
            used
        };
        tags[w] = tag;
        meta[w] = mask as u64 | if write { DIRTY } else { 0 };
        lru[w] = clock;
        Access {
            miss_mask: mask,
            writeback_bytes: wb,
        }
    }

    pub fn invalidate_all(&mut self) {
        self.len.fill(0);
    }
}

/// Per-request result from the L2/HBM model.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemResult {
    pub done: u64,
    pub l2_hit_sectors: u32,
    pub l2_miss_sectors: u32,
    pub dram_bytes: u64,
}

/// Cycles per HBM utilization window (loaded latency).
const HBM_LOAD_WINDOW: u64 = 4096;

/// Per-die L2 + HBM + die-to-die link.
pub struct MemSystem {
    dies: u32,
    l2: Vec<Cache>,
    l2_bw: Vec<Server>,
    hbm: Vec<Server>,
    d2d: Vec<Server>,
    l2_lat: u64,
    hbm_lat: u64,
    /// Per die: (window index, bytes read in it, utilization of the last
    /// full window), for the load-dependent part of the miss latency.
    hbm_load: Vec<(u64, f64, f64)>,
    hbm_window_bytes: f64,
    /// `[utilization, cycles]`, from `HbmSpec::loaded_latency`.
    loaded: rsim_hw::HbmSpec,
    clock_ghz: f64,
    d2d_lat: u64,
    far_fraction: Option<f64>,
    sector: u32,
    sm_count: u32,
    pub dram_read_bytes: u64,
    pub dram_write_bytes: u64,
}

impl MemSystem {
    pub fn new(g: &GpuSpec) -> Self {
        let dies = g.dies.max(1);
        let clock_ghz = g.sm_clock_mhz / 1000.0;
        let hbm_rate = g.hbm.bw_gbps * g.hbm.efficiency / clock_ghz / dies as f64; // bytes/cycle/die
        let l2_rate = g.l2_bytes_per_cycle / dies as f64;
        let d2d_rate = g
            .d2d
            .as_ref()
            .map(|d| d.gbps_per_dir / clock_ghz)
            .unwrap_or(1e12);
        Self {
            dies,
            l2: (0..dies)
                .map(|_| Cache::new(&g.l2, g.l2.size_bytes / dies as u64))
                .collect(),
            l2_bw: (0..dies).map(|_| Server::new(l2_rate)).collect(),
            hbm: (0..dies).map(|_| Server::new(hbm_rate)).collect(),
            d2d: (0..dies).map(|_| Server::new(d2d_rate)).collect(),
            l2_lat: g.l2.hit_latency as u64,
            hbm_lat: (g.hbm.latency_ns * clock_ghz).round() as u64,
            hbm_load: vec![(0, 0.0, 0.0); dies as usize],
            hbm_window_bytes: hbm_rate * HBM_LOAD_WINDOW as f64,
            loaded: g.hbm.clone(),
            clock_ghz,
            d2d_lat: g.d2d.as_ref().map(|d| d.extra_latency as u64).unwrap_or(0),
            far_fraction: g.d2d.as_ref().and_then(|d| d.far_fraction),
            sector: g.l2.sector_bytes,
            sm_count: g.sm_count,
            dram_read_bytes: 0,
            dram_write_bytes: 0,
        }
    }

    #[inline]
    fn home_die(&self, addr: u64) -> usize {
        // 4 KiB interleave across dies (est; real hashing is undisclosed).
        ((addr >> 12) % self.dies as u64) as usize
    }
    #[inline]
    pub fn sm_die(&self, sm: u32) -> usize {
        (sm as u64 * self.dies as u64 / self.sm_count as u64) as usize
    }

    /// Load-dependent extra miss latency: utilization of the die's HBM over
    /// the last full window, mapped through the measured curve.
    fn loaded_cycles(&mut self, die: usize, now: u64, bytes: u64) -> u64 {
        if self.loaded.loaded_latency.is_empty() {
            return 0;
        }
        let w = now / HBM_LOAD_WINDOW;
        let l = &mut self.hbm_load[die];
        if w > l.0 {
            l.2 = if w == l.0 + 1 {
                l.1 / self.hbm_window_bytes
            } else {
                0.0
            };
            *l = (w, 0.0, l.2);
        }
        l.1 += bytes as f64;
        let util = l.2;
        (self.loaded.loaded_extra_ns(util) * self.clock_ghz).round() as u64
    }

    /// Access one line's sectors. Returns completion cycle.
    pub fn access_line(
        &mut self,
        sm: u32,
        now: u64,
        addr: u64,
        mask: u8,
        write: bool,
    ) -> MemResult {
        let die = self.home_die(addr);
        let bytes = mask.count_ones() as u64 * self.sector as u64;
        let mut t = now;
        let mut extra = 0;
        let far = match self.far_fraction {
            Some(f) => {
                let h = (addr >> 7) ^ ((self.sm_die(sm) as u64) << 40);
                let h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11;
                (h as f64 / (1u64 << 53) as f64) < f
            }
            None => die != self.sm_die(sm),
        };
        if far {
            t = self.d2d[die].serve(t, bytes);
            extra = self.d2d_lat;
        }
        let a = self.l2[die].access(addr, mask, write);
        if a.writeback_bytes > 0 {
            self.hbm[die].serve(now, a.writeback_bytes);
            self.dram_write_bytes += a.writeback_bytes;
        }
        let served = self.l2_bw[die].serve(t, bytes);
        let miss_sectors = a.miss_mask.count_ones();
        let mut done = served + self.l2_lat + extra;
        let mut dram = 0;
        if miss_sectors > 0 && !write {
            dram = miss_sectors as u64 * self.sector as u64;
            let h = self.hbm[die].serve(served, dram);
            let q = self.loaded_cycles(die, served, dram);
            done = done.max(h + self.hbm_lat + q + self.l2_lat + extra);
            self.dram_read_bytes += dram;
        }
        MemResult {
            done,
            l2_hit_sectors: mask.count_ones() - miss_sectors,
            l2_miss_sectors: miss_sectors,
            dram_bytes: dram,
        }
    }

    /// Bulk contiguous transfer (TMA). `l2_bytes` may be smaller than
    /// `bytes` when multicast shares the fetch across a cluster.
    pub fn bulk(
        &mut self,
        sm: u32,
        now: u64,
        addr: u64,
        bytes: u64,
        l2_fraction: f64,
        write: bool,
    ) -> MemResult {
        let line = 128u64;
        let start = addr & !(line - 1);
        let n_lines = (addr + bytes).div_ceil(line) - start / line;
        let n_fetch = ((n_lines as f64) * l2_fraction).ceil().max(1.0) as u64;
        let mut r = MemResult {
            done: now,
            ..Default::default()
        };
        for i in 0..n_fetch {
            let x = self.access_line(sm, now, start + i * line, 0xF, write);
            r.done = r.done.max(x.done);
            r.l2_hit_sectors += x.l2_hit_sectors;
            r.l2_miss_sectors += x.l2_miss_sectors;
            r.dram_bytes += x.dram_bytes;
        }
        r
    }

    pub fn l2_hits(&self) -> u64 {
        self.l2.iter().map(|c| c.hits).sum()
    }
    pub fn l2_misses(&self) -> u64 {
        self.l2.iter().map(|c| c.misses).sum()
    }
    pub fn hbm_busy_cycles(&self) -> f64 {
        self.hbm.iter().map(|s| s.busy).fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsim_hw::presets;

    #[test]
    fn server_queues() {
        let mut s = Server::new(32.0);
        assert_eq!(s.serve(0, 64), 2);
        assert_eq!(s.serve(0, 64), 4);
        assert_eq!(s.serve(10, 32), 11);
    }

    #[test]
    fn cache_hits_after_fill() {
        let g = presets::h100_sxm();
        let mut c = Cache::new(&g.l1, g.l1.size_bytes);
        let m = c.sector_mask_of(0x1000, 128);
        assert_eq!(m, 0xF);
        assert_eq!(c.access(0x1000, m, false).miss_mask, 0xF);
        assert_eq!(c.access(0x1000, m, false).miss_mask, 0);
        assert_eq!(c.sector_mask_of(0x1020, 4), 0b0010);
    }

    #[test]
    fn hbm_streaming_rate() {
        let g = presets::h100_sxm();
        let mut m = MemSystem::new(&g);
        // Stream 64 MiB of distinct lines at t=0: completion ≈ bytes / HBM rate.
        let bytes: u64 = 64 << 20;
        let mut done = 0;
        for i in 0..(bytes / 128) {
            done = done.max(m.access_line(0, 0, (1 << 40) + i * 128, 0xF, false).done);
        }
        let ns = done as f64 / (g.sm_clock_mhz / 1000.0);
        let gbps = bytes as f64 / ns;
        let want = g.hbm.bw_gbps * g.hbm.efficiency;
        assert!(
            (gbps - want).abs() / want < 0.05,
            "achieved {gbps} GB/s, want {want}"
        );
    }
}
