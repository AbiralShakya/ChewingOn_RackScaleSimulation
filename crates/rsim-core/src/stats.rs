//! Lightweight statistics.

use serde::Serialize;

/// Log2-bucketed histogram (cheap, mergeable, good enough for latencies).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Histogram {
    pub buckets: Vec<u64>,
    pub count: u64,
    pub sum: u128,
    pub min: u64,
    pub max: u64,
}

impl Histogram {
    pub fn record(&mut self, v: u64) {
        let b = 64 - v.leading_zeros() as usize;
        if self.buckets.len() <= b {
            self.buckets.resize(b + 1, 0);
        }
        self.buckets[b] += 1;
        if self.count == 0 || v < self.min {
            self.min = v;
        }
        self.max = self.max.max(v);
        self.count += 1;
        self.sum += v as u128;
    }
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum as f64 / self.count as f64
        }
    }
    pub fn merge(&mut self, o: &Histogram) {
        if o.count == 0 {
            return;
        }
        if self.buckets.len() < o.buckets.len() {
            self.buckets.resize(o.buckets.len(), 0);
        }
        for (a, b) in self.buckets.iter_mut().zip(&o.buckets) {
            *a += b;
        }
        self.min = if self.count == 0 {
            o.min
        } else {
            self.min.min(o.min)
        };
        self.max = self.max.max(o.max);
        self.count += o.count;
        self.sum += o.sum;
    }
}
