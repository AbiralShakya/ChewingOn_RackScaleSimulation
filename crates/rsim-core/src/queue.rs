//! Deterministic event queue.
//!
//! Events at equal timestamps pop in insertion order, so a simulation is
//! bit-for-bit reproducible regardless of heap internals or thread count.

use crate::Time;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

struct Entry<E> {
    t: Time,
    seq: u64,
    ev: E,
}

impl<E> PartialEq for Entry<E> {
    fn eq(&self, o: &Self) -> bool {
        self.t == o.t && self.seq == o.seq
    }
}
impl<E> Eq for Entry<E> {}
impl<E> PartialOrd for Entry<E> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl<E> Ord for Entry<E> {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max-heap: invert for earliest-first.
        o.t.cmp(&self.t).then_with(|| o.seq.cmp(&self.seq))
    }
}

pub struct EventQueue<E> {
    heap: BinaryHeap<Entry<E>>,
    seq: u64,
}

impl<E> Default for EventQueue<E> {
    fn default() -> Self {
        Self {
            heap: BinaryHeap::new(),
            seq: 0,
        }
    }
}

impl<E> EventQueue<E> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push(&mut self, t: Time, ev: E) {
        self.seq += 1;
        self.heap.push(Entry {
            t,
            seq: self.seq,
            ev,
        });
    }
    pub fn peek_time(&self) -> Option<Time> {
        self.heap.peek().map(|e| e.t)
    }
    pub fn pop(&mut self) -> Option<(Time, E)> {
        self.heap.pop().map(|e| (e.t, e.ev))
    }
    /// Pop the next event only if it is strictly before `until`.
    pub fn pop_before(&mut self, until: Time) -> Option<(Time, E)> {
        match self.heap.peek() {
            Some(e) if e.t < until => self.pop(),
            _ => None,
        }
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fifo_on_ties() {
        let mut q = EventQueue::new();
        q.push(5, 'b');
        q.push(1, 'a');
        q.push(5, 'c');
        q.push(5, 'd');
        let v: Vec<_> = std::iter::from_fn(|| q.pop()).collect();
        assert_eq!(v, vec![(1, 'a'), (5, 'b'), (5, 'c'), (5, 'd')]);
    }
}
