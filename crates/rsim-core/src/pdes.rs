//! Conservative window-synchronous parallel discrete-event simulation.
//!
//! A rack is a set of *logical processes* (LPs): one per GPU plus the
//! interconnect fabric. LPs interact only through timestamped messages, and
//! every message is guaranteed to arrive at least `lookahead` picoseconds
//! after it was sent (the minimum physical link latency). This lets all LPs
//! advance through the window `[t, t + lookahead)` fully in parallel, then
//! exchange messages at a barrier — the classic YAWNS/bounded-lag scheme.
//!
//! With NVLink hop latencies of a few hundred ns and SM clocks near 2 GHz the
//! window is hundreds of cycles wide, so 72 cycle-level GPU models scale
//! almost linearly across host cores.

use crate::Time;
use rayon::prelude::*;

/// A message addressed to LP `dst`, to be delivered at time `at`.
#[derive(Debug, Clone)]
pub struct Envelope<M> {
    pub dst: usize,
    pub at: Time,
    pub msg: M,
}

pub trait LogicalProcess: Send {
    type Msg: Send;

    /// Advance local state through every event strictly before `until`.
    /// Outgoing messages are appended to `out`; each must satisfy
    /// `at >= until` (the lookahead contract).
    fn advance(&mut self, until: Time, out: &mut Vec<Envelope<Self::Msg>>);

    /// Receive a message (always called between windows).
    fn deliver(&mut self, at: Time, msg: Self::Msg);

    /// Earliest pending local event, or `None` if the LP is quiescent.
    fn next_event_time(&self) -> Option<Time>;

    /// True once the LP has finished its workload.
    fn done(&self) -> bool;
}

#[derive(Debug, Clone, Default)]
pub struct PdesStats {
    pub windows: u64,
    pub messages: u64,
    pub skipped_windows: u64,
}

/// Run all LPs to completion (or until `horizon`). Returns final time.
pub fn run<L: LogicalProcess>(
    lps: &mut [L],
    lookahead: Time,
    horizon: Time,
    stats: &mut PdesStats,
) -> Time {
    assert!(lookahead > 0, "lookahead must be positive");
    let mut now: Time = 0;
    let mut pending: Vec<Envelope<L::Msg>> = Vec::new();
    loop {
        // Deliver messages from the previous window.
        for env in pending.drain(..) {
            lps[env.dst].deliver(env.at, env.msg);
        }
        if lps.iter().all(|lp| lp.done()) {
            break;
        }
        // Skip idle gaps: jump to the earliest pending event.
        let next = lps.iter().filter_map(|lp| lp.next_event_time()).min();
        let Some(next) = next else { break };
        if next > now + lookahead {
            stats.skipped_windows += (next - now) / lookahead;
            now = next;
        }
        if now >= horizon {
            break;
        }
        let until = now + lookahead;
        let outs: Vec<Vec<Envelope<L::Msg>>> = lps
            .par_iter_mut()
            .map(|lp| {
                let mut out = Vec::new();
                lp.advance(until, &mut out);
                out
            })
            .collect();
        for out in outs {
            for env in out {
                debug_assert!(
                    env.at >= until,
                    "lookahead violation: message at {} inside window ending {}",
                    env.at,
                    until
                );
                stats.messages += 1;
                pending.push(env);
            }
        }
        stats.windows += 1;
        now = until;
    }
    now
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::EventQueue;

    /// Ping-pong between two LPs with fixed latency.
    struct Pinger {
        id: usize,
        q: EventQueue<u32>,
        lat: Time,
        received: Vec<(Time, u32)>,
        limit: u32,
    }
    impl LogicalProcess for Pinger {
        type Msg = u32;
        fn advance(&mut self, until: Time, out: &mut Vec<Envelope<u32>>) {
            while let Some((t, n)) = self.q.pop_before(until) {
                self.received.push((t, n));
                if n < self.limit {
                    out.push(Envelope { dst: 1 - self.id, at: t + self.lat, msg: n + 1 });
                }
            }
        }
        fn deliver(&mut self, at: Time, msg: u32) {
            self.q.push(at, msg);
        }
        fn next_event_time(&self) -> Option<Time> {
            self.q.peek_time()
        }
        fn done(&self) -> bool {
            self.q.is_empty()
        }
    }

    #[test]
    fn ping_pong_is_exact() {
        let mk = |id| Pinger { id, q: EventQueue::new(), lat: 100, received: vec![], limit: 10 };
        let mut lps = vec![mk(0), mk(1)];
        lps[0].q.push(0, 0);
        let mut st = PdesStats::default();
        run(&mut lps, 100, Time::MAX, &mut st);
        let all: Vec<_> = lps[0].received.iter().chain(lps[1].received.iter()).copied().collect();
        assert_eq!(all.len(), 11);
        for (t, n) in all {
            assert_eq!(t, n as Time * 100);
        }
    }
}
