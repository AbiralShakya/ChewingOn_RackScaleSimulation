//! Exact rack runs at the cost of the distinct ranks: symmetric-rank folding.
//!
//! Ranks of an SPMD program are often identical: same GPU, same kernels in
//! the same order, and, until the fabric tells them apart, the same
//! collective completion times. A rank LP is deterministic in its own state
//! and the messages delivered to it, so a rank whose inputs have so far been
//! exactly those of a lower *leader* rank is in the leader's state (up to
//! its index within each collective group). Such a *follower* is not
//! simulated: its outgoing messages are the leader's, re-addressed, and its
//! report is the leader's.
//!
//! The first window in which a follower's deliveries differ from its
//! leader's, it *forks*: its untouched initial LP replays the leader's
//! logged windows (the same deliveries and the same `advance` bounds), which
//! puts it in exactly the state it would have had, and from then on it runs
//! like any other rank. Results are therefore identical to simulating every
//! rank, whatever the fabric does; only the work saved depends on how long
//! ranks stay in step.
//!
//! The window loop is [`rsim_core::pdes::run`]'s, with the fold added.

use crate::sim::{FlatOp, Lp, Msg, RankLp};
use rayon::prelude::*;
use rsim_core::pdes::{Envelope, LogicalProcess, PdesStats};
use rsim_core::Time;
use std::collections::HashMap;

/// What a rank's LP is, up to the rank's own index in each collective:
/// equal signatures start in the same state and react to equal inputs
/// equally.
pub(crate) fn signature(r: &RankLp) -> String {
    let mut s = format!(
        "{:?}|{:?}|{:?}|{}|",
        r.gpu.spec, r.gpu.cfg, r.clock, r.launch_gap
    );
    for st in r.stream_ops() {
        s.push('[');
        for op in st {
            match op {
                FlatOp::Kernel { label, k } => {
                    s += &format!("K({label:?},{:#x})", r.kernel_key(k));
                }
                FlatOp::Coll {
                    label,
                    uid,
                    desc,
                    sms,
                    ..
                } => s += &format!("C({label:?},{uid},{desc:?},{sms})"),
                o => s += &format!("{o:?}"),
            }
        }
        s.push(']');
    }
    s
}

/// The group index a rank holds in each of its collectives, by uid.
fn coll_idx(r: &RankLp) -> HashMap<u64, usize> {
    r.stream_ops()
        .flatten()
        .filter_map(|op| match op {
            FlatOp::Coll { uid, idx, .. } => Some((*uid, *idx)),
            _ => None,
        })
        .collect()
}

struct Window {
    until: Time,
    inbox: Vec<(Time, Msg)>,
}

fn same(a: &[(Time, Msg)], b: &[(Time, Msg)]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.0 == y.0
                && match (&x.1, &y.1) {
                    (Msg::Done { uid: u }, Msg::Done { uid: v }) => u == v,
                    _ => false,
                }
        })
}

/// Run all LPs to completion (or `horizon`), folding each rank onto the
/// lowest rank with the same signature (`classes[i]` = leader of LP `i`,
/// `i` itself for leaders and the fabric). Returns the final time.
pub(crate) fn run(
    lps: &mut [Lp],
    classes: &[usize],
    lookahead: Time,
    horizon: Time,
    stats: &mut PdesStats,
) -> Time {
    assert!(lookahead > 0, "lookahead must be positive");
    let n = lps.len();
    // Leader of each folded follower (None once forked, or never folded).
    let mut leader: Vec<Option<usize>> = (0..n)
        .map(|i| (classes[i] != i).then_some(classes[i]))
        .collect();
    let is_leader: Vec<bool> = (0..n).map(|i| leader.contains(&Some(i))).collect();
    let idx: Vec<HashMap<u64, usize>> = lps
        .iter()
        .map(|lp| match lp {
            Lp::Rank(r) => coll_idx(r),
            Lp::Net(_) => HashMap::new(),
        })
        .collect();
    // Every window of each leader, while it still has followers to fork.
    let mut log: Vec<Vec<Window>> = (0..n).map(|_| Vec::new()).collect();
    let mut now: Time = 0;
    let mut pending: Vec<Envelope<Msg>> = Vec::new();
    loop {
        // Deliver messages from the previous window.
        let mut inbox: Vec<Vec<(Time, Msg)>> = (0..n).map(|_| Vec::new()).collect();
        for env in pending.drain(..) {
            if leader[env.dst].is_some() || is_leader[env.dst] {
                inbox[env.dst].push((env.at, env.msg.clone()));
            }
            if leader[env.dst].is_none() {
                lps[env.dst].deliver(env.at, env.msg);
            }
        }
        // Fork the followers whose inputs left their leader's.
        let fork: Vec<usize> = (0..n)
            .filter(|&f| leader[f].is_some_and(|l| !same(&inbox[f], &inbox[l])))
            .collect();
        if !fork.is_empty() {
            let mut slots: Vec<(usize, &mut Lp)> = lps
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| fork.contains(i))
                .collect();
            slots.par_iter_mut().for_each(|(f, lp)| {
                let l = leader[*f].unwrap();
                let mut sink = Vec::new();
                for w in &log[l] {
                    for (at, m) in &w.inbox {
                        lp.deliver(*at, m.clone());
                    }
                    lp.advance(w.until, &mut sink);
                    sink.clear();
                }
                for (at, m) in &inbox[*f] {
                    lp.deliver(*at, m.clone());
                }
            });
            stats.forked += fork.len() as u64;
            for f in fork {
                leader[f] = None;
            }
        }
        let live = |i: usize, leader: &[Option<usize>]| leader[i].is_none();
        if (0..n).all(|i| lps[leader[i].unwrap_or(i)].done()) {
            break;
        }
        // Skip idle gaps: jump to the earliest pending event (a folded
        // follower's is its leader's).
        let next = (0..n)
            .filter(|&i| live(i, &leader))
            .filter_map(|i| lps[i].next_event_time())
            .min();
        let Some(next) = next else { break };
        if next > now + lookahead {
            stats.skipped_windows += (next - now) / lookahead;
            now = next;
        }
        if now >= horizon {
            break;
        }
        let until = now + lookahead;
        for l in 0..n {
            if is_leader[l] && leader.contains(&Some(l)) {
                log[l].push(Window {
                    until,
                    inbox: std::mem::take(&mut inbox[l]),
                });
            } else if is_leader[l] {
                // No follower left to fork: the log is no longer needed.
                log[l] = Vec::new();
            }
        }
        let mut outs: Vec<Vec<Envelope<Msg>>> = lps
            .par_iter_mut()
            .enumerate()
            .map(|(i, lp)| {
                let mut out = Vec::new();
                if leader[i].is_none() {
                    lp.advance(until, &mut out);
                }
                out
            })
            .collect();
        for f in 0..n {
            if let Some(l) = leader[f] {
                outs[f] = outs[l]
                    .iter()
                    .map(|e| Envelope {
                        dst: e.dst,
                        at: e.at,
                        msg: match &e.msg {
                            Msg::Post { uid, desc, .. } => Msg::Post {
                                uid: *uid,
                                desc: desc.clone(),
                                idx: idx[f][uid],
                            },
                            m => m.clone(),
                        },
                    })
                    .collect();
            }
        }
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
    // Followers still in step report what their leader did.
    for f in 0..n {
        if let Some(l) = leader[f] {
            stats.folded += 1;
            let src = match &lps[l] {
                Lp::Rank(r) => r.result(),
                Lp::Net(_) => unreachable!("the fabric is never folded"),
            };
            if let Lp::Rank(r) = &mut lps[f] {
                r.take_result(src);
            }
        }
    }
    now
}
