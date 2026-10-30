"""Closed forms for the tail: stragglers, link faults and congestion control.

Chapter 16 of the book. Each function is the formula a simulator run is
checked against, written so it can be evaluated by hand.

Stragglers. A bulk-synchronous step runs compute phases, each ended by a
collective that every rank must reach. If each rank's kernel takes
``t * (1 + sigma * |Z|)`` (the simulator's ``variability.kernel_sigma``), the
step takes the *sum of the maxima* over ranks, and the expected maximum of
``n`` half-normal draws grows like ``sqrt(2 ln n)``. ``straggler_overhead``
evaluates it (numerical integration for one kernel per sync, Monte Carlo
for several).

Link faults. ``degraded_plane_slowdown`` is the slowdown of a collective
striped in lockstep over several links when one of them runs at ``scale`` of
its rate; ``dead_plane_slowdown`` the same link removed from the stripe.

Hash collisions. ``ecmp_max_load`` is the expected load of the busiest of
``links`` equal-cost links when ``flows`` equal flows are hashed onto them,
which is what a static-hash fabric's slowest link carries.
"""

from __future__ import annotations

import math
import random

HALF_NORMAL_MEAN = math.sqrt(2.0 / math.pi)  # E|Z| = 0.798


def expected_max_half_normal(n: int, steps: int = 4000) -> float:
    """E[max of n i.i.d. |Z|], Z standard normal: the integral of 1 - F(x)^n."""
    if n <= 0:
        return 0.0
    hi = math.sqrt(2.0 * math.log(max(n, 2))) + 6.0
    h = hi / steps
    tot = 0.0
    for i in range(steps):
        x = (i + 0.5) * h
        tot += 1.0 - math.erf(x / math.sqrt(2.0)) ** n
    return tot * h


def straggler_overhead(n: int, sigma: float, sync_every: int = 1, trials: int = 20000,
                       seed: int = 0) -> float:
    """Expected fractional slowdown of the compute part of a bulk-synchronous
    step over a jitter-free one: E[max over n ranks of the sum of `sync_every`
    kernel factors] / sync_every - 1, each factor ``1 + sigma |Z|``.

    With ``sync_every = 1`` this is ``sigma * expected_max_half_normal(n)``.
    """
    if sync_every <= 1:
        return sigma * expected_max_half_normal(n)
    rng = random.Random(seed)
    tot = 0.0
    for _ in range(trials):
        m = 0.0
        for _r in range(n):
            s = sum(abs(rng.gauss(0.0, 1.0)) for _ in range(sync_every))
            m = max(m, s)
        tot += m
    return sigma * tot / trials / sync_every


def mean_overhead(sigma: float) -> float:
    """What a single GPU loses to the same jitter: sigma * E|Z|."""
    return sigma * HALF_NORMAL_MEAN


def degraded_plane_slowdown(planes: int, scale: float) -> float:
    """A collective striped evenly over `planes` links in lockstep, one of
    them at `scale` of its rate: the slow plane carries 1/planes of the data
    and sets the finish time, so the slowdown is max(1, 1/scale) on the
    whole collective's bandwidth term."""
    if scale <= 0:
        return math.inf
    return max(1.0, 1.0 / scale)


def dead_plane_slowdown(planes: int) -> float:
    """The same link removed from the stripe: the rest carry its share."""
    return planes / (planes - 1)


def ecmp_max_load(flows: int, links: int, groups: int = 1, trials: int = 4000, seed: int = 0) -> float:
    """E[max flows on one link] / (flows / links) when `flows` equal flows are
    hashed uniformly onto `links` links, in each of `groups` independent
    places (every leaf's uplinks and downlinks, say), and the busiest link of
    all of them sets the finish time: balls into bins."""
    rng = random.Random(seed)
    tot = 0
    for _ in range(trials):
        m = 0
        for _g in range(groups):
            c = [0] * links
            for _f in range(flows):
                c[rng.randrange(links)] += 1
            m = max(m, max(c))
        tot += m
    return tot / trials / (flows / links)
