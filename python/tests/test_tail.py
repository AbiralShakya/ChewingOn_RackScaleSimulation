import math

from rackscale import tail


def test_expected_max_half_normal():
    assert abs(tail.expected_max_half_normal(1) - math.sqrt(2 / math.pi)) < 1e-3
    # max of two |Z| has mean 2/sqrt(pi)
    assert abs(tail.expected_max_half_normal(2) - 2 / math.sqrt(math.pi)) < 1e-3
    assert tail.expected_max_half_normal(72) < math.sqrt(2 * math.log(144)) + 0.5


def test_syncing_less_often_shrinks_the_straggler_tax():
    one = tail.straggler_overhead(72, 0.05)
    four = tail.straggler_overhead(72, 0.05, sync_every=4, trials=500)
    assert tail.mean_overhead(0.05) < four < one


def test_link_and_hash_closed_forms():
    assert tail.dead_plane_slowdown(18) == 18 / 17
    assert tail.degraded_plane_slowdown(18, 0.5) == 2.0
    assert 1.0 < tail.ecmp_max_load(448, 8, trials=300) < 1.3
    assert tail.ecmp_max_load(8, 8, trials=300) > 2.0
