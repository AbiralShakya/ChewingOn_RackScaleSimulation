"""Register patterns of rfport.py (shared with harness.py, which has no GPU)."""


def patterns(nsrc):
    P = {}
    # 3 sources, all even registers (same bank mod 2, 4 and 8), rotating.
    P["same_bank"] = lambda k: (44 + k % 16, [20 + 8 * (k % 3), 21 + 8 * (k % 3) + 7, 22 + 8 * (k % 3) + 14][:nsrc], 0)
    P["even_even_even"] = lambda k: (44 + k % 16, [20 + 2 * (k % 4), 28 + 2 * (k % 4), 36 + 2 * (k % 4)][:nsrc], 0)
    P["even_odd_even"] = lambda k: (44 + k % 16, [20 + 2 * (k % 4), 29 + 2 * (k % 4), 36 + 2 * (k % 4)][:nsrc], 0)
    P["odd_even_odd"] = lambda k: (44 + k % 16, [21 + 2 * (k % 4), 28 + 2 * (k % 4), 37 + 2 * (k % 4)][:nsrc], 0)
    P["mod4_0_1_2"] = lambda k: (44 + k % 16, [20 + 4 * (k % 3), 33 + 4 * (k % 2), 42 - 4 * (k % 2)][:nsrc], 0)
    P["mod4_0_2_0"] = lambda k: (44 + k % 16, [20 + 4 * (k % 3), 34 + 4 * (k % 2), 32 - 4 * (k % 2)][:nsrc], 0)
    P["alternating_parity"] = lambda k: (44 + k % 16, [20 + k % 8, 28 + k % 8, 36 + k % 8][:nsrc], 0)
    # One register read by every instruction in slot a, reuse on/off.
    P["a_fixed_noreuse"] = lambda k: (44 + k % 16, [20, 29 + 2 * (k % 4), 36 + 2 * (k % 4)][:nsrc], 0)
    P["a_fixed_reuse_a"] = lambda k: (44 + k % 16, [20, 29 + 2 * (k % 4), 36 + 2 * (k % 4)][:nsrc], 1)
    P["ab_fixed_reuse_ab"] = lambda k: (44 + k % 16, [20, 29, 36 + 2 * (k % 4)][:nsrc], 3)
    P["abc_fixed_reuse_abc"] = lambda k: (44 + k % 16, [20, 29, 38][:nsrc], 7)
    P["abc_fixed_noreuse"] = lambda k: (44 + k % 16, [20, 29, 38][:nsrc], 0)
    # Same register twice in one instruction.
    P["a_eq_b"] = lambda k: (44 + k % 16, [20 + 2 * (k % 4), 20 + 2 * (k % 4), 37 + 2 * (k % 4)][:nsrc], 0)
    return P
