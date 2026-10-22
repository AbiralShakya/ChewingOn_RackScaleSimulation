"""clock64 probes for FlashAttention-4 (flash_attn.cute, CuTe DSL) without
editing its source: every pipeline wait/acquire/commit/release of every warp
role is wrapped so that, after the original operation, the warp stores
(%clock64, tag) to a record buffer.

Records go to a per-warp ring indexed by clock64 >> 4 (16-cycle buckets), so
no counter, atomic or loop-carried state is needed; two events of one warp
within 16 cycles share a bucket and the later one wins. Only CTAs < NCTA
record. Layout: buf[(cta * 32 + warp) * NSLOT + slot] = (clock64, tag),
16 bytes each.

Set FA4_PROBE_ADDR (device address of a zeroed buffer of buffer_bytes())
before the kernel is JIT-compiled, then call install().
"""
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr

PROBE = int(os.environ.get("FA4_PROBE_ADDR", "0"))
NCTA = 8
NSLOT = 1 << 15

METHODS = {"consumer_wait": 1, "producer_acquire": 2, "producer_commit": 3, "consumer_release": 4}


def buffer_bytes():
    return NCTA * 32 * NSLOT * 16


@cute.jit
def stamp(tag: cutlass.Constexpr):
    if const_expr(PROBE != 0):
        bid = cute.arch.block_idx()[0]
        if bid < NCTA:
            w = cute.arch.warp_idx()
            t = cute.arch.clock64()
            slot = (t >> 4) & (NSLOT - 1)
            addr = Int64(PROBE) + (Int64(bid * 32 + w) * NSLOT + slot) * 16
            p0 = cute.make_ptr(Int64, addr, cute.AddressSpace.gmem, assumed_align=16)
            p1 = cute.make_ptr(Int64, addr + 8, cute.AddressSpace.gmem, assumed_align=8)
            cute.arch.store(p0, t)
            cute.arch.store(p1, Int64(tag))


_depth = [0]


def install():
    """Wrap the pipeline methods FA4 uses. Tag = 16 * class id + method id."""
    import cutlass.pipeline as cp

    import flash_attn.cute.pipeline as fp
    classes = []
    for mod in (fp, cp):
        for name in dir(mod):
            c = getattr(mod, name)
            if isinstance(c, type) and name.startswith("Pipeline") and c not in classes:
                classes.append(c)
    ids = {}
    for c in classes:
        cid = ids.setdefault(c.__name__.replace("Og", ""), len(ids) + 1)
        for m, mid in METHODS.items():
            f = c.__dict__.get(m)
            if f is None or getattr(f, "_probed", False):
                continue

            def wrap(f=f, tag=16 * cid + mid):
                def g(self, *a, **k):
                    # FA4's pipeline classes call their cutlass base methods:
                    # stamp only the outermost call (tracing is sequential).
                    _depth[0] += 1
                    try:
                        r = f(self, *a, **k)
                    finally:
                        _depth[0] -= 1
                    if _depth[0] == 0:
                        stamp(tag)
                    return r
                g._probed = True
                return g
            setattr(c, m, wrap())
    return {v: k for k, v in ids.items()}
