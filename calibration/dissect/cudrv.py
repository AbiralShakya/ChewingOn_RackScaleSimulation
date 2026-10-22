"""Tiny CUDA driver-API binding (ctypes, no dependencies) to load a cubin and
launch a kernel with pointer/int arguments. Used to run hand-patched cubins."""
import ctypes as C
import os
import struct

_cu = C.CDLL("libcuda.so.1")


def ck(r, what=""):
    if r != 0:
        s = C.c_char_p()
        _cu.cuGetErrorString(r, C.byref(s))
        raise RuntimeError(f"{what}: CUDA error {r} {s.value.decode() if s.value else ''}")


class Device:
    def __init__(self, ordinal=0):
        ck(_cu.cuInit(0), "cuInit")
        self.dev = C.c_int()
        ck(_cu.cuDeviceGet(C.byref(self.dev), ordinal), "cuDeviceGet")
        self.ctx = C.c_void_p()
        ck(_cu.cuDevicePrimaryCtxRetain(C.byref(self.ctx), self.dev), "ctx")
        ck(_cu.cuCtxSetCurrent(self.ctx), "ctxset")
        self.on_reset = []

    def reset(self):
        """Recover from a sticky error (illegal instruction/address)."""
        _cu.cuDevicePrimaryCtxRelease(self.dev)
        _cu.cuDevicePrimaryCtxReset(self.dev)
        ck(_cu.cuDevicePrimaryCtxRetain(C.byref(self.ctx), self.dev), "ctx")
        ck(_cu.cuCtxSetCurrent(self.ctx), "ctxset")
        for f in self.on_reset:
            f()

    def attr(self, a):
        v = C.c_int()
        ck(_cu.cuDeviceGetAttribute(C.byref(v), a, self.dev))
        return v.value

    def alloc(self, n):
        p = C.c_uint64()
        ck(_cu.cuMemAlloc_v2(C.byref(p), C.c_size_t(n)), "alloc")
        return p.value

    def free(self, p):
        _cu.cuMemFree_v2(C.c_uint64(p))

    def memset(self, p, n, byte=0):
        ck(_cu.cuMemsetD8_v2(C.c_uint64(p), C.c_ubyte(byte), C.c_size_t(n)), "memset")

    def h2d(self, p, b):
        ck(_cu.cuMemcpyHtoD_v2(C.c_uint64(p), C.c_char_p(bytes(b)), C.c_size_t(len(b))), "h2d")

    def d2h(self, p, n):
        buf = C.create_string_buffer(n)
        ck(_cu.cuMemcpyDtoH_v2(buf, C.c_uint64(p), C.c_size_t(n)), "d2h")
        return buf.raw

    def sync(self):
        ck(_cu.cuCtxSynchronize(), "sync")

    def load(self, image):
        m = C.c_void_p()
        ck(_cu.cuModuleLoadData(C.byref(m), C.c_char_p(bytes(image))), "load")
        return Module(self, m)


class Module:
    def __init__(self, dev, m):
        self.dev, self.m = dev, m

    def fn(self, name):
        f = C.c_void_p()
        ck(_cu.cuModuleGetFunction(C.byref(f), self.m, name.encode()), f"getfn {name}")
        return Kernel(self.dev, f)

    def unload(self):
        _cu.cuModuleUnload(self.m)


class Kernel:
    def __init__(self, dev, f):
        self.dev, self.f = dev, f

    def set_smem(self, n):
        # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
        ck(_cu.cuFuncSetAttribute(self.f, 8, n), "smem attr")

    def launch(self, grid, block, args, smem=0, sync=True):
        """args: list of ('p', int) pointer or ('i', int) int32 or ('q', int) u64."""
        holders = []
        for kind, v in args:
            if kind == "i":
                holders.append(C.c_int32(v))
            else:
                holders.append(C.c_uint64(v))
        arr = (C.c_void_p * len(holders))(*[C.cast(C.byref(h), C.c_void_p) for h in holders])
        g = grid if isinstance(grid, tuple) else (grid, 1, 1)
        b = block if isinstance(block, tuple) else (block, 1, 1)
        ck(_cu.cuLaunchKernel(self.f, *g, *b, smem, None, arr, None), "launch")
        if sync:
            self.dev.sync()


def u64s(b):
    return list(struct.unpack(f"<{len(b)//8}Q", b))


def u32s(b):
    return list(struct.unpack(f"<{len(b)//4}I", b))


def f32s(b):
    return list(struct.unpack(f"<{len(b)//4}f", b))
