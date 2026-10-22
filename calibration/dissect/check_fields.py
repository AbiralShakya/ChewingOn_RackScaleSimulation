"""Verify encoding field positions by patching one instruction and re-disassembling."""
import subprocess
import sys

from cubin import Cubin
from sasslib import parse

cub, sass, fn = sys.argv[1:4]
img = open(cub, "rb").read()
insts = parse(open(sass).read())[fn]
tgt = next(i for i in insts if i.base == "FFMA")
print("orig :", tgt.text, tgt.ctrl)
c = Cubin(img)
c.set_reg(fn, tgt.pc, "rd", 201)
c.set_reg(fn, tgt.pc, "ra", 130)
c.set_reg(fn, tgt.pc, "rb", 141)
c.set_reg(fn, tgt.pc, "rc", 152)
c.set_ctrl(fn, tgt.pc, stall=13, yld=0, wbar=3, rbar=4, wait=0b100101, reuse=0b0101)
c.set_regcount(fn, 255)
open("/tmp/cf.cubin", "wb").write(c.data)
out = subprocess.run(["cuobjdump", "-sass", "/tmp/cf.cubin"], capture_output=True, text=True).stdout
p = parse(out)[fn]
t2 = next(i for i in p if i.pc == tgt.pc)
print("patch:", t2.text, t2.ctrl)
print("regcount header:", c.regcount(fn))
out = subprocess.run(["nvdisasm", "-print-instruction-encoding", "/tmp/cf.cubin"], capture_output=True, text=True)
print(out.stderr[:300])
