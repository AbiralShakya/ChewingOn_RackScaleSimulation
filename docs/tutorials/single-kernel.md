# 1. One kernel at cycle level

**What you will do:** simulate a GEMM and a decode-attention kernel on Hopper and Blackwell,
read the scheduler-cycle anatomy, run real SASS, and open the kernel in the viewer.
**Time:** 5 minutes. **Needs:** `pip install rackscale` (no GPU).

## A GEMM on two generations

```bash
rackscale kernel --gpu h100_sxm gemm 8192 8192 8192
rackscale kernel --gpu b200     gemm 8192 8192 8192
```

```
sm100_gemm_bf16_8192x8192x8192_tile128x256
  time        791.43 us   (1468893 cycles, 2048 CTAs, 2 CTA/SM)
  compute     1389.3 TFLOPS  (61.7% of dense BF16 peak)
  memory      509 GB/s model-bytes (6.6% of HBM peak)
  dram        372.8 MB read, 134.2 MB written; L2 hit 95.3%
  scheduler-cycle anatomy:
    async_wait            82.02%  #########################################
    tma_wait              13.68%  #######
    tensor_wait            1.82%  #
    ...
```

The synthetic kernel is built the way the vendor library would build it: on Blackwell a
warp-specialised `tcgen05` kernel (TMA producer warp, one MMA-issuing thread, epilogue warps,
mbarrier handoffs), on Hopper a `wgmma` kernel with the same roles. The H100 run takes
1616.8 µs at 680 TFLOPS, so the B200 is 2.04x faster on this shape.

**Reading the anatomy.** Every scheduler cycle of every SM is attributed to one state, the way
Nsight Compute's warp-state sampling does it:

| state | meaning | in this GEMM |
|---|---|---|
| `async_wait` | warps parked on an mbarrier while TMA or UMMA work they issued is in flight | the expected signature of a warp-specialised kernel: most warps are waiting by design |
| `tma_wait` | the MMA warp waits for a TMA load to land | the tensor core is starved by loads, the bubble that matters here |
| `tensor_wait` | waiting for the tensor core to accept or finish work | |
| `long_scoreboard` / `short_scoreboard` | waiting on a global / shared-memory or MUFU result | dominates memory-bound kernels |
| `selected` | an instruction issued | |

## Memory-bound: decode attention

```bash
rackscale kernel --gpu b200 attention 8 64 8 1 8192 128     # batch, q heads, kv heads, sq, skv, head dim
```

```
sm100_fmha_decode_b8_hq64_hkv8_sq1_skv8192_d128
  time        62.06 us   (115189 cycles, 256 CTAs, 1 CTA/SM)
  compute     34.6 TFLOPS  (1.5% of dense BF16 peak)
  memory      4329 GB/s model-bytes (56.2% of HBM peak)
  dram        268.4 MB read, 8.4 MB written; L2 hit 2.3%
```

256 CTAs on 148 SMs is 1.7 waves: the second wave runs on 108 SMs and the rest sit idle, which is
why the kernel reaches only 56% of HBM bandwidth. Try `--gpu h100_sxm` (132 SMs, so 1.9 waves)
and compare.

Other kinds: `elementwise N`, `rownorm ROWS COLS`, `ssm_update B H P N G`,
`conv1d B DIM L [W]`, `ssd_<stage> B L H P N G [CHUNK]`. Add `--dtype fp8`, or `--json` for every
counter. From Python: `rs.kernel("b200", "gemm", 4096, 4096, 4096)`.

## Real SASS

Dump any kernel with `cuobjdump -sass` (or `nvdisasm`) and simulate exactly those instructions,
control bits included:

```bash
rackscale sass examples/sass/vecadd_sm90.sass --inspect                       # parsed program
rackscale sass examples/sass/vecadd_sm90.sass --gpu h100_sxm --grid 1024 --block 256
```

```
_Z6vecAddPKfS0_Pfi: 16 instructions, control bits: true
  0000  LDC                      Ldc S02 Y W- R- wait000000
  ...
  00a0  LDG.E                    Mem S04 Y W2 R- wait000000
  00c0  FADD                     Fp32 S05   W- R- wait000100
```

Each line shows the opcode, its pipe class, and the decoded control bits: stall count (`S04`),
yield (`Y`), write and read barriers (`W2`), and the wait mask. The simulated run is dominated by
`long_scoreboard` (the `FADD` waiting on the two `LDG`s), as expected for a vector add.

To replay a kernel *exactly* as it ran (every warp's path, lane addresses, mbarriers, TMA and
UMMA operands), record it with NVBit and use `rackscale path`; see
[Capture and replay](capture-replay.md).

## See it

```bash
rackscale viz kernel --gpu b200 gemm 4096 4096 4096 -o gemm.html
rackscale view gemm.html
```

The page shows the kernel's warp roles (producer, MMA, epilogue) and what each waits on, the
per-SM CTA timeline, every warp of SM 0 cycle by cycle, and the SASS listing with stall samples
per instruction. Click an instruction to mark its issues in the warp timeline.
See [the viewer guide](../VIZ.md).

## Check it against hardware

The reference kernels measured on a B300 replay in a few seconds each:

```bash
rackscale validate calibration/results/b300-sxm6-ac-sm103/ncu_reference.json --gpu b300 \
    --only dfma_latency_chain,l2_pointer_chase,hmma_bf16_throughput
```

```
kernel                       ncu cycles   sim cycles    error    trips
dfma_latency_chain (path)         40875        42171     3.2%        0
l2_pointer_chase (path)         3458464      3331353    -3.7%        0
hmma_bf16_throughput (path)       139972       139781    -0.1%        0
```

Each reference is replayed from an exact recording of the kernel as it ran on the B300. CI runs the same check on every
pull request. [Calibration](../CALIBRATION.md) explains how the reference data was taken.
