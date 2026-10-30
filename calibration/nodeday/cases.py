"""The node day's cases, shared by predict.py (the simulator) and the measuring scripts.

Keeping one list means a measured run and its prediction have the same
model, depth, batch, sequence length and parallel plan by construction.
"""

# Training: forward + backward of the transformer layers only (no embedding,
# LM head, loss or optimizer), one micro-batch of 4096 tokens, BF16. This is
# what rackscale.lower.train_step lowers and book chapter 11 reports.
T8 = dict(model="llama3-8b", phase="train", micro_batch=1, seqlen=4096)
TRAIN = [
    dict(name="1 GPU", spec=T8, plan=dict(), layers=[2, 8], script="single"),
    dict(name="DDP", spec=T8, plan=dict(dp=8, zero=0), layers=[2], script="ddp"),
    dict(name="FSDP", spec=T8, plan=dict(dp=8, zero=3), layers=[2, 8], script="fsdp"),
    dict(name="HSDP 4x2", spec=T8, plan=dict(dp=8, zero=3, shard=4), layers=[2], script="hsdp"),
    dict(name="TP8", spec=T8, plan=dict(tp=8), layers=[2, 8], script="tp"),
    dict(name="TP8 + SP", spec=T8, plan=dict(tp=8, sp=True), layers=[2], script="tp_sp"),
    dict(name="PP2 GPipe m8", spec={**T8, "micro_batches": 8}, plan=dict(pp=2, schedule="gpipe"), layers=[2],
         script="pp_gpipe"),
    dict(name="PP2 1F1B m8", spec={**T8, "micro_batches": 8}, plan=dict(pp=2, schedule="1f1b"), layers=[2],
         script="pp_1f1b"),
    dict(name="PP2 ZB-H1 m8", spec={**T8, "micro_batches": 8}, plan=dict(pp=2, schedule="zb"), layers=[2],
         script="pp_zb"),
]

# Decode: one token for every sequence. "full" is the whole model (served by
# vLLM with dummy weights, where engine overhead is small next to the step);
# 2 or 4 layers match chapter 11's truncated cases. `batch` is per replica
# for TP and per rank for EP (DeepSeek's data-parallel attention).
DECODE = [
    dict(name="decode 1 GPU b64", spec=dict(model="llama3-8b", phase="decode", batch=64, context=8192), plan=dict(),
         layers=["full", 2], hf="llama3-8b", tp=1, dp=1, ep=False, concurrency=64),
    dict(name="decode TP8 b64 (70B)", spec=dict(model="llama3-70b", phase="decode", batch=64, context=8192),
         plan=dict(tp=8), layers=["full", 2], hf="llama3-70b", tp=8, dp=1, ep=False, concurrency=64),
    dict(name="decode MoE TP4 (Qwen3-30B-A3B)",
         spec=dict(model="qwen3-30b-a3b", phase="decode", batch=64, context=8192), plan=dict(tp=4),
         layers=["full", 2], hf="Qwen/Qwen3-30B-A3B", tp=4, dp=1, ep=False, concurrency=64),
    # 671B in FP8 needs ~690 GB: it fits 8x B200/B300 (1.4-2.3 TB), not 8x H100 (640 GB).
    dict(name="decode EP8 (DeepSeek-V3)", spec=dict(model="deepseek-v3", phase="decode", batch=64, context=4096),
         plan=dict(ep=8), layers=["full", 4], hf="deepseek-ai/DeepSeek-V3", tp=1, dp=8, ep=True, concurrency=512,
         skip=("h100",)),
]

# nccl-tests binaries and the simulator's names for them.
NCCL_COLLS = ["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast", "send_recv"]
NCCL_RANKS = [2, 4, 8]
# Forced algorithms at 8 GPUs (NCCL_ALGO), for the collectives NCCL can run several ways.
NCCL_ALGOS = {"all_reduce": ["ring", "tree", "nvls"], "all_gather": ["ring", "nvls"],
              "reduce_scatter": ["ring", "nvls"]}
NCCL_HEADLINE_SIZES = [1 << 20, 32 << 20, 1 << 30]
NCCL_TESTS_BIN = {"all_reduce": "all_reduce_perf", "all_gather": "all_gather_perf",
                  "reduce_scatter": "reduce_scatter_perf", "all_to_all": "alltoall_perf",
                  "broadcast": "broadcast_perf", "send_recv": "sendrecv_perf"}

# Square GEMMs for clock-under-power and kernel time.
GEMM_SIZES = [2048, 4096, 8192, 16384]

# A GEMM with NCCL running beside it on R SMs (NCCL_MIN_CTAS = NCCL_MAX_CTAS = R).
OVERLAP_RESERVE = dict(gemm=[4096, 4096, 4096], reserve=[0, 2, 4, 8, 16, 24, 32])
