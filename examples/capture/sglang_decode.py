"""SGLang offline decode: measure it natively, or run it under `rackscale capture`.

    python sglang_decode.py --model meta-llama/Llama-3.2-1B-Instruct --batch 32 --measure
    rackscale capture -o cap/sglang --region armed:0:9 -- python sglang_decode.py --model ... --batch 32 --eager

SGLang runs the model in a scheduler subprocess, so the capture region is set
by counting model forwards there (`--region armed:SKIP:COUNT`) once this
script calls `rsim_capture.arm()`: 9 forwards = the prefill + 8 decode steps.
Same flags as vllm_decode.py for shapes and sharding.
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shapes import SHAPES, shard  # noqa: E402

try:
    from rsim_capture import arm
except ImportError:  # not under rackscale capture
    def arm():
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (required for capture)")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--capture-steps", type=int, default=8)
    ap.add_argument("--shape", choices=sorted(SHAPES))
    ap.add_argument("--layers", type=int)
    ap.add_argument("--shard-tp", type=int, default=1)
    ap.add_argument("--shard-ep", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    a = ap.parse_args()

    import sglang as sgl
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model).to_dict()
    over = dict(SHAPES[a.shape]) if a.shape else {}
    if a.layers:
        over["num_hidden_layers"] = a.layers
    cfg.update(over)
    over.update(shard(cfg, a.shard_tp, a.shard_ep))
    vocab = over.get("vocab_size", cfg["vocab_size"])
    steps = 64 if a.measure else 1 + a.capture_steps
    eng = sgl.Engine(model_path=a.model, dtype="bfloat16", random_seed=0, mem_fraction_static=a.gpu_mem,
                     disable_cuda_graph=a.eager, max_running_requests=a.batch,
                     chunked_prefill_size=max(8192, a.batch * a.prompt_len),
                     context_length=a.prompt_len + 2 * steps + 16, load_format="dummy" if over else "auto",
                     json_model_override_args=json.dumps(over) if over else "{}", log_level="error")
    rng = random.Random(0)
    ids = [[rng.randrange(100, min(vocab, 32000)) for _ in range(a.prompt_len)] for _ in range(a.batch)]

    def gen(n):
        t = time.perf_counter()
        eng.generate(input_ids=ids, sampling_params={"max_new_tokens": n, "ignore_eos": True, "temperature": 0.0})
        return time.perf_counter() - t

    gen(4)
    gen(steps)
    if a.measure:
        arm()  # opens the profiler window of a timed run (RSIM_CAP_PROFILE_WINDOW)
        t1 = min(gen(1) for _ in range(3))
        n1, n2 = 16, 64
        ta = min(gen(n1) for _ in range(3))
        tb = min(gen(n2) for _ in range(3))
        print(json.dumps({"engine": "sglang", "batch": a.batch, "prompt_len": a.prompt_len, "eager": a.eager,
                          "shape": a.shape, "shard_tp": a.shard_tp, "prefill_ms": 1e3 * t1,
                          "decode_step_us": 1e6 * (tb - ta) / (n2 - n1)}))
    else:
        arm()
        gen(steps)
        # One more forward ends the region inside the scheduler process: the engine kills that
        # process at shutdown, before an exit hook could close it.
        gen(1)
    eng.shutdown()


if __name__ == "__main__":
    main()
