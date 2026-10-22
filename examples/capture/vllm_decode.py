"""vLLM offline decode: measure it natively, or run it under `rackscale capture`.

    # native step times (CUDA graphs on, as served)
    python vllm_decode.py --model meta-llama/Llama-3.2-1B-Instruct --batch 32 --measure
    # capture: eager, in-process engine, region = prefill + K decode steps
    rackscale capture -o cap/vllm -- python vllm_decode.py --model ... --batch 32 --eager --capture-steps 8

`--shard-tp N` runs one tensor-parallel rank's share of the model on one GPU
(heads, KV heads, MLP width and vocabulary divided by N, dummy weights), and
`--shape` replaces the model's dimensions (e.g. a Llama-3.1-70B-shaped model
on the 1B model's code and tokenizer), so a rack deployment can be captured
on one GPU. Prompts are random token ids, generation ignores EOS.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shapes import SHAPES, shard  # noqa: E402

try:
    from rsim_capture import region
except ImportError:  # not under rackscale capture
    from contextlib import nullcontext as region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (required for capture)")
    ap.add_argument("--measure", action="store_true", help="time prefill and decode steps natively")
    ap.add_argument("--capture-steps", type=int, default=8, help="decode steps inside the capture region")
    ap.add_argument("--shape", choices=sorted(SHAPES), help="replace the model's dimensions")
    ap.add_argument("--layers", type=int, help="override the number of layers")
    ap.add_argument("--shard-tp", type=int, default=1)
    ap.add_argument("--shard-ep", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.30)
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine in this process: the region marks reach it
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cfg = AutoConfig.from_pretrained(a.model).to_dict()
    over = dict(SHAPES[a.shape]) if a.shape else {}
    if a.layers:
        over["num_hidden_layers"] = a.layers
    cfg.update(over)
    over.update(shard(cfg, a.shard_tp, a.shard_ep))
    dummy = bool(over)
    vocab = over.get("vocab_size", cfg["vocab_size"])
    steps = 64 if a.measure else 1 + a.capture_steps
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=a.eager, seed=0,
              gpu_memory_utilization=a.gpu_mem, max_model_len=a.prompt_len + 2 * steps + 16,
              max_num_seqs=a.batch, max_num_batched_tokens=max(8192, a.batch * a.prompt_len),
              hf_overrides=over or None, load_format="dummy" if dummy else "auto")
    import random

    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(100, min(vocab, 32000)) for _ in range(a.prompt_len)])
               for _ in range(a.batch)]

    def gen(n):
        sp = SamplingParams(max_tokens=n, ignore_eos=True, temperature=0.0)
        t = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t

    gen(4)  # warm up every kernel the region will run
    gen(steps)
    if a.measure:
        t1 = min(gen(1) for _ in range(3))
        n1, n2 = 16, 64
        ta = min(gen(n1) for _ in range(3))
        tb = min(gen(n2) for _ in range(3))
        print(json.dumps({"engine": "vllm", "batch": a.batch, "prompt_len": a.prompt_len, "eager": a.eager,
                          "shape": a.shape, "shard_tp": a.shard_tp, "prefill_ms": 1e3 * t1,
                          "decode_step_us": 1e6 * (tb - ta) / (n2 - n1)}))
        return
    with region("vllm generate"):
        gen(steps)


if __name__ == "__main__":
    main()
