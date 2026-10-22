"""Model shapes to capture on one GPU with dummy weights, and per-rank sharding."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
try:
    from rackscale.replay import shard_overrides as shard  # noqa: F401
except ImportError:  # rackscale not importable here: same rule, inlined
    def shard(config, tp=1, ep=1):
        o = {}
        hd = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
        if tp > 1:
            o["num_attention_heads"] = config["num_attention_heads"] // tp
            o["num_key_value_heads"] = max(1, config["num_key_value_heads"] // tp)
            for k in ("intermediate_size", "moe_intermediate_size", "shared_expert_intermediate_size"):
                if config.get(k):
                    o[k] = config[k] // tp
            o["vocab_size"] = -(-config["vocab_size"] // tp)
            o["head_dim"] = hd
        if ep > 1:
            for k in ("num_experts", "n_routed_experts", "num_local_experts"):
                if config.get(k):
                    o[k] = config[k] // ep
        return o

# Dimensions only; the architecture (code path) is the loaded model's.
SHAPES = {
    "llama3-70b": {"hidden_size": 8192, "intermediate_size": 28672, "num_hidden_layers": 80,
                   "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 128256},
    "llama3-8b": {"hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
                  "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 128256},
}
