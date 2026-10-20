"""Asynchronous RL on a heterogeneous cluster: 8 H100 rollout samplers decode
while 8 B200 trainers run a training step, then the trainers push fresh
weights to the samplers over InfiniBand.

    python examples/async_rl_mixed.py
"""

import rackscale as rs
from rackscale import lower, models

wl = lower.async_rl(models.LLAMA3_8B, rollout_ranks=8, trainer_ranks=8, rollout_batch=64, context=2048,
                    decode_steps=4, train_micro_batch=1, seqlen=2048, layers=2)
rep = rs.simulate("configs/mixed_rl_h100_b200.yaml", wl, fast=True)
print(rep.summary())
for r in rep["ranks"][:1] + rep["ranks"][8:9]:
    print({k: r[k] for k in r if not isinstance(r[k], (list, dict))})
