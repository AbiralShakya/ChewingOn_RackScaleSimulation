## FLOPs per byte (8 KiB sequences)

| model | params | GFLOP/byte | paper | error |
|---|---|---|---|---|
| hnet-1stage-xl | 1.26 B | 0.672 | 0.72 | -6.7% |
| hnet-2stage-xl | 1.59 B | 0.670 | 0.69 | -2.8% |
| bpe-transformer-xl | 1.20 B | 0.696 | 0.69 | +0.8% |
| byte-transformer-xl | 1.20 B | 3.197 |  |  |

## Prefill, one B300, 8 prompts of 8 KiB

| model | step µs | MB/s of text |
|---|---|---|
| hnet-1stage-xl | 47995 | 1.37 |
| hnet-2stage-xl | 54705 | 1.20 |
| bpe-transformer-xl | 33464 | 1.96 |
| byte-transformer-xl | 175665 | 0.37 |

## Decode, one B300, batch 64, 8 KiB context

| model | routing | µs per byte | KB/s of text | main-net batch mean (min-max) |
|---|---|---|---|---|
| hnet-1stage-xl | sampled | 2413.0 | 27 | 13.3 (6-18) |
| hnet-1stage-xl | sampled, evened | 2466.1 | 26 | 13.3 (13-14) |
| hnet-1stage-xl | mean field | 2468.2 | 26 | 13.6 (13-14) |
| hnet-2stage-xl | sampled | 2703.2 | 24 | 8.8 (3-12) |
| hnet-2stage-xl | sampled, evened | 2596.8 | 25 | 8.8 (8-9) |
| hnet-2stage-xl | mean field | 2676.4 | 24 | 9.2 (9-10) |
| bpe-transformer-xl | token | 1137.5 | 56 |  |
| byte-transformer-xl | byte | 16583.5 | 4 |  |

## Decode by batch (µs per byte; KB/s of text per GPU)

| model | batch 16 | batch 64 | batch 256 |
|---|---|---|---|
| hnet-1stage-xl | 1597 (10) | 2467 (26) | 5354 (48) |
| hnet-2stage-xl | 2016 (8) | 2676 (24) | 5320 (48) |
| bpe-transformer-xl | 500 (32) | 1137 (56) | 3807 (67) |

## Where the time goes

Prefill: ms per step of 8 prompts. Decode: µs per byte of every sequence (a BPE step makes 4.45 bytes; H-Net rows are the mean-field run).

| run | model | ops | outer stages | main attention | main matmuls | main norms and elementwise | LM head | launch gaps |
|---|---|---|---|---|---|---|---|---|
| prefill | hnet-1stage-xl | 338 | 18.1 | 2.4 | 24.0 | 3.2 | 0.0 | 0.2 |
| prefill | hnet-2stage-xl | 485 | 31.2 | 1.5 | 19.3 | 2.4 | 0.0 | 0.3 |
| prefill | bpe-transformer-xl | 240 | 0.0 | 2.8 | 27.1 | 3.4 | 0.0 | 0.1 |
| prefill | byte-transformer-xl | 240 | 0.0 | 39.6 | 118.9 | 17.1 | 0.0 | 0.1 |
| decode | hnet-1stage-xl | 328 per byte | 367 | 1061 | 744 | 111 | 4 | 182 |
| decode | hnet-2stage-xl | 446 per byte | 834 | 631 | 836 | 124 | 4 | 247 |
| decode | bpe-transformer-xl | 60 per byte | 0 | 892 | 167 | 27 | 18 | 33 |
| decode | byte-transformer-xl | 266 per byte | 0 | 15569 | 744 | 119 | 4 | 147 |
