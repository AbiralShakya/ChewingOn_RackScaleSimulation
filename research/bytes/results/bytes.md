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
| hnet-1stage-xl ÷ BPE, simulated | 3.19x | 2.17x | 1.41x |
| hnet-1stage-xl ÷ BPE, closed form | 1.95x | 1.32x | 1.09x |
| hnet-2stage-xl ÷ BPE, simulated | 4.03x | 2.35x | 1.40x |
| hnet-2stage-xl ÷ BPE, closed form | 2.13x | 1.22x | 0.89x |

## Decode in closed form (bytes moved; µs at 6.82 TB/s)

| model | batch | GB per step | GB per byte | µs per byte |
|---|---|---|---|---|
| hnet-1stage-xl | 16 | 2.39 main weights + 0.12 outer weights + 1.17 main KV + 0.14 Mamba state | 3.82 | 560 |
| hnet-2stage-xl | 16 | 2.69 main weights + 0.49 outer weights + 0.61 main KV + 0.18 outer attention KV + 0.21 Mamba state | 4.18 | 613 |
| bpe-transformer-xl | 16 | 2.39 weights + 0.53 LM head + 5.79 KV | 1.96 | 287 |
| byte-transformer-xl | 16 | 2.39 weights + 25.77 KV | 28.16 | 4131 |
| hnet-1stage-xl | 64 | 2.39 main weights + 0.12 outer weights + 4.67 main KV + 0.55 Mamba state | 7.73 | 1134 |
| hnet-2stage-xl | 64 | 2.69 main weights + 0.49 outer weights + 2.44 main KV + 0.72 outer attention KV + 0.83 Mamba state | 7.16 | 1050 |
| bpe-transformer-xl | 64 | 2.39 weights + 0.53 LM head + 23.16 KV | 5.86 | 860 |
| byte-transformer-xl | 64 | 2.39 weights + 103.08 KV | 105.47 | 15470 |
| hnet-1stage-xl | 256 | 2.39 main weights + 0.12 outer weights + 18.67 main KV + 2.20 Mamba state | 23.38 | 3429 |
| hnet-2stage-xl | 256 | 2.69 main weights + 0.49 outer weights + 9.74 main KV + 2.86 outer attention KV + 3.31 Mamba state | 19.09 | 2801 |
| bpe-transformer-xl | 256 | 2.39 weights + 0.53 LM head + 92.66 KV | 21.48 | 3150 |
| byte-transformer-xl | 256 | 2.39 weights + 412.32 KV | 414.71 | 60826 |

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
