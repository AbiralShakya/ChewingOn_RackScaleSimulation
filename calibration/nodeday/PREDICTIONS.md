# Node-day predictions

Written by `predict.py` before any measurement. `compare.py` scores the node day against these
files as committed; they are not regenerated afterwards. Commits they came from:

- `hgx_b300_8`: `1161306f402df201f309360b1040d93e22012bc2` at 2026-10-30T18:23:07+00:00
- `hgx_b200_8`: `74249a969c8b506e7c31110ccec24cd050f41dd8` at 2026-10-30T18:38:36+00:00
- `hgx_h100_8`: `74249a969c8b506e7c31110ccec24cd050f41dd8` at 2026-10-30T18:45:53+00:00

## Collectives (nccl-tests out-of-place time, default algorithm, 8 GPUs)

| case | hgx_b300_8 | hgx_b200_8 | hgx_h100_8 |
|---|---|---|---|
| all_reduce 1 MiB | 18.5 µs · 99 GB/s · Ring/LL | 18.5 µs · 99 GB/s · Ring/LL | 30.7 µs · 60 GB/s · Ring/LL |
| all_reduce 32 MiB | 91.1 µs · 645 GB/s · Nvls/Simple | 91.0 µs · 645 GB/s · Nvls/Simple | 150.8 µs · 389 GB/s · Nvls/Simple |
| all_reduce 1024 MiB | 1910.3 µs · 984 GB/s · Nvls/Simple | 1912.8 µs · 982 GB/s · Nvls/Simple | 3700.1 µs · 508 GB/s · Nvls/Simple |
| all_gather 1 MiB | 13.2 µs · 70 GB/s · Ring/LL | 13.2 µs · 70 GB/s · Ring/LL | 19.6 µs · 47 GB/s · Ring/LL |
| all_gather 32 MiB | 65.9 µs · 446 GB/s · Ring/LL128 | 65.9 µs · 446 GB/s · Ring/LL128 | 109.2 µs · 269 GB/s · Ring/Simple |
| all_gather 1024 MiB | 1324.0 µs · 710 GB/s · Ring/Simple | 1324.0 µs · 710 GB/s · Ring/Simple | 2882.8 µs · 326 GB/s · Ring/Simple |
| reduce_scatter 1 MiB | 13.2 µs · 70 GB/s · Ring/LL | 13.2 µs · 70 GB/s · Ring/LL | 19.6 µs · 47 GB/s · Ring/LL |
| reduce_scatter 32 MiB | 65.9 µs · 446 GB/s · Ring/LL128 | 65.9 µs · 446 GB/s · Ring/LL128 | 109.2 µs · 269 GB/s · Ring/Simple |
| reduce_scatter 1024 MiB | 1324.0 µs · 710 GB/s · Ring/Simple | 1324.0 µs · 710 GB/s · Ring/Simple | 2882.8 µs · 326 GB/s · Ring/Simple |
| all_to_all 1 MiB | 17.3 µs · 53 GB/s · Direct/Simple | 17.3 µs · 53 GB/s · Direct/Simple | 19.5 µs · 47 GB/s · Direct/Simple |
| all_to_all 32 MiB | 56.7 µs · 518 GB/s · Direct/Simple | 56.7 µs · 518 GB/s · Direct/Simple | 126.8 µs · 232 GB/s · Direct/Simple |
| all_to_all 1024 MiB | 1318.0 µs · 713 GB/s · Direct/Simple | 1318.0 µs · 713 GB/s · Direct/Simple | 2888.2 µs · 325 GB/s · Direct/Simple |
| broadcast 1 MiB | 70.6 µs · 15 GB/s · Direct/Simple | 70.6 µs · 15 GB/s · Direct/Simple | 111.6 µs · 9 GB/s · Direct/Simple |
| broadcast 32 MiB | 122.4 µs · 274 GB/s · Direct/Simple | 122.4 µs · 274 GB/s · Direct/Simple | 223.9 µs · 150 GB/s · Direct/Simple |
| broadcast 1024 MiB | 3657.6 µs · 294 GB/s · Direct/Simple | 3657.6 µs · 294 GB/s · Direct/Simple | 7036.7 µs · 153 GB/s · Direct/Simple |
| send_recv 1 MiB | 18.2 µs · 58 GB/s · Direct/Simple | 18.2 µs · 58 GB/s · Direct/Simple | 20.9 µs · 50 GB/s · Direct/Simple |
| send_recv 32 MiB | 108.8 µs · 308 GB/s · Direct/Simple | 108.8 µs · 308 GB/s · Direct/Simple | 184.9 µs · 181 GB/s · Direct/Simple |
| send_recv 1024 MiB | 3080.8 µs · 349 GB/s · Direct/Simple | 3080.8 µs · 349 GB/s · Direct/Simple | 5560.6 µs · 193 GB/s · Direct/Simple |

## Training steps (forward + backward of the transformer layers, sim)

| case | hgx_b300_8 | hgx_b200_8 | hgx_h100_8 |
|---|---|---|---|
| 1 GPU, 2 layers | 9141.5 µs (replay 5116.5 µs) | 9467.9 µs (replay 5555.3 µs) | 18.12 ms (replay 12.61 ms) |
| 1 GPU, 8 layers | 36.57 ms (replay 20.47 ms) | 37.87 ms (replay 22.22 ms) | 72.48 ms (replay 50.44 ms) |
| DDP, 2 layers | 9960.6 µs (replay 5606.1 µs) | 10.23 ms (replay 6045.0 µs) | 19.73 ms (replay 13.59 ms) |
| FSDP, 2 layers | 10.35 ms (replay 5974.7 µs) | 10.55 ms (replay 6413.5 µs) | 20.76 ms (replay 14.32 ms) |
| FSDP, 8 layers | 39.62 ms (replay 21.32 ms) | 39.52 ms (replay 23.08 ms) | 77.36 ms (replay 52.15 ms) |
| HSDP 4x2, 2 layers | 10.42 ms (replay 5979.7 µs) | 10.63 ms (replay 6418.5 µs) | 20.95 ms (replay 14.32 ms) |
| TP8, 2 layers | 2019.1 µs (replay 1174.3 µs) | 2294.8 µs (replay 1226.4 µs) | 3980.0 µs (replay 2628.7 µs) |
| TP8, 8 layers | 8076.6 µs (replay 4697.2 µs) | 9179.2 µs (replay 4905.5 µs) | 15.93 ms (replay 10.51 ms) |
| TP8 + SP, 2 layers | 2253.1 µs (replay 1265.8 µs) | 2521.9 µs (replay 1320.6 µs) | 4224.0 µs (replay 2726.5 µs) |
| PP2 GPipe m8, 2 layers | 41.46 ms (replay 23.10 ms) | 42.93 ms (replay 25.08 ms) | 82.12 ms (replay 56.90 ms) |
| PP2 1F1B m8, 2 layers | 42.56 ms (replay 23.35 ms) | 44.03 ms (replay 25.32 ms) | 84.00 ms (replay 57.37 ms) |
| PP2 ZB-H1 m8, 2 layers | 40.00 ms (replay 22.38 ms) | 41.48 ms (replay 24.28 ms) | 79.20 ms (replay 55.10 ms) |
| DDP x4, 2 layers | 10.03 ms (replay 5606.1 µs) | 10.29 ms (replay 6045.0 µs) | 19.86 ms (replay 13.59 ms) |
| FSDP x4, 2 layers | 10.12 ms (replay 5853.5 µs) | 10.39 ms (replay 6292.3 µs) | 20.46 ms (replay 14.08 ms) |
| TP4, 2 layers | 3027.7 µs (replay 1785.8 µs) | 3384.0 µs (replay 1893.1 µs) | 6229.7 µs (replay 4145.6 µs) |

## Decode steps (one token for every sequence, sim)

| case | hgx_b300_8 | hgx_b200_8 | hgx_h100_8 |
|---|---|---|---|
| decode 1 GPU b64, 32 layers | 14.97 ms (replay 12.62 ms) | 18.09 ms (replay 12.42 ms) | 32.84 ms (replay 28.30 ms) |
| decode 1 GPU b64, 2 layers | 1087.0 µs (replay 936.5 µs) | 1281.2 µs (replay 921.6 µs) | 2388.1 µs (replay 2102.0 µs) |
| decode TP8 b64 (70B), 80 layers | 12.12 ms (replay 7409.7 µs) | 13.81 ms (replay 7313.8 µs) | 29.08 ms (replay 15.12 ms) |
| decode TP8 b64 (70B), 2 layers | 347.8 µs (replay 224.6 µs) | 398.7 µs (replay 221.6 µs) | 819.7 µs (replay 465.9 µs) |
| decode MoE TP4 (Qwen3-30B-A3B), 48 layers | 6472.5 µs (replay 5045.3 µs) | 7742.6 µs (replay 4977.6 µs) | 12.93 ms (replay 10.39 ms) |
| decode MoE TP4 (Qwen3-30B-A3B), 2 layers | 296.1 µs (replay 233.9 µs) | 349.4 µs (replay 230.8 µs) | 605.2 µs (replay 485.2 µs) |
| decode EP8 (DeepSeek-V3), 61 layers | 27.44 ms (replay 33.45 ms) | 33.47 ms (replay 32.94 ms) | – |
| decode EP8 (DeepSeek-V3), 4 layers | 1666.2 µs (replay 1510.4 µs) | 1834.5 µs (replay 1486.8 µs) | – |

## GEMMs (square, one GPU; time and sustained SM clock)

| case | hgx_b300_8 | hgx_b200_8 | hgx_h100_8 |
|---|---|---|---|
| bf16 2048³ | 16.0 µs · 1075 TFLOP/s, no clock drop | 16.9 µs · 1015 TFLOP/s, no clock drop | 27.0 µs · 636 TFLOP/s, no clock drop |
| bf16 4096³ | 91.6 µs · 1500 TFLOP/s, 1650 MHz | 102.0 µs · 1348 TFLOP/s, no clock drop | 203.7 µs · 675 TFLOP/s, no clock drop |
| bf16 8192³ | 778.4 µs · 1413 TFLOP/s, 1303 MHz | 791.1 µs · 1390 TFLOP/s, no clock drop | 1617.0 µs · 680 TFLOP/s, no clock drop |
| bf16 16384³ | 6513.1 µs · 1351 TFLOP/s, 1235 MHz | 6236.8 µs · 1410 TFLOP/s, no clock drop | 12.76 ms · 690 TFLOP/s, no clock drop |
| fp8 2048³ | 11.9 µs · 1438 TFLOP/s, no clock drop | 11.1 µs · 1555 TFLOP/s, no clock drop | 14.4 µs · 1189 TFLOP/s, no clock drop |
| fp8 4096³ | 41.0 µs · 3353 TFLOP/s, no clock drop | 53.6 µs · 2565 TFLOP/s, no clock drop | 105.3 µs · 1305 TFLOP/s, no clock drop |
| fp8 8192³ | 259.7 µs · 4233 TFLOP/s, no clock drop | 402.6 µs · 2731 TFLOP/s, no clock drop | 817.7 µs · 1345 TFLOP/s, no clock drop |
| fp8 16384³ | 2018.0 µs · 4359 TFLOP/s, no clock drop | 3159.4 µs · 2784 TFLOP/s, no clock drop | 6428.7 µs · 1368 TFLOP/s, no clock drop |

## GEMM beside NCCL (4096³ BF16 with R SMs held by communication)

| case | hgx_b300_8 | hgx_b200_8 | hgx_h100_8 |
|---|---|---|---|
| R = 0 | 91.6 µs · 4 waves | 102.0 µs · 4 waves | 203.7 µs · 4 waves |
| R = 2 | 91.3 µs · 4 waves | 101.7 µs · 4 waves | 203.6 µs · 4 waves |
| R = 4 | 90.5 µs · 4 waves | 102.0 µs · 4 waves | 203.6 µs · 4 waves |
| R = 8 | 90.0 µs · 4 waves | 101.9 µs · 4 waves | 232.1 µs · 5 waves |
| R = 16 | 90.2 µs · 4 waves | 102.0 µs · 4 waves | 220.9 µs · 5 waves |
| R = 24 | 116.9 µs · 5 waves | 123.3 µs · 5 waves | 210.1 µs · 5 waves |
| R = 32 | 117.7 µs · 5 waves | 118.1 µs · 5 waves | 234.4 µs · 6 waves |

## H-Net ÷ BPE, time per byte (B300, sim)

| batch | model | ratio |
|---|---|---|
| 16 | hnet-1stage-xl | 3.19× |
| 16 | hnet-2stage-xl | 4.03× |
| 64 | hnet-1stage-xl | 2.17× |
| 64 | hnet-2stage-xl | 2.35× |
| 256 | hnet-1stage-xl | 1.41× |
| 256 | hnet-2stage-xl | 1.40× |

## Stragglers

tax(N) = sigma * E[max of N standard normals]; E[max_8] = 1.4236, E[max_4] = 1.0294, E[max_2] = 0.5642.

