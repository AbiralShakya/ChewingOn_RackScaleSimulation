| case | GPUs | textbook µs | replay µs | sim µs | sim vs replay | sim vs textbook | kernel efficiency µs | power clock µs | collectives µs | overlap + idle µs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 GPU | 1 | 4685 | 5116 | 9141 | +78.7% | +95.1% | +1378 | +2620 | +0 | +28 |
| DDP | 8 | 4871 | 5606 | 9967 | +77.8% | +104.6% | +1415 | +2615 | +612 | -282 |
| ZeRO-2 | 8 | 4685 | 5546 | 9705 | +75.0% | +107.1% | +1372 | +2619 | +280 | -112 |
| ZeRO-2 + optimizer | 8 | 8399 | 6559 | 11000 | +67.7% | +31.0% | +1414 | +2615 | +523 | -111 |
| FSDP | 8 | 7243 | 5975 | 10377 | +73.7% | +43.3% | +1488 | +2605 | +849 | -540 |
| HSDP 4x2 | 8 | 6401 | 5980 | 10425 | +74.3% | +62.9% | +1448 | +2609 | +934 | -547 |
| TP8 | 8 | 924 | 1174 | 2057 | +75.2% | +122.7% | +355 | +101 | +398 | +29 |
| TP8 + SP | 8 | 1188 | 1266 | 2506 | +98.0% | +111.1% | +405 | +101 | +705 | +29 |
| TP2 x DP4 | 8 | 2681 | 3196 | 5638 | +76.4% | +110.3% | +763 | +1116 | +590 | -26 |
| TP2 x FSDP4 (HSDP 2) | 8 | 3099 | 3387 | 6238 | +84.2% | +101.3% | +769 | +1109 | +1343 | -370 |
| PP2 GPipe m8 | 2 | 21164 | 23105 | 41490 | +79.6% | +96.0% | +5470 | +10384 | +2805 | -274 |
| PP2 1F1B m8 | 2 | 21164 | 23346 | 42676 | +82.8% | +101.6% | +5456 | +10339 | +2805 | +729 |
| PP2 ZB-H1 m8 | 2 | 19602 | 22377 | 40024 | +78.9% | +104.2% | +5456 | +10340 | +2805 | -954 |
| TP2 x PP2 x DP2 1F1B m4 | 8 | 6745 | 7735 | 14293 | +84.8% | +111.9% | +1545 | +2236 | +1899 | +878 |
| prefill TP4 | 4 | 1155 | 1415 | 2435 | +72.2% | +110.9% | +303 | +386 | +320 | +12 |
| prefill TP4 + SP | 4 | 1155 | 1436 | 2444 | +70.2% | +111.6% | +333 | +386 | +277 | +12 |
| prefill CP8 ring | 8 | 6470 | 7727 | 13690 | +77.2% | +111.6% | +4060 | +1877 | +4916 | -4889 |
| prefill CP8 Ulysses | 8 | 6816 | 7231 | 12770 | +76.6% | +87.4% | +3383 | +1874 | +271 | +11 |
| decode 1 GPU b64 | 1 | 1066 | 936 | 1087 | +16.1% | +2.0% | +137 | +0 | +0 | +13 |
| decode TP8 b64 (70B) | 8 | 243 | 225 | 366 | +62.8% | +50.3% | +60 | +0 | +67 | +14 |
| decode MoE TP4 (Qwen3-30B-A3B) | 4 | 237 | 234 | 304 | +29.8% | +28.1% | +23 | +0 | +30 | +16 |
| decode EP8 (DeepSeek-V3) | 8 | 1740 | 1510 | 1672 | +10.7% | -3.9% | +100 | +1 | +30 | +30 |

Collectives, simulated vs closed form (mean per call):

| case | kind | algo | size | ranks | n | closed form µs | sim µs | error |
|---|---|---|---|---|---|---|---|---|
| DDP | all_reduce | Nvls | 416.0 MiB | 8 | 2 | 489.7 | 795.9 | +62.5% |
| ZeRO-2 | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 569.1 | +32.6% |
| ZeRO-2 + optimizer | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 569.1 | +32.6% |
| ZeRO-2 + optimizer | all_gather | Ring | 832.0 MiB | 8 | 1 | 853.2 | 1096.0 | +28.5% |
| FSDP | all_gather | Ring | 416.0 MiB | 8 | 4 | 429.1 | 568.8 | +32.6% |
| FSDP | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 573.9 | +33.7% |
| HSDP 4x2 | all_gather | Ring | 416.0 MiB | 4 | 8 | 368.5 | 473.0 | +28.3% |
| HSDP 4x2 | reduce_scatter | Ring | 416.0 MiB | 4 | 4 | 368.5 | 473.6 | +28.5% |
| HSDP 4x2 | all_reduce | Nvls | 104.0 MiB | 2 | 8 | 126.2 | 279.3 | +121.3% |
| TP8 | all_reduce | Nvls | 32.0 MiB | 8 | 8 | 42.3 | 92.0 | +117.6% |
| TP8 + SP | all_gather | Ring | 32.0 MiB | 8 | 8 | 37.6 | 82.0 | +118.0% |
| TP8 + SP | reduce_scatter | Ring | 32.0 MiB | 8 | 8 | 37.6 | 81.4 | +116.3% |
| TP2 x DP4 | all_reduce | Nvls | 32.0 MiB | 2 | 32 | 42.3 | 101.5 | +140.0% |
| TP2 x DP4 | reduce_scatter | Ring | 208.0 MiB | 4 | 4 | 186.8 | 245.2 | +31.3% |
| TP2 x FSDP4 (HSDP 2) | all_gather | Ring | 208.0 MiB | 2 | 16 | 126.2 | 170.1 | +34.8% |
| TP2 x FSDP4 (HSDP 2) | all_reduce | Nvls | 32.0 MiB | 2 | 32 | 42.3 | 129.7 | +206.6% |
| TP2 x FSDP4 (HSDP 2) | reduce_scatter | Ring | 208.0 MiB | 2 | 8 | 126.2 | 169.0 | +33.9% |
| TP2 x FSDP4 (HSDP 2) | all_reduce | Nvls | 104.0 MiB | 2 | 8 | 126.2 | 317.5 | +151.6% |
| PP2 GPipe m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| PP2 1F1B m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| PP2 ZB-H1 m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| TP2 x PP2 x DP2 1F1B m4 | all_reduce | Nvls | 32.0 MiB | 2 | 64 | 42.3 | 104.3 | +146.7% |
| TP2 x PP2 x DP2 1F1B m4 | send_recv | Direct | 16.0 MiB | 2 | 32 | 21.6 | 126.9 | +486.6% |
| TP2 x PP2 x DP2 1F1B m4 | reduce_scatter | Ring | 208.0 MiB | 2 | 4 | 126.2 | 191.0 | +51.4% |
| prefill TP4 | all_reduce | Nvls | 64.0 MiB | 4 | 4 | 79.6 | 159.6 | +100.6% |
| prefill TP4 + SP | all_gather | Ring | 64.0 MiB | 4 | 4 | 60.9 | 95.6 | +57.0% |
| prefill TP4 + SP | reduce_scatter | Ring | 64.0 MiB | 4 | 4 | 60.9 | 95.4 | +56.7% |
| prefill CP8 ring | send_recv | Direct | 32.0 MiB | 2 | 112 | 40.3 | 215.8 | +435.8% |
| prefill CP8 Ulysses | all_to_all | Direct | 96.0 MiB | 8 | 2 | 102.9 | 176.4 | +71.5% |
| prefill CP8 Ulysses | all_to_all | Direct | 64.0 MiB | 8 | 2 | 70.2 | 132.3 | +88.3% |
| decode TP8 b64 (70B) | all_reduce | Ring | 1.0 MiB | 8 | 4 | 6.2 | 23.0 | +272.9% |
| decode MoE TP4 (Qwen3-30B-A3B) | all_reduce | Ring | 0.2 MiB | 4 | 4 | 5.3 | 12.9 | +144.0% |
| decode EP8 (DeepSeek-V3) | all_to_all | Direct | 3.5 MiB | 8 | 1 | 8.6 | 21.7 | +153.7% |
| decode EP8 (DeepSeek-V3) | all_to_all | Direct | 7.0 MiB | 8 | 1 | 12.1 | 29.4 | +142.0% |
