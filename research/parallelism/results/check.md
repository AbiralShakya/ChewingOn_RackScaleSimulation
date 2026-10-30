| case | GPUs | textbook µs | replay µs | sim µs | sim vs replay | sim vs textbook | kernel efficiency µs | power clock µs | collectives µs | overlap + idle µs |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 GPU | 1 | 4685 | 5116 | 9141 | +78.7% | +95.1% | +1378 | +2620 | +0 | +28 |
| DDP | 8 | 4871 | 5606 | 9980 | +78.0% | +104.9% | +1415 | +2615 | +675 | -331 |
| ZeRO-2 | 8 | 4685 | 5546 | 9702 | +74.9% | +107.1% | +1372 | +2619 | +277 | -112 |
| ZeRO-2 + optimizer | 8 | 8399 | 6559 | 11009 | +67.8% | +31.1% | +1414 | +2615 | +543 | -122 |
| FSDP | 8 | 7243 | 5975 | 10377 | +73.7% | +43.3% | +1488 | +2605 | +859 | -550 |
| HSDP 4x2 | 8 | 6401 | 5980 | 10423 | +74.3% | +62.8% | +1448 | +2609 | +926 | -540 |
| TP8 | 8 | 924 | 1174 | 2056 | +75.1% | +122.6% | +355 | +101 | +397 | +29 |
| TP8 + SP | 8 | 1188 | 1266 | 2493 | +96.9% | +109.9% | +405 | +101 | +691 | +29 |
| TP2 x DP4 | 8 | 2681 | 3196 | 5672 | +77.5% | +111.6% | +763 | +1116 | +606 | -8 |
| TP2 x FSDP4 (HSDP 2) | 8 | 3099 | 3387 | 6251 | +84.5% | +101.7% | +769 | +1109 | +1336 | -350 |
| PP2 GPipe m8 | 2 | 21164 | 23105 | 41490 | +79.6% | +96.0% | +5470 | +10384 | +2805 | -274 |
| PP2 1F1B m8 | 2 | 21164 | 23346 | 42676 | +82.8% | +101.6% | +5456 | +10339 | +2805 | +729 |
| PP2 ZB-H1 m8 | 2 | 19602 | 22377 | 40024 | +78.9% | +104.2% | +5456 | +10340 | +2805 | -954 |
| TP2 x PP2 x DP2 1F1B m4 | 8 | 6745 | 7735 | 14325 | +85.2% | +112.4% | +1545 | +2236 | +1911 | +899 |
| prefill TP4 | 4 | 1155 | 1415 | 2426 | +71.5% | +110.1% | +303 | +386 | +311 | +12 |
| prefill TP4 + SP | 4 | 1155 | 1436 | 2443 | +70.1% | +111.5% | +333 | +386 | +276 | +12 |
| prefill CP8 ring | 8 | 6470 | 7727 | 13690 | +77.2% | +111.6% | +4060 | +1877 | +4916 | -4889 |
| prefill CP8 Ulysses | 8 | 6816 | 7231 | 12749 | +76.3% | +87.0% | +3383 | +1874 | +250 | +11 |
| decode 1 GPU b64 | 1 | 1066 | 936 | 1087 | +16.1% | +2.0% | +137 | +0 | +0 | +13 |
| decode TP8 b64 (70B) | 8 | 243 | 225 | 359 | +59.9% | +47.7% | +60 | +0 | +61 | +14 |
| decode MoE TP4 (Qwen3-30B-A3B) | 4 | 237 | 234 | 302 | +29.1% | +27.5% | +23 | +0 | +29 | +16 |
| decode EP8 (DeepSeek-V3) | 8 | 1740 | 1510 | 1671 | +10.6% | -4.0% | +100 | +1 | +30 | +30 |

Collectives, simulated vs closed form (mean per call):

| case | kind | algo | size | ranks | n | closed form µs | sim µs | error |
|---|---|---|---|---|---|---|---|---|
| DDP | all_reduce | Nvls | 416.0 MiB | 8 | 2 | 489.7 | 827.3 | +68.9% |
| ZeRO-2 | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 567.8 | +32.3% |
| ZeRO-2 + optimizer | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 575.0 | +34.0% |
| ZeRO-2 + optimizer | all_gather | Ring | 832.0 MiB | 8 | 1 | 853.2 | 1104.3 | +29.4% |
| FSDP | all_gather | Ring | 416.0 MiB | 8 | 4 | 429.1 | 570.8 | +33.0% |
| FSDP | reduce_scatter | Ring | 416.0 MiB | 8 | 2 | 429.1 | 575.0 | +34.0% |
| HSDP 4x2 | all_gather | Ring | 416.0 MiB | 4 | 8 | 368.5 | 472.7 | +28.3% |
| HSDP 4x2 | reduce_scatter | Ring | 416.0 MiB | 4 | 4 | 368.5 | 473.3 | +28.4% |
| HSDP 4x2 | all_reduce | Nvls | 104.0 MiB | 2 | 8 | 126.2 | 276.1 | +118.8% |
| TP8 | all_reduce | Nvls | 32.0 MiB | 8 | 8 | 42.3 | 91.9 | +117.4% |
| TP8 + SP | all_gather | Ring | 32.0 MiB | 8 | 8 | 37.6 | 80.0 | +112.7% |
| TP8 + SP | reduce_scatter | Ring | 32.0 MiB | 8 | 8 | 37.6 | 81.6 | +117.0% |
| TP2 x DP4 | all_reduce | Nvls | 32.0 MiB | 2 | 32 | 42.3 | 102.6 | +142.6% |
| TP2 x DP4 | reduce_scatter | Ring | 208.0 MiB | 4 | 4 | 186.8 | 248.5 | +33.1% |
| TP2 x FSDP4 (HSDP 2) | all_gather | Ring | 208.0 MiB | 2 | 16 | 126.2 | 170.3 | +35.0% |
| TP2 x FSDP4 (HSDP 2) | all_reduce | Nvls | 32.0 MiB | 2 | 32 | 42.3 | 129.5 | +206.2% |
| TP2 x FSDP4 (HSDP 2) | reduce_scatter | Ring | 208.0 MiB | 2 | 8 | 126.2 | 169.8 | +34.6% |
| TP2 x FSDP4 (HSDP 2) | all_reduce | Nvls | 104.0 MiB | 2 | 8 | 126.2 | 313.5 | +148.5% |
| PP2 GPipe m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| PP2 1F1B m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| PP2 ZB-H1 m8 | send_recv | Direct | 32.0 MiB | 2 | 16 | 40.3 | 215.6 | +435.3% |
| TP2 x PP2 x DP2 1F1B m4 | all_reduce | Nvls | 32.0 MiB | 2 | 64 | 42.3 | 105.1 | +148.7% |
| TP2 x PP2 x DP2 1F1B m4 | send_recv | Direct | 16.0 MiB | 2 | 32 | 21.6 | 126.8 | +485.8% |
| TP2 x PP2 x DP2 1F1B m4 | reduce_scatter | Ring | 208.0 MiB | 2 | 4 | 126.2 | 190.0 | +50.6% |
| prefill TP4 | all_reduce | Nvls | 64.0 MiB | 4 | 4 | 79.6 | 157.3 | +97.7% |
| prefill TP4 + SP | all_gather | Ring | 64.0 MiB | 4 | 4 | 60.9 | 94.7 | +55.4% |
| prefill TP4 + SP | reduce_scatter | Ring | 64.0 MiB | 4 | 4 | 60.9 | 96.1 | +57.7% |
| prefill CP8 ring | send_recv | Direct | 32.0 MiB | 2 | 112 | 40.3 | 215.8 | +435.8% |
| prefill CP8 Ulysses | all_to_all | Direct | 96.0 MiB | 8 | 2 | 102.9 | 170.2 | +65.4% |
| prefill CP8 Ulysses | all_to_all | Direct | 64.0 MiB | 8 | 2 | 70.2 | 128.0 | +82.2% |
| decode TP8 b64 (70B) | all_reduce | Ring | 1.0 MiB | 8 | 4 | 6.2 | 21.4 | +247.3% |
| decode MoE TP4 (Qwen3-30B-A3B) | all_reduce | Ring | 0.2 MiB | 4 | 4 | 5.3 | 12.5 | +136.7% |
| decode EP8 (DeepSeek-V3) | all_to_all | Direct | 3.5 MiB | 8 | 1 | 8.6 | 21.7 | +153.7% |
| decode EP8 (DeepSeek-V3) | all_to_all | Direct | 7.0 MiB | 8 | 1 | 12.1 | 28.6 | +135.4% |
