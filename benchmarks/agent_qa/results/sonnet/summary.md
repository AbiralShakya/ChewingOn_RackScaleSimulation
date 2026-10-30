# Rack-scale questions: sonnet with and without rackscale

| | with tools | without tools |
|---|---|---|
| score | 16.33 | 13.67 |
| measured | 2.5 / 4 | 4 / 4 |
| simulated | 9.833 / 11 | 5.667 / 11 |
| closed_form | 4 / 4 | 4 / 4 |
| cost_usd | 0.961 | 0.335 |

| question | truth | reference | with tools | score | without tools | score |
|---|---|---|---|---|---|---|
| gemm_m1 | measured | time_us 41.25 | time_us 38 (-8%) | 1.0 | time_us 38 (-8%) | 1.0 |
| gemm_m256 | measured | time_us 34.14 | time_us 23 (-33%) | 0.0 | time_us 31 (-9%) | 1.0 |
| gemm_m1024 | measured | time_us 93.9 | time_us 78 (-17%) | 0.5 | time_us 95 (+1%) | 1.0 |
| attn_decode | measured | time_us 619.9 | time_us 640 (+3%) | 1.0 | time_us 660 (+6%) | 1.0 |
| splitk | simulated | best_split 4; time_us 25.26 | best_split 4; time_us 25.3 (+0%) | 1.0 | best_split 2 ✗; time_us 23 (-9%) | 0.5 |
| waves | closed_form | waves 5; efficiency 0.8216 | waves 5; efficiency 0.8216 (-0%) | 1.0 | waves 5; efficiency 0.8216 (-0%) | 1.0 |
| plan_llama70b | simulated | tp 4; batch 128; tokens_per_s_per_gpu 1257 | tp 4; batch 256 ✗; tokens_per_s_per_gpu 2005 (+60%) | 0.333 | tp 8 ✗; batch 256 ✗; tokens_per_s_per_gpu 1700 (+35%) | 0.167 |
| explain_idle | simulated | dominant communication; idle_fraction 0.1918 | dominant communication; idle_fraction 0.19 (-1%) | 1.0 | dominant idle_gaps ✗; idle_fraction 0.7 (+265%) | 0.0 |
| explain_gap_train | simulated | ratio 1.951; largest_cause power_capped_clock | ratio 1.95 (-0%); largest_cause power_capped_clock | 1.0 | ratio 1.9 (-3%); largest_cause kernel_inefficiency ✗ | 0.5 |
| predict_hbm | simulated | speedup 1.047 | speedup 1.05 (+0%) | 1.0 | speedup 1.27 (+21%) | 0.0 |
| predict_tensor | simulated | speedup 1.034 | speedup 1.03 (-0%) | 1.0 | speedup 1.05 (+2%) | 1.0 |
| power_cap | simulated | slowdown_pct 9.24 | slowdown_pct 7.5 (-19%) | 1.0 | slowdown_pct 14 (+52%) | 0.5 |
| hnet | simulated | ratio 0.433 | ratio 0.43 (-1%) | 1.0 | ratio 0.6 (+39%) | 0.5 |
| allreduce_nvl72 | simulated | time_us 174.4 | time_us 125 (-28%) | 0.5 | time_us 190 (+9%) | 1.0 |
| nvls_fault | simulated | nvls_slowdown 1.5; ring_slowdown 1.001 | nvls_slowdown 1.5 (-0%); ring_slowdown 1.001 (+0%) | 1.0 | nvls_slowdown 1.027 (-32%); ring_slowdown 1.015 (+1%) | 0.5 |
| stragglers | simulated | overhead_pct 11.2 | overhead_pct 13.4 (+20%) | 1.0 | overhead_pct 13.3 (+19%) | 1.0 |
| ecmp | closed_form | busiest_over_mean 1.338 | busiest_over_mean 1.337 (-0%) | 1.0 | busiest_over_mean 1.34 (+0%) | 1.0 |
| bubble | closed_form | bubble_fraction 0.1795 | bubble_fraction 0.1795 (+0%) | 1.0 | bubble_fraction 0.1795 (+0%) | 1.0 |
| kv_bytes | closed_form | kv_bytes_per_token 327680 | kv_bytes_per_token 3.277e+05 (+0%) | 1.0 | kv_bytes_per_token 3.277e+05 (+0%) | 1.0 |
