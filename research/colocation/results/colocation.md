### partition

```
2 tenants on GB300 NVL72 (4 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
deepseek     deepseek-v3        decode   ep64               64     1520.4     1520.4    +0.00%
llama8b      llama3-8b          decode   tp8                 8      535.7      535.7    +0.00%
```

### mixed

```
6 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
trainer      llama3-70b         train    tp8 dp2 fsdp       16     6433.3     6433.3    +0.00%
moe          qwen3-30b-a3b      decode   tp4                 4      397.3      397.3    +0.00%
1b-0         llama3.2-1b        decode   1 GPU               1      478.1      478.1    +0.00%
1b-1         llama3.2-1b        decode   1 GPU               1      478.1      478.1    +0.00%
1b-2         llama3.2-1b        decode   1 GPU               1      478.1      478.1    +0.00%
1b-3         llama3.2-1b        decode   1 GPU               1      478.1      478.1    +0.00%
```

### disagg 2 GB/s

```
1 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
decode       llama3-70b         decode   tp8                 8      365.3      365.2    +0.02%
flow kv: 8 GPU pairs x 2 x 4.0 MiB on 4 SMs, 2.0 GB/s offered, 2.0 GB/s delivered per pair
```

### disagg 10 GB/s

```
1 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
decode       llama3-70b         decode   tp8                 8      366.4      365.2    +0.32%
flow kv: 8 GPU pairs x 7 x 4.0 MiB on 4 SMs, 10.0 GB/s offered, 10.0 GB/s delivered per pair
```

### disagg 25 GB/s

```
1 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
decode       llama3-70b         decode   tp8                 8      367.6      365.2    +0.65%
flow kv: 8 GPU pairs x 18 x 4.0 MiB on 4 SMs, 25.0 GB/s offered, 25.0 GB/s delivered per pair
```

### disagg 50 GB/s

```
1 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
decode       llama3-70b         decode   tp8                 8      372.0      365.2    +1.87%
flow kv: 8 GPU pairs x 35 x 4.0 MiB on 4 SMs, 50.0 GB/s offered, 50.0 GB/s delivered per pair
```

### disagg 100 GB/s

```
1 tenants on GB300 NVL72 (2 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
decode       llama3-70b         decode   tp8                 8      376.9      365.2    +3.19%
flow kv: 8 GPU pairs x 70 x 4.0 MiB on 4 SMs, 100.0 GB/s offered, 100.0 GB/s delivered per pair
```

### shared (fast)

```
2 tenants on GB300 NVL72 (1 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
8b           llama3-8b          decode   tp4                 4      210.4      145.6   +44.45%
1b           llama3.2-1b        decode   tp4                 4      165.7      124.6   +32.96%
```

### shared (exact)

```
2 tenants on GB300 NVL72 (1 layers each)
tenant       model              phase    plan             GPUs    step us   alone us  slowdown
8b           llama3-8b          decode   tp4                 4      173.4      138.3   +25.39%
1b           llama3.2-1b        decode   tp4                 4      144.6      120.3   +20.18%
```
