# Command line

Generated from `--help` by `python docs/reference/gen_cli.py`. Every command below is also
available as `rsim <command>` for the simulator-core ones, which is the same program as the
standalone Rust binary.

```text
usage: rackscale [-h] [--version] COMMAND ...

``rackscale`` command line: lower a model or a captured trace and simulate it.

rackscale demo --open                  # 30 seconds: H100 vs B200 on Llama-3-70B, then the viewer
rackscale decode deepseek-v3 --system gb200_nvl72 --ep 72 --batch 64 --context 4096 --layers 2 --trace out/ds.json
rackscale train llama3-70b --system hgx_h100_8 --tp 8 --micro-batch 1 --seqlen 4096 --layers 2
rackscale train llama3-70b --system hgx_b200_8 --pp 8 --micro-batches 16 --schedule zb --fast
rackscale capture vllm_trace.json.gz --system gb200_nvl72 --world 8
rackscale capture -o cap/ --region forward:4:2 -- python my_engine.py ...   # any CUDA program, under NVBit
rackscale replay cap/ --system gb300_nvl72 --tp 8 --dp 9

positional arguments:
  COMMAND
    decode    one decode iteration
    prefill   one prefill pass
    train     forward+backward of one micro-batch
    capture   capture any CUDA program under the NVBit tool (`-- cmd ...`), or simulate a torch-
              profiler trace (PATH)
    replay    simulate a capture on a system with a parallelism plan
    run       simulate a workload JSON/YAML file
    demo      a 30-second tour: Llama-3-70B decode on HGX H100 vs B200, then the viewer
    models    list the built-in model architectures
    view      open a viewer page (.html) or a Perfetto trace (.json) in a browser

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

simulator-core commands (same as the `rsim` binary; `rackscale CMD --help` for each):
  presets    list built-in GPU, switch and system presets
  system     resolve a system YAML (or preset) and print its topology
  kernel     simulate one synthetic kernel on one GPU (gemm, attention, ssm_update, ...)
  sass       simulate one kernel from a cuobjdump/nvdisasm listing
  path       simulate a kernel recorded on hardware with NVBit
  timeline   per-warp issue timeline of one kernel (compare with clock64 probes)
  validate   replay measured reference kernels and report error vs Nsight Compute
  net        collective sweeps, NCCL cost model, validation against published numbers
  viz        write the zoomable rack -> GPU -> SM -> warp -> SASS viewer page

docs: https://abiralshakya.github.io/ChewingOn_RackScaleSimulation/
```

## Model and workload commands

### `rackscale demo`

```text
usage: rackscale demo [-h] [-o OUT] [--open]

options:
  -h, --help     show this help message and exit
  -o, --out OUT  viewer page to write
  --open         open the page in a browser
```

### `rackscale decode`

```text
usage: rackscale decode [-h] [--tp TP] [--ep EP] [--dp DP] [--pp PP] [--comm-sms COMM_SMS]
                        [--layers LAYERS] [--batch BATCH] [--context CONTEXT] [--overlap]
                        [--system SYSTEM] [--trace TRACE] [--report REPORT]
                        [--dump-workload DUMP_WORKLOAD] [--launch-gap-ns LAUNCH_GAP_NS] [--fast]
                        [--per-sm] [--viz PAGE.html] [--open]
                        MODEL

positional arguments:
  MODEL                 one of: deepseek-v3, llama3-405b, llama3-70b, llama3-8b, mamba2-70b,
                        mamba2-70b-ssu8, mamba2-8b, mamba2-hybrid-70b, mamba2-hybrid-8b,
                        mixtral-8x22b, qwen3-235b-a22b (see `rackscale models`)

options:
  -h, --help            show this help message and exit
  --tp TP
  --ep EP
  --dp DP
  --pp PP               pipeline stages (train)
  --comm-sms COMM_SMS
  --layers LAYERS       simulate only the first N layers (memory is still checked for the full
                        model)
  --batch BATCH         sequences per rank
  --context CONTEXT
  --overlap             two-micro-batch compute/comm overlap (DeepSeek style)
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale prefill`

```text
usage: rackscale prefill [-h] [--tp TP] [--ep EP] [--dp DP] [--pp PP] [--comm-sms COMM_SMS]
                         [--layers LAYERS] [--batch BATCH] [--seqlen SEQLEN] [--system SYSTEM]
                         [--trace TRACE] [--report REPORT] [--dump-workload DUMP_WORKLOAD]
                         [--launch-gap-ns LAUNCH_GAP_NS] [--fast] [--per-sm] [--viz PAGE.html]
                         [--open]
                         MODEL

positional arguments:
  MODEL                 one of: deepseek-v3, llama3-405b, llama3-70b, llama3-8b, mamba2-70b,
                        mamba2-70b-ssu8, mamba2-8b, mamba2-hybrid-70b, mamba2-hybrid-8b,
                        mixtral-8x22b, qwen3-235b-a22b (see `rackscale models`)

options:
  -h, --help            show this help message and exit
  --tp TP
  --ep EP
  --dp DP
  --pp PP               pipeline stages (train)
  --comm-sms COMM_SMS
  --layers LAYERS       simulate only the first N layers (memory is still checked for the full
                        model)
  --batch BATCH
  --seqlen SEQLEN
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale train`

```text
usage: rackscale train [-h] [--tp TP] [--ep EP] [--dp DP] [--pp PP] [--comm-sms COMM_SMS]
                       [--layers LAYERS] [--micro-batch MICRO_BATCH] [--seqlen SEQLEN]
                       [--micro-batches MICRO_BATCHES] [--schedule {gpipe,1f1b,zb}] [--optimizer]
                       [--system SYSTEM] [--trace TRACE] [--report REPORT]
                       [--dump-workload DUMP_WORKLOAD] [--launch-gap-ns LAUNCH_GAP_NS] [--fast]
                       [--per-sm] [--viz PAGE.html] [--open]
                       MODEL

positional arguments:
  MODEL                 one of: deepseek-v3, llama3-405b, llama3-70b, llama3-8b, mamba2-70b,
                        mamba2-70b-ssu8, mamba2-8b, mamba2-hybrid-70b, mamba2-hybrid-8b,
                        mixtral-8x22b, qwen3-235b-a22b (see `rackscale models`)

options:
  -h, --help            show this help message and exit
  --tp TP
  --ep EP
  --dp DP
  --pp PP               pipeline stages (train)
  --comm-sms COMM_SMS
  --layers LAYERS       simulate only the first N layers (memory is still checked for the full
                        model)
  --micro-batch MICRO_BATCH
  --seqlen SEQLEN
  --micro-batches MICRO_BATCHES
                        micro-batches per step (pipeline)
  --schedule {gpipe,1f1b,zb}
                        pipeline schedule: GPipe, 1F1B, or ZB-H1 zero-bubble
  --optimizer           end the step with an Adam update and DP all-gather
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale capture`

```text
usage: rackscale capture [-h] [-o OUT] [--region REGION] [--key {config,func}] [--passes PASSES]
                         [--no-modules] [--max-paths MAX_PATHS] [--max-mb MAX_MB]
                         [--filter FILTER] [--gpu-lock GPU_LOCK] [--build-only] [--world WORLD]
                         [--group GROUP] [--mode {resimulate,replay}] [--max-kernels MAX_KERNELS]
                         [--system SYSTEM] [--trace TRACE] [--report REPORT]
                         [--dump-workload DUMP_WORKLOAD] [--launch-gap-ns LAUNCH_GAP_NS] [--fast]
                         [--per-sm] [--viz PAGE.html] [--open]
                         [path]

positional arguments:
  path                  torch-profiler trace (when no `-- cmd` is given)

options:
  -h, --help            show this help message and exit
  -o, --out OUT         capture directory (with `-- cmd`)
  --region REGION       marked (rsim_capture.region() in the program) | all |
                        forward:SKIP:COUNT[:CLASS_REGEX]
  --key {config,func}   what makes two launches the same kernel (one recording each)
  --passes PASSES       graph (launch graph), paths (warp paths), timing (native nsys run for
                        exact times)
  --no-modules          do not mark module calls (cleaner host timeline)
  --max-paths MAX_PATHS
  --max-mb MAX_MB       drop recordings larger than this
  --filter FILTER       only record kernels whose mangled name contains this
  --gpu-lock GPU_LOCK   flock this file around each pass
  --build-only          rebuild capture.json from existing passes
  --world WORLD
  --group GROUP         communicator group for captured NCCL ops
  --mode {resimulate,replay}
  --max-kernels MAX_KERNELS
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale replay`

```text
usage: rackscale replay [-h] [--plan PLAN] [--tp TP] [--ep EP] [--pp PP] [--dp DP]
                        [--microbatches MICROBATCHES] [--topk TOPK] [--mode {paths,measured}]
                        [--host] [--steps STEPS] [--kernels] [--system SYSTEM] [--trace TRACE]
                        [--report REPORT] [--dump-workload DUMP_WORKLOAD]
                        [--launch-gap-ns LAUNCH_GAP_NS] [--fast] [--per-sm] [--viz PAGE.html]
                        [--open]
                        capture

positional arguments:
  capture

options:
  -h, --help            show this help message and exit
  --plan PLAN           plan YAML (tp, ep, pp, dp, microbatches, collectives)
  --tp TP
  --ep EP
  --pp PP
  --dp DP
  --microbatches MICROBATCHES
  --topk TOPK           experts per token (all-to-all bytes)
  --mode {paths,measured}
                        re-simulate recorded kernels, or keep every kernel's measured duration
  --host                replay the captured host issue time of every launch
  --steps STEPS         engine steps to replay, A:B (0-based, B exclusive)
  --kernels             also compare every recorded kernel alone to its measurement
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale run`

```text
usage: rackscale run [-h] [--system SYSTEM] [--trace TRACE] [--report REPORT]
                     [--dump-workload DUMP_WORKLOAD] [--launch-gap-ns LAUNCH_GAP_NS] [--fast]
                     [--per-sm] [--viz PAGE.html] [--open]
                     workload

positional arguments:
  workload

options:
  -h, --help            show this help message and exit
  --system SYSTEM       preset name or system YAML path
  --trace TRACE         write a Perfetto trace (open in ui.perfetto.dev)
  --report REPORT       write the full JSON report
  --dump-workload DUMP_WORKLOAD
                        write the lowered workload JSON and continue
  --launch-gap-ns LAUNCH_GAP_NS
  --fast                simulate each distinct kernel once and reuse it across ranks/layers
  --per-sm              add one Perfetto track per SM with a span per CTA (exact mode)
  --viz PAGE.html       also write the zoomable viewer page (rack down to SASS) for rank 0
  --open                open the --viz page (or Perfetto for --trace) in a browser
```

### `rackscale models`

```text
usage: rackscale models [-h]

options:
  -h, --help  show this help message and exit
```

### `rackscale view`

```text
usage: rackscale view [-h] file

positional arguments:
  file

options:
  -h, --help  show this help message and exit
```

## Simulator-core commands

### `rackscale presets`

```text
List built-in GPU, switch and system presets

Usage: rackscale presets

Options:
  -h, --help  Print help
```

### `rackscale system`

```text
Resolve a system YAML and print its topology

Usage: rackscale system <PATH>

Arguments:
  <PATH>  

Options:
  -h, --help  Print help
```

### `rackscale kernel`

```text
Simulate one synthetic kernel on one GPU

Usage: rackscale kernel [OPTIONS] <KIND> [DIMS]...

Arguments:
  <KIND>     gemm M N K | attention B HQ HKV SQ SKV D | elementwise N | rownorm ROWS COLS | ssm_update B H P N G | conv1d B DIM L [W] | ssd_<stage> B L H P N G [CHUNK]
  [DIMS]...  

Options:
      --gpu <GPU>                  [default: h100_sxm]
      --dtype <DTYPE>              [default: bf16]
      --calibration <CALIBRATION>  Apply a rackscale-calibration/v1 JSON on top of the preset
      --json                       
  -h, --help                       Print help
```

### `rackscale sass`

```text
Parse a cuobjdump/nvdisasm listing and simulate one kernel from it

Usage: rackscale sass [OPTIONS] <PATH>

Arguments:
  <PATH>  

Options:
      --gpu <GPU>      [default: h100_sxm]
      --grid <GRID>    [default: 1]
      --block <BLOCK>  [default: 128]
      --regs <REGS>    [default: 32]
      --smem <SMEM>    [default: 0]
      --inspect        Only print the parsed program (classes, control bits)
  -h, --help           Print help
```

### `rackscale path`

```text
Simulate a kernel recorded on hardware with the NVBit warp_path tool (exact per-warp paths, mbarrier/TMA/UMMA operands; see research/flash_attention)

Usage: rackscale path [OPTIONS] <PATH>

Arguments:
  <PATH>  

Options:
      --gpu <GPU>                  [default: b300]
      --calibration <CALIBRATION>  rackscale-calibration/v1 JSON measured on the target GPU
      --regs <REGS>                Override registers per thread (NVBit reports 0 for some JIT kernels)
      --json                       
  -h, --help                       Print help
```

### `rackscale timeline`

```text
Simulate one kernel from a SASS listing and dump every warp's issue timeline (cycle and PC of each issued instruction) on SM 0. Used to compare against clock64 probes read on hardware (calibration/dissect/timeline.py)

Usage: rackscale timeline [OPTIONS] --out <OUT> <PATH>

Arguments:
  <PATH>  

Options:
      --kernel <KERNEL>            Function name in the listing (default: the first one)
      --gpu <GPU>                  [default: b300]
      --calibration <CALIBRATION>  
      --grid <GRID>                [default: 1]
      --block <BLOCK>              [default: 32]
      --regs <REGS>                [default: 64]
      --smem <SMEM>                [default: 0]
      --lrr                        Loose round-robin warp scheduling instead of greedy-then-oldest
      --gty                        Greedy-then-youngest warp scheduling
      --no-control-bits            Ignore the compiler's control bits (per-register scoreboarding)
      --sms <SMS>                  SMs whose warps are traced (default: all)
      --warm <WARM>                Run the kernel this many times first (warm caches, as a hardware measurement after warm-up launches); the timeline is the last run's [default: 0]
  -o, --out <OUT>                  
  -h, --help                       Print help
```

### `rackscale validate`

```text
Replay measured reference kernels (ncu_reference.json) and report simulated vs measured SM cycles

Usage: rackscale validate [OPTIONS] <REFERENCE>

Arguments:
  <REFERENCE>  

Options:
      --gpu <GPU>                  [default: b300]
      --calibration <CALIBRATION>  Apply a rackscale-calibration/v1 JSON on top of the preset
      --only <ONLY>                Only these reference labels (comma separated)
      --max-error <MAX_ERROR>      Fail if any kernel's |error| exceeds this many percent
      --json                       
  -h, --help                       Print help
```

### `rackscale net`

```text
Network: nccl-tests-style sweeps, NCCL's cost model, and validation against published numbers (see docs/NETWORK.md)

Usage: rackscale net <COMMAND>

Commands:
  sweep     nccl-tests-style sweep of one collective (all ranks post at t = 0)
  model     NCCL 2.32.3's own cost model: every (algorithm, protocol) it considers
  validate  Compare against published numbers (calibration/net/published.json)
  fit       Fit the NCCL latency/throughput parameters to the `fit` points
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

#### `rackscale net sweep`

```text
nccl-tests-style sweep of one collective (all ranks post at t = 0)

Usage: rackscale net sweep [OPTIONS] --system <SYSTEM>

Options:
      --system <SYSTEM>      System preset name or YAML path
      --coll <COLL>          all_reduce | all_gather | reduce_scatter | all_to_all | broadcast | send_recv [default: all_reduce]
      --ranks <RANKS>        Communicator size (default: every GPU)
      --algo <ALGO>          auto | ring | tree | nvls | nvls_tree | collnet | direct [default: auto]
      --proto <PROTO>        ll | ll128 | simple
      --fidelity <FIDELITY>  transfer | packet [default: transfer]
  -b, --begin <BEGIN>        [default: 8]
  -e, --end <END>            [default: 1G]
  -f, --factor <FACTOR>      [default: 4]
      --set <SET>            Override a NetConfig field, e.g. `nccl.step_ns=[300,1200,2400]`
      --json                 
  -h, --help                 Print help
```

#### `rackscale net model`

```text
NCCL 2.32.3's own cost model: every (algorithm, protocol) it considers

Usage: rackscale net model [OPTIONS] --system <SYSTEM>

Options:
      --system <SYSTEM>  
      --coll <COLL>      [default: all_reduce]
      --ranks <RANKS>    
  -b, --begin <BEGIN>    [default: 8]
  -e, --end <END>        [default: 1G]
  -f, --factor <FACTOR>  [default: 4]
  -h, --help             Print help
```

#### `rackscale net validate`

```text
Compare against published numbers (calibration/net/published.json)

Usage: rackscale net validate [OPTIONS] <FILE>

Arguments:
  <FILE>  

Options:
      --fidelity <FIDELITY>  transfer | packet | both [default: both]
      --set <SET>            
      --out <OUT>            Write per-point results as JSON
      --only <ONLY>          Only points whose id contains this
  -h, --help                 Print help
```

#### `rackscale net fit`

```text
Fit the NCCL latency/throughput parameters to the `fit` points

Usage: rackscale net fit [OPTIONS] <FILE>

Arguments:
  <FILE>  

Options:
      --fidelity <FIDELITY>  [default: transfer]
      --set <SET>            
      --rounds <ROUNDS>      [default: 3]
  -h, --help                 Print help
```

### `rackscale viz`

```text
Simulate with full capture and write a zoomable viewer: rack, GPU, SMs, warps and SASS (see docs/VIZ.md)

Usage: rackscale viz <COMMAND>

Commands:
  run     Run a workload on a system (like `rsim run`) and capture it
  path    A kernel recorded on hardware (`.rpath`, see `rsim path`)
  kernel  A synthetic kernel (same kinds as `rsim kernel`)
  sass    A cuobjdump/nvdisasm listing (same flags as `rsim sass`)
  merge   Merge bundles (`.json` or `.html` written by `rsim viz`) into one page
  help    Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

#### `rackscale viz run`

```text
Run a workload on a system (like `rsim run`) and capture it

Usage: rackscale viz run [OPTIONS] --system <SYSTEM> --workload <WORKLOAD> --out <OUT>

Options:
      --system <SYSTEM>
          
      --workload <WORKLOAD>
          
      --fast
          Fast mode (memoized kernels; see `rsim run --fast`)
      --detail-ranks <DETAIL_RANKS>
          Ranks with per-SM, per-warp and per-PC capture [default: 0]
      --link-bin-ns <LINK_BIN_NS>
          Link utilisation time-bin width, ns [default: 1000]
      --launch-gap-ns <LAUNCH_GAP_NS>
          
  -o, --out <OUT>
          Output file: `.html` (self-contained viewer) or `.json` (bundle)
      --name <NAME>
          Run name shown in the viewer (default: derived from the input)
      --title <TITLE>
          Page title
      --note <NOTE>
          Explanatory text shown with the run (blank lines separate paragraphs)
      --warp-sms <WARP_SMS>
          SMs that get per-warp timelines (comma separated) [default: 0]
      --max-warp-events <MAX_WARP_EVENTS>
          Cap on warp-timeline events per traced SM [default: 400000]
  -h, --help
          Print help
```

#### `rackscale viz path`

```text
A kernel recorded on hardware (`.rpath`, see `rsim path`)

Usage: rackscale viz path [OPTIONS] --out <OUT> <PATH>

Arguments:
  <PATH>  

Options:
      --gpu <GPU>
          [default: b300]
      --calibration <CALIBRATION>
          
      --regs <REGS>
          
  -o, --out <OUT>
          Output file: `.html` (self-contained viewer) or `.json` (bundle)
      --name <NAME>
          Run name shown in the viewer (default: derived from the input)
      --title <TITLE>
          Page title
      --note <NOTE>
          Explanatory text shown with the run (blank lines separate paragraphs)
      --warp-sms <WARP_SMS>
          SMs that get per-warp timelines (comma separated) [default: 0]
      --max-warp-events <MAX_WARP_EVENTS>
          Cap on warp-timeline events per traced SM [default: 400000]
  -h, --help
          Print help
```

#### `rackscale viz kernel`

```text
A synthetic kernel (same kinds as `rsim kernel`)

Usage: rackscale viz kernel [OPTIONS] --out <OUT> <KIND> [DIMS]...

Arguments:
  <KIND>     
  [DIMS]...  

Options:
      --gpu <GPU>
          [default: b200]
      --dtype <DTYPE>
          [default: bf16]
      --calibration <CALIBRATION>
          
  -o, --out <OUT>
          Output file: `.html` (self-contained viewer) or `.json` (bundle)
      --name <NAME>
          Run name shown in the viewer (default: derived from the input)
      --title <TITLE>
          Page title
      --note <NOTE>
          Explanatory text shown with the run (blank lines separate paragraphs)
      --warp-sms <WARP_SMS>
          SMs that get per-warp timelines (comma separated) [default: 0]
      --max-warp-events <MAX_WARP_EVENTS>
          Cap on warp-timeline events per traced SM [default: 400000]
  -h, --help
          Print help
```

#### `rackscale viz sass`

```text
A cuobjdump/nvdisasm listing (same flags as `rsim sass`)

Usage: rackscale viz sass [OPTIONS] --out <OUT> <PATH>

Arguments:
  <PATH>  

Options:
      --gpu <GPU>
          [default: h100_sxm]
      --grid <GRID>
          [default: 1]
      --block <BLOCK>
          [default: 128]
      --regs <REGS>
          [default: 32]
      --smem <SMEM>
          [default: 0]
  -o, --out <OUT>
          Output file: `.html` (self-contained viewer) or `.json` (bundle)
      --name <NAME>
          Run name shown in the viewer (default: derived from the input)
      --title <TITLE>
          Page title
      --note <NOTE>
          Explanatory text shown with the run (blank lines separate paragraphs)
      --warp-sms <WARP_SMS>
          SMs that get per-warp timelines (comma separated) [default: 0]
      --max-warp-events <MAX_WARP_EVENTS>
          Cap on warp-timeline events per traced SM [default: 400000]
  -h, --help
          Print help
```

#### `rackscale viz merge`

```text
Merge bundles (`.json` or `.html` written by `rsim viz`) into one page

Usage: rackscale viz merge [OPTIONS] --out <OUT> [INPUTS]...

Arguments:
  [INPUTS]...  

Options:
  -o, --out <OUT>      
      --title <TITLE>  
  -h, --help           Print help
```
