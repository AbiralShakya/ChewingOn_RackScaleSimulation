# rackscale_caps: vLLM vs Hazy Llama-1B megakernel on B300 (della-tridao GPU 3)

Della work dir: /scratch/gpfs/MZHASAN/as0714/mkviz (scripts, logs/, out/, runs/, allcta/).
Captures came from /scratch/gpfs/MZHASAN/as0714/capture (env: capture/scratch/env.sh; repo = capture/repo).
`$L1B` = unsloth/Llama-3.2-1B-Instruct snapshot; all GPU runs `flock /home/as0714/.gpu3.lock`, CUDA_VISIBLE_DEVICES=3.
`./collect_della.sh` (this folder) copies the step 2-4 results from Della once the queue is done.

## Step 1: existing captures

| tarball | what | command |
|---|---|---|
| vllm_l1b_b32_nographs.tar.gz | vLLM 0.32, batch 32, decode at ~520 ctx, torch.compile, no CUDA graphs: graph+paths capture (`vllm_l1b_c/`), plus the native CUDA-graph nsys run (`nsys_vllm_l1b.sqlite`), its step summary (`logs/nsys_vllm_l1b.json`: 193 kernels, 941.8 us GPU span, 916.6 us busy per step) and `graphrun_vllm_l1b.json` | `MODE=--no-graphs ./cap_engine.sh vllm_l1b_c vllm marked --batch 32 --gpu-mem 0.1` = `rackscale capture -o runs/vllm_l1b_c --region marked --passes graph,paths --gpu-lock $LOCK -- python examples/capture/vllm_decode.py --model $L1B --no-graphs --batch 32 --gpu-mem 0.1`; native: `nsys profile -t cuda --cuda-graph-trace=node ... vllm_decode.py --model $L1B --batch 32 --measure` then `nsys_graph_steps.py` and `compare_graph_run.py - nsys_vllm_l1b.sqlite --summarize graphrun_vllm_l1b.json` |
| vllm_l1b_b32_eager.tar.gz | same model/batch, `--eager` (no compile, no graphs; module marks): `vllm_l1b/` with graph/, paths/, paths1/, kernels/, and a later nsys timing pass (`timing/timing.sqlite`); `logs/vllm_measure--eager.log` (eager decode step 10.15 ms) | `./cap_engine.sh vllm_l1b vllm marked --batch 32` (graph,paths), then `PASSES=timing` (command.json records the last pass only) |
| mk_l1b_hazy.tar.gz | Hazy low-latency Llama-1B megakernel, graph pass of the whole run (one 148-CTA x 640-thread launch per token); `logs/mk_native.log`: 1644.7 tok/s = 608 us/token (stock build, ntok=32) | `rackscale capture -o runs/mk_l1b --region all --passes graph -- python megakernels/scripts/generate.py mode=mk model=$L1B ntok=8 num_warmup=1 num_iters=1` (scratch/mk.sh) |
| vllm_70b_tp8_rank.tar.gz | one TP=8 rank of a Llama-3.1-70B-shaped model (dummy weights), vLLM eager, batch 32: paths passes, nsys timing; native CUDA-graph run `nsys_vllm_70b_tp8.sqlite` + `graphrun_vllm_70b_tp8.json` (6.74 ms decode step) | `./cap_engine.sh vllm_70b_tp8 vllm marked --batch 32 --shape llama3-70b --shard-tp 8`; native: queue14.sh nsys line with `--measure --shape llama3-70b --shard-tp 8 --gpu-mem 0.1` |

## Step 2: megakernel per-SM schedule and timing (pending on Della)

Built a timing copy (`mkviz/Megakernels_timing`, patch in hazy_src/mkviz_timing_patch.diff.txt); dump script hazy_src/mkviz_mk_timing_dump.py.txt.
Outputs (Della `mkviz/out/`): megakernel_timing.json (default 8-token prompt), megakernel_timing_ctx512.json (512 random tokens, like vLLM),
megakernel_wall_stock*.json (stock build, wall time only). Command: `mkviz/mk_runs.sh`.

Timing units: `timings[sm][instr][slot]` int32 = clock64() cycles since that CTA's kernel entry (0 = slot not written).
Slots: 0 controller start, 1 instruction fetch done, 2 page alloc done, 3 semaphores set up, 4 controller end (instruction retired);
5/6 loader start/end, 7/8 launcher start/end, 9/10 storer start/end, 11+2w / 12+2w consumer warp w start/end (w = 0..15);
44/45 at / done global-memory barrier wait (spin on Bar counter of the producing opcode), 46/47 at / done gmem store + atomicAdd to Bar;
48-53 first/last load, use, store (matvec pipeline); 54 output ready; 55-82 op-specific (llama.cuh: ATOMIC_ADD, EPILOGUE, ACT/WEIGHT waits, RMS);
123 attention-partial QOL stored. mkviz additions: 116/117 absolute clock64 at controller start (lo/hi), 118/119 %globaltimer ns (lo/hi), 126/127 CTA start clock64 (lo/hi).
Inter-SM sync: `Bar[layer, opcode-1, idx]` uint32 counters; producers atomicAdd after their store, consumers spin `while (Bar < expected) __nanosleep(GMEM_SPIN_LOOP_SLEEP_NANOS)`
(declared `static constexpr bool` = 20, so it sleeps 1 ns). Barriers are refilled with 0 before every token (MK_Generator.fill).

## Step 3: vLLM batch 1 (pending): mkviz/vllm_b1.sh -> vllm_l1b_b1.tar.gz, logs/nsys_vllm_l1b_b1.json, logs/vllm_measure_b1.log, logs/sglang_measure_b1.log
## Step 4: all-CTA megakernel passes (pending): mkviz/mk_allcta.sh -> mk_allcta/pass_<a>.{rpath.gz,raw.tar.gz,ptrs.json,timings.json,summary.json}

## Files
(to be filled with sizes when done)
