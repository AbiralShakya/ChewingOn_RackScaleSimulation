#!/bin/bash
#SBATCH --job-name=fa-a100cp2
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:40:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/flashattn/results/a100cp2_%j.log
W=/scratch/gpfs/MZHASAN/as0714/flashattn
cd $W/repo; . remote/env.sh; hostname
TAG=a100_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -d '()')
R=$W/results/$TAG; mkdir -p $R
for C in 1024 2048 4096; do for HKV in 8 64; do
  python remote/fa_bench.py --impl fa2 --b 1 --h 64 --hkv $HKV --s $C --iters 70 2>/dev/null | tail -1 >> $R/timing_cp2.jsonl
  timeout 900 python remote/ncu_attn.py --impl fa2 --b 1 --h 64 --hkv $HKV --s $C --out $R/ncu_fa2_cp_c${C}_kv${HKV}.json 2>&1 | tail -1
done; done
for i in naive sdpa_flash sdpa_cudnn fa2 fa4; do for S in 1024 4096 16384; do B=$((16384 / S))
  python remote/fa_bench.py --impl $i --b $B --h 16 --s $S --iters 35 2>/dev/null | tail -1 >> $R/timing2.jsonl
done; done
echo DONE
