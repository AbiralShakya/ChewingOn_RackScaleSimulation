#!/bin/bash
#SBATCH --job-name=fa-a100p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/flashattn/results/a100p_%j.log
W=/scratch/gpfs/MZHASAN/as0714/flashattn
cd $W/repo
. remote/env.sh
hostname
TAG=a100_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -d '()')
R=$W/results/$TAG; mkdir -p $R
T=$W/traces
export WP_LIST=1
bash remote/trace.sh $T/${TAG}_fa2_s4096 flash_fwd 0,1,32,128 --impl fa2 --b 4 --h 16 --s 4096
WP_TMA_ALL=1 bash remote/trace.sh $T/${TAG}_fa4_s4096 FlashAttentionForward 0,1,32,128 --impl fa4 --b 4 --h 16 --s 4096
for i in sdpa_flash sdpa_cudnn fa2 fa4; do
  timeout 1200 python remote/ncu_attn.py --impl $i --b 4 --h 16 --s 4096 --out $R/ncu_${i}_s4096.json 2>&1 | tail -4
done
echo DONE
