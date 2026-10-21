#!/bin/bash
#SBATCH --job-name=fa-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/flashattn/results/a100_%j.log
set -x
W=/scratch/gpfs/MZHASAN/as0714/flashattn
cd $W/repo
. remote/env.sh
hostname; nvidia-smi
export NCU=$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | tail -1); [ -x "$NCU" ] || export NCU=$CUDA_HOME/bin/ncu
TAG=a100_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -d '()')
bash remote/run_gpu.sh $TAG naive sdpa_flash sdpa_cudnn fa2 fa4
T=$W/traces
bash remote/trace.sh $T/${TAG}_fa2_s4096 flash_fwd_kernel 0,1,32,128 --impl fa2 --b 4 --h 16 --s 4096
WP_TMA_ALL=1 bash remote/trace.sh $T/${TAG}_fa4_s4096 FlashAttentionForward 0,1,32,128 --impl fa4 --b 4 --h 16 --s 4096
echo DONE
