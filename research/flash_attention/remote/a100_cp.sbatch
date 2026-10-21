#!/bin/bash
#SBATCH --job-name=fa-a100cp
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/flashattn/results/a100cp_%j.log
W=/scratch/gpfs/MZHASAN/as0714/flashattn
cd $W/repo; . remote/env.sh; hostname
TAG=a100_$(nvidia-smi --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -d '()')
bash remote/cp_shapes.sh $TAG fa2 flash_fwd auto
echo DONE
