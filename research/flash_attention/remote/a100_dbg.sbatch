#!/bin/bash
#SBATCH --job-name=fa-dbg
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/flashattn/results/a100dbg_%j.log
W=/scratch/gpfs/MZHASAN/as0714/flashattn
cd $W/repo; . remote/env.sh; hostname; which cuobjdump nvdisasm
WP_LIST=1 WP_KERNEL=flash_fwd WP_LAUNCH=1 WP_OUT=/tmp LD_PRELOAD=$WARP_PATH_SO python remote/fa_bench.py --once --impl fa2 --b 1 --h 2 --s 1024 2>&1 | grep -v "Warning" | tail -30
