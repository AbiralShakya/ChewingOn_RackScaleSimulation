#!/usr/bin/env bash
#SBATCH --job-name=ssm-vs-attn-a100
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100&gpu80&pcie
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/MZHASAN/as0714/ssmstudy_a100/slurm-%j.log
# A100 80GB PCIe (Della, the GPU the A100 calibration was measured on):
# timing sweep, then Nsight Compute + SASS.
set -uo pipefail
cd /scratch/gpfs/MZHASAN/as0714/ssmstudy_a100
module load cudatoolkit/12.8 >/dev/null 2>&1
export PATH=/scratch/gpfs/MZHASAN/as0714/rackcal-ampere/tools:$PATH
. .venv/bin/activate
export TRITON_CACHE_DIR=$PWD/${OUT:-out_a100pcie}/triton_cache
mkdir -p ${OUT:-out_a100pcie}
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv > ${OUT:-out_a100pcie}/gpu.txt
python bench_kernels.py --out ${OUT:-out_a100pcie}/timing.json 2> ${OUT:-out_a100pcie}/timing.log
echo TIMING_DONE
./run_ncu.sh ${OUT:-out_a100pcie}
echo ALL_DONE
