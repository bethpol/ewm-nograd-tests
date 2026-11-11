#!/bin/bash
#SBATCH --job-name=clean-random-groups
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --output=cleancode/random_groups_%A_%a.out
#SBATCH --error=cleancode/random_groups_%A_%a.err
set -eo pipefail
module load python/miniforge3_pytorch
set -u
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/clean-random-mpl-${SLURM_JOB_ID}"
: "${SWEEP_OUTPUT:?}"
srun .venv/bin/python cleancode/random_group_sweep.py --output-dir "$SWEEP_OUTPUT" --task "$SLURM_ARRAY_TASK_ID"
