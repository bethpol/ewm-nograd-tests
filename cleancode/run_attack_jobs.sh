#!/bin/bash
#SBATCH --job-name=clean-attacks
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=cleancode/attack_jobs_%j.out
#SBATCH --error=cleancode/attack_jobs_%j.err
set -eo pipefail
module load python/miniforge3_pytorch
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/clean-attack-mpl-${SLURM_JOB_ID}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
srun .venv/bin/python cleancode/run_attack_jobs.py
