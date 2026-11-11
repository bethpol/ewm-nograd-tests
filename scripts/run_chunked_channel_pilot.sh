#!/bin/bash
#SBATCH --job-name=ewm-chunks
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:55:00
#SBATCH --output=chunked_channel_%j.out
#SBATCH --error=chunked_channel_%j.err

set -eo pipefail
module load python/miniforge3_pytorch
set -u
cd "${SLURM_SUBMIT_DIR:?}"
export EWM_RUN="$PWD/outputs/chunked_channel_${SLURM_JOB_ID}"
export EWM_SAMPLES="$PWD/outputs/grouped_pilot_3487318/channel_training/samples"
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$EWM_RUN"
trap 'echo failed > "$EWM_RUN/status"' ERR
echo running > "$EWM_RUN/status"
echo "Job $SLURM_JOB_ID on $(hostname); results: $EWM_RUN"
nvidia-smi
srun "$PWD/.venv/bin/python" -m evals.chunked_channel_pilot
