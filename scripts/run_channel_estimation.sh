#!/bin/bash
#SBATCH --job-name=ewm-channel-fit
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=channel_fit_%j.out
#SBATCH --error=channel_fit_%j.err

set -eo pipefail
module load python/miniforge3_pytorch
set -u
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit using sbatch}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/ewm-channel-mpl-${SLURM_JOB_ID}"
EWM_OUTPUT="${EWM_OUTPUT:-$PWD/outputs/channel_fit_${SLURM_JOB_ID}}"
EWM_TRAINING_CLIPS="${EWM_TRAINING_CLIPS:-512}"
EWM_VALIDATION_CLIPS="${EWM_VALIDATION_CLIPS:-128}"
EWM_BOOTSTRAP="${EWM_BOOTSTRAP:-300}"
mkdir -p "$EWM_OUTPUT"
exec 9>"$EWM_OUTPUT/run.lock"
flock -n 9 || { echo "Another channel collector holds the output lock" >&2; exit 1; }
trap 'echo failed > "$EWM_OUTPUT/status"' ERR
echo running > "$EWM_OUTPUT/status"
echo "Channel-estimation job ${SLURM_JOB_ID}; output ${EWM_OUTPUT}"
nvidia-smi
srun "$PWD/.venv/bin/python" -m evals.channel_estimation collect \
    --output_dir "$EWM_OUTPUT" \
    --training_clips "$EWM_TRAINING_CLIPS" \
    --validation_clips "$EWM_VALIDATION_CLIPS" \
    --bootstrap "$EWM_BOOTSTRAP"
echo complete > "$EWM_OUTPUT/status"
