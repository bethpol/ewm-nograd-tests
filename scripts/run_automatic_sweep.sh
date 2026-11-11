#!/bin/bash
#SBATCH --job-name=ewm-auto-sweep
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=automatic_sweep_%j.out
#SBATCH --error=automatic_sweep_%j.err

set -eo pipefail
module load python/miniforge3_pytorch
set -u
cd "${SLURM_SUBMIT_DIR:?}"
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${EWM_CHANNEL_DIR:?Set channel-estimation output directory}"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/ewm-auto-sweep-mpl-${SLURM_JOB_ID}"
EWM_SWEEP_OUTPUT="${EWM_SWEEP_OUTPUT:-$PWD/outputs/automatic_sweep_${SLURM_JOB_ID}}"
mkdir -p "$EWM_SWEEP_OUTPUT"
exec 9>"$EWM_SWEEP_OUTPUT/run.lock"
flock -n 9 || { echo "Another process holds this sweep output lock" >&2; exit 1; }
trap 'echo failed > "$EWM_SWEEP_OUTPUT/status"' ERR
echo "Automatic quality-gated sweep ${SLURM_JOB_ID}; output $EWM_SWEEP_OUTPUT"
srun "$PWD/.venv/bin/python" -m evals.automatic_sweep \
    --channel_dir "$EWM_CHANNEL_DIR" --output_dir "$EWM_SWEEP_OUTPUT" \
    --clips "${EWM_SWEEP_CLIPS:-16}" --null_clips "${EWM_SWEEP_NULL_CLIPS:-16}"
