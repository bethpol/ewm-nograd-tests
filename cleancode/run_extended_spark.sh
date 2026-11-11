#!/bin/bash
#SBATCH --job-name=spark-groups-32-64
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --array=0-5%2
#SBATCH --output=cleancode/extended_spark_%A_%a.out
#SBATCH --error=cleancode/extended_spark_%A_%a.err
set -eo pipefail
module load python/miniforge3_pytorch
set -u
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/extended-spark-${SLURM_JOB_ID}"
exec 9>"outputs/extended_spark_detection_32_64/task_${SLURM_ARRAY_TASK_ID}.lock"
flock -n 9 || exit 1
srun .venv/bin/python -m cleancode.run_extended_spark --task "$SLURM_ARRAY_TASK_ID"
