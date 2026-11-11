#!/bin/bash
#SBATCH --job-name=ewm-pilot
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=grouped_pilot_%j.out
#SBATCH --error=grouped_pilot_%j.err

set -eo pipefail
module load python/miniforge3_pytorch
set -u
: "${SLURM_JOB_ID:?Submit with sbatch from repository root}"
cd "${SLURM_SUBMIT_DIR:?}"
EWM_PYTHON="$PWD/.venv/bin/python"
MODEL_DIR="$PWD/checkpoints/sparktts/Spark-TTS-0.5B"
EWM_RUN="$PWD/outputs/grouped_pilot_${SLURM_JOB_ID}"
mkdir -p "$EWM_RUN"
trap 'echo "failed" > "$EWM_RUN/status"' ERR
echo "running" > "$EWM_RUN/status"
echo "Job $SLURM_JOB_ID on $(hostname); outputs: $EWM_RUN"
nvidia-smi
export OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
EWM_TRAINING_DIR="${EWM_TRAINING_DIR:-$EWM_RUN/channel_training}"
# Reuse an explicitly supplied completed training set when restarting a sweep.
if [ ! -f "$EWM_TRAINING_DIR/transition_counts.npz" ]; then
  srun "$EWM_PYTHON" -m evals.sparktts_transition generate \
    --model_dir "$MODEL_DIR" --device cuda --num_samples 16 --steps 400 \
    --seed 202610290 --output_dir "$EWM_TRAINING_DIR"
fi
echo "$EWM_TRAINING_DIR" > "$EWM_RUN/channel_training_source.txt"
EWM_PROMPT_TEXT="$(cat data/audio_prompts/prompt_00000.txt)"
for ewm_groups in {2..16}; do
  srun "$EWM_PYTHON" -m evals.keyed_sparktts generate \
    --model_dir "$MODEL_DIR" --device cuda --nsamples 4 --steps 400 \
    --prompt_audio data/audio_prompts/prompt_00000.wav --prompt_text "$EWM_PROMPT_TEXT" \
    --num_groups "$ewm_groups" --partition_seed 20261029 \
    --output_dir "$EWM_RUN/audio/partition_20261029/groups_${ewm_groups}"
done
srun "$EWM_PYTHON" -m evals.grouped_audio_report \
  --counts "$EWM_TRAINING_DIR/transition_counts.npz" \
  --audio_dir "$EWM_RUN/audio" --output_dir "$EWM_RUN/report"
echo "complete" > "$EWM_RUN/status"
