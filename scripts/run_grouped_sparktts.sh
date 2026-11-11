#!/bin/bash
#SBATCH --job-name=ewm-groups
#SBATCH --account=biry-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=grouped_%j.out
#SBATCH --error=grouped_%j.err

set -euo pipefail
module load python/miniforge3_pytorch
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${MODEL_DIR:?Set MODEL_DIR to the Spark-TTS checkpoint directory}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
nvidia-smi
EWM_PYTHON="${EWM_PYTHON:-python3}"
EWM_GROUP_COUNTS="${EWM_GROUP_COUNTS:-1 2 4 8 16 32}"
EWM_PARTITION_SEEDS="${EWM_PARTITION_SEEDS:-20261029 20261030 20261031}"
EWM_REFERENCE="${EWM_REFERENCE:-data/audio_prompts/prompt_00000.wav}"
EWM_TRANSCRIPT="${EWM_REFERENCE%.wav}.txt"
EWM_PROMPT_TEXT="$(cat "$EWM_TRANSCRIPT")"
for ewm_partition in $EWM_PARTITION_SEEDS; do
    for ewm_groups in $EWM_GROUP_COUNTS; do
        srun "$EWM_PYTHON" -m evals.keyed_sparktts generate \
            --model_dir "$MODEL_DIR" --device cuda \
            --prompt_audio "$EWM_REFERENCE" --prompt_text "$EWM_PROMPT_TEXT" \
            --prompt_file data/tts_prompts.txt --nsamples "${EWM_NSAMPLES:-10}" \
            --num_groups "$ewm_groups" --partition_seed "$ewm_partition" \
            --output_dir "outputs/grouped_audio/partition_${ewm_partition}/groups_${ewm_groups}"
    done
done
