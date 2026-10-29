#!/usr/bin/env bash
# Run inside a detached tmux session. Re-running resumes completed clips.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/sparktts_transition_4096"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
if ! flock -n 9; then
    echo "A Spark-TTS channel run already holds $OUT/run.lock" >&2
    exit 1
fi
if [ -f "$OUT/collection_complete.json" ]; then
    echo "Collection already finalized. See $OUT/collection_complete.json"
    exit 0
fi
printf '%s\n' "$$" > "$OUT/runner.pid"
printf 'running\n' > "$OUT/exit_code"
export HF_HOME="$ROOT/.cache/huggingface"
export MPLCONFIGDIR=/tmp/ewm-matplotlib
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/home/cusgadmin/miniforge3/envs/wmar_audio/bin/python}"
set +e
"$PYTHON" -u -m evals.sparktts_transition generate \
    --num_samples 4096 --seed 20270000 --resume --checkpoint_every 256 \
    --heldout_samples_dir outputs/sparktts_transition_64/samples \
    --output_dir "$OUT" >> "$OUT/run.log" 2>&1
run_status=$?
printf '%s\n' "$run_status" > "$OUT/exit_code"
exit "$run_status"
