#!/usr/bin/env bash
# Finite prefix-ASR feasibility experiment; no watermark embedding/hashing.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/rolling_stt_pilot"
mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || exit 1
trap 'printf "%s\n" "$?" > "$OUT/exit_code"' EXIT
printf 'running\n' > "$OUT/exit_code"
printf '%s\n' "$$" > "$OUT/runner.pid"
export HF_HOME="$ROOT/.cache/huggingface"
export MPLCONFIGDIR=/tmp/ewm-matplotlib
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-/home/cusgadmin/miniforge3/envs/wmar_audio/bin/python}"
"$PYTHON" -u -m evals.rolling_stt --clips 4 --output_dir "$OUT" \
    >> "$OUT/run.log" 2>&1
