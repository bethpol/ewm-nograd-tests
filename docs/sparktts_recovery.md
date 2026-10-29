# Bounded Spark-TTS recovery experiments

The initial transition collection is finalized at 1,709 clips and 347,050
aligned token pairs. Its artifacts are at `outputs/sparktts_transition_4096/`
(the directory retains its original target name). `collection_complete.json`
records the actual size. Generation is not resumed beyond that finalized target.

The first robustness batch uses the first 16 clips from the separate 64-clip
held-out set, with no further language-model generation. The runner reuses
`training/augmentations.py` and `SparkTTSTokenizerWrapper` from the existing
Spark-TTS evaluator. It has exactly 176 clip/attack conditions:

- Identity.
- Gaussian noise standard deviations 0.001, 0.01, and 0.05 in waveform units.
- Low-pass at 3000 Hz and high-pass at 1000 Hz.
- Smoothing with a 1 ms averaging window.
- Speed factor 1.1, shifts of 10 and 40 ms, and a random 50% crop.

These strengths are arguments to the existing implementations, not alternative
attack implementations. Noise is not normalized to a fixed SNR. Speed uses the
repo's resampling-based transform, which also affects pitch. Its crop returns a
shorter waveform, despite a misleading zero-padding description in that class.

## Recovery comparisons

`evals/sparktts_recovery.py` reports exact token recovery plus equality after
mapping both streams through fixed groups:

- The shipped selected Leiden map: 4,311 groups.
- A shuffled version of that map, preserving its exact group-size distribution.
- Balanced random partitions into 64 and 256 groups.

Maps are saved in `groupings.npz`. These are recovery diagnostics on ordinary
generated audio. They do not yet implement group-first keyed embedding, a
channel-based sequential detector, or false-positive testing. Fewer groups raise
chance agreement; raw group recovery percentages cannot alone establish better
watermark detection. A fair next experiment must account for the actual
per-position group probabilities and compare evidence growth.

The denominator is the original token count. Missing recovered positions count
as failures; shared-prefix recovery is also reported. Repeated token positions
are included. No alignment search is performed. Speed, shift, and crop results
are separated in the plot because they can violate index correspondence.

The batch explicitly excludes MP3 (FFmpeg is not installed) and neural-codec
attacks (additional packages/weights are not set up). It is a bounded first
suite, not a claim to have run every attack in the repository.

## Run, monitor, and resume

```bash
conda activate wmar_audio
python -m evals.sparktts_recovery --nsamples 16
```

Or launch the same experiment detached:

```bash
tmux -L ewm-channel new-session -d -s sparktts-recovery-16 -c "$PWD" \
    'exec bash scripts/run_sparktts_recovery.sh'
```

All results are in `outputs/sparktts_recovery_16/`:

| File | Meaning |
| --- | --- |
| `progress.json` | Condition count, current attack, completion/failure status |
| `run.log` | Persistent log for the detached job |
| `config.json` | Clip selection, seed, exclusions, and settings |
| `per_clip.csv` | Per-clip/per-attack/per-grouping recovery metrics |
| `summary.csv` | Aggregated counts and token-weighted recovery rates |
| `recovery_comparison.png` / `.pdf` | Final recovery comparison plot |
| `transition_counts/*.npz` | Raw aligned token counts for each attack |
| `samples/<clip>/<attack>/` | Recovered tokens and scores; transformed WAVs for the first two clips |

Each completed condition is saved and reused on a restart with the same config.
The detached launcher holds a lock to avoid duplicate jobs. Intermediate CSVs
are updated after each complete clip. The plot is written when all conditions
finish. No long-running collection continues after this finite batch finishes.
