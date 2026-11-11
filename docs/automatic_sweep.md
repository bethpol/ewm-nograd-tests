# Automatic channel-gated watermark sweep

The automatic successor runs after the channel-collection job terminates. It
first verifies successful collection, full requested split sizes, a final report,
and the original screening thresholds. It recomputes the per-channel screening
decision from numeric diagnostics rather than accepting a candidate flag alone.
Every accepted archive is checked against its partition, counts, bootstrap
intervals, and metadata, then copied and hashed before watermark evaluation.

Only configurations passing **all** support, uncertainty, stability, held-out
gain, and train/validation consistency criteria are used. No thresholds are
relaxed. If collection fails, artifacts are incomplete, or no channel qualifies,
the job exits without loading Spark-TTS or generating watermark audio. The output
contains `gate.json`, `gate.txt`, and `status=blocked_by_channel_quality`.

## Automatic successor

Submit from the repository root, substituting the collection job ID and output:

```bash
sbatch --dependency=afterany:COLLECTION_JOB_ID \
  --export=ALL,EWM_CHANNEL_DIR=/absolute/path/to/channel_collection \
  scripts/run_automatic_sweep.sh
```

`afterany` lets the gate explain a collection failure rather than waiting forever
on an unsatisfied success dependency. The gate still requires success. The
successor reserves one GPU, eight CPUs, 32 GiB and up to four hours. It evaluates
all approved configurations in one process, sharing model weights. It does not
submit further jobs. An output-directory lock prevents concurrent writers.
`EWM_SWEEP_OUTPUT`, `EWM_SWEEP_CLIPS` and `EWM_SWEEP_NULL_CLIPS` override defaults.
To resume, use the same output path and settings; private keys must be retained.

## Watermark and observations

For each approved (group count, partition) pair, generate 16 watermarked clips
and 16 ordinary controls, using the four texts in data/tts_prompts.txt in rotation
and the collection's reference voices in rotation. These are fresh generations
rather than channel-fitting recordings. Maximum generation is 800 model tokens,
with temperature 1, top-k 50, top-p .95. Only clean audio is evaluated initially.

For every semantic position t, calculate U_t = HMAC-SHA256(key, uint64_be(t)),
convert it to an open-interval uniform, draw the group by inverse CDF of the
actual per-position group masses, then sample an ordinary token within that
group. Only semantic tokens are keyed. Nonsemantic/global/control probabilities
remain ordinary. The index is absolute within the recording and does not reset
at chunks. Each recording uses an independent private key; the index is the
**only PRF message**. There is no context, STT, audio feature, token-history,
nonce, or seed extraction function in this rule. Keys are stored with mode 0600
under `.cache/automatic_sweep_keys/`; traces contain no secret key or uniforms.

The detector reconstructs group masses and seeds from saved generation p0 and
the recording key. This is an oracle-distribution replay experiment, not a blind
detector that reconstructs model distributions from recovered audio. Generation
and replay assert exact original-group agreement. Trace version
`sparktts-position-group-cdf-v2` distinguishes direct group CDF from the older
candidate-token-then-group construction.

Each waveform is re-encoded separately for its approved policies, using exactly
the same chunk boundaries, context, preprocessing, and nominal absolute-index
policy as channel estimation. Missing frames are skipped, never compacted. Each
observation contributes once. Log e-factors accumulate over chunks and threshold
checks occur at completed audio arrivals. Paths continue beyond crossing only
for offline diagnostics. Arrival time includes lookahead but excludes encoder
runtime and network delay. This simulates streaming reception of saved audio;
it does not implement streaming TTS generation.

Matched channels are the main comparison. A full-channel mismatch ablation is
included only when that pair's full-audio channel also passes the gate. A rejected
full channel is not silently used as a reference or denominator.

Ordinary controls use ordinary filtered categorical sampling without consulting
the detector key, while still saving p0. Wrong-key replay of watermarked audio
is an additional diagnostic. The bounded control set does not establish a 1%
false-positive rate or replace the stated ideal-randomness/conditional-null
assumptions behind the e-process.

## Artifacts and plots

- `gate.json`, `gate.txt`: approved/rejected policies and reasons.
- `frozen_channels/`: accepted channel/partition archives, metadata, hashes.
- `channel_collection_config.json`, `sweep_config.json`: frozen experiment setup.
- `progress.json`, `status`, `error.json` on failure: job state.
- `samples/mM_seedS/{watermarked,ordinary}/clip_XXXX/`: audio, p0 trace,
  metadata with private-key path, recovered observations with positions and
  arrivals, per-policy encoder timings, evidence paths, results, completion marker.
- `report/per_clip.csv`, `summary.csv`, `summary.json`: scored conditions and
  aggregate detection/censoring results. Conditions with missing channel support
  are absent rather than treated as failed detections.
- Five plot families: streaming detection, grouping versus latency, controls,
  channel matching, and example evidence paths. Each has PNG/PDF versions and a
  matching TXT explaining sources, axes, pooling, censoring, timing, and limits.

Restricted comparisons use the shortest completed recording duration as a
shared horizon. Restricted mean time includes undetected clips at that horizon.
Available partition subsets can differ between policies after screening, so
curves are not automatically balanced comparisons. Report sample counts and
inspect the gate before interpreting differences. Plots show empirical summaries
without confidence bands; partition and ordinary generation variation remain.
