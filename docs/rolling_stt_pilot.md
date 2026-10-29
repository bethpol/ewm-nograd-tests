# Rolling speech-to-text synchronization pilot

This is a bounded feasibility experiment. It does not embed a watermark, hash
words, assert independent keyed randomness, or establish anytime validity.

## Experimental grid

- Backend: faster-whisper 1.2.1, `small.en`, CPU int8 with four threads.
- Clips: four of the longest saved Spark-TTS clips, with distinct source texts.
  The source corpus contains only short utterances (at most about 5.5 seconds).
- Updates: every 200 ms or 500 ms on the same recording-time origin.
- Commitment: agreement across two or three successive prefix recognitions.
- Word context: most recent one or two committed words.
- Safety rules: withhold the trailing word; require its estimated ending to be
  at least 200 ms behind the available audio; activate on the token grid at or
  after confirmation plus 100 ms.
- Token grid: the existing 320-sample/20 ms Spark-TTS grid, never restarted at
  a word boundary. Absolute token positions remain the indexing convention.

The model is decoded deterministically with beam size one, temperature zero,
fixed English language, no VAD, no target text or initial prompt, and no text
carried from a preceding recognition call. The full available prefix is decoded
again at each update. This is prefix simulation, not a native streaming model.

## Four audio paths

1. `generation_prefix`: decode only the original semantic-token prefix available
   at each update, using the original reference speaker tokens. Later generated
   semantic tokens are not passed to the codec. Extending the decoded sequence
   may change earlier audio; this is measured explicitly.
2. `detection_clean`: growing prefixes of the finished clean waveform.
3. `detection_noise`: growing prefixes with Gaussian noise of waveform standard
   deviation 0.001, using the repository's `NoiseInjection` transform.
4. `detection_lowpass`: growing prefixes filtered at 3000 Hz using the repo's
   causal biquad `LowpassFilter` transform.

The three detection paths are compared to the generation-prefix path. This
separates generation-side prefix-decoder effects from additional degradation.
These are replays of already-generated tokens, not a live feedback loop in which
new watermark seeds influence later generation. Prefix preparation uses the GPU;
ASR uses CPU to avoid adding new CUDA/cuDNN requirements.

Noise samples are generated once from a fixed seed. Filtering the full waveform
is safe for prefix simulation here because the selected filter is causal. No
speed, crop, inserted silence, or clock-recovery experiment is included yet.

## What counts as a committed word?

`watermark/word_anchors.py` contains the model-independent policy. Each partial
word is normalized with Unicode NFKC, case folding, punctuation removal, and
apostrophe normalization. No fuzzy matching, spelling repair, or synonyms are
used. The normalized common prefix of the last N eligible hypotheses is the
candidate commitment. Eligibility excludes each hypothesis's trailing word.

A committed word is immutable. If a later partial transcript contradicts that
prefix, the discrepancy is logged and further commitments are blocked until
the committed prefix agrees again. This deliberately conservative starting
policy may get stuck after a wrong early commitment; the logs will show whether
that makes it unsuitable. It is not a guarantee that committed words are true.

Each event records the estimated acoustic start/end, the audio-availability time
at confirmation, and the future activation index. Multiple words committed in
one update share an activation index; the latest context is active there.

We never use the known Spark-TTS script to bias recognition. A complete-audio
transcript is recorded only as `final_audit.json`. It is never passed through
the commitment policy, and no forced EOF flush contributes live anchors.

Activation timing is expressed in available-audio time. ASR compute time is
logged separately. The experiment assumes generation can pause for recognition;
it does not claim this CPU setup meets real-time latency requirements.

## Measurements

Raw partial transcripts and timestamps are saved before applying any policy, so
the two update cadences, agreement counts, and word-context lengths reuse the
same recognition outputs. Event schedules are then compared at every original
semantic-token index, without resetting a local counter or searching for shifts.

`active_context_agreement` uses positions where at least one side has a word
anchor. Initial positions where both sides have no anchor do not inflate the
score. Anchor coverage is reported separately. No-active-anchor cases have a
null score rather than perfect agreement. The figure represents these as zero
height; consult coverage and the JSON score to distinguish them from failures.

Per-clip measurements include committed word counts, contradictions, active
coverage, and activation-index differences for text-matched committed words.
The latter uses offline sequence matching only to measure error; it never
changes either side's live commitment decisions. Word timestamps themselves
remain available in every raw hypothesis and commit file.

Short clips may contain few useful anchors. Low coverage is a feasibility
finding, not evidence that longer continuous speech cannot work. A later phase
should include longer utterances, more speakers, and actual closed-loop
generation before choosing a final hash protocol.

## Run and inspect

Optional dependencies are in `requirements-stt.txt`. Model files are stored in
`checkpoints/stt/faster-whisper-small.en/`.

```bash
conda activate wmar_audio
python -m evals.rolling_stt --clips 4
```

For the detached job:

```bash
tmux -L ewm-channel new-session -d -s rolling-stt-pilot -c "$PWD" \
    'exec bash scripts/run_rolling_stt_pilot.sh'
```

All output is under `outputs/rolling_stt_pilot/`:

| Path | Contents |
| --- | --- |
| `progress.json`, `run.log`, `exit_code` | Live status, persistent log, exit status |
| `config.json`, `manifest.json` | Parameters, package versions, source clip identities |
| `clips/<clip>/preparation.json` | Update schedule, token count, prefix-vs-final waveform differences |
| `clips/<clip>/generation_prefix/*.npy` | Waveform decoded from each available token prefix |
| `clips/<clip>/recognition/<arm>/samples_*.json` | Every causal-prefix hypothesis, word timestamps, compute time |
| `clips/<clip>/recognition/<arm>/final_audit.json` | Separate full-audio transcript, excluded from live commits |
| `clips/<clip>/commits_*.json` | Word confirmation and activation events plus contradictions |
| `per_clip.csv`, `summary.json` | Per-clip metrics and aggregated comparisons |
| `word_agreement.png`, `.pdf` | Context agreement by recognition/commitment policy |

Recognition outputs are cached individually and reused if restarted with the
same configuration. A file lock prevents duplicate detached jobs.

```bash
cat outputs/rolling_stt_pilot/progress.json
tail -f outputs/rolling_stt_pilot/run.log
tmux -L ewm-channel send-keys -t rolling-stt-pilot C-c
```
