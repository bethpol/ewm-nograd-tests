# Generate speech, step by step

Run all commands below from the repository root:

```bash
cd ~/ewm-nograd-tests
```

## Run both models

To generate one sample from **both models** with the included reference recording
and transcript, run:

```bash
.venv/bin/python cleancode/generate_samples.py
```

Outputs are `cleancode/samples/sparktts.wav` and
`cleancode/samples/cosyvoice3.wav`. Override the sentence with `--text "..."`,
or override the CosyVoice3 checkpoint location with
`--cosyvoice3-model-dir /path/to/Fun-CosyVoice3-0.5B`.
Each model runs in a separate process to free memory between runs. If one fails,
the runner still tries the other and exits with an error status.

Start with `generate_audio.py`: two generation functions and one command-line
entry point. It uses the model source already in `models/`, without importing
the experiment runners in `evals/`. Watermarking is disabled.

Use the project's Python environment with the dependencies in the root
`requirements.txt`. Run these commands from the repository root. Supply a real,
clean recording of one speaker and its exact transcript; a few seconds of speech
is a useful starting point. Both models load/resample the reference internally.
The target text is what the model should say: `Say hello` synthesizes those words.

## Run Spark-TTS only

This uses the included reference recording and its paired transcript:

```bash
.venv/bin/python cleancode/generate_audio.py \
  --model sparktts \
  --text "Hello. This is speech generated with Spark TTS." \
  --reference-audio data/audio_prompts/prompt_00000.wav \
  --reference-text "$(cat data/audio_prompts/prompt_00000.txt)" \
  --output cleancode/samples/sparktts.wav
```

## Run CosyVoice3 only

```bash
.venv/bin/python cleancode/generate_audio.py \
  --model cosyvoice3 \
  --text "Hello. This is speech generated with CosyVoice three." \
  --reference-audio data/audio_prompts/prompt_00000.wav \
  --reference-text "$(cat data/audio_prompts/prompt_00000.txt)" \
  --output cleancode/samples/cosyvoice3.wav
```

## How to write your own command

Choose `--model sparktts` or `--model cosyvoice3`, then set these arguments:

| Argument | What to supply |
| --- | --- |
| `--text "..."` | The words you want in the generated audio. |
| `--reference-audio path.wav` | A recording of the voice you want to use. |
| `--reference-text "..."` | The exact words spoken in that recording. |
| `--output path.wav` | Where to save the new audio; parent folders are created automatically. |
| `--model-dir path` | Optional: override the selected model's checkpoint directory. |

`"$(cat data/audio_prompts/prompt_00000.txt)"` tells the shell to read the
transcript from that file and pass it as one argument. You can replace it with
literal quoted text, for example `--reference-text "Today is a beautiful day."`,
if those are the words in your recording. Keep the target text and transcript
in quotes so that spaces stay within one argument.

Each trailing `\` continues the command onto the next line. It must be the last
character on that line, with no spaces after it. The final line has no `\`.
Using an existing output filename overwrites that WAV; choose a new filename to
keep earlier samples. Running one model only does not rerun the other.

The default checkpoint directories are
`checkpoints/sparktts/Spark-TTS-0.5B` and
`checkpoints/cosyvoice/Fun-CosyVoice3-0.5B`; both are now present in this workspace.
On a fresh checkout, download them with:

```bash
.venv/bin/python scripts/download_models.py --targets sparktts cosyvoice --checkpoints_dir checkpoints
```

The generation scripts themselves do not download checkpoints.

## CPU and GPU execution

Model defaults select CUDA if available, otherwise CPU. To select a GPU, set
`CUDA_VISIBLE_DEVICES` before launching; to force CPU, set it to an empty string.
Sampling means repeat runs can produce different audio.

For CPU runs, you can limit PyTorch's CPU threads, for example
`OMP_NUM_THREADS=4 .venv/bin/python cleancode/generate_samples.py`.
CosyVoice's reference loader calls SoundFile directly so that it also works with
newer torchaudio releases that require TorchCodec for `torchaudio.load()`.

## Grouped watermark experiment

The third script, `watermark_audio.py`, implements the first matched-model stage
from `draft.tex` for **both Spark-TTS and CosyVoice3**. Each model has its own
partition, calibration clips, channel estimate, evaluation clips and plots.
Use one command per model:

```bash
OMP_NUM_THREADS=4 .venv/bin/python cleancode/watermark_audio.py \
  --model sparktts --num-groups 4 --calibration-samples 3 --samples 3

OMP_NUM_THREADS=4 .venv/bin/python cleancode/watermark_audio.py \
  --model cosyvoice3 --num-groups 4 --calibration-samples 3 --samples 3
```

`--samples 3` generates three watermarked and three genuinely unwatermarked
evaluation clips, in addition to the three unwatermarked calibration clips.
The default sentence and reference are supplied for you. Change the sentence
with `--text "..."`, and use `--reference-audio` plus `--reference-text-file` to
use your own voice. These flags take paths to the recording and its transcript.
CosyVoice's tokenizer currently requires each recording to be at most 30 seconds.
Keep the sentence short. `--max-tokens` caps speech generation (default 600);
reaching the cap can truncate the spoken sentence.

Results go into separate folders:

```text
cleancode/watermark_samples/sparktts/
cleancode/watermark_samples/cosyvoice3/
    audio/calibration/     WAVs used ONLY to learn this model's transition matrix
    audio/watermarked/     held-out watermarked evaluation WAVs
    audio/unwatermarked/   held-out ordinary evaluation WAVs
    samples.csv            audio inventory with roles, text, reference, and lengths
    groups.npy             token ID -> group ID
    manifest.json          experiment settings and token lengths
    channel.npz            group counts and smoothed channel probabilities
    channel_training.json  exact calibration WAVs and pair counts used by the fit
    *.npz                  original/recovered tokens and per-step group masses
    *.key                  independent experimental secret key for each clip
    *.evidence.npz         cumulative log evidence for each evaluation clip
    detection.json         final evidence and first threshold crossing per clip
    log_e.png / log_e.pdf   mean log e-process with error bars
    log_e.csv              plotted means, sample counts, and standard errors
```

For a new run or group count, choose a **new** `--output-dir`, for example
`--output-dir cleancode/watermark_samples/sparktts_groups8 --num-groups 8`.
Generation refuses to overwrite a run to prevent mixing partitions or channels.
The initial small trials used four groups, two calibration clips and two
evaluation clips per condition, with a 180-token generation cap. Channel
calibration has since been expanded to ten non-replayed clips per model, while
keeping those same evaluation clips. Original fits/plots are in `two_clip_baseline/`.
These are short pipeline checks; `hit_token_cap` in the inventory flags clips
whose speech may have been cut short. New experiments can use the default
600-token cap and more clips.
All experiment outputs must stay inside `cleancode/`; the script rejects an
output directory elsewhere. Its tests are also siloed in `cleancode/tests/`:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m unittest discover -s cleancode/tests -v
```

`samples.csv` labels every calibration file with `used_for_channel_fit=True`.
It records the relative WAV path, model, transcript, reference file paths, audio
duration/sample rate, original/recovered token lengths and aligned pair count.
The inventory and manifest are updated after each clip. WAVs stay available for
listening after channel fitting; they are never removed by later stages.
Keys are stored locally with owner-only permissions. They are experimental
keys, not suitable for publishing alongside a deployed detector.

### Run the stages separately

Run generation once, then inspect or rerun later stages without loading the TTS
model. The following example uses the default Spark output folder:

```bash
# 1. Generate calibration, watermarked and unwatermarked clips; re-encode WAVs.
OMP_NUM_THREADS=4 .venv/bin/python cleancode/watermark_audio.py \
  --model sparktts --stage generate --num-groups 4 --calibration-samples 3 --samples 3

# 2. Fit only on calibration clips. beta is a pseudocount per group transition.
.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage fit --beta 0.5

# 3. Score held-out evaluation clips with the fitted channel.
.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage detect --alpha 0.01

# 4. Plot saved evidence paths.
.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage plot
```

Replace `--model sparktts` with `--model cosyvoice3` for its separate stages.
If you chose a custom `--output-dir`, pass it at every stage. Later stages read
the saved partition rather than using a new `--num-groups` value.

### Improve an existing channel estimate

Append calibration clips without regenerating any evaluation WAVs or keys:

```bash
# Add eight calibration clips to the existing run (two becomes ten).
OMP_NUM_THREADS=4 .venv/bin/python cleancode/watermark_audio.py \
  --model sparktts --stage calibrate --calibration-samples 8

.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage fit
.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage detect
.venv/bin/python cleancode/watermark_audio.py --model sparktts --stage plot
```

Repeat with `--model cosyvoice3` for its separate calibration and channel.
The `calibrate` stage restores the original partition, target text, reference
voice and token cap from the manifest. It generates ordinary unwatermarked
speech, appends numbered WAVs in `audio/calibration/`, and updates the inventory.
It does not refit automatically: run `fit`, `detect`, and `plot` afterward.

For the first expansion, the original estimates and plots are preserved in
`two_clip_baseline/` inside each model's folder. Compare the two estimates on
the **same held-out evaluation clips** with:

```bash
.venv/bin/python cleancode/channel_diagnostics.py --model sparktts
.venv/bin/python cleancode/channel_diagnostics.py --model cosyvoice3
```

This saves `channel_estimates.png/.pdf` (matrix heatmaps and source-row counts),
`channel_comparison.png/.pdf` (mean log evidence before/after, ±1 SEM), and
`channel_comparison.json` (per-clip crossing times, group agreement, anchor
entropy and channel log probabilities). This helper uses the default model
folders and requires their saved `two_clip_baseline/` snapshots.

CosyVoice's checkpoint loader resets the PyTorch RNG. The script reapplies the
batch seed **after loading** so added calibration does not replay earlier clips.
During the initial expansion, two confirmed replayed clips were retained in
`audio/calibration_excluded/`, with `used_for_channel_fit=False` and an exclusion
reason in `samples.csv`. Two replacement clips bring the fitted channel to ten
non-replayed calibration clips. Excluded files never contribute to channel counts.

More calibration reduces estimation noise, but need not increase detection
speed. Retokenization loses group information, and a stationary group channel
can differ across speech contexts. Low entropy in the per-step group distribution
also limits the amount of watermark evidence available. These plots diagnose
the effect of more calibration; they do not tune a channel on evaluation data.

### What is being embedded and detected?

1. Partition the model's **speech-token** vocabulary into `M = --num-groups`
   groups. This initial baseline uses a seeded random partition balanced by
   token count. It does not yet optimize channel mutual information or balance
   the model's probability mass across groups.
2. At speech position `t`, sum the model's token probabilities within each group
   to obtain `pi_t`. Compute `U_t = HMAC(key, t)` as a uniform number. Choose
   `A_t = inverse_CDF(pi_t, U_t)`, then sample a token from the model conditional
   on belonging to `A_t`. The key determines the group, not the token inside it.
   Averaged over fresh ideal keyed uniforms, this preserves the filtered token
   distribution. EOS/non-speech decisions remain ordinary categorical draws.
3. Decode to audio, save PCM-16 WAVs, and encode them again with the **same
   model's** speech tokenizer. Compare original and recovered tokens at direct
   indices. No alignment, offset search, STT, resynchronization or attacks.
4. On separate calibration clips, count original group `a` -> recovered group
   `b`. Estimate `C[a,b] = (count[a,b] + beta) / (row_count[a] + M*beta)`.
   Positive smoothing avoids infinite negative log evidence on unseen transitions.
   A few groups reduce the matrix to `M*M` entries, but sample sufficiency depends
   on row coverage and channel stability. The script prints calibration counts.
5. On held-out clips, reconstruct `A_t` from the key and saved `pi_t`. If `B_t`
   is the recovered group, accumulate natural logarithms of
   `e_t = C[A_t,B_t] / sum_a pi_t[a]*C[a,B_t]`, as in the draft.
   Detection occurs at the first crossing of `log(1/alpha)`; logs start at zero.

Both adapters use explicit temperature 0.8, top-k 50, and top-p 0.95 filtering.
For CosyVoice3 this **replaces its stock repetition-aware sampling rule** in all
three conditions. The preservation statement refers to this chosen filtered
categorical distribution. Calibration uses the same sampling settings as testing.

The plot shows both recovered-audio evidence (solid) and original-token evidence
(dashed), for watermarked and unwatermarked clips. Original-token curves use the
same fitted channel; they are a diagnostic for codec loss, not a separately
optimized identity-channel detector. Each curve is an arithmetic mean of
**cumulative log e**, not the log of mean e. Error bars are **±1 standard error
across independent clips**. Curves stop at the shortest clip in each condition
to keep a fixed cohort; they do not pad missing tokens or average only survivors.
With one clip, a standard error cannot be estimated and bars are omitted.

### Limits of this first stage

Detection currently needs the saved per-position model group probabilities.
It is a generation-trace-assisted experiment, not an audio-only detector.
The key input is only the zero-based speech-token position; reference tokens
do not count. Each clip uses a fresh key to avoid reusing the same random stream
across replications. Grouping and the channel are frozen before evaluation.
Direct index comparisons score only the shared prefix; missing/extra token counts
are reported. Insertions or deletions can therefore destroy the signal.

Under the ideal fresh-uniform null, with each current seed independent of the
null observation and its predictable anchor, the factors have conditional mean
one: `sum_a pi_t[a]*e_t(a,b) = 1` for every observed `b`. HMAC approximates that
ideal stream. This baseline does not establish validity conditional on a fixed
key, handle arbitrary temporal alignment errors, or prove that the fitted
stationary channel describes every context. Small-sample error bars are
descriptive. The channel fit affects detection power; its counts should be
inspected before making conclusions from a small calibration set.

## What happens under each call

| Stage | Spark-TTS | CosyVoice3 |
| --- | --- | --- |
| Load | Text tokenizer, causal LLM, BiCodec | Text/audio frontend, LLM, flow model, HiFT vocoder |
| Reference | BiCodec global voice tokens and semantic speech tokens | Speech tokens, speaker embedding, mel features |
| Generate | LLM predicts new semantic tokens conditioned on text and reference | LLM predicts speech tokens conditioned on text and reference |
| Decode | BiCodec converts new semantic tokens + reference global tokens to audio | Flow converts tokens to mel features; HiFT converts features to audio |
| Save | Mono WAV at model sample rate | Concatenate every yielded piece; mono WAV at model sample rate |

Speech tokens are discrete model symbols, not waveform samples. Generation
predicts these symbols first; the audio decoder turns them into audible speech.
The reference transcript describes the input recording; the target text describes
the new output. CosyVoice3 also needs `<|endofprompt|>` between its instruction
and reference transcript; the script inserts it for you.

To follow the actual implementation, open these files in order:

1. `cleancode/generate_audio.py`: the complete entry point for this example.
2. `models/Spark-TTS/cli/SparkTTS.py`: `process_prompt()` and `inference()`.
3. `models/Spark-TTS/sparktts/models/audio_tokenizer.py`: `tokenize()` / `detokenize()`.
4. `models/CosyVoice/cosyvoice/cli/cosyvoice.py`: `CosyVoice3.__init__()` and inherited `inference_zero_shot()`.
5. `models/CosyVoice/cosyvoice/cli/frontend.py`: `frontend_zero_shot()`.
6. `models/CosyVoice/cosyvoice/cli/model.py`: `tts()`, `llm_job()`, and `token2wav()` across the model classes.

The existing `scripts/demo_*.py` instead launch `evals/wm_*.py`, which wrap
synthesis with watermark sampling, clustering options, audio transformations,
re-tokenization and detection scores. In those experiments, `base` still refers
to a watermark configuration. This example isolates ordinary synthesis first.

## Random grouping sweep (2026-11-10)

`random_group_sweep.py` runs Spark-TTS and CosyVoice3 with 2, 4, 8, and 16
balanced random groups. Each configuration uses seed 42, a fixed partition,
10 independent calibration clips, 20 watermarked clips and 20 unwatermarked
controls, a 600-token cap, beta 0.5, and alpha 0.01. A trial is a fresh clip/key,
not a newly drawn partition. Text and voice follow `watermark_audio.py` defaults.
This measures clip variability conditional on one random partition per group
count; it does not estimate variability over randomly redrawn partitions.

The submitted array is job **3573638**, task order Spark-TTS 2/4/8/16 followed
by CosyVoice3 2/4/8/16. Outputs are under
`cleancode/watermark_samples/random_groups_20261110/<model>_groups<M>/`.
Logs are `cleancode/random_groups_3573638_<task>.out/.err`.

To submit a new sweep, choose a new output directory:

```bash
sbatch --export=ALL,SWEEP_OUTPUT="$PWD/cleancode/watermark_samples/random_groups_NEW" cleancode/run_random_group_sweep.sh
```

After completion, regenerate the aggregate plot with:

```bash
.venv/bin/python cleancode/random_group_sweep.py \
  --output-dir cleancode/watermark_samples/random_groups_20261110 --summarize
```

This writes `detection_rates.png`, `.pdf`, and `.csv` in the sweep directory.
Rates count any threshold crossing within each clip's shared token prefix,
with 95% Wilson binomial intervals. Both recovered-audio and original-token
diagnostics are shown, along with unwatermarked false positives. CSV rows
include trial counts and generation-cap counts. Per-configuration cumulative
log-e plots and all audio/traces remain available. Summary runs can show partial
results, but each included condition must contain at least 20 trials. Intervals
are descriptive clip-level intervals, conditional on the fitted channel and
partition; calibration uncertainty is not included.

## Attack-specific channel calibration

After selecting a group count for each model from the clean sweep, run the
attack experiment on a GPU allocation. Pass the completed configuration as
`--source` and choose a fresh output directory. For example, if Spark-TTS
selects eight groups:

```bash
OMP_NUM_THREADS=4 .venv/bin/python cleancode/attack_sweep.py \
  --source cleancode/watermark_samples/random_groups_20261110/sparktts_groups8 \
  --output-dir cleancode/watermark_samples/sparktts_attack_sweep
```

Use `--list-attacks` to see the existing local recovery suite. Repeat
`--attack NAME` to run a subset. Identity provides the clean grouped channel
and its matrix plot. The suite also includes three noise strengths, filtering,
smoothing, speed changes, shifts, and cropping.

Each condition transforms the original saved WAVs and re-encodes them with the
source model. It estimates its own smoothed P_c from transformed calibration
clips using the original group labels and recovered groups at the same indices.
The channel then scores the transformed held-out watermarked and control clips
with the source partition, keys, and generation traces. Attack severity is
known at scoring time. Timing changes can break token correspondence because
the detector compares direct positions.

Each condition saves a channel matrix and heatmap, training counts, transformed
audio, per-clip detections, and average log e-value plots. The output root
contains `attack_rates.csv`, `attack_rates.png`, and `attack_rates.pdf`.
The same clips are used across conditions, so intervals describe each rate
individually. For a confirmatory estimate after choosing K, generate a fresh
source run at the chosen K before running this experiment.
