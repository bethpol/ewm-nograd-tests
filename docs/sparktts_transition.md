# Spark-TTS retokenization transition matrix

`evals/sparktts_transition.py` generates ordinary, unwatermarked Spark-TTS speech,
decodes it to waveform, and re-encodes it with the same BiCodec. Original and
recovered semantic IDs are paired at the same token index. No alignment search
or audio attack is applied. Pairs include unchanged tokens on the diagonal.

## Generate data and create the figure

```bash
conda activate wmar_audio
python -m evals.sparktts_transition generate \
    --num_samples 64 --output_dir outputs/sparktts_transition_64
```

Defaults use 32 varied texts in `data/sparktts_channel_prompts.txt`, reference
recordings/transcripts in `data/audio_prompts`, a new ordinary sampling seed per
clip, temperature 1, top-k 50, and top-p 0.95. The 64-clip run repeats each text
twice. Source references are cycled, and all assignments and settings are saved.
The generated marginal is the ordinary model distribution; this experiment does
not apply Maryland bias or the keyed sampler.

Outputs in the selected directory:

| File | Meaning |
| --- | --- |
| `transition_scatter.png` | High-resolution scatter plot of all observed pairs |
| `transition_scatter.pdf` | Vector version of the same figure |
| `transition_counts.npz` | Sparse integer counts N[v, v_prime], including matches |
| `transition_empirical.npz` | Sparse empirical row probabilities; unseen rows are zero |
| `transition_matrix.npy` | Dense float32, smoothed, row-stochastic detector matrix |
| `input_token_counts.npy` | Number of observed transitions from each input token |
| `transition_metadata.json` | Orientation, smoothing, coverage, length differences, summary |
| `generation_config.json` | Prompt list, references, generation settings, base seed |
| `samples.csv` | Per-clip token lengths, paired counts, matches, and unpaired tails |
| `samples/sample_XXXX/` | WAV, original/recovered token files, and clip metadata |

The plot's vertical axis is the original token ID; the horizontal axis is the
recovered token ID. Every pair with a positive count gets a dot, including
one-occurrence pairs. Dot size is constant and large enough to remain visible;
color represents the observed count on a logarithmic scale. Smoothing never adds
dots to this plot. The dotted diagonal marks perfect recovery.

Rebuild matrices and plots without generating again:

```bash
python -m evals.sparktts_transition build \
    --samples_dir outputs/sparktts_transition_64/samples \
    --output_dir outputs/sparktts_transition_64 --beta 0.0001
```

## Use the matrix in a detector

The convention is **row = generated token v, column = recovered token v_prime**.

```python
import numpy as np
from scipy import sparse

root = "outputs/sparktts_transition_64"
P = np.load(f"{root}/transition_matrix.npy", mmap_mode="r")
N = sparse.load_npz(f"{root}/transition_counts.npz")
usage = np.load(f"{root}/input_token_counts.npy")

# P[v, y] estimates Pr(recovered token y | generated token v).
# p0_t is the matched semantic-token distribution for this position.
column = np.asarray(P[:, recovered_id], dtype=np.float64)
q_noisy = p0_t @ column
e_value = column[keyed_sample_id] / q_noisy
```

Here `keyed_sample_id` means the reconstructed **semantic token** S_t, not the
recording's public nonce. This snippet supplies the channel-based factor from
the draft; it does not implement a full sequential detector or establish its
conditional-null assumptions.

The smoothing rule is `(N[v,y] + beta) / (N[v] + K*beta)`, with K=8192 and default
beta=0.0001. Every dense row sums to one, and all entries are positive. For an
unobserved input token, this rule returns the uniform distribution. That row is
a prior assumption, not an empirically learned channel. Keep the usage counts
and raw counts with the matrix so future experiments can inspect uncertainty or
change the regularizer. The dense matrix takes about 256 MiB; counts and
empirical probabilities are stored sparsely to avoid extra dense copies.

## Interpretation and limits

This is a generated-speech estimate, not complete coverage of all 8192 inputs.
Metadata reports observed input-token coverage and the number of aligned pairs.
Reference voices, prompts, generation settings, and the codec affect the estimate.
Use separate held-out generations to assess the eventual detector.

If lengths differ, only the shared prefix supplies pairs. Unpaired tails are
counted separately. Positions are never shifted to improve agreement. This
follows the initial index-alignment assumption; it cannot distinguish codec
substitution from any real internal alignment drift.

For this checkpoint there is also a predictable boundary length difference.
The decoder emits 320 waveform samples per semantic token. Wav2Vec2's convolution
stack has an effective receptive field of 400 samples and stride 320 without
boundary padding. Therefore a decoded waveform of length 320*T has
`floor((320*T - 400)/320) + 1 = T - 1` recovered frames. The first clips match this
prediction exactly. One fewer frame is not evidence of an internal token
deletion; equal index correspondence remains the experiment's stated assumption.
No padding is introduced to force equal lengths.

The earlier `clustering.py` confusion builder omits diagonal matches by design.
Those matrices should not be used as full transition counts for this detector.
This experiment explicitly counts both matches and mismatches.

## Background collection: 4,096 new clips

The larger run is launched in a detached tmux session:

```bash
tmux -L ewm-channel new-session -d -s sparktts-channel-4096 -c "$PWD" \
    'exec bash scripts/run_sparktts_transition_background.sh'
```

The script uses the `wmar_audio` Conda interpreter by default; `PYTHON` can
override its path. It collects 4,096 new clips using seeds starting at 20270000,
with the same 32 texts and reference-voice pool. The earlier 64 clips are held
out of fitting and used for predictive evaluation. Thus this tests improved
estimation under the same text/reference distribution, not transfer to unseen
texts or voices.

All files are under `outputs/sparktts_transition_4096/`:

- `run.log`: persistent generation and checkpoint log.
- `progress.json`: current status, completed/target clips, process ID, and newest
  completed checkpoint. It is replaced atomically after every generated clip.
- `latest_checkpoint.json`: paths and predictive score of the newest complete
  checkpoint, available after 64 new clips.
- `checkpoint_history.jsonl`: successive coverage and held-out log-likelihood
  measurements. Higher mean held-out log probability means better prediction;
  improvement is an empirical question, not guaranteed.
- `checkpoints/clips_00064/`, `checkpoints/clips_00256/`, etc.: complete plots,
  counts, smoothed matrices, and held-out evaluation. Checkpoints continue every
  256 clips through 4096. Each dense matrix is about 256 MiB.
- `samples/`: completed audio, semantic token pairs, and generation metadata.
- `exit_code`: `running` while active, `0` on success, or a nonzero failure code.
- Final plot/matrix and `heldout_evaluation.json` appear directly in this output
  directory when collection completes.

Monitor from the repository root:

```bash
cat outputs/sparktts_transition_4096/progress.json
tail -f outputs/sparktts_transition_4096/run.log
tmux -L ewm-channel list-sessions
```

To stop gracefully, send Ctrl-C to the job:

```bash
tmux -L ewm-channel send-keys -t sparktts-channel-4096 C-c
```

The launcher uses a file lock to prevent duplicate collectors. To resume after
the tmux session exits, repeat the launch command. Completed clips are retained;
an incomplete clip is regenerated. Changes to sampling settings, texts,
reference assignments/transcripts, or the seed are rejected on resume. The
detached session can continue after this assistant turn ends, but the machine
and WSL session must remain running.

Spark-TTS can occasionally terminate without generating any semantic tokens.
The collector retries this specific failure up to three total attempts, using
seed offsets of 10,000,000 and 20,000,000 for the additional attempts. Failures
are recorded in `generation_retries.jsonl`; each successful clip's metadata
records its actual seed and retry count. Such empty outputs supply no token
pairs to the estimate. Other errors still stop the job and are reported in
`progress.json` and `run.log`.

## Visualizing concentration and predictive usefulness

```bash
python -m evals.plot_channel_diagnostics
```

This writes `diagnostics/channel_diagnostics.png`, a PDF, and `summary.json`
inside the large run directory. It uses the newest completed checkpoint and the
original 64 held-out clips. It does not regenerate audio or change the channel.

The upper panels rank destinations separately for each input using training
counts, and plot cumulative top-k mass on training and held-out pairs. The right
panel removes diagonal matches before ranking so copying cannot conceal a
diffuse mismatch distribution. Mass is weighted by token frequency. Ties in
training counts are broken by output ID. Unobserved destinations receive no
top-k credit even when k exceeds the number seen in training.

The lower-left panel shows observed support against observations per input.
Small observed support with few observations is not strong evidence of true
sparsity. The lower-right panel compares held-out log loss for the full smoothed
matrix, a copy-plus-pooled-errors model, and an output marginal ignoring input.
The second model copies at the training recovery rate; otherwise its pooled
error distribution is conditioned on the output differing from the input.

Lower log loss means better held-out prediction. If the smaller model wins, the
full matrix may need more data or stronger shrinkage. Both are noisy-channel
models; neither requires a small number of mismatch destinations. These are
predictive-fit checks, not watermark-power measurements or proof of stationarity.
