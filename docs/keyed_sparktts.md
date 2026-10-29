# Keyed Spark-TTS: the first matched-distribution experiment

This implementation follows the agreed starting point: delta = 0, one semantic
stream, ordinary categorical sampling driven by a secret key, and match-count
detection. Token positions are assumed aligned. There is no synchronization
search, learned grouping, channel estimation, or e-value test in this version.

## Run it

From the repository root:

```bash
conda activate wmar_audio
python -m evals.keyed_sparktts generate \
    --prompt_audio data/audio_prompts/prompt_00000.wav \
    --prompt_text "Describe the most effective ways to reduce stress and anxiety in today's fast-paced world." \
    --nsamples 1 --steps 800 \
    --output_dir outputs/keyed_sparktts
```

The default key file is `.cache/keyed_sparktts/secret.key`. It is created on the
first generation run with 32 random bytes and reused on later runs. `--key_file`
selects another file. Alternatively, `EWM_SECRET_KEY` supplies a nonempty UTF-8
key through the environment and overrides the file. Neither key value is written
to traces, metadata, results, or command output. Keep the key file available for
later detection. The trace contains original-token information and model
distributions, so it is a research artifact, not a proposed public watermark
certificate.

Each recording has a directory named `sample_<index>_<public sample ID>`:

| File | Contents |
| --- | --- |
| `generated.wav` | Generated speech before any augmentation |
| `aug_00_00.wav` | Audio from the identity/round-trip evaluation |
| `original_tokens.txt` | Generated semantic IDs, one per line |
| `aug_00_00_tokens.txt` | Semantic IDs recovered from that audio |
| `trace.pt` | Public sample ID, original semantic IDs, saved p0 distributions, sampling settings |
| `metadata.json` | Text, reference information, model path, and experiment settings |
| `results.json` | Original and recovered matches with the same key and a wrong key |

`results.csv` in the parent directory aggregates recovered-token scores. Read
`results.json` for the clean original-token replay check as well.

To repeat detection without running generation, substitute actual file paths:

```bash
python -m evals.keyed_sparktts detect \
    --trace outputs/keyed_sparktts/sample_<index>_<id>/trace.pt \
    --tokens outputs/keyed_sparktts/sample_<index>_<id>/aug_00_00_tokens.txt
```

Replace `--tokens ...` with `--audio .../generated.wav` to re-encode audio again.
Audio input must be mono, 16 kHz, as produced by this experiment. Token-file
detection needs no GPU or model weights; audio detection loads the codec.

## What is being sampled?

The language model vocabulary includes semantic audio tokens, global speaker
tokens, text tokens, and control markers. Spark-TTS's semantic vocabulary has
8,192 IDs. The recording's reference audio supplies the global speaker tokens.
Only newly generated semantic tokens are keyed; reference tokens are excluded.

At a language-model step, define `Q` as the distribution after the model's normal
logits processors, temperature, top-k, and top-p. Filtering order matches Hugging
Face: temperature, then top-k, then top-p. Here p0_t = q_t is the conditional
distribution over semantic IDs, given that this step emits a semantic token:

```
p0_t[s] = Q[model_vocab_id(s)] / sum(Q[semantic_model_vocab_ids])
```

The semantic ID order is ascending, from 0 to 8191. These IDs differ from the
corresponding IDs in the language model's much larger vocabulary.

`KeyedSemanticSampler.__call__` first takes an ordinary categorical draw from Q.
If it is nonsemantic, the token is left alone and the semantic index does not
advance. If it is semantic, its identity is replaced by a keyed categorical draw
from p0_t. Thus the semantic/nonsemantic probability mass is preserved, and
semantic IDs have the same marginal distribution under ideal uniform keyed
randomness. The extra ordinary draw chooses the category; it does not apply a
watermark bias. Its randomness is controlled by `--seed` for experiment setup.

For a fixed key and sample ID, the semantic choices are deterministic conditional
on the distributions. Distribution preservation refers to averaging over fresh
ideal keyed randomness. HMAC supplies computational pseudorandomness, represented
here on a finite 52-bit grid, rather than literal independent continuous draws.

## How does the key choose a token?

Read `watermark/keyed_sampling.py` in this order:

1. `keyed_uniform`: HMAC-SHA256 maps secret key, public sample ID, and semantic
   index t to U_t in (0, 1). A version/domain label separates this use of HMAC
   from other uses. A fresh public sample ID namespaces each recording, so
   separate recordings do not reuse one random stream. It is saved in the trace;
   no synchronization mechanism is needed to reconstruct it in this experiment.
2. `inverse_cdf`: normalize p0_t, form cumulative probabilities, and select the
   first cumulative interval strictly above U_t. Zero-probability tokens have no
   interval. CPU float64 arithmetic and the same function are used for sampling
   and replay.
3. `KeyedSemanticSampler`: apply the normal filters, choose a semantic/nonsemantic
   category, use the keyed draw for semantic output, and record p0_t.
4. `trace`: export the sampling information without the key or uniform draws.
5. `replay_tokens`: reconstruct S_t from saved p0_t and the supplied key.
6. `match_statistics`: compare recovered semantic token t with reconstructed S_t.

For a vocabulary with probabilities `[0.2, 0.5, 0.3]`, the CDF is
`[0.2, 0.7, 1.0]`. A keyed uniform of `0.4` selects token 1. The detector obtains
the same token from that distribution and the same key/index/sample ID.

## How is it connected to Spark-TTS?

`SparkTTS.inference` accepts an optional `keyed_sampler`. With this argument it
attaches our processor and disables Hugging Face's temperature/top-k/top-p
warpers, because the sampler already applied them. The processor returns logits
with exactly one possible token. HF's final multinomial operation therefore
emits the token already chosen by the sampler. This point mass is an interface
adapter; the choice itself was made by categorical sampling from Q/p0_t.

The original Maryland path still runs when no keyed sampler is supplied. Keyed
sampling rejects simultaneous Maryland watermarking and controlled-generation
mode; it currently supports reference-conditioned TTS and batch size one only.

`evals/keyed_sparktts.py` orchestrates the experiment:

1. Load the secret key, Spark-TTS, and a reference recording/transcript.
2. Construct a new sampler for each text prompt and public sample ID.
3. Generate speech and assert that extracted semantic IDs equal the trace IDs.
4. Assert that same-key replay matches every original token.
5. Save the waveform and trace.
6. Reuse the existing augmentation and semantic re-encoding code.
7. Compare recovered tokens with same-key and wrong-key replay.

The re-encoder shares Spark-TTS's already-loaded codec to avoid a second GPU copy.
The wrong-key control uses a separate HMAC-derived key, which is also never
exported. These scores answer whether a different key reproduces tokens using
the **same saved distributions**, not whether independently generated ordinary
audio passes a calibrated statistical test.

## What does detection mean at this stage?

In the clean matched case, V_t = S_t, so original-token replay must match 100%.
After audio decoding and re-encoding, V'_t can differ from V_t. We count
`matches = sum(V'_t == S_t)` by aligned semantic position, including repeated
token occurrences. There is no token deduplication as in the old greenlist
detector. Missing positions count as failures in `match_rate`, whose denominator
is the original trace length. `compared_match_rate` uses only the shared prefix;
missing and extra token counts are reported explicitly. No alignment is inferred
even when the two lengths differ.

Wrong-key matches can be common when p0_t is concentrated. Under independent
draws from a fixed p0_t, the match probability is sum_s p0_t(s)^2, not 1/8192.
Saved p0_t depends on the generated history, so a wrong-key replay is a useful
control but is not a complete unwatermarked-null experiment.

This detector intentionally uses saved generation distributions. The key and
position reproduce U_t; they do not by themselves reproduce the token S_t
without p0_t. No model reconstruction from noisy history is attempted here.
No p-values, false-positive guarantee, or anytime-valid stopping rule are claimed.
Those require the next statistical layer and its conditional-null assumptions.

## How this connects to the paper and existing evaluations

The code above implements the delta=0 clean coupling S_t = V_t plus match-count
diagnostics after the codec channel. Your channel-aware detector can later use
the same traces and estimated smoothed transition matrix to compute

```
e_t(v_prime, s) = P_hat(v_prime | s) / sum_v p0_t(v) P_hat(v_prime | v)
```

That detector and channel estimation are not implemented yet. Its statistical
validity must be checked under the intended sequential null; match counts alone
are not an e-process.

`generate --eval_aug` connects to the existing `get_validation_augs` attack suite.
It requires the suite's additional codec packages, weights, and FFmpeg. The smoke
test uses the identity path only. Speed/crop attacks that break the agreed index
alignment assumption will be compared literally by index; synchronization stays
out of scope. Audio-quality metrics and full multi-prompt attack evaluations are
separate follow-up experiments.

## Tests and suggested reading order

```bash
python -m unittest discover -s tests -v
```

The CPU tests check key/index/sample separation, inverse-CDF boundaries,
sampling frequencies, semantic/nonsemantic mass preservation, filtering,
Hugging Face integration using a tiny random model, replay, and length reporting.
These checks do not require downloaded checkpoints.

The initial GPU smoke test generated 3.90 seconds of audio with 195 semantic
tokens. Same-key replay matched all 195 original tokens. Audio re-encoding
recovered 194 tokens; 85 matched the keyed sequence, compared with 3 under the
wrong key. Standalone audio detection on CPU reproduced those counts. This is
one functional smoke test, not a detection-rate or false-positive benchmark.

To understand the implementation incrementally, start with `keyed_uniform` and
`inverse_cdf`, then read `replay_tokens` and `match_statistics`. Next read the
sampler's `__call__` method and the small hook in `SparkTTS.inference`. Finish
with `generate` and `detect` in the experiment runner. The tests provide small
executable examples of each layer.
