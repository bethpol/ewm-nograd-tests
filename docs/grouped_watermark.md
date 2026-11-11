# Random-group e-value experiment

## Implementation review

`watermark/keyed_sampling.py` currently draws individual semantic tokens using
position-keyed inverse CDF sampling, saves the filtered conditional semantic
distribution, and provides match counts. It does not implement group sampling
or a sequential e-value detector. `docs/sparktts_transition.md` supplies the
channel-factor formula; its snippet is not a complete detector.

The older Maryland cluster bias in `watermark/wm_sampling_methods.py` is a
different watermark. Its green-group bias does not implement the keyed
categorical construction. Repeated context hashes there also cannot simply be
treated as fresh independent evidence in a new sequential test.

## Proposed grouped construction

Fix a random balanced token-to-group map independently of evaluation outcomes.
For filtered semantic probabilities p_t(v), let a_t(g) = sum_{v in g} p_t(v).
Draw a candidate token S_t from p_t using the existing position-keyed token
sampler. Independently draw an output token within g(S_t), with probabilities
p_t(v)/a_t(g(S_t)). Both candidate and output are in the token alphabet.
This preserves the token
marginal averaged over fresh key randomness. The reusable CPU implementation
is `watermark/grouped.py`; the caller supplies the keyed uniform. The actual
Spark-TTS integration is `KeyedSemanticSampler(..., groups=...)`, enabled with
`evals.keyed_sparktts generate --num_groups G --partition_seed SEED`.
It retains the existing candidate-token HMAC protocol and saves a versioned
`sparktts-keyed-grouped-v1` trace including the partition and emitted tokens.
Replay reconstructs candidate tokens, whose groups must match emitted groups.

Key derivation uses only the secret key, public recording nonce, and semantic
position. Token history, word anchors, and experimental context keys are not
used. Context still determines the model probabilities, as in ordinary generation.

With observed recovered group Y_t and a fixed positive channel C_t, use
e_t = C_t[g(S_t),Y_t] / sum_g a_t(g) C_t[g,Y_t]. Accumulate log factors and detect
at log E_t >= log(1/alpha). Under the random-key null, fresh g(S_t) must be
independent of Y_t conditional on past information and the probabilities used
for the draw. The identity sum_g a_t(g)e_t(g,y)=1 gives the conditional null
normalization (here S_t is token-valued, and g(S_t) has law a_t).
A fixed-key or adaptive replay deployment needs its own argument;
empirical null crossings alone do not establish validity.

Saved p_t from one generation cannot be treated as counterfactual model logits
after changing groups or sampled tokens: future autoregressive contexts change.
Actual watermark-power comparisons require regenerating audio for each setting.

## CPU pilot

```bash
module load python/miniforge3_pytorch
python3 -m unittest discover -s tests -p 'test_grouped_watermark.py'
python3 -m evals.grouped_watermark --output_dir outputs/grouped_synthetic
python3 -m evals.grouped_watermark \
  --counts PATH/TO/TRAINING/transition_counts.npz \
  --groups 1 2 4 8 16 32 64 --partitions 5 --trials 200 \
  --tokens 800 --alpha .01 --output_dir outputs/grouped_channel_pilot
```

Without `--counts`, this uses a synthetic 128-token channel. With counts, it
pools original/recovered token pairs into groups before smoothing. Rows are
generated groups; columns are recovered groups. The within-group token mixture
is the observed source frequency in the training data. A real effective channel
depends on context: C_t[g,h] = sum_{v in g} p_t(v)/a_t(g)
sum_{y in h} P[y|v]. The fitted stationary matrix approximates this mixture.
Group-level smoothing is a separate prior from smoothing the full token matrix
and then aggregating it; the pilot records beta explicitly.

Outputs include per-trial first crossings and censoring indicators in `trials.csv`,
settings/results in `summary.json`, PNG/PDF detection plots, and group maps,
counts, masses, channels, and transition heatmaps in per-partition directories.
The time plot uses mean min(T,H), including failures at the horizon H. The
crossing-fraction panel makes censoring visible. These are fitted-model Monte
Carlo results, not measurements of watermark recovery on new audio. The null
simulates outputs independently of the keyed group; it is a calibration check
for the implemented model, not a real-audio false-positive estimate.

## Next audio experiment

Submit the prepared generation sweep from the repository root after installing
the repository dependencies in the selected Python environment:

```bash
MODEL_DIR=/path/to/Spark-TTS-0.5B sbatch scripts/run_grouped_sparktts.sh
```

`EWM_PYTHON`, `EWM_GROUP_COUNTS`, `EWM_PARTITION_SEEDS`, `EWM_REFERENCE`, and
`EWM_NSAMPLES` customize the launcher. No job is submitted by preparing it.
Score one saved observation using an explicitly chosen fitted group channel:

```bash
python3 -m evals.grouped_detect --trace SAMPLE/trace.pt \
  --tokens SAMPLE/aug_00_00_tokens.txt \
  --channel outputs/grouped_channel_pilot/groups_8_partition_0/channel.npz \
  --output SAMPLE/group_detection.json
```

The detector refuses a mismatched partition and saves the full log-e path.
Its denominator uses each saved p_t, not the stationary mass used by the pilot.
These are offline replay diagnostics; the detector requires saved generation
distributions. `keyed_sparktts` match statistics become group matches for grouped
traces; wrong-key comparisons remain diagnostics.

Use disjoint channel-training and evaluation clips. Freeze maps, smoothing,
alpha, and token horizon before evaluation. Sweep groups and several independent
partition seeds; include one group (no signal) and the full vocabulary baseline.
Generate fresh watermarked clips per setting using group-first sampling, saving
p_t and a versioned replay trace. Decode/re-encode, map recovered tokens to
groups, and score the shared aligned prefix. Report unpaired tails and test the
index-alignment assumption separately. Compare pooled stationary channels with
context-weighted token channels. Report detection fraction, restricted mean time,
and uncertainty across clips and partitions; test null audio with independent
keys/nonces without selecting the best key or alignment.

GPU generation on this server must follow `/u/epolito/test.sh`: allocate through
Slurm account `biry-dtai-gh`, partition `ghx4`, request GPU resources explicitly,
load `python/miniforge3_pytorch`, and launch with `srun`. Adjust memory/time to
the actual workload; do not run generation on the login node.

## Complete real-audio pilot on DeltaAI

`sbatch scripts/run_grouped_pilot.sh` uses the repository `.venv` with the
DeltaAI PyTorch module and the downloaded checkpoint at
`checkpoints/sparktts/Spark-TTS-0.5B`. It allocates one GPU for at most one hour.
The pilot collects 16 ordinary channel-fitting clips, then four fresh evaluation
clips for each consecutive group count from 2 through 16, with one fixed random partition seed.
All generation is launched with `srun`. Key derivation remains position based.

Each job writes `outputs/grouped_pilot_JOBID/`: `channel_training/` contains the
ordinary training pairs and full token-channel counts; `audio/` contains grouped
generation traces and recovered audio tokens; `report/` contains group channels,
heatmaps, first-crossing CSV, log-e paths, and detection-vs-groups PNG/PDF plots.
The `status` file reads running, complete, or failed. Slurm logs are
`grouped_pilot_JOBID.out` and `.err` in the repository root.

The report estimates group channels exclusively from ordinary training clips.
It uses a common observed token horizon across all evaluation configurations and
includes undetected clips in restricted mean detection time. This small pilot
has no real-audio null calibration or uncertainty intervals yet; it is intended
to verify the complete pipeline before expanding clips and partition seeds.
