# Channel estimation before chunked watermark experiments

This experiment collects **ordinary unwatermarked** Spark-TTS speech. No seed
extraction, STT, keyed sampling, or watermark generation is used. The subsequent
watermark detector sweep remains on hold.

The default collection is 512 training and 128 validation recordings. A fixed
split puts 24 of the 32 texts in training and eight in validation. Voices are
shared between splits. Repeated texts within a split are independent ordinary
sampling runs; this is not a 640-unique-text benchmark. Generation uses
600 maximum model tokens, temperature 1, top-k 50, and top-p .95. Actual audio
lengths and retry seeds are recorded. Only empty-semantic-output failures retry.

Group maps are fixed random vocabulary-count-balanced partitions of 8192
semantic tokens: 2, 4, 8, and 16 groups; seeds 20261104, 20261105, and 20261106.
These are not learned clusters and are not probability-balanced. The exact maps
are saved. This experiment estimates channels for these candidate groupings;
it does not select a winner using watermark results.

Seven encoding policies are compared: full recording; chunks of 25, 50, or 100
nominal tokens (0.5, 1, or 2 seconds), each with 0 or 25 tokens (0 or 0.5 seconds)
of right lookahead and 100 tokens (2 seconds) of left context. Window origins
are multiples of 320 samples at 16 kHz. The wrapper retains its normal per-window
volume normalization. The encoder is evaluated in inference/eval mode.

Recovered frames are assigned absolute nominal positions. Missing boundary
frames are omitted without shifting later indices. These are assumed positional
correspondences, not a demonstrated token-center alignment. Each policy fits
its own available positions; paired comparisons may therefore have different
frame coverage. No padding, time-shift search, crop correction, or deduplication
is introduced. Counts include both matching and mismatching group pairs.

For each partition and policy, fit rows=original group and columns=recovered
group, with `(N[a,b]+beta)/(N[a]+M*beta)` and beta=0.0001. No validation counts
enter fitting. Within-group source mixtures follow ordinary generated training
audio; deployment mixtures can differ with token probabilities and context.

## Run and resume

Submit `sbatch scripts/run_channel_estimation.sh` from the repository root.
The launcher uses one Slurm GPU, eight CPU threads, 32 GiB memory, and an
eight-hour time limit. `EWM_OUTPUT`, `EWM_TRAINING_CLIPS`,
`EWM_VALIDATION_CLIPS`, and `EWM_BOOTSTRAP` customize the bounded run. To resume,
submit the same settings with the same EWM_OUTPUT. Configuration mismatches
are rejected. A complete.json marker commits a clip only after all seven
encodings exist. Incomplete clips reuse saved audio and completed encodings.
Only one writer should run against a given output directory.

A smoke job uses eight training and four validation clips; it verifies plumbing,
not estimation adequacy. The full job can depend on smoke success. No detector
jobs are submitted by this launcher.

## Artifacts and interpretation

- `config.json`: immutable split texts, references, settings, policies, and scope.
- `progress.json`, `status`: progress and completion/failure status.
- `samples/{training,validation}/clip_XXXX/`: original WAV, original IDs,
  seven recovered streams with absolute positions/arrival times, timing JSON,
  metadata and completion marker. These permit later regrouping without TTS.
- `groupings/`: exact vocabulary maps.
- `channels/`: training counts, validation diagnostic counts, fitted matrices,
  source-row clip coverage, final pointwise bootstrap intervals, and metadata.
- `report/learning_curves.csv`: fixed-validation log loss and independent-half
  weighted row TV at increasing training sizes.
- `report/diagnostics.csv`: support, interval width, final-vs-half stability,
  held-out gain over an output-marginal baseline, train/validation discrepancy,
  and screening flags for every partition/policy/group count.
- `report/summary.json`: candidates for later review, never automatic approval.
- Every plot has PNG and PDF versions and a same-stem TXT explanation covering
  sources, axes, aggregation, uncertainty, smoothing, and practical limits.

Uncertainty uses 300 recording-cluster bootstrap replicates: training recordings
are resampled for cell intervals, validation recordings for predictive-gain and
loss-change intervals. This preserves dependence within a recording but assumes
independent recordings; intervals are pointwise, not simultaneous. Partition
ranges on plots are sensitivity ranges, not confidence intervals. Empirical
mutual information is a potentially biased plug-in diagnostic, not achieved
watermark evidence growth.

Predeclared practical screens require every source row to have >=500 pairs and
appear in >=32 training recordings; maximum cell interval half-width <=.05;
weighted TV between final and first-half channel <=.02; TV between channels
fitted on disjoint halves <=.05; and the 95% bootstrap interval for final minus
half-data held-out loss within [-.01,.01] nats/pair. Additional flags require
positive held-out predictive gain (lower interval >0) and train/validation
weighted row TV <=.10. These are conservative operational checks, not proofs of
channel stationarity, estimation accuracy, or anytime validity. If they fail,
inspect the reason and gather more data or revise/shrink the model before
launching a watermark sweep.

## Automatic successor

An optional quality-gated successor is documented in [automatic_sweep.md](automatic_sweep.md). The collector itself submits no jobs. The connected successor requires all original criteria to pass and freezes accepted channels before generating any watermark evaluation audio. Current submitted job IDs are recorded in `outputs/channel_estimation_launch/jobs.json`.
