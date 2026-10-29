# Recovering from an early STT word disagreement

The original `WordCommitter` confirms only an agreeing full transcript prefix.
A committed initial insertion such as "with" can therefore block all later
anchors when subsequent partial recognition omits that word.

`LocalWordCommitter` in `watermark/word_anchors.py` is an alternative policy.
It treats each new word as a time-local occurrence, not as an ordinal position
in a permanently frozen transcript prefix.

## Causal algorithm

At each scheduled recognition update:

1. Normalize words using exactly the original policy's normalization.
2. Withhold the trailing word and retain the last two or three hypotheses.
3. Align the current eligible words separately against each earlier hypothesis.
   Matches require identical normalized text, chronological one-to-one ordering,
   and start/end estimates within 200 ms. Earlier insertions or deletions may be
   skipped; the rest of the old prefix need not match.
4. A candidate must match in every required hypothesis and finish at least
   200 ms before the current available-audio time.
5. To avoid committing an old occurrence twice, require its midpoint to advance
   beyond the latest committed midpoint, and reject an interval overlapping a
   committed interval by at least half the shorter interval's length. This
   permits small overlaps in estimates of adjacent word boundaries.
6. Commit it only now, activating at the next token-grid position at or after
   confirmation plus the existing 100 ms delay.

Previous commitment events never change. Old-prefix contradictions are logged
but do not clear the agreement history or stop new local anchors. No final
transcript, counterpart stream, known generation script, or future update is
consulted by either committer. The two sides run independently.

An early wrong word still affects its initial interval. A one-word context
recovers once both sides use the next common anchor; a two-word context requires
two common anchors. Recovery requires those anchors and their activation times
to agree. Wrong timestamps, missed words, and overlapping speech remain failure
modes. This is not a guarantee that all early errors become irrelevant.

## Replay without STT or GPU work

```bash
conda activate wmar_audio
python -m evals.replay_word_anchors
```

This reads the original raw hypotheses in `outputs/rolling_stt_pilot`, replays
both policies, verifies that the baseline reproduces the original summary
exactly, and saves new outputs in `outputs/rolling_stt_local_replay_v2`.
Original transcripts, policies' original outputs, and plots are not overwritten.

| Path in replay output | Contents |
| --- | --- |
| `config.json` | Source experiment and fixed local-policy parameters |
| `prefix_baseline/` | Independently reproduced original-policy results |
| `timestamp_local/` | New event logs, per-clip metrics, and all policy-grid results |
| `comparison.json` | Agreement and coverage differences for every configuration |
| `policy_comparison.png` / `.pdf` | Side-by-side comparison for 200 ms, two-update confirmation |

The first development implementation (`outputs/rolling_stt_local_replay`) used
a strict non-overlapping time frontier. Inspection showed that small overlaps
in ASR timestamps incorrectly suppressed adjacent words. Version 2 uses the
midpoint/overlap rule above. Both development outputs are retained. The final
code implements version 2; these experiments were used to develop the policy,
so a new held-out sample is needed before making general performance claims.

## Pilot findings

All four source clips are retained. With 200 ms updates, agreement across two
updates, and one-word context, active-position agreement changes as follows:

| Detection audio | Full-prefix baseline | Local policy v2 |
| --- | ---: | ---: |
| Clean | 70.2% | 88.8% |
| Noise, waveform std 0.001 | 70.8% | 84.7% |
| Low-pass, 3 kHz | 64.4% | 82.1% |

The active-position denominators are unchanged: 803, 783, and 783, respectively.
Embedder coverage remains about 76%; clean-detector mean coverage changes from
about 78% to 77%. The previously failing clip improves from zero to 77.2% clean
agreement. Some previously successful clips lose agreement because the local
policy changes individual commit timing, so it does not dominate on every clip.

These are development replays on four short utterances, not independent
validation, closed-loop watermark experiments, or a proof of anytime validity.
The original policy remains available rather than being silently replaced.

## Aligned sample-by-sample example

```bash
python -m evals.plot_word_timeline --clip sample_1050
python -m evals.plot_word_timeline --clip sample_1050 --policy prefix_baseline
```

Plots and exact interval JSON files are saved under
`outputs/rolling_stt_local_replay_v2/timelines/`. The two main rows share the
audio sample axis (16 kHz) and show the most recent active committed word on
the 20 ms token grid. A word has the same color in either row. Gray means no
anchor yet. The third strip is green for equal active contexts and red for
different contexts, including an anchor existing on only one side. Joint gray
startup is excluded from the active-position agreement denominator.

These intervals show when a word would be used as the hash context, not when
the word was acoustically spoken. Confirmation and activation delays account
for the difference. When multiple words commit in one update, only the latest
word has a nonzero active interval in this one-word-context view.

Both sides use identical STT weights and recognition/commitment settings. The
generation side repeatedly decodes only its available semantic-token prefix;
the clean detector observes the same-time prefix of the finished waveform.
Contextual decoding means those waveforms can differ even without an attack.
The detector STT listens directly to that waveform, not to a second decoded
waveform after codec re-encoding. The noisy/filtered variants additionally
transform detection audio. No actual keyed watermark is embedded in this pilot.
