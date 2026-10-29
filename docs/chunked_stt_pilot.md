# Chunk-based embedder/detector STT rerun

This reruns the same four clips with immutable released audio instead of
comparing revised prefix decoding to prefixes of a globally decoded waveform.
It uses saved ordinary semantic tokens; it is not closed-loop watermarked
generation, a native streaming vocoder, or streaming codec-token detection.

## Audio release protocol

At 16 kHz, each semantic token corresponds to a nominal hop of 320 samples
(20 ms). Release chunks contain five hops: 1,600 samples or 100 ms.
The prefix decoder may use ten additional tokens (200 ms) of lookahead.

For a chunk ending at token b, the simulator makes tokens 0 through b+10
available, decodes that prefix, and copies only the chunk's new sample interval
into the output stream. Earlier released chunks are never replaced by samples
from a later decode. Near EOF, remaining chunks are flushed with the available
tokens; updates that would require EOF flushing are excluded from rolling
commitments. EOF is used only for a separate final transcript audit.

The detector accumulates separate copies of those exact float32 chunk bytes.
Preparation asserts byte-identical clean inputs, and stores chunk hashes and
sample ranges. Independent STT calls are made on each side, with the same model
and settings. Noise (std 0.001) and causal low-pass filtering (3 kHz) are applied
only to detection variants. Chunk boundaries may create acoustic artifacts;
clean equality is a controlled sanity check, not a speech-quality guarantee.

## Which flags do we recover?

There is no STT oracle identifying the exact physical sample at a word boundary.
The logs distinguish several different coordinates:

1. **Chunk end:** an exact exclusive sample index determined by the release
   protocol. Receivers know it by counting samples; STT does not estimate it.
2. **Estimated word end:** the STT word-end timestamp multiplied by 16,000 and
   rounded to an integer. This remains an acoustic estimate with uncertainty.
3. **Confirmation sample:** how many released audio samples are available when
   the word meets the local agreement rule. The rule withholds trailing words,
   requires two or three observations and a 200 ms ending margin.
4. **Generation frontier:** tokens already generated to support that audio.
   On non-EOF updates it corresponds to confirmation sample + 3,200 samples.
5. **Activation index:** the first future nominal token index at or after the
   generation frontier plus a 100 ms safety delay:

```
activation_token = ceil((generation_frontier_sample + 1600) / 320)
activation_nominal_sample = 320 * activation_token
```

The frontier prevents retroactively changing tokens already consumed by decoder
lookahead. For example, when 1.6 seconds of audio is available, 1.8 seconds'
worth of tokens have already been generated. The new context activates at token
index 95 (nominal 1.9 seconds), not at the estimated word ending or at 1.7 seconds.
Both sides use the fixed agreed chunk/lookahead protocol to compute this
coordinate. It assumes a known recording origin and unchanged sample clock.

These are sample/token coordinate flags, not in-band markers in the audio.
They do not recover unknown cropping offsets or guarantee codec retokenization
produces the original symbols. Timing synchronization, ASR agreement, and the
eventual anytime-valid evidence process remain separate parts of the design.

## Results and rerunning

```bash
conda activate wmar_audio
python -m evals.chunked_stt
```

Detached launch:

```bash
tmux -L ewm-channel new-session -d -s chunked-stt-pilot -c "$PWD" \
    'exec bash scripts/run_chunked_stt_pilot.sh'
```

All output is under `outputs/chunked_stt_pilot/`:

| Path | Contents |
| --- | --- |
| `progress.json`, `run.log`, `exit_code` | Run status, log, and exit code |
| `config.json`, `manifest.json` | Release/ASR parameters and the same four source clips |
| `clips/<clip>/preparation.json` | Exact chunk ranges, hashes, causal generation frontiers |
| `clips/<clip>/released_audio.wav` | The immutable chunk stream actually presented to STT |
| `clips/<clip>/recognition/` | Independent prefix transcripts and timestamps for every arm |
| `clean_control.json` | Verification of clean independent-recognition agreement |
| `sample_flags.csv` | Estimated word ends, confirmation samples, generation frontiers, activation indices |
| `per_clip.csv`, `summary.json` | Agreement, coverage, delays, and boundary errors |
| `word_agreement.png`, `.pdf` | Aggregate agreement figure |
| `timelines/` | Aligned word-color timelines for sample_1050 under clean/noisy/filtered detection |

Local word commitments are immutable. All STT updates use only released audio;
no known target text or full-audio transcript repairs rolling decisions.
STT computation time is logged separately; this is an audio-time simulation
that can pause generation for recognition, not a measured real-time deployment.
