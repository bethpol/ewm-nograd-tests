"""Causal buffered chunk release, followed by rolling STT on identical clean chunks.

This replays saved tokens; it is not closed-loop keyed generation or a streaming
codec-token detector. Released audio is immutable and decoder lookahead is
included in the future token activation rule.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from evals.rolling_stt import SR, HOP, ARMS, transcribe, analyze, update_schedule
from evals.sparktts_transition import atomic_json
from watermark.word_anchors import LocalWordCommitter


def chunk_schedule(total_tokens, chunk_tokens, lookahead_tokens):
    if total_tokens < 1 or chunk_tokens < 1 or lookahead_tokens < 0:
        raise ValueError("Invalid chunk schedule")
    chunks = []
    for start in range(0, total_tokens, chunk_tokens):
        end = min(total_tokens, start + chunk_tokens)
        frontier = min(total_tokens, end + lookahead_tokens)
        chunks.append({"start_token": start, "end_token": end,
                       "generated_tokens_available": frontier,
                       "tail_flush": end + lookahead_tokens > total_tokens})
    return chunks


def release_chunks(total_tokens, chunk_tokens, lookahead_tokens, decode_prefix):
    """decode_prefix only receives the causal generated-token frontier.

    Returned chunks are copies: later decoder calls cannot revise released data.
    Never splice chunks out of a waveform decoded using future unavailable tokens.
    """
    released, records = [], []
    for item in chunk_schedule(total_tokens, chunk_tokens, lookahead_tokens):
        frontier = item["generated_tokens_available"]
        decoded = np.asarray(decode_prefix(frontier), dtype=np.float32).reshape(-1)
        if len(decoded) != frontier * HOP:
            raise ValueError("Decoder output length differs from nominal token grid")
        start, end = item["start_token"] * HOP, item["end_token"] * HOP
        chunk = decoded[start:end].copy()
        released.append(chunk)
        records.append({**item, "start_sample": start, "end_sample_exclusive": end,
                        "generation_frontier_sample": frontier * HOP,
                        "sha256_float32": hashlib.sha256(chunk.tobytes()).hexdigest()})
    return released, records


def prepare(args, manifest, out):
    import torch
    from evals.wm_sparktts import SparkTTSTokenizerWrapper
    from training.augmentations import NoiseInjection, LowpassFilter

    encoder = SparkTTSTokenizerWrapper(args.codec_dir, torch.device(args.codec_device))
    chunk_tokens = args.chunk_ms * SR // 1000 // HOP
    lookahead_tokens = args.lookahead_ms * SR // 1000 // HOP
    for index, clip in enumerate(manifest):
        source = Path(clip["source"])
        folder = out / "clips" / source.name
        folder.mkdir(parents=True, exist_ok=True)
        original = np.loadtxt(source / "original_tokens.txt", dtype=np.int64, ndmin=1)
        tensor = torch.tensor(original, device=args.codec_device).reshape(1, -1)
        with torch.inference_mode():
            global_tokens, _ = encoder.tok.tokenize(clip["metadata"]["reference"])

        def decode(frontier):
            with torch.inference_mode():
                return encoder.tok.detokenize(global_tokens.squeeze(0), tensor[:, :frontier])

        chunks, records = release_chunks(len(original), chunk_tokens, lookahead_tokens, decode)
        # Simulate separate receivers accumulating the exact released chunk bytes.
        embedder_audio = np.concatenate(chunks)
        received_chunks = [np.frombuffer(c.tobytes(), dtype=np.float32).copy() for c in chunks]
        detector_audio = np.concatenate(received_chunks)
        if not np.array_equal(embedder_audio, detector_audio):
            raise RuntimeError("Clean receiver did not receive identical samples")
        sf.write(folder / "released_audio.wav", embedder_audio, SR, subtype="FLOAT")
        np.save(folder / "detection_clean.npy", detector_audio)
        torch.manual_seed(args.seed + index)
        audio = torch.tensor(detector_audio, device=args.codec_device).reshape(1, 1, -1)
        with torch.inference_mode():
            noisy, _ = NoiseInjection()(audio, None, .001)
            filtered, _ = LowpassFilter(sample_rate=SR)(audio, None, 3000)
        np.save(folder / "detection_noise.npy", noisy.cpu().numpy().reshape(-1))
        np.save(folder / "detection_lowpass.npy", filtered.cpu().numpy().reshape(-1))
        # Avoid EOF-dependent flushing in the rolling experiment. Both sides
        # know the fixed chunk/lookahead protocol, not the future clip length.
        schedule = [n for n in update_schedule(len(embedder_audio), args.strides_ms)
                    if n % (chunk_tokens * HOP) == 0
                    and n + lookahead_tokens * HOP <= len(embedder_audio)]
        prefix_dir = folder / "generation_prefix"
        prefix_dir.mkdir(exist_ok=True)
        frontier_map = {}
        for n in schedule:
            # This prefix contains only permanently released chunks.
            np.save(prefix_dir / f"samples_{n:08d}.npy", embedder_audio[:n])
            frontier_map[str(n)] = n + lookahead_tokens * HOP
        n = len(embedder_audio)
        np.save(prefix_dir / f"samples_{n:08d}.npy", embedder_audio)
        frontier_map[str(n)] = n  # final audit only, never used to commit
        atomic_json(folder / "preparation.json", {
            "schedule_samples": schedule, "total_samples": n,
            "total_tokens": len(original), "chunk_ms": args.chunk_ms,
            "lookahead_ms": args.lookahead_ms,
            "generation_frontier_by_samples": frontier_map,
            "clean_receiver_bit_identical": True,
            "audio_sha256_float32": hashlib.sha256(embedder_audio.tobytes()).hexdigest(),
            "chunks": records})
        print(f"Prepared {source.name}: {len(chunks)} immutable chunks, "
              f"{len(schedule)} rolling updates; clean bytes identical", flush=True)
    del encoder
    if args.codec_device.startswith("cuda"):
        torch.cuda.empty_cache()


def export_flags(manifest, out):
    import csv
    fields = ["clip", "arm", "stride_ms", "agreement_updates", "word",
              "estimated_word_end_sample", "confirmation_audio_sample",
              "generated_token_frontier_sample", "activation_token_index",
              "activation_nominal_sample"]
    rows = []
    for clip in manifest:
        folder = out / "clips" / Path(clip["source"]).name
        for path in sorted(folder.glob("commits_*.json")):
            data = json.loads(path.read_text())
            name = path.stem.removeprefix("commits_")
            arm, stride, agreement = name.rsplit("_", 2)
            for e in data["events"]:
                rows.append(dict(zip(fields, [folder.name, arm, int(stride.removesuffix("ms")),
                    int(agreement.removesuffix("agree")), e["word"], e["word_end_samples"],
                    e["confirmed_samples"], e["generation_frontier_samples"],
                    e["activation_token"], e["activation_token"] * HOP])))
    with (out / "sample_flags.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_clean(manifest, out):
    count = 0
    for clip in manifest:
        folder = out / "clips" / Path(clip["source"]).name
        for path in sorted((folder / "recognition/generation_prefix").glob("*.json")):
            a = json.loads(path.read_text())
            b = json.loads((folder / "recognition/detection_clean" / path.name).read_text())
            if a["words"] != b["words"]:
                raise RuntimeError(f"Clean ASR outputs differ for identical input: {folder.name}/{path.name}")
            count += 1
    atomic_json(out / "clean_control.json", {"identical_waveform": True,
        "identical_independently_computed_asr": True, "checked_prefixes_and_audits": count})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/chunked_stt_pilot")
    parser.add_argument("--manifest", default="outputs/rolling_stt_pilot/manifest.json")
    parser.add_argument("--codec_dir", default="checkpoints/sparktts/Spark-TTS-0.5B")
    parser.add_argument("--codec_device", default="cuda")
    parser.add_argument("--stt_model", default="checkpoints/stt/faster-whisper-small.en")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--chunk_ms", type=int, default=100)
    parser.add_argument("--lookahead_ms", type=int, default=200)
    args = parser.parse_args()
    if args.chunk_ms <= 0 or args.chunk_ms % 20 or args.lookahead_ms < 0 or args.lookahead_ms % 20:
        parser.error("Chunk and lookahead must be whole 20 ms token hops")
    args.strides_ms, args.agreements, args.context_words = [200, 500], [2, 3], [1, 2]
    args.margin_ms, args.delay_ms = 200, 100
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        config = {**vars(args), "versions": {x: importlib.metadata.version(x)
                  for x in ("faster-whisper", "ctranslate2", "torch", "numpy")},
                  "scope": "immutable chunk replay of saved tokens; no watermark or streaming re-encoder"}
        if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Existing output configuration differs")
        atomic_json(out / "config.json", config)
        manifest = json.loads(Path(args.manifest).read_text())
        atomic_json(out / "manifest.json", manifest)
        atomic_json(out / "progress.json", {"status": "preparing_chunks"})
        prepare(args, manifest, out)
        transcribe(args, manifest, out)
        verify_clean(manifest, out)
        analyze(args, manifest, out, committer_class=LocalWordCommitter)
        export_flags(manifest, out)
        from types import SimpleNamespace
        from evals.plot_word_timeline import render
        for arm in ARMS[1:]:
            render(SimpleNamespace(clip="sample_1050", arm=arm, policy="timestamp_local",
                replay_dir=str(out), source_dir=str(out), events_dir=str(out),
                stride_ms=200, agreement=2, embedder_label="immutable released chunks"))
        atomic_json(out / "progress.json", {"status": "complete", "clips": len(manifest)})
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
