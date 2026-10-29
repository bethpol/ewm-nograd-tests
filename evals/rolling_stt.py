"""Small rolling-STT pilot: prefix-decoded generation vs detection audio.

Recognition is run on prefixes only. No source transcript is given to ASR.
Final recognition is a separate audit and never participates in live commits.
"""

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

from evals.sparktts_transition import atomic_json
from watermark.word_anchors import WordCommitter, agreement_statistics

SR, HOP = 16000, 320
ARMS = ("generation_prefix", "detection_clean", "detection_noise", "detection_lowpass")


def select_clips(source_dir, n):
    candidates = []
    for p in Path(source_dir).glob("sample_*/metadata.json"):
        m = json.loads(p.read_text())
        if (p.parent / "original_tokens.txt").exists():
            candidates.append((m["seconds"], str(p.parent), m))
    selected, texts = [], set()
    for _, folder, metadata in sorted(candidates, reverse=True):
        if metadata["text"] not in texts:
            selected.append({"source": folder, "metadata": metadata})
            texts.add(metadata["text"])
        if len(selected) == n:
            return selected
    raise ValueError("Not enough distinct completed source texts")


def update_schedule(total_samples, strides_ms):
    schedule = set()
    for stride in strides_ms:
        step = stride * SR // 1000
        if step <= 0 or step % HOP:
            raise ValueError("Update steps must be positive multiples of the 20 ms token hop")
        schedule.update(range(step, total_samples, step))
    return sorted(schedule)


def prepare(args, manifest, out):
    import torch
    from evals.wm_sparktts import SparkTTSTokenizerWrapper
    from training.augmentations import NoiseInjection, LowpassFilter

    encoder = SparkTTSTokenizerWrapper(args.codec_dir, torch.device(args.codec_device))
    for index, clip in enumerate(manifest):
        source = Path(clip["source"])
        folder = out / "clips" / source.name
        folder.mkdir(parents=True, exist_ok=True)
        clean, sr = sf.read(source / "generated.wav", dtype="float32")
        tokens = np.loadtxt(source / "original_tokens.txt", dtype=np.int64, ndmin=1)
        if sr != SR or clean.ndim != 1 or len(clean) != len(tokens) * HOP:
            raise ValueError("Expected mono 16 kHz decoded audio with 320 samples per token")
        schedule = update_schedule(len(clean), args.strides_ms)
        audio = torch.tensor(clean, device=args.codec_device).reshape(1, 1, -1)
        torch.manual_seed(args.seed + index)
        with torch.inference_mode():
            noisy, _ = NoiseInjection()(audio, None, .001)
            filtered, _ = LowpassFilter(sample_rate=SR)(audio, None, 3000)
            global_tokens, _ = encoder.tok.tokenize(clip["metadata"]["reference"])
        np.save(folder / "detection_clean.npy", clean)
        np.save(folder / "detection_noise.npy", noisy.cpu().numpy().reshape(-1))
        np.save(folder / "detection_lowpass.npy", filtered.cpu().numpy().reshape(-1))
        prefix_dir = folder / "generation_prefix"
        prefix_dir.mkdir(exist_ok=True)
        tensor = torch.tensor(tokens, device=args.codec_device).reshape(1, -1)
        waveform_errors = []
        for n in schedule + [len(clean)]:
            path = prefix_dir / f"samples_{n:08d}.npy"
            if path.exists():
                prefix = np.load(path)
            else:
                with torch.inference_mode():
                    prefix = encoder.tok.detokenize(global_tokens.squeeze(0), tensor[:, :n // HOP])
                np.save(path, prefix)
            if len(prefix) != n:
                raise ValueError("Prefix decoder changed nominal waveform length")
            difference = np.asarray(prefix, dtype=float) - clean[:n]
            waveform_errors.append({"available_samples": n,
                                    "rms_difference_from_final_prefix": float(np.sqrt(np.mean(difference ** 2))),
                                    "max_absolute_difference": float(np.max(np.abs(difference)))})
        atomic_json(folder / "preparation.json", {"schedule_samples": schedule,
            "total_samples": len(clean), "total_tokens": len(tokens),
            "prefix_waveform_differences": waveform_errors})
        print(f"Prepared {source.name}: {len(clean) / SR:.2f}s, {len(schedule)} rolling updates", flush=True)
    del encoder
    if args.codec_device.startswith("cuda"):
        torch.cuda.empty_cache()


def recognize(model, audio):
    started = time.monotonic()
    segments, _ = model.transcribe(
        np.asarray(audio, dtype=np.float32), language="en", beam_size=1, best_of=1,
        temperature=0., condition_on_previous_text=False, word_timestamps=True,
        vad_filter=False, initial_prompt=None,
    )
    words = []
    for segment in segments:
        for w in segment.words or []:
            words.append({"word": w.word, "start_samples": round(w.start * SR),
                          "end_samples": round(w.end * SR), "probability": w.probability})
    return words, time.monotonic() - started


def transcribe(args, manifest, out):
    from faster_whisper import WhisperModel
    # CPU int8 avoids changing the codec environment's CUDA/cuDNN dependencies.
    model = WhisperModel(args.stt_model, device="cpu", compute_type="int8",
                         cpu_threads=args.cpu_threads, num_workers=1,
                         local_files_only=True)
    total = 0
    for clip in manifest:
        folder = out / "clips" / Path(clip["source"]).name
        prep = json.loads((folder / "preparation.json").read_text())
        total += len(ARMS) * (len(prep["schedule_samples"]) + 1)
    completed = 0
    for clip in manifest:
        folder = out / "clips" / Path(clip["source"]).name
        prep = json.loads((folder / "preparation.json").read_text())
        for arm in ARMS:
            observations_dir = folder / "recognition" / arm
            observations_dir.mkdir(parents=True, exist_ok=True)
            full = np.load(folder / f"{arm}.npy") if arm != "generation_prefix" else None
            for n in prep["schedule_samples"] + [prep["total_samples"]]:
                # Last item is an EOF audit; not a commitment update.
                audit = n == prep["total_samples"]
                path = observations_dir / ("final_audit.json" if audit else f"samples_{n:08d}.json")
                if not path.exists():
                    audio = (np.load(folder / arm / f"samples_{n:08d}.npy")
                             if arm == "generation_prefix" else full[:n])
                    words, seconds = recognize(model, audio)
                    atomic_json(path, {"available_samples": n, "words": words,
                                       "asr_compute_seconds": seconds, "eof_audit": audit,
                                       **({"generation_frontier_samples": prep["generation_frontier_by_samples"][str(n)]}
                                          if "generation_frontier_by_samples" in prep else {})})
                completed += 1
                atomic_json(out / "progress.json", {"status": "transcribing",
                    "completed_updates": completed, "target_updates": total,
                    "clip": folder.name, "arm": arm, "available_seconds": n / SR})
                print(f"STT {completed}/{total}: {folder.name} {arm} {n / SR:.2f}s"
                      + (" [EOF audit]" if audit else ""), flush=True)


def commit_trace(folder, arm, stride_ms, agreement, margin_ms, delay_ms, committer_class=WordCommitter):
    committer = committer_class(agreement=agreement, margin_samples=margin_ms * SR // 1000,
                              delay_samples=delay_ms * SR // 1000)
    step = stride_ms * SR // 1000
    for path in sorted((folder / "recognition" / arm).glob("samples_*.json")):
        observation = json.loads(path.read_text())
        if observation["available_samples"] % step == 0:
            committer.update(observation)
    return committer


def analyze(args, manifest, out, source_dir=None, committer_class=WordCommitter):
    rows = []
    for clip in manifest:
        folder = (source_dir or out) / "clips" / Path(clip["source"]).name
        event_folder = out / "clips" / folder.name
        event_folder.mkdir(parents=True, exist_ok=True)
        n = json.loads((folder / "preparation.json").read_text())["total_tokens"]
        for stride in args.strides_ms:
            for agreement in args.agreements:
                committers = {}
                for arm in ARMS:
                    state = commit_trace(folder, arm, stride, agreement, args.margin_ms, args.delay_ms, committer_class)
                    committers[arm] = state
                    atomic_json(event_folder / f"commits_{arm}_{stride}ms_{agreement}agree.json",
                                {"events": state.events, "contradictions": state.contradictions})
                for context in args.context_words:
                    for arm in ARMS[1:]:
                        a, b = committers[ARMS[0]], committers[arm]
                        stats = agreement_statistics(a.events, b.events, n, context)
                        differences = stats.pop("activation_index_absolute_differences")
                        latencies = [(e["confirmed_samples"] - e["word_end_samples"]) / SR
                                     for e in a.events]
                        rows.append({"clip": folder.name, "detector_arm": arm,
                            "stride_ms": stride, "agreement_updates": agreement, "context_words": context,
                            **stats, "median_activation_error_tokens": float(np.median(differences)) if differences else None,
                            "p95_activation_error_tokens": float(np.percentile(differences, 95)) if differences else None,
                            "median_embedder_confirmation_delay_s": float(np.median(latencies)) if latencies else None,
                            "embedder_contradictory_updates": len(a.contradictions),
                            "detector_contradictory_updates": len(b.contradictions)})
    with (out / "per_clip.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregated = []
    for stride in args.strides_ms:
        for agreement in args.agreements:
            for context in args.context_words:
                for arm in ARMS[1:]:
                    group = [r for r in rows if (r["stride_ms"], r["agreement_updates"], r["context_words"], r["detector_arm"])
                             == (stride, agreement, context, arm)]
                    active = sum(r["active_union_positions"] for r in group)
                    matches = sum(r["matching_active_positions"] for r in group)
                    aggregated.append({"stride_ms": stride, "agreement_updates": agreement,
                        "context_words": context, "detector_arm": arm, "clips": len(group),
                        "active_positions": active, "matching_positions": matches,
                        "active_context_agreement": matches / active if active else None,
                        "mean_embedder_active_fraction": float(np.mean([r["embedder_active_fraction"] for r in group])),
                        "mean_detector_active_fraction": float(np.mean([r["detector_active_fraction"] for r in group]))})
    atomic_json(out / "summary.json", aggregated)
    plot(aggregated, out)


def plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = ("#3178a8", "#cf6b30", "#40864d")
    for ax, context in zip(axes, (1, 2)):
        labels = []
        for index, (stride, agreement) in enumerate([(200, 2), (200, 3), (500, 2), (500, 3)]):
            labels.append(f"{stride}ms / {agreement} updates")
            for j, arm in enumerate(ARMS[1:]):
                values = [r for r in summary if (r["context_words"], r["stride_ms"], r["agreement_updates"], r["detector_arm"])
                          == (context, stride, agreement, arm)]
                if values:
                    value = values[0]["active_context_agreement"]
                    ax.bar(index + (j - 1) * .23, value if value is not None else 0,
                           width=.22, color=colors[j], label=arm if index == 0 else None)
        ax.set_xticks(range(4), labels, rotation=20, ha="right")
        ax.set(ylim=(0, 1), title=f"Previous {context} committed word(s)",
               ylabel="Context agreement where either side has an anchor")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Rolling STT feasibility pilot — no hashing or watermark embedding")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"word_agreement.{ext}", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/rolling_stt_pilot")
    parser.add_argument("--source_dir", default="outputs/sparktts_transition_4096/samples")
    parser.add_argument("--clips", type=int, default=4)
    parser.add_argument("--codec_dir", default="checkpoints/sparktts/Spark-TTS-0.5B")
    parser.add_argument("--codec_device", default="cuda")
    parser.add_argument("--stt_model", default="checkpoints/stt/faster-whisper-small.en")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261027)
    # This first plot intentionally has a fixed small experimental grid.
    args = parser.parse_args()
    args.strides_ms, args.agreements, args.context_words = [200, 500], [2, 3], [1, 2]
    args.margin_ms, args.delay_ms = 200, 100
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        config = vars(args).copy()
        config["versions"] = {name: importlib.metadata.version(name)
                              for name in ("faster-whisper", "ctranslate2", "torch", "numpy")}
        config["scope"] = "prefix replay of saved generations, not closed-loop watermarked generation"
        if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Output directory contains a different experiment config")
        atomic_json(out / "config.json", config)
        manifest_path = out / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else select_clips(args.source_dir, args.clips)
        atomic_json(manifest_path, manifest)
        atomic_json(out / "progress.json", {"status": "preparing_prefix_audio"})
        prepare(args, manifest, out)
        transcribe(args, manifest, out)
        analyze(args, manifest, out)
        atomic_json(out / "progress.json", {"status": "complete", "clips": len(manifest)})
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
