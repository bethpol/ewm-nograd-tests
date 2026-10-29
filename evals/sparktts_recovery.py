"""Bounded recovery benchmark using existing repo attacks and generated clips.

This measures token/group recovery, not a watermark test or synchronization
algorithm. Timing-changing attacks are explicitly separated in reports.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy import sparse
import soundfile as sf
import torch

from evals.sparktts_transition import atomic_json
from watermark.keyed_sampling import match_statistics


def groupings(path, seed=1234):
    with open(path, "rb") as handle:
        labels = np.asarray(pickle.load(handle)[0][(10, 1.0)], dtype=np.int64)
    if labels.shape != (8192,) or (labels < 0).any():
        raise ValueError("Expected a complete nonnegative Spark-TTS grouping")
    _, labels = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(seed)
    m = int(labels.max()) + 1
    return {
        "token": np.arange(8192),
        f"shipped_leiden_{m}": labels,
        f"shuffled_leiden_{m}": rng.permutation(labels),
        "random_64": rng.permutation(np.arange(8192) % 64),
        "random_256": rng.permutation(np.arange(8192) % 256),
    }


def attack_suite(sr):
    from training.augmentations import (
        Identity, Speed, NoiseInjection, LowpassFilter, HighpassFilter,
        Smooth, TimeShift, TemporalCrop,
    )
    return [
        ("identity", Identity(), 0., False),
        ("noise_std_0.001", NoiseInjection(), .001, False),
        ("noise_std_0.01", NoiseInjection(), .01, False),
        ("noise_std_0.05", NoiseInjection(), .05, False),
        ("lowpass_3000hz", LowpassFilter(sample_rate=sr), 3000, False),
        ("highpass_1000hz", HighpassFilter(sample_rate=sr), 1000, False),
        ("smooth_1ms", Smooth(sample_rate=sr), .001, False),
        ("speed_1.1", Speed(sample_rate=sr), 1.1, True),
        ("shift_10ms", TimeShift(sample_rate=sr), 10, True),
        ("shift_40ms", TimeShift(sample_rate=sr), 40, True),
        ("crop_keep_50pct", TemporalCrop(), .5, True),
    ]


def summarize(rows):
    totals = {}
    for row in rows:
        key = (row["attack"], row["grouping"])
        if key not in totals:
            totals[key] = {"attack": key[0], "grouping": key[1],
                           "timing_changed": row["timing_changed"], "clips": 0,
                           "matches": 0, "expected_tokens": 0, "compared_tokens": 0,
                           "missing_tokens": 0, "extra_tokens": 0}
        record = totals[key]
        record["clips"] += 1
        for field in ("matches", "expected_tokens", "compared_tokens", "missing_tokens", "extra_tokens"):
            record[field] += int(row[field])
    for record in totals.values():
        record["recovery_rate"] = record["matches"] / record["expected_tokens"]
        record["compared_recovery_rate"] = record["matches"] / record["compared_tokens"]
    return list(totals.values())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    groups = list(dict.fromkeys(r["grouping"] for r in summary))
    for ax, timing in zip(axes, (False, True)):
        attacks = list(dict.fromkeys(r["attack"] for r in summary if r["timing_changed"] == timing))
        x = np.arange(len(attacks))
        for index, group in enumerate(groups):
            values = [next(r["recovery_rate"] for r in summary
                           if r["attack"] == attack and r["grouping"] == group) for attack in attacks]
            ax.bar(x + (index - (len(groups) - 1) / 2) * .15, values,
                   width=.145, label=group)
        ax.set_xticks(x, attacks, rotation=35, ha="right")
        ax.set(ylim=(0, 1), ylabel="Index-matched recovery / original token count",
               title="Timing-changing attacks: no synchronization correction" if timing
               else "Waveform changes preserving nominal timing")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Spark-TTS recovery: {summary[0]['clips']} held-out clips per condition", fontsize=15)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"recovery_comparison.{ext}", dpi=250)
    plt.close(fig)


def run(args):
    from evals.wm_sparktts import SparkTTSTokenizerWrapper
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    maps = groupings(args.grouping_file)
    source = [p for p in sorted(Path(args.samples_dir).glob("sample_*"))
              if (p / "generated.wav").exists() and (p / "original_tokens.txt").exists()][:args.nsamples]
    if len(source) != args.nsamples:
        raise ValueError("Not enough complete source clips")
    config = {k: v for k, v in vars(args).items()}
    config.update({"source_clips": [str(p.resolve()) for p in source],
                   "metric": "same-index recovery, no synchronization or detector p-values",
                   "excluded_attacks": {"MP3": "ffmpeg executable unavailable in current environment",
                       "neural_codecs": "additional codec packages/weights not set up for this bounded run"}})
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Output directory already has a different configuration")
    atomic_json(out / "config.json", config)
    np.savez(out / "groupings.npz", **maps)
    atomic_json(out / "progress.json", {"status": "loading_codec", "completed_conditions": 0,
                                         "target_conditions": args.nsamples * 11})
    tokenizer = SparkTTSTokenizerWrapper(args.model_dir, torch.device(args.device))
    suite = attack_suite(tokenizer.sr)
    rows = []
    all_counts = defaultdict(Counter)
    started = time.time()
    done = 0
    for clip_index, folder in enumerate(source):
        original = np.loadtxt(folder / "original_tokens.txt", dtype=np.int64, ndmin=1)
        wav, sr = sf.read(folder / "generated.wav")
        if sr != tokenizer.sr or wav.ndim != 1:
            raise ValueError("Expected mono audio at codec sample rate")
        audio = torch.tensor(wav, dtype=torch.float32, device=args.device).reshape(1, 1, -1)
        for attack_index, (name, aug, strength, timing) in enumerate(suite):
            destination = out / "samples" / folder.name / name
            destination.mkdir(parents=True, exist_ok=True)
            result_path = destination / "result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                recovered = np.loadtxt(destination / "recovered_tokens.txt", dtype=np.int64, ndmin=1)
                rows.extend(saved)
            else:
                torch.manual_seed(args.seed + clip_index * 100 + attack_index)
                with torch.inference_mode():
                    attacked, _ = aug.to(args.device)(audio.clone(), None, strength)
                    recovered = tokenizer.encode_semantic(attacked).cpu().numpy()
                if not len(recovered):
                    raise RuntimeError(f"No recovered tokens for {folder.name}/{name}")
                np.savetxt(destination / "recovered_tokens.txt", recovered, fmt="%d")
                if clip_index < args.save_audio_clips:
                    sf.write(destination / "audio.wav", attacked.detach().cpu().numpy().reshape(-1),
                             tokenizer.sr, subtype="FLOAT")
                saved = []
                for group, mapping in maps.items():
                    saved.append({"sample": folder.name, "attack": name, "strength": strength,
                                  "timing_changed": timing, "grouping": group,
                                  **match_statistics(mapping[recovered], mapping[original])})
                atomic_json(result_path, saved)
                rows.extend(saved)
            n = min(len(original), len(recovered))
            all_counts[name].update(zip(original[:n].tolist(), recovered[:n].tolist()))
            done += 1
            atomic_json(out / "progress.json", {"status": "running", "completed_conditions": done,
                         "target_conditions": len(source) * len(suite), "elapsed_seconds": time.time() - started,
                         "last_sample": folder.name, "last_attack": name})
            print(f"Condition {done}/{len(source) * len(suite)}: {folder.name} {name}", flush=True)
        write_csv(out / "per_clip.csv", rows)
        write_csv(out / "summary.csv", summarize(rows))
    counts_dir = out / "transition_counts"
    counts_dir.mkdir(exist_ok=True)
    for name, counts in all_counts.items():
        r, c = np.asarray(list(counts), dtype=np.int64).T
        matrix = sparse.csr_matrix((list(counts.values()), (r, c)), shape=(8192, 8192))
        sparse.save_npz(counts_dir / f"{name}.npz", matrix)
    summary = summarize(rows)
    plot_results(summary, out)
    atomic_json(out / "progress.json", {"status": "complete", "completed_conditions": done,
                 "target_conditions": done, "elapsed_seconds": time.time() - started})
    print(f"Complete. Plot: {(out / 'recovery_comparison.png').resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", default="outputs/sparktts_transition_64/samples")
    parser.add_argument("--output_dir", default="outputs/sparktts_recovery_16")
    parser.add_argument("--model_dir", default="checkpoints/sparktts/Spark-TTS-0.5B")
    parser.add_argument("--grouping_file", default="models/embeddings/clusterings_sparktts/leiden_clusterings_trainonly_allparams.pkl")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--save_audio_clips", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.nsamples < 1:
        parser.error("nsamples must be positive")
    try:
        run(args)
    except BaseException as exc:
        path = Path(args.output_dir) / "progress.json"
        if path.exists():
            record = json.loads(path.read_text())
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(path, record)
        raise


if __name__ == "__main__":
    main()
