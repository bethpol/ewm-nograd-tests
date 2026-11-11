#!/usr/bin/env python3
"""Fit and evaluate a separate grouped channel for each audio transform."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from codec_attacks import CODEC_NAMES, codec_conditions
from random_group_sweep import wilson
from watermark_audio import ROOT, detect, fit_channel, load_model, plot, recover_tokens, save_sample_index

sys.path.insert(0, str(ROOT))


def transform_split(source, run, records, manifest, model, transform, strength, seed):
    import soundfile as sf
    import torch

    for index, record in records:
        name = record["name"]
        wav, sr = sf.read(source / record["audio_path"], dtype="float32")
        if wav.ndim != 1 or sr != model.sample_rate:
            raise ValueError(f"Unexpected audio format for {name}")
        torch.manual_seed(seed + index)
        with torch.inference_mode():
            attacked, _ = transform(torch.from_numpy(wav).reshape(1, 1, -1), None, strength)
            path = run / record["audio_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, attacked.cpu().numpy().reshape(-1), sr, subtype="FLOAT")
            observed = recover_tokens(model, path, manifest["model"])
        if not len(observed):
            raise ValueError(f"No recovered tokens for {path}")
        with np.load(source / f"{name}.npz") as trace:
            original = trace["original"]
            np.savez_compressed(run / f"{name}.npz", original=original, observed=observed,
                                group_mass=trace["group_mass"])
        if record["kind"] != "calibration":
            key = run / f"{name}.key"
            key.touch(mode=0o600, exist_ok=False)
            key.write_bytes((source / key.name).read_bytes())
        manifest["records"].append(dict(record, observed_tokens=len(observed),
                                        aligned_pairs=min(len(original), len(observed)),
                                        duration_seconds=attacked.shape[-1] / sr))
    save_sample_index(run, manifest)


def write_summary(out, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (out / "attack_rates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    attacks = list(dict.fromkeys(row["attack"] for row in rows))
    for kind, label, offset in (("watermarked", "Detection", -.12),
                                ("unwatermarked", "False positive", .12)):
        selected = [row for row in rows if row["kind"] == kind]
        ax.errorbar(np.arange(len(attacks)) + offset, [r["rate"] for r in selected],
                    yerr=[[max(0, r["rate"] - r["ci95_low"]) for r in selected],
                          [max(0, r["ci95_high"] - r["rate"]) for r in selected]],
                    fmt="o", capsize=3, label=label)
    ax.set_xticks(range(len(attacks)), attacks, rotation=35, ha="right")
    ax.set(ylabel="Fraction crossing threshold (95% Wilson interval)", ylim=(-.04, 1.04))
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(out / f"attack_rates.{extension}", dpi=180)
    plt.close(fig)


def plot_channel(run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(run / "channel.npz") as archive:
        channel = archive["channel"]
    fig, ax = plt.subplots(layout="constrained")
    fig.colorbar(ax.imshow(channel, vmin=0, vmax=1), ax=ax, label="Conditional probability")
    ax.set(title=run.name, xlabel="Recovered group", ylabel="Original group")
    for extension in ("png", "pdf"):
        fig.savefig(run / f"channel.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed random grouping run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attack", action="append", help="Repeat to select conditions. Default is the local suite.")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--list-attacks", action="store_true")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "cleancode"):
        parser.error("Keep output inside cleancode/")
    from evals.sparktts_recovery import attack_suite

    base = json.loads((source / "manifest.json").read_text())
    records = list(enumerate(r for r in base["records"] if not r.get("exclude_from_channel_fit", False)))
    if not records:
        parser.error("Source has no usable clips")
    suite = attack_suite(records[0][1]["sample_rate"])
    if args.list_attacks:
        for name, _, _, _ in suite:
            print(name)
        for name in CODEC_NAMES:
            print(f"{name} (opt-in)")
        return
    names = {name for name, _, _, _ in suite} | set(CODEC_NAMES)
    if args.attack and not set(args.attack) <= names:
        parser.error(f"Unknown attack. Choose from {', '.join(sorted(names))}")
    cal = [(i, r) for i, r in records if r["kind"] == "calibration"]
    heldout = [(i, r) for i, r in records if r["kind"] in ("watermarked", "unwatermarked")]
    if not cal or any(sum(r["kind"] == kind for _, r in heldout) < 20
                      for kind in ("watermarked", "unwatermarked")):
        parser.error("Source needs separate calibration and at least 20 clips per evaluation condition")
    if not (source / "COMPLETE").exists():
        parser.error("Use a completed random grouping sweep configuration")
    args.model = base["model"]
    defaults = {"sparktts": "sparktts/Spark-TTS-0.5B", "cosyvoice3": "cosyvoice/Fun-CosyVoice3-0.5B"}
    args.model_dir = args.model_dir or ROOT / "checkpoints" / defaults[args.model]
    beta = json.loads((source / "channel_training.json").read_text())["beta"]
    alpha = json.loads((source / "detection.json").read_text())["alpha"]
    suite.extend(codec_conditions(records[0][1]["sample_rate"], args.attack or ()))
    out.mkdir(parents=True, exist_ok=False)
    model, mapping = load_model(args)
    groups = np.load(source / "groups.npy")
    if len(groups) != len(mapping):
        raise ValueError("Model and source partition vocabularies disagree")
    rows = []
    for name, transform, strength, timing_changed in suite:
        if args.attack and name not in args.attack:
            continue
        print(f"Running {name}", flush=True)
        run = out / name
        run.mkdir()
        np.save(run / "groups.npy", groups)
        manifest = dict(base, records=[], source_run=str(source), attack=name,
                        strength=strength, timing_changed=timing_changed, attack_seed=args.seed)
        transform_split(source, run, cal, manifest, model, transform, strength, args.seed)
        fit_channel(run, beta)
        plot_channel(run)
        transform_split(source, run, heldout, manifest, model, transform, strength, args.seed)
        detect(run, alpha)
        plot(run)
        result = json.loads((run / "detection.json").read_text())
        for kind in ("watermarked", "unwatermarked"):
            trials = [r for r in result["samples"] if r["kind"] == kind]
            hits = sum(r["first_detection"] is not None for r in trials)
            low, high = wilson(hits, len(trials))
            rows.append(dict(model=args.model, num_groups=base["num_groups"], attack=name,
                             strength=strength, timing_changed=timing_changed, kind=kind,
                             alpha=alpha, trials=len(trials), detections=hits, rate=hits / len(trials),
                             ci95_low=low, ci95_high=high))
        write_summary(out, rows)


if __name__ == "__main__":
    main()
