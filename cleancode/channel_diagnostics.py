#!/usr/bin/env python3
"""Compare old and expanded channel estimates on identical evaluation clips."""
import argparse
import json
from pathlib import Path

import numpy as np

from watermark_audio import ROOT, average_paths, log_e_path


def compare(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((out / "manifest.json").read_text())
    baseline = out / "two_clip_baseline"
    matrices, counts, sizes = {}, {}, {}
    for label, folder in (("original", baseline), ("expanded", out)):
        with np.load(folder / "channel.npz") as archive:
            matrices[label] = archive["channel"].copy()
            counts[label] = archive["counts"].copy()
            if not np.array_equal(archive["groups"], np.load(out / "groups.npy")):
                raise ValueError("Channel comparison requires the same partition.")
        training = json.loads((folder / "channel_training.json").read_text())
        sizes[label] = len(training["calibration_files"])

    # Plot the matrices with actual row coverage; off-diagonal mass represents
    # observed codec loss, not merely uncertainty about a matrix entry.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, label in zip(axes, matrices):
        channel = matrices[label]
        im = ax.imshow(channel, vmin=0, vmax=1, cmap="Blues")
        for a in range(len(channel)):
            for b in range(len(channel)):
                ax.text(b, a, f"{channel[a,b]:.2f}", ha="center", va="center",
                        color="white" if channel[a,b] > .55 else "black")
        ax.set(xticks=range(len(channel)), yticks=range(len(channel)),
               xlabel="Recovered group", ylabel="Original group",
               title=f'{sizes[label]} calibration clips; {counts[label].sum()} pairs\n'
                     f'Row counts: {counts[label].sum(1).tolist()}')
    fig.colorbar(im, ax=axes, label="P(recovered group | original group)")
    fig.suptitle(f'{manifest["model"]}: channel estimates (same partition)')
    for extension in ("png", "pdf"):
        fig.savefig(out / f"channel_estimates.{extension}", dpi=180)
    plt.close(fig)

    alpha = json.loads((out / "detection.json").read_text())["alpha"]
    threshold = np.log(1 / alpha)
    groups = np.load(out / "groups.npy")
    paths = {(kind, label): [] for kind in ("watermarked", "unwatermarked") for label in matrices}
    reports = []
    for record in manifest["records"]:
        if record["kind"] == "calibration":
            continue
        name = record["name"]
        with np.load(out / f"{name}.npz") as trace:
            original, observed = groups[trace["original"]], groups[trace["observed"]]
            mass = trace["group_mass"]
            n = min(len(original), len(observed))
            row = dict(name=name, kind=record["kind"], compared_tokens=n,
                       group_agreement=float(np.mean(original[:n] == observed[:n])),
                       mean_anchor_entropy=float(-(mass * np.log(np.maximum(mass, 1e-300))).sum(1).mean()))
            for label, channel in matrices.items():
                path = log_e_path(mass, observed, channel, (out / f"{name}.key").read_bytes())
                if label == "original":
                    with np.load(baseline / f"{name}.evidence.npz") as saved:
                        if path.shape != saved["roundtrip"].shape or not np.allclose(path, saved["roundtrip"], rtol=1e-12, atol=1e-12):
                            raise ValueError(f"Original evidence cannot be reproduced for {name}; evaluation inputs changed.")
                paths[record["kind"], label].append(path)
                hits = np.flatnonzero(path >= threshold)
                row[label] = dict(final_log_e=float(path[-1]),
                                  first_detection=int(hits[0]) if len(hits) else None,
                                  mean_channel_log_probability=float(np.log(channel[original[:n], observed[:n]]).mean()))
        reports.append(row)
    (out / "channel_comparison.json").write_text(json.dumps(dict(
        model=manifest["model"], calibration_clip_counts=sizes, alpha=alpha,
        scope="Fixed evaluation clips; diagnostic comparison, not model selection or new independent evaluation.",
        samples=reports), indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, kind in zip(axes, ("watermarked", "unwatermarked")):
        for label in matrices:
            mean, sem = average_paths(paths[kind, label])
            t = np.arange(len(mean))
            line, = ax.plot(t, mean, label=f"{sizes[label]} calibration clips")
            if len(paths[kind, label]) > 1:
                step = max(1, len(t) // 12)
                ax.errorbar(t[::step], mean[::step], yerr=sem[::step], fmt="none",
                            ecolor=line.get_color(), capsize=2, alpha=.6)
        ax.axhline(threshold, color="black", linestyle=":", label="log(1/alpha)")
        ax.set(title=f'{kind} (same {len(paths[kind, "original"])} clips)',
               xlabel="Token count", ylabel="Mean cumulative log e")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'{manifest["model"]}: expanded calibration; error bars = ±1 SEM')
    for extension in ("png", "pdf"):
        fig.savefig(out / f"channel_comparison.{extension}", dpi=180)
    plt.close(fig)
    for row in reports:
        print(row["name"], "original:", row["original"], "expanded:", row["expanded"])
    print(f"Saved channel_estimates and channel_comparison plots in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["sparktts", "cosyvoice3"])
    args = parser.parse_args()
    compare(ROOT / "cleancode/watermark_samples" / args.model)
