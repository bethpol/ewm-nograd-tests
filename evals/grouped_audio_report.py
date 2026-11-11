"""Fit group channels on separate training counts; report real audio crossings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from evals.grouped_detect import score
from evals.keyed_sparktts import read_key
from watermark.grouped import effective_channel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--key_file", default=".cache/keyed_sparktts/secret.key")
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--beta", type=float, default=1e-4)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("Require alpha in (0,1)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = sparse.load_npz(args.counts)
    key = read_key(args.key_file)
    records = []
    archives = {}
    for trace_path in sorted(Path(args.audio_dir).glob("partition_*/groups_*/sample_*/trace.pt")):
        trace = torch.load(trace_path, map_location="cpu", weights_only=True)
        configuration = trace_path.parent.parent
        tag = str(configuration.relative_to(args.audio_dir))
        if tag not in archives:
            groups = trace["groups"].numpy()
            pooled, channel = effective_channel(counts, groups, args.beta)
            archives[tag] = {"groups": groups, "channel": channel}
            target = out / tag
            target.mkdir(parents=True, exist_ok=True)
            np.savez(target / "channel.npz", groups=groups, counts=pooled, channel=channel)
            fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
            image = ax.imshow(channel, vmin=0, vmax=1, cmap="viridis")
            ax.set(xlabel="Recovered group", ylabel="Original group", title=tag)
            fig.colorbar(image, ax=ax, label="Fitted conditional probability")
            for ext in ("png", "pdf"):
                fig.savefig(target / f"transition.{ext}", dpi=180)
            plt.close(fig)
        for observation in sorted(trace_path.parent.glob("aug_*_tokens.txt")):
            observed = np.loadtxt(observation, dtype=np.int64, ndmin=1)
            result, path = score(trace, observed, key, archives[tag], args.alpha)
            destination = out / tag / trace_path.parent.name
            destination.mkdir(exist_ok=True)
            np.save(destination / f"{observation.stem}_log_e.npy", path)
            records.append({"configuration": tag, "groups": int(trace["groups"].max()) + 1,
                            "sample": trace_path.parent.name, "observation": observation.name,
                            **result})
    if not records:
        raise ValueError("No grouped traces and recovered observations found")
    with (out / "detections.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for k in sorted({r["groups"] for r in records}):
        rows = [r for r in records if r["groups"] == k]
        # Common horizon avoids comparing restricted means at different clip lengths.
        horizon = min(r["compared_tokens"] for r in rows)
        crossings = [r["first_detection"] for r in rows]
        summary.append({"groups": k, "clips": len(rows), "common_horizon": horizon,
                        "detection_fraction": float(np.mean([t is not None and t <= horizon for t in crossings])),
                        "restricted_mean_tokens": float(np.mean([min(t, horizon) if t is not None else horizon for t in crossings]))})
    # Use one horizon across all group counts for an honest comparison.
    horizon = min(s["common_horizon"] for s in summary)
    for item in summary:
        rows = [r for r in records if r["groups"] == item["groups"]]
        item["common_horizon"] = horizon
        item["detection_fraction"] = float(np.mean([r["first_detection"] is not None and r["first_detection"] <= horizon for r in rows]))
        item["restricted_mean_tokens"] = float(np.mean([min(r["first_detection"], horizon) if r["first_detection"] is not None else horizon for r in rows]))
    (out / "summary.json").write_text(json.dumps({"settings": vars(args), "scope": "real generated audio; separate ordinary channel-training clips; stationary pooled channel; saved-p0 replay", "results": summary}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = [s["groups"] for s in summary]
    axes[0].plot(x, [s["restricted_mean_tokens"] for s in summary], "o-")
    axes[0].set(xlabel="Number of groups", ylabel=f"Mean min(first detection, {horizon})")
    axes[1].plot(x, [s["detection_fraction"] for s in summary], "o-")
    axes[1].set(xlabel="Number of groups", ylabel=f"Detected by token {horizon}", ylim=(-.02, 1.02))
    fig.suptitle("Spark-TTS audio pilot: stationary channel, saved-p0 replay")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"detection_vs_groups.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
