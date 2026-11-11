"""CPU channel-model pilot, not an audio-generation watermark-power result."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from watermark.grouped import effective_channel, group_probabilities, log_e_path, random_groups


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.counts:
        counts = sparse.load_npz(args.counts)
        source = "fitted channel simulation; no new audio generated"
    else:
        # Deliberately simple synthetic channel for testing the pipeline.
        v = 128
        counts = sparse.csr_matrix(100 * np.eye(v) + np.ones((v, v)))
        source = "synthetic copy-plus-uniform channel; no audio data"
    p = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    if not p.sum():
        raise ValueError("Empty training counts")
    p /= p.sum()
    records = []
    for k in args.groups:
        for partition in range(args.partitions):
            seed = args.seed + partition
            mapping = random_groups(len(p), k, seed)
            pooled, channel = effective_channel(counts, mapping, args.beta)
            mass = group_probabilities(p, mapping)
            root = out / f"groups_{k}_partition_{partition}"
            root.mkdir(exist_ok=True)
            np.savez(root / "channel.npz", groups=mapping, counts=pooled,
                     channel=channel, group_mass=mass)
            if partition == 0:
                fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
                im = ax.imshow(channel, vmin=0, vmax=1, cmap="viridis", origin="upper")
                ax.set(xlabel="Recovered group", ylabel="Generated group",
                       title=f"{k} random groups: fitted channel")
                fig.colorbar(im, ax=ax, label="Conditional probability")
                for ext in ("png", "pdf"):
                    fig.savefig(root / f"transition.{ext}", dpi=180)
                plt.close(fig)
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, k, partition]))
            for trial in range(args.trials):
                expected = rng.choice(k, args.tokens, p=mass)
                for null in (False, True):
                    # Null output is independent of the keyed group. Under the
                    # alternative it passes through the fitted group channel.
                    latent = rng.choice(k, args.tokens, p=mass) if null else expected
                    observed = np.array([rng.choice(k, p=channel[g]) for g in latent])
                    path = log_e_path(expected, observed, mass, channel)
                    crossed = np.flatnonzero(path >= np.log(1 / args.alpha))
                    records.append(dict(groups=k, partition=partition, trial=trial,
                                        null=null, detected=bool(len(crossed)),
                                        first_detection=int(crossed[0] + 1) if len(crossed) else None,
                                        horizon=args.tokens))
    with (out / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = []
    for k in args.groups:
        alt = [r for r in records if r["groups"] == k and not r["null"]]
        null = [r for r in records if r["groups"] == k and r["null"]]
        # E[min(T,H)] includes all censored trials; never silently drop failures.
        summary.append(dict(groups=k, detection_rate=np.mean([r["detected"] for r in alt]),
                            restricted_mean_tokens=np.mean([r["first_detection"] or args.tokens for r in alt]),
                            null_crossing_rate=np.mean([r["detected"] for r in null])))
    (out / "summary.json").write_text(json.dumps({"scope": source, "settings": vars(args), "results": summary}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([s["groups"] for s in summary], [s["restricted_mean_tokens"] for s in summary], "o-")
    axes[0].set(ylabel=f"Mean min(first detection, {args.tokens})", xlabel="Number of groups")
    for field, label in (("detection_rate", "Watermark"), ("null_crossing_rate", "Independent null")):
        axes[1].plot([s["groups"] for s in summary], [s[field] for s in summary], "o-", label=label)
    axes[1].axhline(args.alpha, linestyle="--", color="gray", label="alpha")
    axes[1].set(xlabel="Number of groups", ylabel="Crossing fraction", ylim=(-.02, 1.02))
    axes[1].legend()
    fig.suptitle(source)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"detection_vs_groups.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", help="Training transition_counts.npz; omitted uses synthetic data")
    parser.add_argument("--output_dir", default="outputs/grouped_pilot")
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--partitions", type=int, default=3)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    if min(args.partitions, args.trials, args.tokens) <= 0 or not 0 < args.alpha < 1:
        parser.error("Require positive repetitions/horizon and alpha in (0,1)")
    run(args)


if __name__ == "__main__":
    main()
