"""Plot predictive concentration and held-out fit, rather than matrix occupancy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from evals.sparktts_transition import count_transitions, heldout_score


def ranked_mass(train, heldout, max_k=64, errors_only=False):
    train = train.copy().tocsr()
    heldout = heldout.copy().tocsr()
    if errors_only:
        train.setdiag(0)
        heldout.setdiag(0)
        train.eliminate_zeros()
        heldout.eliminate_zeros()
    totals = [int(train.sum()), int(heldout.sum())]
    masses = np.zeros((2, max_k))
    for row in range(train.shape[0]):
        start, stop = train.indptr[row:row + 2]
        cols, values = train.indices[start:stop], train.data[start:stop]
        order = np.lexsort((cols, -values))[:max_k]
        selected = cols[order]
        a = np.zeros(max_k)
        b = np.zeros(max_k)
        a[:len(selected)] = values[order]
        b[:len(selected)] = heldout[row, selected].toarray().ravel()
        masses += np.vstack([np.cumsum(a), np.cumsum(b)])
    return masses / np.array(totals)[:, None]


def baseline_scores(train, heldout, beta):
    k = train.shape[0]
    pairs = heldout.tocoo()
    n = int(train.sum())
    q = (np.asarray(train.sum(axis=0)).ravel() + beta) / (n + k * beta)
    marginal = float(pairs.data @ np.log(q[pairs.col]) / pairs.data.sum())
    off = train.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    error_q = (np.asarray(off.sum(axis=0)).ravel() + beta) / (off.sum() + k * beta)
    recovery = train.diagonal().sum() / n
    copy = np.where(pairs.row == pairs.col, recovery,
                    (1 - recovery) * error_q[pairs.col] / (1 - error_q[pairs.row]))
    return {"output_marginal": marginal,
            "copy_plus_pooled_errors": float(pairs.data @ np.log(copy) / pairs.data.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default="outputs/sparktts_transition_4096")
    parser.add_argument("--heldout_samples_dir", default="outputs/sparktts_transition_64/samples")
    args = parser.parse_args()
    root = Path(args.run_dir)
    checkpoints = sorted((root / "checkpoints").glob("clips_*/evaluation.json"))
    if not checkpoints:
        raise ValueError("No completed checkpoints")
    latest = checkpoints[-1].parent
    train = sparse.load_npz(latest / "transition_counts.npz")
    heldout, _ = count_transitions(args.heldout_samples_dir, train.shape[0])
    all_mass = ranked_mass(train, heldout)
    error_mass = ranked_mass(train, heldout, errors_only=True)
    learning = []
    for evaluation_path in checkpoints:
        folder = evaluation_path.parent
        counts = sparse.load_npz(folder / "transition_counts.npz")
        meta = json.loads((folder / "transition_metadata.json").read_text())
        learning.append({"clips": meta["clips"],
                         "channel": heldout_score(folder / "transition_matrix.npy", heldout)["heldout_mean_log_probability"],
                         **baseline_scores(counts, heldout, meta["beta"])})
    usage = np.asarray(train.sum(axis=1)).ravel()
    degree = np.diff(train.indptr)
    summary = {
        "checkpoint": str(latest), "training_pairs": int(train.sum()),
        "heldout_pairs": int(heldout.sum()), "median_input_observations": float(np.median(usage)),
        "observed_outputs_per_input_median": float(np.median(degree)),
        "rank_tie_break": "ascending output token ID; ranking learned from training only",
        "mass_weighting": "token-frequency weighted; not an equal average over vocabulary rows",
        "top_k": {str(k): {"training_all": float(all_mass[0, k - 1]),
                          "heldout_all": float(all_mass[1, k - 1]),
                          "training_errors": float(error_mass[0, k - 1]),
                          "heldout_errors": float(error_mass[1, k - 1])}
                  for k in (1, 5, 10, 20, 64)},
        "learning_curve": learning,
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, data, title in zip(axes[0], (all_mass, error_mass),
                               ("All transitions (including exact recovery)", "Mismatch transitions only (diagonal removed)")):
        for values, label, color in zip(data, ("Training", "Held-out"), ("#3178a8", "#cf6b30")):
            ax.plot(np.arange(1, 65), values, label=label, color=color, linewidth=2)
        ax.set(xlabel="Top k outputs per input (ranked using training counts)",
               ylabel="Fraction of transitions captured", title=title,
               xlim=(1, 64), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    ax = axes[1, 0]
    valid = usage > 0
    ax.scatter(usage[valid], degree[valid], s=8, alpha=.35, linewidths=0, color="#3178a8")
    lim = max(usage.max(), degree.max())
    ax.plot([1, lim], [1, lim], "--", color="gray", linewidth=1, label="Every observation is a distinct output")
    ax.set(xscale="log", yscale="log", xlabel="Training observations of input token",
           ylabel="Distinct recovered outputs observed", title="Observed support depends on sample count")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for key, label in (("channel", "Estimated token channel"),
                       ("copy_plus_pooled_errors", "Copy + pooled error distribution"),
                       ("output_marginal", "Output marginal (ignores input)")):
        ax.plot([r["clips"] for r in learning], [-r[key] for r in learning], marker="o", label=label)
    ax.set(xlabel="Training clips", ylabel="Held-out log loss (nats per token; lower is better)",
           title="Does knowing the input improve prediction?", xscale="log")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle(f"Spark-TTS channel diagnostics — {latest.name}", fontsize=16)
    out = root / "diagnostics"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"channel_diagnostics.{ext}", dpi=250)
    plt.close(fig)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
