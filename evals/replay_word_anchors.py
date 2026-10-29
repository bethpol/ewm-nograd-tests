"""Replay causal commitment policies on saved rolling STT hypotheses, without ASR."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evals.rolling_stt import analyze
from evals.sparktts_transition import atomic_json
from watermark.word_anchors import LocalWordCommitter, WordCommitter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="outputs/rolling_stt_pilot")
    parser.add_argument("--output_dir", default="outputs/rolling_stt_local_replay_v2")
    args = parser.parse_args()
    source, out = Path(args.source_dir), Path(args.output_dir)
    if source.resolve() == out.resolve():
        parser.error("Replay output must differ from the source experiment")
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    atomic_json(out / "config.json", {"source_dir": str(source.resolve()),
        "source_config": config, "policy": "timestamp-local-v2",
        "timestamp_tolerance_ms": 200, "future_audio_used": False,
        "scope": "policy development replay; not an independent held-out experiment"})
    for name, policy in (("prefix_baseline", WordCommitter), ("timestamp_local", LocalWordCommitter)):
        destination = out / name
        destination.mkdir(exist_ok=True)
        analyze(SimpleNamespace(**config), manifest, destination,
                source_dir=source, committer_class=policy)
    baseline = json.loads((out / "prefix_baseline/summary.json").read_text())
    revised = json.loads((out / "timestamp_local/summary.json").read_text())
    # Reproducing the old summary is a guard against changed data/denominators.
    original = json.loads((source / "summary.json").read_text())
    if baseline != original:
        raise RuntimeError("Baseline replay differs from original summary")
    comparisons = []
    for a, b in zip(baseline, revised):
        fields = ("stride_ms", "agreement_updates", "context_words", "detector_arm")
        if any(a[k] != b[k] for k in fields):
            raise RuntimeError("Policy summaries are not aligned")
        comparisons.append({**{k: a[k] for k in fields},
            "baseline_agreement": a["active_context_agreement"],
            "local_agreement": b["active_context_agreement"],
            "baseline_active_positions": a["active_positions"],
            "local_active_positions": b["active_positions"],
            "baseline_embedder_coverage": a["mean_embedder_active_fraction"],
            "local_embedder_coverage": b["mean_embedder_active_fraction"],
            "baseline_detector_coverage": a["mean_detector_active_fraction"],
            "local_detector_coverage": b["mean_detector_active_fraction"]})
    atomic_json(out / "comparison.json", comparisons)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    arms = ("detection_clean", "detection_noise", "detection_lowpass")
    for ax, context in zip(axes, (1, 2)):
        selected = [next(r for r in comparisons if r["stride_ms"] == 200
                         and r["agreement_updates"] == 2 and r["context_words"] == context
                         and r["detector_arm"] == arm) for arm in arms]
        x = np.arange(len(arms))
        for offset, field, label in ((-.18, "baseline_agreement", "Full-prefix baseline"),
                                     (.18, "local_agreement", "Timestamp-local policy")):
            ax.bar(x + offset, [r[field] if r[field] is not None else np.nan for r in selected],
                   width=.35, label=label)
        ax.set_xticks(x, ["Clean", "Noise σ=0.001", "Low-pass 3 kHz"])
        ax.set(ylim=(0, 1), title=f"Last {context} committed word(s)",
               ylabel="Agreement on positions where either side has an anchor")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Word-anchor policy replay: 200 ms updates, two-update confirmation\n"
                 "All four clips retained; no retroactive event changes")
    for extension in ("png", "pdf"):
        fig.savefig(out / f"policy_comparison.{extension}", dpi=250)
    plt.close(fig)
    print(json.dumps([r for r in comparisons if r["stride_ms"] == 200
                      and r["agreement_updates"] == 2], indent=2))
    print(f"Plot: {(out / 'policy_comparison.png').resolve()}")


if __name__ == "__main__":
    main()
