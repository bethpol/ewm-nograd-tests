"""Plot aligned active-word contexts on the original audio sample grid."""

import argparse
import json
from pathlib import Path

import numpy as np

from watermark.word_anchors import context_timeline, agreement_statistics
from evals.sparktts_transition import atomic_json


def runs(values):
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or values[end] != values[start]:
            yield start, end, values[start]
            start = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="sample_1050")
    parser.add_argument("--arm", choices=["detection_clean", "detection_noise", "detection_lowpass"], default="detection_clean")
    parser.add_argument("--policy", choices=["timestamp_local", "prefix_baseline"], default="timestamp_local")
    parser.add_argument("--replay_dir", default="outputs/rolling_stt_local_replay_v2")
    parser.add_argument("--source_dir", default="outputs/rolling_stt_pilot")
    parser.add_argument("--stride_ms", type=int, default=200)
    parser.add_argument("--agreement", type=int, default=2)
    parser.add_argument("--events_dir", default=None)
    parser.add_argument("--embedder_label", default="token-prefix decoding")
    args = parser.parse_args()
    render(args)


def render(args):
    source = Path(args.source_dir)
    data = (Path(args.events_dir) if args.events_dir else Path(args.replay_dir) / args.policy) / "clips" / args.clip
    prep = json.loads((source / "clips" / args.clip / "preparation.json").read_text())
    def load(arm):
        return json.loads((data / f"commits_{arm}_{args.stride_ms}ms_{args.agreement}agree.json").read_text())["events"]
    a_events, b_events = load("generation_prefix"), load(args.arm)
    a = context_timeline(a_events, prep["total_tokens"], 1)
    b = context_timeline(b_events, prep["total_tokens"], 1)
    stats = agreement_statistics(a_events, b_events, prep["total_tokens"], 1)
    stats.pop("activation_index_absolute_differences")
    words = list(dict.fromkeys(x[0] for row in (a, b) for x in row if x))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    palette = plt.get_cmap("tab20").colors
    colors = {word: palette[i % len(palette)] for i, word in enumerate(words)}
    neutral = "#e6e7e9"
    fig, ax = plt.subplots(figsize=(17, 6))
    fig.subplots_adjust(left=.14, right=.98, top=.73, bottom=.33)
    segments = []
    for values, y, role in ((a, 1.65, "embedder"), (b, .65, "detector")):
        for start, end, context in runs(values):
            left, width = start * 320, (end - start) * 320
            color = colors[context[0]] if context else neutral
            ax.broken_barh([(left, width)], (y, .65), facecolors=[color], edgecolors="white", linewidth=.7)
            word = context[0] if context else "no anchor"
            # Short regions remain represented; their labels are in the legend.
            if width >= 4000:
                r, g, blue = matplotlib.colors.to_rgb(color)
                text_color = "white" if .2126*r + .7152*g + .0722*blue < .48 else "black"
                ax.text(left + width / 2, y + .325, word, ha="center", va="center", fontsize=9,
                        color=text_color, clip_on=True)
            segments.append({"role": role, "start_sample": left, "end_sample": left + width,
                             "active_word": context[0] if context else None})
    statuses = ["no_anchor" if not x and not y else "agree" if x == y else "disagree"
                for x, y in zip(a, b)]
    status_colors = {"no_anchor": neutral, "agree": "#3b9974", "disagree": "#d35454"}
    for start, end, status in runs(statuses):
        ax.broken_barh([(start * 320, (end - start) * 320)], (.12, .19),
                      facecolors=status_colors[status], linewidth=0)
    ax.set_xlim(0, prep["total_samples"])
    ax.set_ylim(0, 2.45)
    ax.set_yticks([1.975, .975, .215], ["Embedder", "Detector", "Agreement"])
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.set_xlabel("Audio sample index (16,000 samples per second)", fontsize=11)
    ax.ticklabel_format(axis="x", style="plain")
    top = ax.secondary_xaxis("top", functions=(lambda s: s/16000, lambda t: t*16000))
    top.set_xlabel("Time in recording (seconds)", labelpad=9)
    ax.grid(axis="x", alpha=.2)
    for spine in ("left", "right"):
        ax.spines[spine].set_visible(False)
    legend = [Patch(facecolor=neutral, label="No confirmed anchor")]
    legend += [Patch(facecolor=colors[word], label=word) for word in words]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.55, .055),
               ncol=7, frameon=False, fontsize=10)
    label = {"detection_clean": "clean finished audio", "detection_noise": "noisy finished audio (σ=0.001)",
             "detection_lowpass": "low-pass finished audio (3 kHz)"}[args.arm]
    fig.suptitle(f"{args.clip}: active last-confirmed-word context, {args.policy}\n"
                 f"Embedder: {args.embedder_label}  |  Detector: {label}", fontsize=15, y=.98)
    percentage = stats["active_context_agreement"]
    fig.text(.55, .82, f"{args.stride_ms} ms updates · {args.agreement}-update confirmation · "
             f"active-position agreement: {percentage:.1%}" if percentage is not None else "No active anchors",
             ha="center", fontsize=11)
    fig.text(.55, .012, "Color means active committed word, not the acoustic interval where it was spoken.  "
             "Agreement strip: green = same word; red = different context (including one-sided no anchor).",
             ha="center", fontsize=9)
    out = Path(args.replay_dir) / "timelines"
    out.mkdir(exist_ok=True)
    name = f"{args.clip}_{args.arm}_{args.policy}"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=250)
    plt.close(fig)
    atomic_json(out / f"{name}.json", {"sample_rate": 16000, "token_hop_samples": 320,
                                       "segments": segments, "metrics": stats})
    print((out / f"{name}.png").resolve())


if __name__ == "__main__":
    main()
