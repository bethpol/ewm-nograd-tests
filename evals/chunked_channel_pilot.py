"""Small aligned group-channel comparison using existing generated audio."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.keyed_sparktts import sparktts_classes

out = Path(os.environ["EWM_RUN"])
out.mkdir(parents=True, exist_ok=True)
source = Path(os.environ["EWM_SAMPLES"])
files = sorted(source.glob("sample_*/generated.wav"))[:16]
if not files:
    raise RuntimeError(f"No audio found in {source}")
for path in files:
    if not path.with_name("original_tokens.txt").exists():
        raise RuntimeError(f"Missing original tokens: {path.parent}")

started = time.monotonic()
deadline = started + 45 * 60
torch.set_num_threads(8)
_, Wrapper = sparktts_classes()
device = torch.device("cuda")
encoder = Wrapper(
    "checkpoints/sparktts/Spark-TTS-0.5B", device
)
encoder.tok.model.eval()
encoder.tok.feature_extractor.eval()

hop = 320
chunk = 50
left = 100
looks = [0, 10, 25, 50]
names = ["full"] + [f"look_{r:02d}" for r in looks]
counts = {
    (seed, groups, name): np.zeros((groups, groups), dtype=np.int64)
    for seed in [20261104, 20261105, 20261106]
    for groups in [2, 4]
    for name in names
}
maps = {}
for seed in [20261104, 20261105, 20261106]:
    order = np.random.default_rng(seed).permutation(8192)
    for groups in [2, 4]:
        mapping = np.empty(8192, dtype=np.int64)
        mapping[order] = np.arange(8192) % groups
        maps[seed, groups] = mapping
        np.save(out / f"map_seed{seed}_groups{groups}.npy", mapping)

records = []

def encode(audio):
    with torch.inference_mode():
        result = encoder.encode_semantic(
            torch.as_tensor(audio.copy(), dtype=torch.float32)
        )
    return result.detach().cpu().numpy().reshape(-1).astype(np.int64)

def report():
    rows = []
    for (seed, groups, name), matrix in counts.items():
        total = int(matrix.sum())
        if not total:
            continue
        joint = matrix / total
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        conditional = np.divide(
            joint, px[:, None], out=np.zeros_like(joint),
            where=px[:, None] > 0
        )
        positive = joint > 0
        hcond = float(-np.sum(
            joint[positive] * np.log2(conditional[positive])
        ))
        hy = float(-np.sum(py[py > 0] * np.log2(py[py > 0])))
        row = dict(
            seed=seed, groups=groups, condition=name,
            lookahead_seconds=0 if name == "full" else int(name[5:]) * .02,
            pairs=total, recovery=float(np.trace(joint)),
            conditional_entropy_bits=hcond,
            mutual_information_bits=hy-hcond,
            input_masses=json.dumps(px.tolist()),
            output_masses=json.dumps(py.tolist()),
        )
        rows.append(row)
        np.savez(
            out / f"channel_seed{seed}_groups{groups}_{name}.npz",
            counts=matrix, empirical_channel=conditional,
            input_masses=px, output_masses=py
        )
    if not rows:
        return
    baseline = {
        (r["seed"], r["groups"]): r for r in rows if r["condition"] == "full"
    }
    for row in rows:
        base = baseline[row["seed"], row["groups"]]
        row["delta_conditional_entropy_bits"] = (
            row["conditional_entropy_bits"] - base["conditional_entropy_bits"]
        )
        row["delta_mutual_information_bits"] = (
            row["mutual_information_bits"] - base["mutual_information_bits"]
        )
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "clips.json").write_text(json.dumps(records, indent=2))
    fig, axes = plt.subplots(2, len(names), figsize=(15, 6), squeeze=False)
    for i, groups in enumerate([2, 4]):
        for j, name in enumerate(names):
            matrix = counts[20261104, groups, name]
            totals = matrix.sum(axis=1, keepdims=True)
            channel = np.divide(
                matrix, totals, out=np.zeros_like(matrix, dtype=float),
                where=totals > 0
            )
            ax = axes[i, j]
            ax.imshow(channel, vmin=0, vmax=1, cmap="Blues")
            for a in range(groups):
                for b in range(groups):
                    ax.text(b, a, f"{channel[a,b]:.2f}", ha="center", va="center")
            ax.set_title(f"{groups} groups / {name}")
            ax.set_xlabel("Recovered group")
            ax.set_ylabel("Original group")
            ax.set_xticks(range(groups))
            ax.set_yticks(range(groups))
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"channels.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for groups in [2, 4]:
        for metric, ax in zip(
            ["conditional_entropy_bits", "mutual_information_bits"], axes
        ):
            values = [
                [r[metric] for r in rows
                 if r["groups"] == groups and r["condition"] == name]
                for name in names
            ]
            ax.plot(
                range(len(names)), np.mean(values, axis=1),
                marker="o", label=f"{groups} groups"
            )
            ax.set_xticks(range(len(names)), names, rotation=25)
            ax.set_ylabel(metric.replace("_", " "))
            ax.legend()
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"entropy_information.{ext}", dpi=180)
    plt.close(fig)

(out / "config.json").write_text(json.dumps({
    "source": str(source.resolve()), "max_clips": len(files),
    "chunk_tokens": chunk, "left_context_tokens": left,
    "lookahead_tokens": looks, "hop_samples": hop,
    "sample_rate": encoder.sr,
    "alignment": "known recording start; sample offsets are multiples of 320",
    "grouping": "fixed random partitions balanced by vocabulary size, not probability",
    "entropy": "empirical plugin estimates; no smoothing",
    "comparison": "common valid positions across all conditions within each clip",
    "uncertainty": "three partition seeds show partition sensitivity, not confidence intervals",
    "preprocessing": "existing wrapper applies its normal per-window audio preprocessing",
}, indent=2))

for path in files:
    if time.monotonic() > deadline:
        print("Stopping between clips at the internal time budget", flush=True)
        break
    clip_started = time.monotonic()
    audio, sr = sf.read(path, dtype="float32")
    if sr != encoder.sr or audio.ndim != 1:
        raise ValueError(f"Expected mono {encoder.sr} Hz: {path}")
    original = np.loadtxt(
        path.with_name("original_tokens.txt"), dtype=np.int64, ndmin=1
    ).reshape(-1)
    full = encode(audio)
    n = min(len(original), len(full))
    recovered = {"full": full[:n]}
    for look in looks:
        result = np.full(n, -1, dtype=np.int64)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            start = max(0, a-left)
            stop_sample = min(len(audio), (b+look)*hop)
            window = encode(audio[start*hop:stop_sample])
            offset = a-start
            available = min(b-a, max(0, len(window)-offset))
            if available:
                result[a:a+available] = window[offset:offset+available]
        recovered[f"look_{look:02d}"] = result
    valid = np.ones(n, dtype=bool)
    for result in recovered.values():
        valid &= (result >= 0) & (result < 8192)
    valid &= (original[:n] >= 0) & (original[:n] < 8192)
    if not valid.any():
        raise RuntimeError(f"No common aligned positions: {path}")
    for (seed, groups), mapping in maps.items():
        x = mapping[original[:n][valid]]
        for name, result in recovered.items():
            y = mapping[result[valid]]
            np.add.at(counts[seed, groups, name], (x, y), 1)
    index = len(records)
    np.savez(
        out / f"clip_{index:03d}_tokens.npz",
        original=original[:n], valid=valid, **recovered
    )
    records.append({
        "source": str(path), "original_tokens": len(original),
        "full_tokens": len(full), "common_pairs": int(valid.sum()),
        "excluded_from_shared_prefix": int(n-valid.sum()),
        "unpaired_original_tail": len(original)-n,
        "unpaired_full_tail": len(full)-n,
        "seconds": time.monotonic()-clip_started,
        "token_agreement_with_full": {
            name: float(np.mean(result[valid] == full[:n][valid]))
            for name, result in recovered.items()
        }
    })
    report()
    print("CLIP " + json.dumps(records[-1]), flush=True)

if not records:
    raise RuntimeError("No clips completed")
(out / "status").write_text("complete\n")
print(f"COMPLETE: {len(records)} clips; results in {out}", flush=True)
