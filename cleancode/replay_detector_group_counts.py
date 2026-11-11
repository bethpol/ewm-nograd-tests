"""CPU detector regrouping on fixed K=16 audio; not new K=32/64 embedding."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from watermark.position_grouped import position_uniform

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/automatic_sweep_3531161'
CHANNELS = ROOT / 'outputs/channel_fit_extended_32_64/channels'
OUT = ROOT / 'cleancode/watermark_samples/local_detector_group_replay'
SEEDS = (20261104, 20261105, 20261106)
COUNTS = (16, 32, 64)


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    threshold = np.log(100)
    rows = []
    for seed in SEEDS:
        channels = {}
        for k in COUNTS:
            archive = (SOURCE / 'frozen_channels' if k == 16 else CHANNELS) / f'm{k}_seed{seed}_full.npz'
            with np.load(archive) as f:
                channels[k] = (f['groups'].copy(), f['channel'].copy())
        for mode in ('watermarked', 'ordinary'):
            folders = sorted((SOURCE / 'samples' / f'm16_seed{seed}' / mode).glob('clip_*'))
            assert len(folders) == 16
            for folder in folders:
                trace = torch.load(folder / 'trace.pt', map_location='cpu', weights_only=False)
                meta = json.loads((folder / 'metadata.json').read_text())
                key_path = Path(meta['key_file'])
                key = (key_path if key_path.is_absolute() else ROOT / key_path).read_bytes()
                probabilities = np.stack([p.numpy() for p in trace['p0']])
                uniforms = np.array([position_uniform(key, t) for t in range(len(probabilities))])
                with np.load(folder / 'full_observation.npz') as f:
                    valid = (f['positions'] >= 0) & (f['positions'] < len(probabilities))
                    positions, observed = f['positions'][valid], f['tokens'][valid]
                for k, (groups, channel) in channels.items():
                    masses = np.stack([np.bincount(groups, weights=p, minlength=k) for p in probabilities])
                    masses /= masses.sum(1, keepdims=True)
                    cdf = np.cumsum(masses, axis=1)
                    cdf[:, -1] = 1.
                    expected = (cdf <= uniforms[:, None]).sum(1)
                    recovered = groups[observed]
                    increments = np.log(channel[expected[positions], recovered]) - np.log(
                        (masses[positions] * channel[:, recovered].T).sum(1))
                    path = np.r_[0., np.cumsum(increments)]
                    assert np.isfinite(path).all()
                    if k == 16:
                        saved = np.load(folder / 'full_matched_same_key_path.npy')
                        np.testing.assert_allclose(path[1:], saved[:, 1], atol=1e-9, rtol=1e-9)
                        if mode == 'watermarked':
                            np.testing.assert_array_equal(expected, trace['selected_groups'].numpy())
                    dest = OUT / f'detector_k{k}' / f'seed{seed}' / mode
                    dest.mkdir(parents=True, exist_ok=True)
                    path_file = dest / f'{folder.name}.npy'
                    np.save(path_file, path)
                    hits = np.flatnonzero(path >= threshold)
                    rows.append(dict(embedding_groups=16, detector_groups=k, partition_seed=seed,
                                     mode=mode, clip=folder.name, matched_embedding=k == 16,
                                     original_group_rule_match_fraction=float(np.mean(expected == groups[trace['original_tokens'].numpy()])),
                                     max_log_e=float(path.max()), final_log_e=float(path[-1]),
                                     detected=bool(len(hits)), first_crossing_token=int(hits[0]) if len(hits) else '',
                                     scored_tokens=len(positions), source=str(folder.relative_to(ROOT)),
                                     evidence_path=str(path_file.relative_to(ROOT))))
            print(f'Replayed partition {seed} on CPU', flush=True)
    assert len(rows) == 288
    save_csv(OUT / 'per_clip.csv', rows)
    summary = []
    for k in COUNTS:
        for mode in ('watermarked', 'ordinary'):
            selected = [r for r in rows if r['detector_groups'] == k and r['mode'] == mode]
            assert len(selected) == 48
            hits = sum(r['detected'] for r in selected)
            summary.append(dict(embedding_groups=16, detector_groups=k, mode=mode,
                                matched_embedding=k == 16, clips=48, detections=hits, rate=hits/48))
    save_csv(OUT / 'summary.csv', summary)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, k in zip(axes, COUNTS):
        for x, mode, color in [(0, 'watermarked', '#0072B2'), (1, 'ordinary', '#D55E00')]:
            values = [r['max_log_e'] for r in rows if r['detector_groups'] == k and r['mode'] == mode]
            if np.ptp(values):
                parts = ax.violinplot(values, positions=[x], widths=.75, showmedians=True, showextrema=False,
                                      quantiles=[[.25, .75]])
                for body in parts['bodies']:
                    body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.5)
                for name in ('cmedians', 'cquantiles'):
                    parts[name].set_color(color)
            else:
                ax.hlines(values[0], x-.3, x+.3, color=color, linewidth=2)
            stat = next(r for r in summary if r['detector_groups'] == k and r['mode'] == mode)
            ax.text(x, .97, f"{stat['detections']}/48 detected", ha='center', va='top',
                    transform=ax.get_xaxis_transform(), color=color, fontsize=9)
        ax.axhline(threshold, color='black', linestyle=':', label='Threshold: log(100)')
        ax.set(title=f'Detector K={k}\n' + ('Matched to embedding' if k == 16 else 'Different from embedding'),
               xticks=(0, 1), xticklabels=('Watermarked', 'Ordinary'))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Maximum cumulative log evidence per clip')
    axes[-1].legend(fontsize=8, loc='center right')
    fig.suptitle('Local detector replay: the same audio was generated with K=16\n48 watermarked + 48 controls; all 3 partitions pooled', fontsize=13)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.text(.5, .01, 'K=32/64 change only the detector; these are not watermarks embedded with 32/64 groups.\nMaximum includes initial log(E₀)=0. Violin bars: median and quartiles. Full WAV observations and saved probabilities reused.\nK=32/64 use less-stable channel fits from the same 512 calibration clips; no new audio or inference.', ha='center', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'detector_group_violins.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'README.md').write_text('''# Local detector regrouping comparison

This runs K=16,32,64 detectors on exactly the same existing K=16-generated Spark-TTS clips: 48 watermarked and 48 ordinary controls across three partitions. It uses saved full-waveform recovered tokens, full generation distributions, and each recording's existing key. No audio generation, retokenization, GPU, new keys, or synthetic observations are used.

Only K=16 is matched to the embedding rule. K=32/64 recompute group probabilities and keyed expected groups under different detector groupings; the existing waveform does not acquire a newly embedded K=32/64 watermark. Their rates measure a grouping mismatch on K=16 watermarks, not the effect of embedding with more groups, and cannot identify diminishing returns from limited calibration data. The original meeting plots are kept separate to avoid mixing these experiments.

All 96 K=16 replay paths are checked numerically against the archived paths, and the keyed expected groups are checked against the saved generation selections for watermarked clips. Maximum evidence includes initial log evidence zero. The 32/64 channels use the existing 512 calibration and 128 validation clips and failed earlier stability screens. Per-clip maxima, first crossings, final evidence, embedding/detector group counts, and artifact paths are in per_clip.csv; aggregate counts are in summary.csv.

Run `.venv/bin/python -m cleancode.replay_detector_group_counts` to reproduce locally on CPU.
''')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
