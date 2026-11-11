"""CPU-only regrouping of saved clean token pairs; not a detection experiment."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'cleancode/watermark_samples'
OUT = BASE / 'cpu_group_diagnostics_20261110'


def partition(size, k, pattern, seed):
    ids = np.arange(size)
    if pattern == 'random':
        return np.random.default_rng(seed).permutation(size) % k
    if pattern == 'contiguous':
        return ids * k // size
    if pattern == 'modulo':
        return ids % k
    raise ValueError(pattern)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ('sparktts', 'cosyvoice3'):
        source = BASE / model
        manifest = json.loads((source / 'manifest.json').read_text())
        size = len(np.load(source / 'groups.npy'))
        calibration = [r for r in manifest['records'] if r['kind'] == 'calibration'
                       and not r.get('exclude_from_channel_fit', False)]
        # Use only unwatermarked held-out clips: old watermark clips were generated
        # using a different partition. No saved group masses are reused here.
        evaluation = [r for r in manifest['records'] if r['kind'] == 'unwatermarked']
        pairs = {}
        for record in calibration + evaluation:
            with np.load(source / (record['name'] + '.npz')) as trace:
                n = min(len(trace['original']), len(trace['observed']))
                pairs[record['name']] = (trace['original'][:n].copy(), trace['observed'][:n].copy())
        for k in (2, 4, 8, 16):
            for pattern in ('random', 'contiguous', 'modulo'):
                for seed in (range(10) if pattern == 'random' else (0,)):
                    groups = partition(size, k, pattern, seed)
                    counts = np.zeros((k, k), dtype=np.int64)
                    for record in calibration:
                        a, b = pairs[record['name']]
                        np.add.at(counts, (groups[a], groups[b]), 1)
                    channel = (counts + .5) / (counts.sum(1, keepdims=True) + .5 * k)
                    held = np.zeros_like(counts)
                    for record in evaluation:
                        a, b = pairs[record['name']]
                        np.add.at(held, (groups[a], groups[b]), 1)
                    agreement = np.trace(held) / held.sum()
                    chance = float((held.sum(0) / held.sum()) @ (held.sum(1) / held.sum()))
                    folder = OUT / model / f'{pattern}_k{k}_seed{seed}'
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez(folder / 'channel.npz', channel=channel, counts=counts,
                             heldout_counts=held, groups=groups)
                    metadata = dict(model=model, pattern=pattern, k=k, seed=seed, beta=.5,
                                    calibration_files=calibration, evaluation_files=evaluation,
                                    calibration_pairs=int(counts.sum()), heldout_pairs=int(held.sum()),
                                    source=str(source.relative_to(ROOT)),
                                    scope='Clean positional token-pair diagnostic; no new generation or watermark detection. Same clips reused across partitions. Token IDs have no assumed semantic ordering.')
                    (folder / 'channel_training.json').write_text(json.dumps(metadata, indent=2) + '\n')
                    row = dict(model=model, pattern=pattern, k=k, seed=seed,
                               calibration_clips=len(calibration), calibration_pairs=int(counts.sum()),
                               heldout_clips=len(evaluation), heldout_pairs=int(held.sum()),
                               heldout_agreement=float(agreement), independent_marginal_agreement=chance,
                               chance_adjusted_agreement=float((agreement - chance) / (1 - chance)) if chance < 1 else None,
                               heldout_mean_log_probability=float((held * np.log(channel)).sum() / held.sum()),
                               channel_path=str((folder / 'channel.npz').relative_to(ROOT)),
                               metadata_path=str((folder / 'channel_training.json').relative_to(ROOT)))
                    rows.append(row)
                    if seed == 0:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        im = ax.imshow(channel, vmin=0, vmax=1, cmap='Blues')
                        ax.set(title=f'{model}: {pattern}, K={k}\n{len(calibration)} calibration clips, {counts.sum()} pairs',
                               xlabel='Recovered group', ylabel='Original group')
                        fig.colorbar(im, ax=ax)
                        fig.tight_layout()
                        for ext in ('png', 'pdf'):
                            fig.savefig(folder / f'channel.{ext}', dpi=160)
                        plt.close(fig)
    with (OUT / 'grouping_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, model in zip(axes, ('sparktts', 'cosyvoice3')):
        for pattern in ('random', 'contiguous', 'modulo'):
            means, lows, highs = [], [], []
            for k in (2, 4, 8, 16):
                values = [r['chance_adjusted_agreement'] for r in rows
                          if r['model'] == model and r['pattern'] == pattern and r['k'] == k]
                means.append(np.mean(values)); lows.append(np.mean(values)-min(values)); highs.append(max(values)-np.mean(values))
            ax.errorbar((2, 4, 8, 16), means, yerr=[lows, highs], marker='o', label=pattern, capsize=3)
        ax.set(title=model, xlabel='Number of groups', xticks=(2, 4, 8, 16))
        ax.axhline(0, color='grey', linestyle=':')
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel('Held-out agreement adjusted for marginal chance')
    fig.suptitle('CPU clean-channel diagnostic; 2 held-out clips/model\nRandom bars: range across 10 partition seeds, not confidence intervals', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'grouping_comparison.{ext}', dpi=180)
    plt.close(fig)
    log = ROOT / 'cleancode/experiment_log.csv'
    with log.open(newline='') as f:
        entries = list(csv.reader(f))
    for model, label in (('sparktts', 'Spark'), ('cosyvoice3', 'Cosy')):
        identifier = f'CPU-group-{label}'
        entries = [r for r in entries if r[0] != identifier]
        example = next(r for r in rows if r['model'] == model)
        entries.append([identifier, model, 'Clean CPU regrouping of existing token pairs',
                        'Grouping comparison; channel heatmaps; per-partition metrics',
                        'K=2,4,8,16; random (10 seeds), contiguous IDs, modulo IDs; beta=0.5.',
                        f"{example['calibration_clips']} calibration clips and {example['calibration_pairs']} pairs per P_c; {example['heldout_clips']} held-out unwatermarked clips and {example['heldout_pairs']} pairs. Diagnostic only, no new watermark detection trials; positional pairing; partition seeds reuse clips.",
                        str(OUT.relative_to(ROOT)) + '/grouping_comparison.png; grouping_comparison.pdf; grouping_metrics.csv; ' + model + '/<pattern>_k<K>_seed<S>/channel.npz; channel_training.json; seed0 channel.png/.pdf'])
    with log.open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(entries)
    print(f'Saved {len(rows)} configurations to {OUT}', flush=True)
    for model in ('sparktts', 'cosyvoice3'):
        for pattern in ('random', 'contiguous', 'modulo'):
            r = [r for r in rows if r['model'] == model and r['pattern'] == pattern and r['k'] == 4]
            print(model, pattern, 'K=4 held-out agreement:', round(float(np.mean([x['heldout_agreement'] for x in r])), 4))


if __name__ == '__main__':
    main()
