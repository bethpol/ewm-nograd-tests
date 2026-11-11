"""Compare finite-calibration channel estimates for K=2 through 64, without inference."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cleancode/watermark_samples/sparktts_meeting_report'
GROUPS = (2, 4, 8, 16, 32, 64)
SEEDS = (20261104, 20261105, 20261106)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def band(ax, x, values, label, color):
    values = np.asarray(values)
    ax.plot(x, values.mean(0), marker='o', markersize=4, color=color, label=label)
    ax.fill_between(x, values.min(0), values.max(0), color=color, alpha=.18, linewidth=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics, learning = [], []
    for source, groups in [('channel_fit_3531082', (2, 4, 8, 16)),
                           ('channel_fit_extended_32_64', (32, 64))]:
        base = ROOT / 'outputs' / source
        with (base / 'report/learning_curves.csv').open() as f:
            checkpoints = [r for r in csv.DictReader(f) if r['policy'] == 'full' and int(r['groups']) in groups]
        with (base / 'report/diagnostics.csv').open() as f:
            stats = [r for r in csv.DictReader(f) if r['policy'] == 'full' and int(r['groups']) in groups]
        for raw in stats:
            k, seed = int(raw['groups']), int(raw['partition_seed'])
            with np.load(base / 'channels' / f'm{k}_seed{seed}_full.npz') as archive:
                counts, val, channel = archive['counts'], archive['validation_counts'], archive['channel']
                np.testing.assert_allclose(channel.sum(1), 1)
                assert int(counts.sum()) == int(raw['training_pairs']) == 102641
                assert int(val.sum()) == int(raw['validation_pairs']) == 26904
                beta = json.loads((base / 'config.json').read_text())['beta']
                marginal = (counts.sum(0)+beta)/(counts.sum()+k*beta)
                baseline_loss = float(-(val*np.log(marginal[None, :])).sum()/val.sum())
                off = counts[~np.eye(k, dtype=bool)]
                row = dict(groups=k, partition_seed=seed, training_clips=512, validation_clips=128,
                           training_pairs=int(counts.sum()), validation_pairs=int(val.sum()),
                           heldout_gain_nats=float(raw['heldout_gain_over_marginal_nats']),
                           full_vs_half_tv=float(raw['full_vs_half_tv']),
                           independent_halves_tv=float(raw['independent_halves_tv']),
                           train_validation_tv=float(raw['train_validation_tv']),
                           mean_offdiagonal_pairs=float(off.mean()),
                           zero_offdiagonal_fraction=float((off == 0).mean()),
                           min_row_pairs=int(raw['min_row_pairs']),
                           passed_original_screen=raw['candidate_for_later_review'],
                           source=str(base.relative_to(ROOT)))
                diagnostics.append(row)
            curve = [r for r in checkpoints if int(r['groups']) == k and int(r['partition_seed']) == seed]
            assert [int(r['training_clips']) for r in curve] == [16, 32, 64, 128, 256, 512]
            for r in curve:
                gain = baseline_loss-float(r['heldout_log_loss'])
                if int(r['training_clips']) == 512:
                    np.testing.assert_allclose(gain, row['heldout_gain_nats'], atol=1e-12)
                learning.append(dict(groups=k, partition_seed=seed, calibration_clips=int(r['training_clips']),
                                     heldout_gain_nats=gain, heldout_loss=float(r['heldout_log_loss']),
                                     fixed_marginal_baseline_loss=baseline_loss, validation_clips=128))
    write_csv('training_limits_by_groups.csv', diagnostics)
    write_csv('training_size_learning_curves.csv', learning)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    panels = [('heldout_gain_nats', 'Held-out channel information increases', 'Predictive gain over marginal (nats/token)', None),
              ('full_vs_half_tv', 'Larger matrices are less stable with fixed data', 'Change when fitting 256 vs 512 clips (TV)', .02),
              ('train_validation_tv', 'Training / validation estimates diverge', 'Training vs validation channel distance (TV)', .10),
              ('mean_offdiagonal_pairs', 'Fewer examples per possible group flip', 'Mean calibration pairs per off-diagonal cell', None)]
    for ax, (metric, title, ylabel, limit) in zip(axes.flat, panels):
        values = [[next(r[metric] for r in diagnostics if r['groups'] == k and r['partition_seed'] == seed)
                   for k in GROUPS] for seed in SEEDS]
        band(ax, GROUPS, values, 'Mean across partitions', '#0072B2')
        if limit is not None:
            ax.axhline(limit, color='#D55E00', linestyle='--', label='Earlier screening cutoff')
        ax.axvspan(24, 80, color='grey', alpha=.07)
        ax.set(xscale='log', title=title, xlabel='Number of random groups, K', ylabel=ylabel, xlim=(1.8, 72))
        ax.set_xticks(GROUPS, labels=GROUPS)
        if metric == 'mean_offdiagonal_pairs':
            ax.set_yscale('log')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Spark-TTS: training-data limits with K = 2, 4, 8, 16, 32, 64\nSame 512 calibration + 128 held-out validation clips at every K', fontsize=13)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.text(.5, .01, 'Shaded bands = min–max across 3 fixed random partitions, not confidence intervals. Grey area: exploratory K=32,64.\nTV is total-variation distance (0 = identical). Held-out predictive gain measures channel quality, not watermark detection.\nNo larger-K detection results yet: do not interpret these curves as detection performance or proof of an optimal K.', ha='center', fontsize=9)
    save(fig, 'training_limits_by_groups')

    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = (16, 32, 64, 128, 256, 512)
    for k, color in zip(GROUPS, plt.get_cmap('tab10').colors):
        values = [[next(r['heldout_gain_nats'] for r in learning if r['groups'] == k and r['partition_seed'] == seed and r['calibration_clips'] == n)
                   for n in sizes] for seed in SEEDS]
        band(ax, sizes, values, f'K={k}' + (' (exploratory)' if k >= 32 else ''), color)
    ax.axhline(0, color='grey', linestyle=':')
    ax.set(xscale='log', xlabel='Calibration clips used to fit the transition matrix',
           ylabel='Held-out predictive gain over fixed marginal (nats/token)',
           title='Does more calibration data still help larger groupings?\nSame 128 validation clips at every checkpoint')
    ax.set_xticks(sizes, labels=sizes); ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.subplots_adjust(bottom=.23, top=.83)
    fig.text(.5, .02, 'Bands = range across 3 partitions, not confidence intervals. Fits use nested prefixes of the same calibration set.\nBaseline: each partition’s marginal fitted on all 512 calibration clips, held fixed across checkpoints.\nThese are channel prediction curves, not generated watermark detection curves.', ha='center', fontsize=9)
    save(fig, 'training_size_learning_curves')
    (OUT / 'TRAINING_LIMITS.md').write_text('''# Larger-group training-data diagnostic

`training_limits_by_groups.png` compares the same 512 calibration recordings (102641 token pairs) and 128 separate validation recordings (26904 pairs) at K=2,4,8,16,32,64. It shows held-out predictive information, sensitivity to fitting on half the training clips, training/validation disagreement, and available examples per off-diagonal matrix cell. Bands show the range across three fixed random partitions; they are not confidence intervals. Earlier screening thresholds are reference lines, not proof of validity or detection failure. K=32,64 failed these earlier checks and are labeled exploratory.

`training_size_learning_curves.png` shows held-out predictive gain for matrices fitted on 16,32,64,128,256,512 nested calibration clips. The reference marginal distribution is always fitted on all 512 calibration clips within each partition and kept fixed across checkpoints. This avoids changing the baseline while comparing training sizes. It uses no validation observations for fitting. Repeated validation comparisons are exploratory diagnostics.

Held-out channel information still increases through K=64 on these data, while estimates become less stable and adding calibration clips helps more at larger K. This is consistent with growing estimation costs; it does not establish diminishing watermark detection returns or a best K. Detection at K=32,64 requires new matched watermarked generations, with the fixed less-stable channel labeled exploratory. The separate mean-evidence and violin figures remain based on completed K=2,4,8,16 recordings until the extension finishes.

The source files are outputs/channel_fit_3531082 and outputs/channel_fit_extended_32_64. Exact statistics are exported to training_limits_by_groups.csv and training_size_learning_curves.csv. No generation is performed by this plotting script. Regenerate with `.venv/bin/python -m cleancode.plot_group_training_limits`.
''')
    log = ROOT / 'cleancode/experiment_log.csv'
    with log.open(newline='') as f:
        entries = list(csv.reader(f))
    entries = [r for r in entries if r[0] != 'Training-limits-Spark']
    entries.append(['Training-limits-Spark', 'Spark-TTS', 'Fixed-calibration group-count and training-size diagnostic',
                    'Predictive gain; stability; support per flip; calibration-size learning curves',
                    'K=2,4,8,16,32,64; 3 random partitions each; full waveform; beta=0.0001. No inference or generated detections.',
                    'Same 512 calibration clips/102641 pairs and 128 validation clips/26904 pairs. Training prefixes 16,32,64,128,256,512. Bands are partition ranges. K32/64 exploratory; no optimal detection K established.',
                    str(OUT.relative_to(ROOT)) + '/training_limits_by_groups.png/.pdf/.csv; training_size_learning_curves.png/.pdf/.csv; TRAINING_LIMITS.md'])
    with log.open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(entries)
    print(f'Saved training-limit plots for {len(diagnostics)} group/partition settings and {len(learning)} checkpoints', flush=True)


if __name__ == '__main__':
    main()
