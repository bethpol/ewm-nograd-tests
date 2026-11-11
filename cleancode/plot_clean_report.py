"""Make collaborator-facing plots from existing clean detection and channel files."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'cleancode/watermark_samples'
OUT = BASE / 'clean_report'
MODELS = ('sparktts', 'cosyvoice3')
LABELS = ('Spark-TTS', 'CosyVoice3')
COLORS = {'watermarked': '#0072B2', 'unwatermarked': '#D55E00'}


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, datasets = [], []
    for model in MODELS:
        folder = BASE / model
        manifest = json.loads((folder / 'manifest.json').read_text())
        detection = json.loads((folder / 'detection.json').read_text())
        training = json.loads((folder / 'channel_training.json').read_text())
        with np.load(folder / 'channel.npz') as archive:
            channel, counts = archive['channel'].copy(), archive['counts'].copy()
            np.testing.assert_array_equal(archive['groups'], np.load(folder / 'groups.npy'))
        np.testing.assert_allclose(channel.sum(1), 1)
        paths = {}
        for record in detection['samples']:
            with np.load(folder / f"{record['name']}.evidence.npz") as archive:
                path = archive['roundtrip'].copy()
            hits = np.flatnonzero(path >= np.log(1 / detection['alpha']))
            first = int(hits[0]) if len(hits) else None
            if first != record['first_detection']:
                raise ValueError('Saved detection does not match evidence')
            paths[record['name']] = path
        datasets.append((manifest, detection, training, channel, counts, paths))
        for kind in COLORS:
            records = [r for r in detection['samples'] if r['kind'] == kind]
            hits = sum(r['first_detection'] is not None for r in records)
            rows.append(dict(model=model, kind=kind, detected=hits, clips=len(records),
                             rate=hits/len(records), alpha=detection['alpha'],
                             num_groups=manifest['num_groups'], partition_seed=manifest['partition_seed'],
                             calibration_clips=len(training['calibration_files']),
                             calibration_pairs=int(counts.sum()),
                             source=str(folder.relative_to(ROOT))))
    with (OUT / 'detection_rates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for offset, kind in zip((-.18, .18), COLORS):
        selected = [r for r in rows if r['kind'] == kind]
        bars = ax.bar(np.arange(2) + offset, [r['rate'] for r in selected], width=.32,
                      color=COLORS[kind], label='Watermarked: detection rate' if kind == 'watermarked' else 'Unwatermarked: false-positive rate')
        for bar, r in zip(bars, selected):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.025,
                    f"{r['detected']}/{r['clips']} ({r['rate']:.0%})", ha='center', fontsize=10)
    ax.set(xticks=(0, 1), xticklabels=LABELS, ylim=(0, 1.19), ylabel='Clips crossing the detection threshold')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.set_title('Clean audio: observed watermark detection\nExisting random partition, K = 4; threshold E ≥ 100 (α = 0.01)', fontsize=12)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.subplots_adjust(bottom=.25, top=.80)
    fig.text(.5, .02, 'Pilot: 2 watermarked + 2 control clips per model; 10 separate calibration clips/model.\nClean = generated waveform → saved WAV → retokenized speech. No added attack.\nZero observed false positives in 2 controls does not establish a population false-positive rate.', ha='center', fontsize=9)
    save(fig, 'detection_rates')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, label, data in zip(axes, LABELS, datasets):
        manifest, detection, training, channel, counts, paths = data
        threshold = np.log(1 / detection['alpha'])
        for kind in COLORS:
            records = [r for r in detection['samples'] if r['kind'] == kind]
            for i, record in enumerate(records):
                path = paths[record['name']]
                ax.plot(np.arange(len(path)), path, color=COLORS[kind],
                        linestyle=('-', '--')[i % 2], label=f"{'Watermarked' if kind == 'watermarked' else 'Control'} {i+1}")
                first = record['first_detection']
                if first is not None:
                    ax.scatter(first, path[first], color=COLORS[kind], s=25, zorder=4)
                    ax.annotate(str(first), (first, path[first]), xytext=(5, 6), textcoords='offset points', fontsize=8)
        ax.axhline(threshold, color='black', linestyle=':', label='Detection threshold: log(100)')
        ax.set(title=label, xlabel='Recovered speech tokens scored', xlim=(0, 185))
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0].set_ylabel('Cumulative log evidence, log(E)')
    fig.suptitle('Clean audio: evidence for each held-out clip\nK = 4 random groups; a clip is detected when its path first crosses the dotted line', fontsize=12)
    fig.text(.5, -.03, 'Each line is one clip; dots label the first crossing token. Paths stop at each clip’s end.\n2 watermarked + 2 controls/model; channel fitted on 10 separate calibration clips/model.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'detection_evidence')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    matrix_rows = []
    for ax, model, label, data in zip(axes, MODELS, LABELS, datasets):
        manifest, detection, training, channel, counts, paths = data
        k = len(channel)
        im = ax.imshow(channel, vmin=0, vmax=1, cmap='Blues')
        for a in range(k):
            for b in range(k):
                ax.text(b, a, f'{channel[a,b]:.1%}', ha='center', va='center',
                        color='white' if channel[a,b] > .5 else 'black', fontsize=10)
                matrix_rows.append(dict(model=model, original_group=a, recovered_group=b,
                                        probability=float(channel[a,b]), observed_pairs=int(counts[a,b]),
                                        source_row_pairs=int(counts[a].sum())))
        ax.set(xticks=range(k), yticks=range(k), xlabel='Recovered group after WAV retokenization',
               ylabel='Original group before audio synthesis',
               title=f'{label}\n{len(training["calibration_files"])} calibration clips; {counts.sum():,} token pairs')
        ax.set_yticklabels([f'{a} (n={counts[a].sum()})' for a in range(k)])
    fig.colorbar(im, ax=axes, format=PercentFormatter(1), label='Conditional transition probability')
    fig.suptitle('Clean-channel transition matrices used by the detector\nP(recovered group = column | original group = row); random K = 4, seed = 42', fontsize=12)
    fig.supxlabel('Rows sum to 100%. Diagonal = stays in group; off-diagonal = flips to another group.\nPositional token pairing; pseudocount β = 0.5 per cell. Group labels are model-specific.', fontsize=9)
    save(fig, 'transition_matrices')
    with (OUT / 'transition_probabilities.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0]))
        writer.writeheader(); writer.writerows(matrix_rows)

    readme = '''# Clean watermark results for collaborators

Start with `detection_rates.png` and `detection_evidence.png`: these report actual detection using existing generated clips, not group agreement as a proxy. Both models use K=4 random groups with seed 42. The detector uses saved generation-time group probabilities and a separately calibrated channel; this is not audio-only blind detection. Clean means synthesis, WAV save, and retokenization, without an additional attack.

`transition_matrices.png` compares the fitted clean channels used by these detectors. Read each row as a distribution over recovered groups. Diagonal entries mean the group survives; off-diagonal entries give flip probabilities. A robust detector can use off-diagonal channel probabilities; agreement alone is not a detection rate. Labels are arbitrary and model-specific. Positional pairing can include synchronization effects. Rows include beta=0.5 smoothing; row labels show actual calibration pair counts.

Each model has 10 calibration clips, two held-out watermarked clips, and two held-out unwatermarked controls. CosyVoice excludes two known replayed calibration clips; see its source channel_training.json. This tiny pilot cannot establish population detection or false-positive rates, the best K, or attack robustness. The existing data only support detection at the partition used during generation; the earlier CPU regrouping figures are supplementary channel diagnostics and do not compare detection across K.

PNG and PDF versions are provided. Exact rates/counts are in detection_rates.csv; cell probabilities/counts are in transition_probabilities.csv. Run `.venv/bin/python cleancode/plot_clean_report.py` to regenerate. No audio generation or model inference is performed.
'''
    (OUT / 'README.md').write_text(readme)
    log = ROOT / 'cleancode/experiment_log.csv'
    with log.open(newline='') as f:
        entries = list(csv.reader(f))
    entries = [r for r in entries if r[0] not in ('Clean-report-Spark', 'Clean-report-Cosy')]
    for label, model in zip(('Spark', 'Cosy'), MODELS):
        r = next(r for r in rows if r['model'] == model)
        entries.append([f'Clean-report-{label}', model, 'Existing clean pilot: actual detection and fitted transition matrices',
                        'Detection rates; per-clip evidence; paired transition matrices',
                        'K=4 random groups, seed=42; alpha=0.01; no added attack. Separate calibration and evaluation clips.',
                        f"{r['calibration_clips']} calibration clips, {r['calibration_pairs']} pairs per P_c; 2 watermarked and 2 unwatermarked clips/model. Pilot only; cannot compare detection across K.",
                        str(OUT.relative_to(ROOT)) + '/: detection_rates.png/.pdf/.csv; detection_evidence.png/.pdf; transition_matrices.png/.pdf; transition_probabilities.csv; README.md'])
    with log.open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(entries)
    print(f'Saved clean report to {OUT}')


if __name__ == '__main__':
    main()
