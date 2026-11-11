#!/usr/bin/env python3
"""Run one random-group configuration or summarize the completed sweep."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('sparktts', 'cosyvoice3')
GROUPS = (2, 4, 8, 16)


def wilson(hits, total):
    z = 1.959963984540054
    p = hits / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0., center-radius), min(1., center+radius)


def summarize(out):
    rows = []
    for model in MODELS:
        for groups in GROUPS:
            run = out / f'{model}_groups{groups}'
            if not (run / 'detection.json').exists():
                continue
            result = json.loads((run / 'detection.json').read_text())
            manifest = json.loads((run / 'manifest.json').read_text())
            for kind in ('watermarked', 'unwatermarked'):
                records = [r for r in result['samples'] if r['kind'] == kind]
                if len(records) < 20:
                    raise ValueError(f'{run}: fewer than 20 {kind} trials')
                for source in ('roundtrip', 'original'):
                    hits = 0
                    for record in records:
                        if source == 'roundtrip':
                            hits += record['first_detection'] is not None
                        else:
                            import numpy as np
                            with np.load(run / f"{record['name']}.evidence.npz") as evidence:
                                hits += bool((evidence[source] >= math.log(1/result['alpha'])).any())
                    low, high = wilson(hits, len(records))
                    rows.append(dict(model=model, num_groups=groups, kind=kind, source=source,
                                     trials=len(records), detections=hits, rate=hits/len(records),
                                     ci95_low=low, ci95_high=high, alpha=result['alpha'],
                                     partition_seed=manifest['partition_seed'],
                                     token_cap_hits=sum(r['hit_token_cap'] for r in records)))
    if not rows:
        raise ValueError('No completed configurations')
    with (out / 'detection_rates.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model in zip(axes, MODELS):
        for kind, label in [('watermarked', 'Detection'), ('unwatermarked', 'False positive')]:
            for source, style in [('roundtrip', '-'), ('original', '--')]:
                selected = [r for r in rows if r['model'] == model and r['kind'] == kind and r['source'] == source]
                if not selected:
                    continue
                rates = [r['rate'] for r in selected]
                ax.errorbar([r['num_groups'] for r in selected], rates,
                            yerr=[[r['rate']-r['ci95_low'] for r in selected],
                                  [r['ci95_high']-r['rate'] for r in selected]],
                            linestyle=style, marker='o', capsize=3,
                            label=f'{label}, {source}')
        ax.set(title=model, xlabel='Random groups', xticks=GROUPS, ylim=(-.04, 1.04))
        ax.set_xlim(1.7, 19)
        ax.set_xscale('log', base=2)
        ax.set_xticks(GROUPS, labels=[str(g) for g in GROUPS])
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    axes[0].set_ylabel('Fraction ever crossing threshold (95% Wilson interval)')
    fig.suptitle('Random grouping: held-out clips, α = 0.01')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'detection_rates.{extension}', dpi=180)
    plt.close(fig)
    print(f'Summarized {len(rows)//4}/8 configurations in {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--task', type=int, choices=range(8))
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / 'cleancode'):
        parser.error('Outputs must stay inside cleancode')
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(out)
        return
    if args.task is None:
        parser.error('Supply --task or --summarize')
    model, groups = [(m, g) for m in MODELS for g in GROUPS][args.task]
    run = out / f'{model}_groups{groups}'
    settings = dict(model=model, num_groups=groups, samples_per_condition=20,
                    calibration_samples=10, seed=42, max_tokens=600, alpha=.01, beta=.5,
                    trial_unit='independent clip with fresh key; fixed random partition per configuration')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'sweep_config.json').write_text(json.dumps(settings, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT / 'cleancode/watermark_audio.py'),
                    '--model', model, '--num-groups', str(groups), '--samples', '20',
                    '--calibration-samples', '10', '--seed', '42', '--max-tokens', '600',
                    '--alpha', '.01', '--beta', '.5', '--output-dir', str(run)], check=True, cwd=ROOT)
    (run / 'COMPLETE').touch()


if __name__ == '__main__':
    main()
