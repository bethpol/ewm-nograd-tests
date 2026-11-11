"""Meeting report of the completed Spark-TTS clean full-audio sweep; no inference."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/automatic_sweep_3531161'
OUT = ROOT / 'cleancode/watermark_samples/sparktts_meeting_report'
GROUPS = (2, 4, 8, 16)
SEEDS = (20261104, 20261105, 20261106)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evidence_path(record):
    return ROOT / record['source_root'] / record['path']


def channel_path(k, seed):
    base = SOURCE if k <= 16 else ROOT / 'outputs/extended_spark_detection_32_64' / f'm{k}_seed{seed}'
    return base / f'frozen_channels/m{k}_seed{seed}_full.npz'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / 'sweep_config.json').read_text())
    alpha = config['alpha']
    threshold = np.log(1 / alpha)
    with (SOURCE / 'report/per_clip.csv').open(newline='') as f:
        records = [r for r in csv.DictReader(f)
                   if r['policy'] == 'full' and r['fit'] == 'matched' and r['key'] == 'same_key']
    for r in records:
        r['source_root'] = str(SOURCE.relative_to(ROOT))
    groups = list(GROUPS)
    for k in (32, 64):
        folders = [ROOT / 'outputs/extended_spark_detection_32_64' / f'm{k}_seed{seed}' for seed in SEEDS]
        if not all((p / 'status').exists() and (p / 'status').read_text().strip() == 'complete' for p in folders):
            continue
        extra = []
        for folder in folders:
            with (folder / 'report/per_clip.csv').open(newline='') as f:
                selected = [r for r in csv.DictReader(f) if r['policy'] == 'full' and r['fit'] == 'matched' and r['key'] == 'same_key']
            assert len(selected) == 32
            for r in selected:
                r['source_root'] = str(folder.relative_to(ROOT))
            extra.extend(selected)
        records.extend(extra)
        groups.append(k)
    identities = [(r['groups'], r['partition_seed'], r['mode'], r['index']) for r in records]
    assert len(records) == len(set(identities)) == 96 * len(groups)
    for r in records:
        path = np.load(evidence_path(r))
        assert path.ndim == 2 and path.shape[1] == 3 and len(path) > 0
        assert np.isclose(path[-1, 1], float(r['final_log_e']))
        # Preserve and verify the archived end-of-clip result, then evaluate the
        # user's running-maximum criterion over the recovered-token prefix path.
        assert (path[-1, 1] >= threshold) == (r['detected'] == 'True')
        assert np.allclose(path[:, 2], float(r['duration_seconds']))
        assert np.isfinite(path[:, 1]).all()
        r['archived_end_detected'] = r['detected']
        r['max_log_e'] = float(max(0., path[:, 1].max()))  # Include E_0 = 1.
        r['detected'] = str(r['max_log_e'] >= threshold)
    write_csv(OUT / 'evaluated_clips.csv', records)
    summary, calibration = [], []
    for k in groups:
        for seed in SEEDS:
            archive = channel_path(k, seed)
            meta = json.loads(archive.with_suffix('.json').read_text())
            calibration.append(dict(groups=k, partition_seed=seed,
                                    calibration_clips=meta['training_clips'],
                                    calibration_pairs=meta['training_pairs'],
                                    validation_clips=meta['validation_clips'], validation_pairs=meta['validation_pairs'],
                                    channel_path=str(archive.relative_to(ROOT))))
            for mode in ('watermarked', 'ordinary'):
                selected = [r for r in records if int(r['groups']) == k and int(r['partition_seed']) == seed and r['mode'] == mode]
                assert len(selected) == 16
        for mode in ('watermarked', 'ordinary'):
            selected = [r for r in records if int(r['groups']) == k and r['mode'] == mode]
            hits = sum(r['detected'] == 'True' for r in selected)
            summary.append(dict(groups=k, mode=mode, detections=hits, recordings=len(selected),
                                rate=hits/len(selected), partitions=3, recordings_per_partition=16,
                                min_duration_seconds=min(float(r['duration_seconds']) for r in selected),
                                max_duration_seconds=max(float(r['duration_seconds']) for r in selected),
                                alpha=alpha, decision='maximum over recovered-token prefixes including E_0=1',
                                calibration_clips_per_channel=512, validation_clips_per_channel=128))
    write_csv(OUT / 'detection_rates.csv', summary)
    write_csv(OUT / 'channel_calibration.csv', calibration)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(groups))
    for mode, offset, color, label in (
        ('watermarked', -.18, '#0072B2', 'Watermarked: detection rate'),
        ('ordinary', .18, '#D55E00', 'Ordinary audio: false-positive rate')):
        selected = [r for r in summary if r['mode'] == mode]
        bars = ax.bar(x + offset, [r['rate'] for r in selected], .32, color=color, label=label)
        for bar, r in zip(bars, selected):
            ax.text(bar.get_x()+bar.get_width()/2, r['rate']+.025,
                    f"{r['detections']}/{r['recordings']}\n{r['rate']:.1%}", ha='center', fontsize=10)
    ax.set(xticks=x, xticklabels=[str(k) + ('*' if k >= 32 else '') for k in groups], xlabel='Number of random token groups, K (* = exploratory calibration)',
           ylabel='Fraction with maximum evidence above threshold', ylim=(0, 1.23))
    ax.set_yticks(np.linspace(0, 1, 5)); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.set_title(f'Spark-TTS: detection by maximum evidence on {len(records)} recordings\n48 watermarked + 48 ordinary clips per K; max E ≥ 100 (α = 0.01)', fontsize=13)
    fig.subplots_adjust(bottom=.32, top=.83)
    fig.text(.5, .015, 'Each K pools 3 random partitions × 16 clips per condition. Maximum over recovered-token prefixes, including E₀ = 1.\nMatched channel: 512 calibration clips; 128 validation clips. Full WAV retokenization; no added attacks.\nSaved generation-probability replay. Prefix scores are computed offline; this is not streaming audio reception.\nObserved rates on this cohort; finite controls do not establish a 1% population false-positive rate.', ha='center', fontsize=9)
    save(fig, 'detection_by_groups')

    fig, axes = plt.subplots(1, len(groups), figsize=(3.5*len(groups), 4.8), sharey=True)
    for ax, k in zip(axes, groups):
        for mode, location, color in [('watermarked', 0, '#0072B2'), ('ordinary', 1, '#D55E00')]:
            selected = [r for r in records if int(r['groups']) == k and r['mode'] == mode]
            values = [r['max_log_e'] for r in selected]
            if np.ptp(values) > 0:
                violin = ax.violinplot([values], positions=[location], widths=.75,
                                       showmeans=False, showmedians=True, showextrema=False,
                                       quantiles=[[.25, .75]])
                for body in violin['bodies']:
                    body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(.55)
                for key in ('cmedians', 'cquantiles'):
                    violin[key].set_color(color)
                    violin[key].set_linewidth(2 if key == 'cmedians' else 1)
            else:
                ax.hlines(values[0], location-.3, location+.3, color=color, linewidth=3)
            hits = sum(r['detected'] == 'True' for r in selected)
            ax.text(location, .98, f'{hits}/{len(selected)} detected', ha='center', va='top',
                    transform=ax.get_xaxis_transform(), color=color, fontsize=8)
        ax.axhline(threshold, color='black', linestyle=':', label='Threshold: log(100)')
        ax.set(title=f'K = {k}' + (' (exploratory)' if k >= 32 else ''), xticks=(0, 1), xticklabels=('Watermarked', 'Ordinary'), xlim=(-.5, 1.5))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Maximum cumulative log evidence, maxₜ log(Eₜ)')
    axes[-1].legend(fontsize=8, loc='center right')
    fig.suptitle('Spark-TTS: distribution of maximum evidence per clip\n48 watermarked + 48 ordinary clips per K, pooled across all 3 partitions', fontsize=12)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.text(.5, .01, 'Each violin summarizes 48 per-clip maxima; bars show median and quartiles. Dotted line: log(100).\nMaximum includes E₀ = 1 (log evidence = 0). Full-waveform retokenization; offline token-prefix scoring.\nMatched channels; saved generation-probability replay; no added attacks.', ha='center', fontsize=9)
    save(fig, 'per_clip_evidence')

    columns = (len(groups)+1)//2
    fig, axes = plt.subplots(2, columns, figsize=(5.5*columns, 8), sharex=False)
    average_rows = []
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    for ax, k in zip(axes.flat, groups):
        selected = [r for r in records if int(r['groups']) == k]
        paths = {}
        for r in selected:
            path = np.load(evidence_path(r))
            # Full-waveform observations have contiguous prefix positions.
            np.testing.assert_array_equal(path[:, 0], np.arange(len(path)))
            paths[r['path']] = np.r_[0., path[:, 1]]
        length = min(len(p) for p in paths.values())
        for mode, color, label in [('watermarked', '#0072B2', 'Watermarked'),
                                    ('ordinary', '#D55E00', 'Ordinary controls')]:
            values = np.stack([paths[r['path']][:length] for r in selected if r['mode'] == mode])
            mean = values.mean(0)
            sem = values.std(0, ddof=1) / np.sqrt(len(values))
            t = np.arange(length)
            ax.plot(t, mean, color=color, label=f'{label} (n={len(values)})')
            ax.fill_between(t, mean-sem, mean+sem, color=color, alpha=.2, linewidth=0)
            average_rows.extend(dict(groups=k, mode=mode, token_index=int(i), clips=len(values),
                                     mean_log_e=float(m), sem_log_e=float(s))
                                for i, m, s in zip(t, mean, sem))
        ax.axhline(threshold, color='black', linestyle=':', label='Threshold: log(100)')
        ax.set(title=f'K = {k}' + (' (exploratory)' if k >= 32 else '') + f'; fixed cohort through {length-1} tokens',
               xlabel='Token index (0 = before scoring)', ylabel='Mean cumulative log evidence')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Spark-TTS: average log evidence versus token index\n48 clips per condition and group count; all 3 partitions pooled; shaded bands = ±1 SEM across clips', fontsize=12)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.text(.5, .01, 'Each panel stops at its shortest clip; all 48 clips per condition contribute at every plotted index.\nSEM describes clip variation, not uncertainty across independent partitions. Full-WAV retokenization; offline prefix scores.', ha='center', fontsize=9)
    save(fig, 'average_log_e_by_token')
    write_csv(OUT / 'average_log_e_by_token.csv', average_rows)

    fig, axes = plt.subplots(3, len(groups), figsize=(3.75*len(groups), 11), layout='constrained')
    cells = []
    for row, seed in enumerate(SEEDS):
        for col, k in enumerate(groups):
            ax = axes[row, col]
            with np.load(channel_path(k, seed)) as f:
                channel, counts = f['channel'], f['counts']
            np.testing.assert_allclose(channel.sum(1), 1)
            meta = next(r for r in calibration if r['groups'] == k and r['partition_seed'] == seed)
            assert int(counts.sum()) == meta['calibration_pairs']
            im = ax.imshow(channel, cmap='Blues', vmin=0, vmax=1)
            for a in range(k):
                for b in range(k):
                    cells.append(dict(groups=k, partition_seed=seed, original_group=a,
                                      recovered_group=b, probability=float(channel[a,b]), pairs=int(counts[a,b])))
                    if k <= 4:
                        ax.text(b, a, f'{channel[a,b]:.0%}', ha='center', va='center',
                                color='white' if channel[a,b] > .5 else 'black', fontsize=9)
            ax.set(title=f'K={k}; {counts.sum():,} calibration pairs', xticks=range(k), yticks=range(k))
            ax.tick_params(labelsize=7)
            if row == 2:
                ax.set_xlabel('Recovered group')
            if col == 0:
                ax.set_ylabel(f'Partition {seed}\nOriginal group')
    fig.colorbar(im, ax=axes, format=PercentFormatter(1), shrink=.8, label='P(recovered group | original group)')
    fig.suptitle(f'Spark-TTS: clean transition matrices for all {3*len(groups)} evaluated groupings\nEach panel: 512 calibration clips; rows sum to 100%; diagonal = stays, off-diagonal = group flip', fontsize=14)
    fig.supxlabel('Same 512 calibration recordings reused across panels; 128 separate validation clips per channel.\nRandom balanced partitions; β = 0.0001 per cell; group labels are arbitrary. These matrices explain the detector, not its detection rate.', fontsize=10)
    save(fig, 'transition_matrix_grid')
    write_csv(OUT / 'transition_probabilities.csv', cells)

    scope_note = ('\n\nCompleted group counts: ' + ', '.join(map(str, groups)) +
                  f'. Evaluated recordings: {len(records)}. K=32,64 are exploratory when present: their channels failed the original stability criteria. '
                  'A larger K is added only after all three partitions finish. No unavailable detection paths are extrapolated.\n')
    (OUT / 'README.md').write_text('''# Spark-TTS results for the meeting

Use `detection_by_groups.png` first: maximum-evidence, matched-channel detection. The original archived K=2,4,8,16 cohort has 384 distinct recordings; completed K=32,64 extensions are added automatically and marked exploratory. The completed group list and total count are recorded at the end of this file. Exact updated counts are in `detection_rates.csv`. A recording is detected if its maximum cumulative log evidence over all recovered-token prefixes reaches log(100). The maximum includes the initial value log(E_0)=0. Every K uses three fixed random balanced partitions, with 16 watermarked and 16 ordinary clips per partition, pooled to 48 clips per condition. Different K and partitions have fresh generations; encoding-policy and wrong-key replays are not counted as extra recordings.

`per_clip_evidence.png` uses violin plots of the per-recording maximum log evidence, pooling all three partitions within each K and condition. Median and quartile bars summarize each distribution; violin width is a smoothed density, not a count. A constant distribution is drawn as a horizontal segment. Values above the threshold are detected, even if evidence falls later. `transition_matrix_grid.png` displays the calibrated channels for completed groups, with rows for partitions and columns for K. A row within each matrix is the probability distribution over recovered groups given an original group; off-diagonal entries are flip probabilities. Read these as explanatory channel matrices, not proxy detection rates. Channel matrices stay separate because their group labels refer to different partitions.

Each channel was fitted on the same 512 separate ordinary calibration recordings, with 128 separate validation recordings used for pre-evaluation screening. Exact per-channel token pair counts and source paths are in `channel_calibration.csv`. Channel maps and estimates were frozen before watermark evaluation. Only random balanced groupings are included; no contiguous or modulo groupings are used.

`average_log_e_by_token.png` shows mean cumulative log evidence (not the mean running maximum) versus token index for every K, separately for watermarked clips and ordinary controls. Translucent shaded bands are plus/minus one standard error across the 48 clips after pooling partitions. Each panel uses the shortest clip across both conditions as its common endpoint; no longer-clip-only tail is plotted. Index zero is the initial log evidence zero. Exact means, SEMs, and sample counts are in `average_log_e_by_token.csv`. These descriptive SEMs do not account for shared prompts or estimate uncertainty across a population of partitions.

These are archived results from `outputs/automatic_sweep_3531161`, not new generations and not the same configuration as the tiny clean pilot. The archived run uses temperature 1, top-k 50, top-p 0.95, pseudocount beta=0.0001, four text prompts, and clips about 2.9–4.6 seconds long. Detection reconstructs probabilities from saved generation traces (oracle-distribution replay), not audio alone. Updated plots evaluate the running maximum of the recovered-token evidence path at E>=100 (alpha=0.01). The archived encoder used the complete waveform; these prefix scores are computed offline and do not demonstrate streaming audio reception. No added attacks are evaluated. Comparing K is descriptive: prompts repeat, partitions vary, and clip lengths vary; this is not a matched-duration causal comparison or a population false-positive guarantee.

The old report's 2.9-second restricted streaming horizon is not used. Updated counts use maximum evidence over all recovered-token prefixes. The original archived reports and channels are preserved. `evaluated_clips.csv` retains `final_log_e` and `archived_end_detected` for provenance, adds `max_log_e`, and updates `detected` to the maximum-evidence criterion.

PNG and PDF versions of every figure are available. `evaluated_clips.csv` gives all evaluated rows and their source roots, `detection_rates.csv` the aggregated counts, and `transition_probabilities.csv` every channel cell. The plotting script validates the archived final scores against all saved evidence paths before computing maxima, and checks channel row normalization and calibration pair counts. No model inference is needed: run `.venv/bin/python cleancode/plot_archived_spark_report.py`.

CosyVoice remains a separate two-watermarked/two-control pilot; these Spark-TTS results do not increase its sample size.
''' + scope_note)
    log = ROOT / 'cleancode/experiment_log.csv'
    with log.open(newline='') as f:
        entries = list(csv.reader(f))
    entries = [r for r in entries if r[0] != 'Meeting-archive-Spark']
    entries.append(['Meeting-archive-Spark', 'Spark-TTS', f'Clean full-audio detection: {len(records)} recordings',
                    'Maximum-evidence detection vs K; pooled maximum-evidence violins; mean log evidence vs token with translucent ±1 SEM bands; transition matrix grid',
                    f'K={groups}; 48 watermarked + 48 ordinary recordings/K; 3 random partitions; matched channels; alpha=0.01. K32/64 exploratory if present; failed original stability criteria.',
                    '512 separate calibration clips per P_c; 128 validation clips; exact token counts in channel_calibration.csv. Same calibration recordings reused across matrices. Detection uses maximum over recovered-token prefixes including E_0=1; full WAV encoded offline; saved generation-probability replay; no attacks.',
                    str(OUT.relative_to(ROOT)) + '/: detection_by_groups.png/.pdf; per_clip_evidence.png/.pdf; average_log_e_by_token.png/.pdf/.csv; transition_matrix_grid.png/.pdf; detection_rates.csv; evaluated_clips.csv; channel_calibration.csv; transition_probabilities.csv; README.md'])
    with log.open('w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerows(entries)
    print(f'Validated {len(records)} recordings and {3*len(groups)} channels. Saved {OUT}')


if __name__ == '__main__':
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'plot.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
