"""Explicit exploratory evaluation of K=32,64 with fixed, less stable channels.

The original quality-gated experiment remains unchanged. This experiment studies
finite calibration effects, so stability failures are recorded rather than used
to suppress the evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from evals.automatic_sweep import rejection_reasons, run
from evals.channel_estimation import write_json

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ROOT / 'outputs/channel_fit_extended_32_64'
OUTPUT = ROOT / 'outputs/extended_spark_detection_32_64'
TASKS = [(k, seed) for k in (32, 64) for seed in (20261104, 20261105, 20261106)]


def prepare(task):
    k, seed = TASKS[task]
    out = OUTPUT / f'm{k}_seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((CHANNELS / 'config.json').read_text())
    config['groups'] = [k]
    config['partition_seeds'] = [seed]
    config['experiment_scope'] = 'Exploratory fixed-calibration evaluation; original stability screen failed.'
    archive = CHANNELS / 'channels' / f'm{k}_seed{seed}_full.npz'
    metadata = json.loads(archive.with_suffix('.json').read_text())
    reasons = rejection_reasons(metadata)
    with np.load(archive) as data:
        assert data['channel'].shape == (k, k)
        np.testing.assert_allclose(data['channel'].sum(1), 1)
        np.testing.assert_array_equal(data['groups'], np.load(CHANNELS / 'groupings' / f'm{k}_seed{seed}.npy'))
        assert int(data['counts'].sum()) == metadata['training_pairs']
    frozen = out / 'frozen_channels'
    frozen.mkdir(exist_ok=True)
    target = frozen / archive.name
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen exploratory channel changed')
    shutil.copyfile(archive, target)
    shutil.copyfile(archive.with_suffix('.json'), target.with_suffix('.json'))
    note = dict(status='exploratory', groups=k, partition_seed=seed,
                original_quality_screen_passed=not reasons, original_screen_failures=reasons,
                calibration_clips=metadata['training_clips'], calibration_pairs=metadata['training_pairs'],
                validation_clips=metadata['validation_clips'],
                scope='Measure detection with fixed limited calibration; not an approved channel under the original gate.',
                frozen_channel=str(target.relative_to(out)), sha256=digest)
    write_json(out / 'exploratory_design.json', note)
    write_json(out / 'channel_collection_config.json', config)
    return out, dict(source=str(CHANNELS), accepted=[dict(groups=k, partition_seed=seed,
                policy='full', archive=str(target.relative_to(out)), sha256=digest)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=int, choices=range(6), required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out, plan = prepare(args.task)
    if args.prepare_only:
        print(f'Prepared {out}', flush=True)
        return
    settings = SimpleNamespace(output_dir=out, clips=16, null_clips=16, steps=800,
                alpha=.01, seed=202611060, prompt_file=ROOT / 'data/tts_prompts.txt',
                quality_gate_description='Exploratory: original stability criteria failed; see exploratory_design.json. Channel frozen before evaluation.')
    (out / 'status').write_text('running\n')
    try:
        run(settings, plan)
        (out / 'status').write_text('complete\n')
    except Exception as error:
        write_json(out / 'error.json', dict(error=repr(error)))
        (out / 'status').write_text('failed\n')
        raise
    subprocess.run([sys.executable, str(ROOT / 'cleancode/plot_archived_spark_report.py')], check=True)


if __name__ == '__main__':
    main()
