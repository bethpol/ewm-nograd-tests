"""Refit K=32,64 clean channels on saved ordinary calibration pairs, on CPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from evals.channel_estimation import report, write_json
from evals.automatic_sweep import gate
from watermark.grouped import random_groups

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'outputs/channel_fit_3531082'
OUT = ROOT / 'outputs/channel_fit_extended_32_64'
SWEEP = ROOT / 'outputs/automatic_sweep_extended_32_64'


def main():
    config = json.loads((SOURCE / 'config.json').read_text())
    config['groups'] = [32, 64]
    config['policies'] = [['full', 0, 0]]
    config['reused_calibration_source'] = str(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    previous = OUT / 'config.json'
    if previous.exists() and json.loads(previous.read_text()) != config:
        raise ValueError('Extended channel configuration changed')
    write_json(previous, config)
    samples = OUT / 'samples'
    if not samples.exists():
        samples.symlink_to(SOURCE / 'samples', target_is_directory=True)
    maps = OUT / 'groupings'
    maps.mkdir(exist_ok=True)
    for k in config['groups']:
        for seed in config['partition_seeds']:
            np.save(maps / f'm{k}_seed{seed}.npy', random_groups(8192, k, seed))
    report(SimpleNamespace(output_dir=OUT, beta=config['beta'], bootstrap=config['bootstrap']))
    write_json(OUT / 'progress.json', dict(status='complete',
               completed=config['training_clips']+config['validation_clips'],
               reused_source=str(SOURCE)))
    (OUT / 'status').write_text('complete\n')
    plan = gate(OUT, SWEEP)
    print(f"Extended calibration: {len(plan['accepted'])} accepted; {len(plan['rejected'])} rejected", flush=True)
    for item in plan['rejected']:
        print(item, flush=True)


if __name__ == '__main__':
    main()
