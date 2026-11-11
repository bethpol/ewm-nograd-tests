import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cleancode.random_group_sweep import summarize, wilson


class SweepTests(unittest.TestCase):
    def test_wilson_extremes(self):
        self.assertAlmostEqual(wilson(0, 20)[0], 0)
        self.assertAlmostEqual(wilson(20, 20)[1], 1)
        self.assertGreater(wilson(0, 20)[1], .1)

    def test_summary_uses_ever_crossing_and_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            run = out / 'sparktts_groups2'
            run.mkdir()
            records = []
            for kind in ('watermarked', 'unwatermarked'):
                for i in range(20):
                    name = f'{kind}_{i}'
                    records.append(dict(name=name, kind=kind, first_detection=1 if kind == 'watermarked' and i < 5 else None, hit_token_cap=False))
                    # Crossing then falling below threshold must still count.
                    np.savez(run / f'{name}.evidence.npz', original=[0., 6., -1.])
            (run / 'manifest.json').write_text(json.dumps(dict(partition_seed=42)))
            (run / 'detection.json').write_text(json.dumps(dict(alpha=.01, samples=records)))
            summarize(out)
            with (out / 'detection_rates.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([float(r['rate']) for r in rows], [.25, 1., 0., 1.])
            self.assertTrue((out / 'detection_rates.png').exists())
            self.assertTrue((out / 'detection_rates.pdf').exists())
