"""Exercise report artifacts without checkpoints or GPU access."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

from evals.grouped_audio_report import main
from watermark.keyed_sampling import KeyedSemanticSampler


class ReportTests(unittest.TestCase):
    def test_report_scores_audio_tokens_and_saves_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = root / "key"
            key.write_bytes(b"test-key")
            sample = root / "audio" / "partition_1" / "groups_2" / "sample_0"
            sample.mkdir(parents=True)
            sampler = KeyedSemanticSampler({i: i for i in range(3)}, b"test-key", "clip",
                                           top_k=0, top_p=1., groups=[0, 1, 0])
            for _ in range(12):
                sampler(torch.tensor([[9]]), torch.tensor([[.2, .3, .5]]).log())
            trace = sampler.trace()
            torch.save(trace, sample / "trace.pt")
            np.savetxt(sample / "aug_00_00_tokens.txt", trace["original_tokens"].numpy(), fmt="%d")
            sparse.save_npz(root / "counts.npz", sparse.csr_matrix(np.eye(3) * 10 + 1))
            report = root / "report"
            with patch.dict("os.environ", {}, clear=True), patch("sys.argv", [
                "report", "--counts", str(root / "counts.npz"), "--audio_dir", str(root / "audio"),
                "--output_dir", str(report), "--key_file", str(key),
            ]):
                main()
            summary = json.loads((report / "summary.json").read_text())
            self.assertEqual(summary["results"][0]["common_horizon"], 12)
            self.assertTrue((report / "detection_vs_groups.png").is_file())
            self.assertTrue((report / "partition_1/groups_2/transition.pdf").is_file())
            self.assertTrue((report / "detections.csv").is_file())


if __name__ == "__main__":
    unittest.main()
