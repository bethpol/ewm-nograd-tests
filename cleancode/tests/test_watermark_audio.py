"""Mathematical checks for the simple grouped watermark experiment."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cleancode.watermark_audio import (
    average_paths, choose_group, fit_channel, generate, log_e_path, make_sampler, position_uniform, save_sample_index,
)


class SimpleWatermarkTests(unittest.TestCase):
    def test_position_prf_and_cdf(self):
        u = position_uniform(b"key", 0)
        self.assertTrue(0 < u < 1)
        self.assertEqual(u, position_uniform(b"key", 0))
        self.assertNotEqual(u, position_uniform(b"key", 1))
        self.assertEqual(choose_group([0, .3, .7], .1), 1)
        self.assertEqual(choose_group([0, .3, .7], .3), 2)

    def test_draft_formula_and_one_group(self):
        mass = np.array([[.2, .8], [.7, .3]])
        channel = np.array([[.9, .1], [.3, .7]])
        observed = np.array([0, 1])
        expected = [choose_group(mass[t], position_uniform(b"key", t)) for t in range(2)]
        increments = [np.log(channel[expected[t], observed[t]] / (mass[t] @ channel[:, observed[t]])) for t in range(2)]
        np.testing.assert_allclose(log_e_path(mass, observed, channel, b"key"), np.r_[0, np.cumsum(increments)])
        np.testing.assert_array_equal(log_e_path(np.ones((4, 1)), np.zeros(4, int), np.ones((1, 1)), b"key"), np.zeros(5))
        # Unit null expectation for EACH observed group, regardless of its law.
        for row in mass:
            for y in range(2):
                self.assertAlmostEqual(row @ (channel[:, y] / (row @ channel[:, y])), 1.)

    def test_channel_orientation_smoothing_and_split(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            np.save(out / "groups.npy", [0, 1])
            (out / "manifest.json").write_text(json.dumps(dict(num_groups=2, records=[
                dict(name="cal", kind="calibration"), dict(name="eval", kind="watermarked"),
                dict(name="replay", kind="calibration", exclude_from_channel_fit=True)])) )
            np.savez(out / "cal.npz", original=[0, 0, 1], observed=[0, 1, 1])
            np.savez(out / "eval.npz", original=[1]*100, observed=[0]*100)
            np.savez(out / "replay.npz", original=[1]*100, observed=[0]*100)
            fit_channel(out, .5)
            with np.load(out / "channel.npz") as archive:
                np.testing.assert_array_equal(archive["counts"], [[1, 1], [0, 1]])
                np.testing.assert_allclose(archive["channel"], [[.5, .5], [.25, .75]])

    def test_fixed_cohort_error_bars(self):
        mean, sem = average_paths([np.array([0., 2., 4.]), np.array([0., 4.])])
        np.testing.assert_allclose(mean, [0., 3.])
        np.testing.assert_allclose(sem, [0., 1.])
        self.assertTrue(np.isnan(average_paths([np.zeros(3)])[1]).all())

    def test_sample_inventory_labels_channel_training_audio(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            record = dict(name="calibration_000", kind="calibration", audio_path="audio/calibration/calibration_000.wav",
                          duration_seconds=2., sample_rate=16000, original_tokens=100,
                          observed_tokens=99, aligned_pairs=99, hit_token_cap=False)
            manifest = dict(model="sparktts", text="Hello", reference_audio="reference.wav",
                            reference_text_file="reference.txt", records=[record])
            save_sample_index(out, manifest)
            with (out / "samples.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["used_for_channel_fit"], "True")
            self.assertEqual(row["audio_path"], record["audio_path"])
            self.assertEqual(row["aligned_pairs"], "99")

    def test_sampler_preserves_nonsemantic_and_group_choice(self):
        import torch
        groups = np.array([0, 1, 0, 1])
        mapping = {i: i for i in range(4)}
        sampler = make_sampler(mapping, groups, b"key", True)
        # Override filters to test the exact categorical marginal.
        sampler.warpers = []
        torch.manual_seed(12)
        p = np.array([.1, .2, .3, .15, .25])  # final symbol is nonsemantic
        counts = np.zeros(5)
        for _ in range(4000):
            old = len(sampler.tokens)
            token = int(sampler(None, torch.tensor(np.log(p)[None, :])).argmax())
            counts[token] += 1
            if token < 4:
                self.assertEqual(groups[token], choose_group(sampler.masses[-1], position_uniform(b"key", old)))
            else:
                self.assertEqual(len(sampler.tokens), old)
        np.testing.assert_allclose(counts / counts.sum(), p, atol=.035)

    def test_expanding_calibration_preserves_partition_and_evaluation(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            ref = out / "reference.txt"
            ref.write_text("Reference transcript")
            np.save(out / "groups.npy", [0, 1])
            record = dict(name="watermarked_000", kind="watermarked", audio_path="audio/watermarked/watermarked_000.wav",
                          duration_seconds=1., sample_rate=16000, original_tokens=2,
                          observed_tokens=2, aligned_pairs=2, hit_token_cap=False)
            manifest = dict(model="sparktts", text="Frozen text", reference_audio=str(out / "reference.wav"),
                            reference_text_file=str(ref), partition_seed=42, num_groups=2,
                            max_tokens=100, records=[record])
            save_sample_index(out, manifest)
            heldout = out / "watermarked_000.npz"
            heldout.write_bytes(b"unchanged held-out data")
            args = SimpleNamespace(output_dir=out, model="sparktts", num_groups=999, calibration_samples=2)
            def load_fake(args):
                import torch
                torch.manual_seed(1986)  # Reproduce CosyVoice's YAML side effect.
                return SimpleNamespace(sample_rate=16000), {0: 0, 1: 1}
            def synthesize_fake(model, sampler, args, reference_text):
                import torch
                self.assertEqual(torch.initial_seed(), 42)
                self.assertEqual(args.text, "Frozen text")
                sampler.tokens = [0, 1]
                sampler.masses = [[.4, .6], [.3, .7]]
                return np.zeros(1600, np.float32), np.array([0, 1])
            with patch("cleancode.watermark_audio.load_model", side_effect=load_fake), \
                 patch("cleancode.watermark_audio.synthesize", side_effect=synthesize_fake), \
                 patch("cleancode.watermark_audio.recover_tokens", return_value=np.array([1, 1])):
                generate(args, append_calibration=True)
            self.assertEqual(heldout.read_bytes(), b"unchanged held-out data")
            np.testing.assert_array_equal(np.load(out / "groups.npy"), [0, 1])
            updated = json.loads((out / "manifest.json").read_text())
            self.assertEqual([r["name"] for r in updated["records"]],
                             ["watermarked_000", "calibration_000", "calibration_001"])
            self.assertFalse((out / "channel.npz").exists())  # No silent refit.


if __name__ == "__main__":
    unittest.main()
