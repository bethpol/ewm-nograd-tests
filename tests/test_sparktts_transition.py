import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

from evals.sparktts_transition import count_transitions, save_transition_matrix, heldout_score, atomic_json


class TransitionTests(unittest.TestCase):
    def test_empty_generations_retry_but_other_errors_propagate(self):
        from unittest.mock import Mock
        from evals.sparktts_transition import generate_nonempty
        events = []
        operation = Mock(side_effect=[RuntimeError("Spark-TTS generated no semantic tokens"), "valid"])
        value, seed, retries = generate_nonempty(operation, 100, events.append)
        self.assertEqual((value, seed, retries), ("valid", 10000100, 1))
        self.assertEqual(len(events), 1)
        operation = Mock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
            generate_nonempty(operation, 100, events.append)
        self.assertEqual(operation.call_count, 1)
        operation = Mock(side_effect=RuntimeError("Spark-TTS generated no semantic tokens"))
        with self.assertRaisesRegex(RuntimeError, "all 3 attempts"):
            generate_nonempty(operation, 100, events.append)
        self.assertEqual(operation.call_count, 3)

    def test_resume_keeps_complete_clips_and_rejects_changed_sampling(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from evals.sparktts_transition import generate

        class Model:
            semantic_vocab_size = 3
            audio_tokenizer = object()

            def __init__(self, *args, **kwargs):
                pass

            def inference(self, **kwargs):
                return np.zeros(640, dtype=np.float32), torch.tensor([0, 1])

        class Encoder:
            sr = 16000

            def __init__(self, *args, **kwargs):
                pass

            def encode_semantic(self, audio):
                return torch.tensor([0, 2])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reference.wav").write_bytes(b"placeholder")
            (root / "reference.txt").write_text("reference speech")
            (root / "prompts.txt").write_text("Test sentence.\n")
            args = SimpleNamespace(output_dir=str(root / "output"), resume=False,
                prompt_file=str(root / "prompts.txt"), reference_dir=str(root),
                model_dir="test", vocab_size=3, device="cpu", seed=100, steps=8,
                temperature=1., top_k=0, top_p=1., beta=.1,
                num_samples=2, heldout_samples_dir=None, checkpoint_every=0)
            with patch("evals.keyed_sparktts.sparktts_classes", return_value=(Model, Encoder)), \
                    patch("evals.sparktts_transition.build"):
                generate(args)
                args.resume = True
                with patch.object(Model, "inference", side_effect=AssertionError("Regenerated complete clip")):
                    generate(args)
                args.temperature = .8
                with self.assertRaisesRegex(ValueError, "different temperature"):
                    generate(args)

    def test_heldout_likelihood_uses_counts_and_correct_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix.npy"
            np.save(path, np.array([[.9, .1], [.3, .7]], dtype=np.float32))
            heldout = sparse.csr_matrix([[2, 1], [3, 0]])
            score = heldout_score(path, heldout)
            expected = (2 * np.log(.9) + np.log(.1) + 3 * np.log(.3)) / 6
            self.assertEqual(score["heldout_pairs"], 6)
            self.assertAlmostEqual(score["heldout_mean_log_probability"], expected, places=6)

    def test_progress_json_is_replaced(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.json"
            atomic_json(path, {"completed": 1})
            atomic_json(path, {"completed": 2})
            self.assertEqual(json.loads(path.read_text()), {"completed": 2})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_orientation_diagonal_repeats_and_unpaired_tails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for index, (original, recovered) in enumerate([
                    ([0, 0, 1, 2], [0, 1, 2]), ([0, 1], [1, 1, 2])]):
                folder = root / f"sample_{index}"
                folder.mkdir()
                np.savetxt(folder / "original_tokens.txt", original, fmt="%d")
                np.savetxt(folder / "roundtrip_tokens.txt", recovered, fmt="%d")
            counts, records = count_transitions(root, 3)
            np.testing.assert_array_equal(counts.toarray(), [[1, 2, 0], [0, 1, 1], [0, 0, 0]])
            self.assertEqual(sum(r["paired_tokens"] for r in records), 5)
            self.assertEqual(sum(r["unpaired_original"] for r in records), 1)
            self.assertEqual(sum(r["unpaired_recovered"] for r in records), 1)

    def test_smoothed_rows_including_unseen_tokens(self):
        counts = sparse.csr_matrix([[2, 1, 0], [0, 0, 0], [1, 0, 3]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "matrix.npy"
            totals = save_transition_matrix(counts, path, beta=.1, block_rows=1)
            matrix = np.load(path, mmap_mode="r")
            np.testing.assert_array_equal(totals, [3, 0, 4])
            np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-7)
            np.testing.assert_allclose(matrix[0], np.array([2.1, 1.1, .1]) / 3.3)
            np.testing.assert_allclose(matrix[1], [1 / 3] * 3)
            self.assertTrue((matrix > 0).all())


if __name__ == "__main__":
    unittest.main()
