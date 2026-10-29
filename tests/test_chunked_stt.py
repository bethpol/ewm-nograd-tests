import unittest
import numpy as np

from evals.chunked_stt import chunk_schedule, release_chunks
from watermark.word_anchors import LocalWordCommitter, WordCommitter, activation_token


class ChunkedSTTTests(unittest.TestCase):
    def test_release_never_revises_previous_chunks_or_uses_unavailable_tokens(self):
        calls = []
        def decode(frontier):
            calls.append(frontier)
            return np.full(frontier * 320, frontier, dtype=np.float32)
        chunks, records = release_chunks(30, 5, 10, decode)
        self.assertEqual(calls, [15, 20, 25, 30, 30, 30])
        self.assertTrue(np.all(chunks[0] == 15))  # not revised to final value 30
        self.assertEqual(sum(map(len, chunks)), 30 * 320)
        self.assertFalse(records[3]["tail_flush"])
        self.assertTrue(records[4]["tail_flush"])
        self.assertEqual(records[1]["generation_frontier_sample"], 20 * 320)

    def test_short_last_chunk_and_receiver_bytes(self):
        chunks, records = release_chunks(13, 5, 2, lambda n: np.arange(n * 320, dtype=np.float32))
        self.assertEqual([len(c) for c in chunks], [1600, 1600, 960])
        received = [np.frombuffer(c.tobytes(), dtype=np.float32).copy() for c in chunks]
        np.testing.assert_array_equal(np.concatenate(chunks), np.concatenate(received))
        self.assertEqual(records[-1]["end_sample_exclusive"], 4160)

    def test_activation_includes_lookahead_for_both_policies(self):
        for policy in (WordCommitter, LocalWordCommitter):
            state = policy(agreement=2, margin_samples=0, delay_samples=1600)
            words = [{"word": "hello", "start_samples": 0, "end_samples": 1600},
                     {"word": "world", "start_samples": 1600, "end_samples": 3200}]
            for n in (6400, 9600):
                state.update({"available_samples": n,
                              "generation_frontier_samples": n + 3200, "words": words})
            e = state.events[0]
            self.assertEqual(e["activation_token"], 45)  # (9600 + 3200 + 1600)/320
            self.assertGreaterEqual(e["activation_token"] * 320, e["generation_frontier_samples"] + 1600)
        with self.assertRaises(ValueError):
            activation_token({"available_samples": 9600, "generation_frontier_samples": 6400}, 0, 320)


if __name__ == "__main__":
    unittest.main()
