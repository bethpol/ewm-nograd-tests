import unittest

from watermark.word_anchors import WordCommitter, LocalWordCommitter, agreement_statistics, context_timeline, normalize_word
from evals.rolling_stt import update_schedule


def observation(n, labels, ends=None):
    if ends is None:
        ends = [1600 * (i + 1) for i in range(len(labels))]
    return {"available_samples": n, "words": [
        {"word": word, "start_samples": max(0, end - 1000), "end_samples": end}
        for word, end in zip(labels, ends)]}


class WordAnchorTests(unittest.TestCase):
    def test_first_word_difference_expires_from_context(self):
        states = [LocalWordCommitter(agreement=2, margin_samples=0, delay_samples=0)
                  for _ in range(2)]
        for state, first in zip(states, ("hello", "yellow")):
            for n in (4000, 5000):
                state.update(observation(n, [first, "world"], [1000, 2400]))
            for n in (7000, 8000):
                state.update(observation(n, [first, "world", "again"], [1000, 2400, 4000]))
            for n in (10000, 11000):
                state.update(observation(n, [first, "world", "again", "later"], [1000, 2400, 4000, 6000]))
        one = [context_timeline(s.events, 50, 1) for s in states]
        two = [context_timeline(s.events, 50, 2) for s in states]
        world_activation = states[0].events[1]["activation_token"]
        again_activation = states[0].events[2]["activation_token"]
        self.assertEqual(one[0][world_activation:], one[1][world_activation:])
        self.assertEqual(two[0][again_activation:], two[1][again_activation:])

    def test_local_policy_allows_small_overlap_between_adjacent_words(self):
        state = LocalWordCommitter(agreement=2, margin_samples=0, delay_samples=0)
        first = [{"word": "hello", "start_samples": 0, "end_samples": 1600},
                 {"word": "world", "start_samples": 1500, "end_samples": 3200},
                 {"word": "again", "start_samples": 3200, "end_samples": 4800}]
        state.update({"available_samples": 6400, "words": first})
        state.update({"available_samples": 8000, "words": first})
        self.assertEqual([e["word"] for e in state.events], ["hello", "world"])
        revised = [dict(w) for w in first]
        revised[1]["word"] = "worlds"
        revised[1]["end_samples"] = 3400
        state.update({"available_samples": 9600, "words": revised})
        state.update({"available_samples": 11200, "words": revised})
        self.assertEqual([e["word"] for e in state.events], ["hello", "world"])

    def test_local_policy_recovers_after_initial_word_insertion(self):
        state = LocalWordCommitter(agreement=2, margin_samples=0, delay_samples=0)
        def obs(n, extra):
            words = ([{"word": "with", "start_samples": 0, "end_samples": 800}] if extra else [])
            words += [{"word": "every", "start_samples": 1000, "end_samples": 2200},
                      {"word": "summer", "start_samples": 2400, "end_samples": 3800},
                      {"word": "comes", "start_samples": 4000, "end_samples": 5400}]
            return {"available_samples": n, "words": words}
        state.update(obs(6400, True))
        state.update(obs(9600, True))
        frozen = [dict(e) for e in state.events]
        # Old prefix changes; a new stable occurrence can still be committed.
        a = obs(12800, False)
        a["words"].append({"word": "again", "start_samples": 5600, "end_samples": 7000})
        self.assertEqual(state.update(a), [])  # comes was previously withheld
        a = {**a, "available_samples": 16000}
        events = state.update(a)
        self.assertIn("comes", [e["word"] for e in events])
        self.assertEqual(state.events[:len(frozen)], frozen)
        self.assertTrue(state.contradictions)
        self.assertTrue(all(e["activation_token"] * 320 >= 16000 for e in events))

    def test_local_policy_distinguishes_repeated_word_occurrences(self):
        state = LocalWordCommitter(agreement=2, margin_samples=0, delay_samples=0,
                                   timestamp_tolerance_samples=100)
        words = [{"word": "the", "start_samples": 0, "end_samples": 1000},
                 {"word": "the", "start_samples": 4000, "end_samples": 5000},
                 {"word": "end", "start_samples": 6000, "end_samples": 7000}]
        state.update({"available_samples": 8000, "words": words})
        state.update({"available_samples": 9600, "words": words})
        self.assertEqual([e["word"] for e in state.events], ["the", "the"])
        state.update({"available_samples": 12800, "words": words})
        self.assertEqual(len(state.events), 2)  # no duplicate re-emission

    def test_local_policy_rejects_future_ends_and_empty_history(self):
        state = LocalWordCommitter(agreement=3, margin_samples=0)
        for n in (3200, 6400, 9600):
            state.update(observation(n, ["hello", "world"], [12000, 13000]))
        self.assertEqual(state.events, [])

    def test_agreement_trailing_word_and_future_activation(self):
        state = WordCommitter(agreement=2, margin_samples=0, delay_samples=1600)
        self.assertEqual(state.update(observation(3200, ["Hello,", "world"])), [])
        events = state.update(observation(6400, ["hello", "world"]))
        self.assertEqual([e["word"] for e in events], ["hello"])
        self.assertEqual(events[0]["activation_token"], 25)
        self.assertGreaterEqual(events[0]["activation_token"] * 320, 6400 + 1600)
        self.assertEqual(len(state.events), 1)  # trailing world never committed

    def test_future_word_timestamp_is_not_committed(self):
        state = WordCommitter(agreement=2, margin_samples=0)
        state.update(observation(3200, ["hello", "world"], [9000, 10000]))
        state.update(observation(6400, ["hello", "world"], [9000, 10000]))
        self.assertEqual(state.events, [])

    def test_contradictions_cannot_repair_previous_commit(self):
        state = WordCommitter(agreement=2, margin_samples=0)
        state.update(observation(3200, ["hello", "world"]))
        state.update(observation(6400, ["hello", "world"]))
        state.update(observation(9600, ["yellow", "world", "again"]))
        self.assertEqual([e["word"] for e in state.events], ["hello"])
        self.assertEqual(len(state.contradictions), 1)

    def test_update_grid_excludes_eof(self):
        schedule = update_schedule(16000, [200, 500])
        self.assertEqual(schedule, [3200, 6400, 8000, 9600, 12800])
        self.assertNotIn(16000, schedule)
        self.assertTrue(all(n % 320 == 0 for n in schedule))

    def test_one_token_activation_error_is_local(self):
        a = [{"word": "hello", "activation_token": 5}, {"word": "world", "activation_token": 10}]
        b = [{"word": "hello", "activation_token": 6}, {"word": "world", "activation_token": 11}]
        result = agreement_statistics(a, b, 20, 1)
        self.assertEqual(result["active_union_positions"], 15)
        self.assertEqual(result["matching_active_positions"], 13)
        self.assertEqual(result["activation_index_absolute_differences"], [1, 1])
        self.assertEqual(agreement_statistics(a, a, 20, 2)["active_context_agreement"], 1.)

    def test_no_anchors_is_not_success(self):
        result = agreement_statistics([], [], 20, 1)
        self.assertIsNone(result["active_context_agreement"])
        self.assertEqual(result["embedder_active_fraction"], 0.)

    def test_normalization_and_monotonic_updates(self):
        self.assertEqual(normalize_word(" Hello! "), "hello")
        self.assertEqual(normalize_word("Don’t"), "don't")
        state = WordCommitter()
        state.update(observation(3200, []))
        with self.assertRaises(ValueError):
            state.update(observation(3200, []))


if __name__ == "__main__":
    unittest.main()
