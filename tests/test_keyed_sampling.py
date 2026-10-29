"""Small CPU tests of the algorithm, independent of downloaded checkpoints."""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from watermark.keyed_sampling import (
    KeyedSemanticSampler, inverse_cdf, keyed_uniform, match_statistics, replay_tokens,
)


class KeyedSamplingTests(unittest.TestCase):
    def test_key_position_and_sample_id(self):
        u = keyed_uniform(b"secret", "clip", 0)
        self.assertEqual(u, keyed_uniform(b"secret", "clip", 0))
        self.assertTrue(0 < u < 1)
        self.assertNotEqual(u, keyed_uniform(b"other", "clip", 0))
        self.assertNotEqual(u, keyed_uniform(b"secret", "other-clip", 0))
        self.assertNotEqual(u, keyed_uniform(b"secret", "clip", 1))

    def test_sparse_cdf_and_invalid_inputs(self):
        p = torch.tensor([0., .25, 0., .75, 0.])
        self.assertEqual(inverse_cdf(p, .1), 1)
        self.assertEqual(inverse_cdf(p, .25), 3)
        self.assertEqual(inverse_cdf(p, .99), 3)
        for p in (torch.zeros(3), torch.tensor([float("nan"), 1.]), torch.tensor([-1., 2.])):
            with self.assertRaises(ValueError):
                inverse_cdf(p, .5)

    def test_distribution_is_not_greenlist_biased(self):
        # Every token is semantic. Repeated fresh nonces must recover q,
        # rather than over-representing a key-selected vocabulary subset.
        p = torch.tensor([.1, .3, .6])
        counts = torch.zeros(3)
        torch.manual_seed(1)
        for i in range(3000):
            sampler = KeyedSemanticSampler({j: j for j in range(3)}, b"test", str(i), top_k=0, top_p=1.)
            token = sampler(torch.tensor([[7]]), p.log()[None]).argmax().item()
            counts[token] += 1
        self.assertTrue(torch.all((counts / counts.sum() - p).abs() < .035))

    def test_semantic_gate_preserves_nonsemantic_probability(self):
        # IDs 0 and 2 are semantic; ID 1 is a marker with 40% probability.
        p = torch.tensor([.1, .4, .5])
        counts = torch.zeros(3)
        torch.manual_seed(7)
        for i in range(3000):
            sampler = KeyedSemanticSampler({0: 0, 1: 2}, b"test", str(i), top_k=0, top_p=1.)
            token = sampler(torch.tensor([[7]]), p.log()[None]).argmax().item()
            counts[token] += 1
            self.assertEqual(len(sampler.tokens), int(token != 1))
        self.assertTrue(torch.all((counts / counts.sum() - p).abs() < .035))

    def test_sampling_filters_are_in_saved_distribution(self):
        sampler = KeyedSemanticSampler({0: 0, 1: 1, 2: 2}, b"key", "clip",
                                       temperature=.5, top_k=2, top_p=1.)
        sampler(torch.tensor([[9]]), torch.tensor([[0., 1., 2.]]))
        expected = torch.tensor([0., 1., torch.exp(torch.tensor(2.)).item()], dtype=torch.float64)
        expected /= expected.sum()
        torch.testing.assert_close(sampler.trace()["p0"][0], expected)
        torch.testing.assert_close(replay_tokens(sampler.trace(), b"key"),
                                   torch.tensor(sampler.tokens))

    def test_huggingface_generation_and_replay(self):
        # Exercise the actual processor -> HF multinomial -> emitted IDs path.
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=12, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, bos_token_id=10,
                             eos_token_id=11, pad_token_id=11)
        model = Qwen2ForCausalLM(config).eval()
        sampler = KeyedSemanticSampler({i: i for i in range(8)}, b"test-key", "clip",
                                       temperature=.8, top_k=5, top_p=.9)
        prompt = torch.tensor([[10, 0, 1]])  # semantic reference tokens do not count
        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=12, do_sample=True,
                                    temperature=1., top_k=0, top_p=1.,
                                    logits_processor=[sampler])
        semantic = torch.tensor([int(t) for t in output[0, 3:] if int(t) < 8])
        self.assertGreater(len(semantic), 0)
        torch.testing.assert_close(semantic, replay_tokens(sampler.trace(), b"test-key"))
        self.assertNotIn("key", sampler.trace())

    def test_missing_extra_tokens_are_explicit(self):
        result = match_statistics([2, 3], [2, 4, 5])
        self.assertEqual(result["matches"], 1)
        self.assertEqual(result["missing_tokens"], 1)
        self.assertEqual(result["match_rate"], 1 / 3)
        self.assertEqual(result["compared_match_rate"], .5)
        self.assertEqual(match_statistics([1, 2, 3], [1])["extra_tokens"], 2)
        self.assertIsNone(match_statistics([], [])["match_rate"])


if __name__ == "__main__":
    unittest.main()
