"""Position-keyed categorical sampling and replay for the matched (delta=0) case.

This module does not bias logits or group tokens. See docs/keyed_sparktts.md
for the probability model, trace format, and limits of this first experiment.
"""

import hashlib
import hmac
import math

import torch
from transformers import (
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def keyed_uniform(key: bytes, sample_id: str, position: int) -> float:
    """An open-interval uniform variate derived from a key and semantic index.

    sample_id is a public recording nonce, not a second secret. Length-prefixing
    keeps its encoding unambiguous. A domain tag versions this sampling rule.
    52 bits allow both midpoint endpoints to be represented strictly inside (0,1).
    """
    if not key or position < 0:
        raise ValueError("A nonempty secret key and nonnegative position are required")
    nonce = sample_id.encode("utf-8")
    message = (b"sparktts-keyed-categorical-v1\0" + len(nonce).to_bytes(4, "big")
               + nonce + position.to_bytes(8, "big"))
    digest = hmac.new(key, message, hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], "big") >> 12
    return (integer + 0.5) / (2 ** 52)


def inverse_cdf(probabilities: torch.Tensor, uniform: float) -> int:
    """Sample in ascending semantic-ID order; use the same rule during replay."""
    p = probabilities.detach().to(device="cpu", dtype=torch.float64)
    if p.ndim != 1 or not torch.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Expected a finite, nonnegative probability vector with positive mass")
    if not 0 < uniform < 1:
        raise ValueError("uniform must lie strictly between zero and one")
    cdf = torch.cumsum(p / p.sum(), dim=0)
    cdf[-1] = 1.0  # absorb floating-point summation error at the upper boundary
    return int(torch.searchsorted(cdf, uniform, right=True).item())


class KeyedSemanticSampler(LogitsProcessor):
    """Sample semantic tokens with a key, preserving the filtered distribution.

    HF calls this after its ordinary logits processors. We apply temperature,
    top-k, and top-p here, so HF's corresponding warpers MUST be disabled.
    An ordinary categorical draw first selects semantic vs nonsemantic output.
    Semantic draws are replaced by a keyed draw from the conditional semantic
    distribution. Nonsemantic tokens retain ordinary sampling. Finally a point
    mass tells HF which token to emit; its final multinomial draw has no choice.

    Only batch size one, one beam, and reference-conditioned TTS are supported.
    No preceding prompt tokens contribute to the semantic position counter.
    """

    def __init__(self, semantic_to_vocab, key, sample_id, temperature=1.0,
                 top_k=50, top_p=0.95):
        if not key or not sample_id:
            raise ValueError("A secret key and public sample ID are required")
        if not math.isfinite(temperature) or temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
            raise ValueError("Require temperature > 0, top_k >= 0, and 0 < top_p <= 1")
        if sorted(semantic_to_vocab) != list(range(len(semantic_to_vocab))):
            raise ValueError("Semantic IDs must be contiguous starting at zero")
        self.vocab_ids = torch.tensor([semantic_to_vocab[i] for i in range(len(semantic_to_vocab))])
        if not len(self.vocab_ids) or len(set(self.vocab_ids.tolist())) != len(self.vocab_ids):
            raise ValueError("Semantic vocabulary must be nonempty with unique model IDs")
        self.semantic_ids = {int(v): i for i, v in enumerate(self.vocab_ids)}
        self.key, self.sample_id = key, sample_id
        self.probabilities = []
        self.tokens = []
        self.warpers = []
        if temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_k:
            self.warpers.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p))

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1:
            raise ValueError("Keyed semantic sampling supports batch size one only")
        for warper in self.warpers:
            scores = warper(input_ids, scores)
        probs = torch.softmax(scores[0].double(), dim=-1)
        if not torch.isfinite(probs).all():
            raise ValueError("Sampling distribution contains nonfinite probabilities")
        candidate = int(torch.multinomial(probs, 1).item())
        if candidate in self.semantic_ids:
            # p0_t = q_t conditional on emitting a semantic token at this step.
            p0 = probs[self.vocab_ids.to(probs.device)].detach().cpu()
            p0 = p0 / p0.sum()
            position = len(self.tokens)
            semantic_id = inverse_cdf(p0, keyed_uniform(self.key, self.sample_id, position))
            self.probabilities.append(p0)
            self.tokens.append(semantic_id)
            candidate = int(self.vocab_ids[semantic_id])
        selected = torch.full_like(scores, -float("inf"))
        selected[0, candidate] = 0.0
        return selected

    def trace(self):
        """Save distributions, not the key, uniforms, or HMAC outputs."""
        return {
            "format": "sparktts-keyed-categorical-v1",
            "sample_id": self.sample_id,
            "p0": (torch.stack(self.probabilities) if self.probabilities
                   else torch.empty((0, len(self.vocab_ids)), dtype=torch.float64)),
            "original_tokens": torch.tensor(self.tokens, dtype=torch.long),
        }


def replay_tokens(trace, key):
    """Reconstruct S_t from the key, public sample ID, index, and saved p0_t."""
    if trace["format"] != "sparktts-keyed-categorical-v1":
        raise ValueError("Unsupported keyed sampling trace format")
    return torch.tensor([
        inverse_cdf(p, keyed_uniform(key, trace["sample_id"], t))
        for t, p in enumerate(trace["p0"])
    ], dtype=torch.long)


def match_statistics(observed, expected):
    """Compare aligned indices without searching, shifting, or deduplicating.

    Missing observations count as failures in match_rate. Extra observations
    are reported but cannot be scored without their corresponding p0_t rows.
    """
    observed = torch.as_tensor(observed).detach().cpu().flatten()
    expected = torch.as_tensor(expected).detach().cpu().flatten()
    n = min(len(observed), len(expected))
    matches = int((observed[:n] == expected[:n]).sum())
    return {
        "matches": matches, "expected_tokens": len(expected),
        "observed_tokens": len(observed), "compared_tokens": n,
        "missing_tokens": max(0, len(expected) - len(observed)),
        "extra_tokens": max(0, len(observed) - len(expected)),
        "match_rate": matches / len(expected) if len(expected) else None,
        "compared_match_rate": matches / n if n else None,
    }
