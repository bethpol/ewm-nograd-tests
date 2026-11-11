"""Direct group-CDF sampling with PRF input equal to absolute semantic index."""
import hashlib
import hmac
import torch
from watermark.keyed_sampling import KeyedSemanticSampler, inverse_cdf


def position_uniform(key, position):
    if not key or position < 0:
        raise ValueError('Require nonempty key and nonnegative absolute position')
    digest = hmac.new(key, position.to_bytes(8, 'big'), hashlib.sha256).digest()
    return ((int.from_bytes(digest[:8], 'big') >> 12) + .5) / 2**52


def masses(p, groups):
    result = torch.zeros(int(groups.max()) + 1, dtype=torch.float64)
    result.scatter_add_(0, groups, p.double())
    return result / result.sum()


class PositionGroupedSampler(KeyedSemanticSampler):
    def __init__(self, semantic_to_vocab, key, groups, record_only=False, **kwargs):
        super().__init__(semantic_to_vocab, key, 'position-only', groups=groups, **kwargs)
        self.selected_groups = []
        self.record_only = record_only

    def __call__(self, input_ids, scores):
        if scores.shape[0] != 1:
            raise ValueError('Only batch size one is supported')
        for warper in self.warpers:
            scores = warper(input_ids, scores)
        probs = torch.softmax(scores[0].double(), dim=-1)
        if not torch.isfinite(probs).all():
            raise ValueError('Nonfinite model probabilities')
        candidate = int(torch.multinomial(probs, 1).item())
        if candidate in self.semantic_ids:
            p = probs[self.vocab_ids.to(probs.device)].detach().cpu()
            p = p / p.sum()
            if self.record_only:
                # Record an ordinary filtered categorical draw without consulting
                # the key; used for genuine unwatermarked audio controls.
                token = self.semantic_ids[candidate]
                group = int(self.groups[token])
            else:
                group = inverse_cdf(masses(p, self.groups), position_uniform(self.key, len(self.tokens)))
                token = int(torch.multinomial(p * (self.groups == group), 1).item())
            self.probabilities.append(p)
            self.tokens.append(token)
            self.selected_groups.append(group)
            candidate = int(self.vocab_ids[token])
        selected = torch.full_like(scores, -float('inf'))
        selected[0, candidate] = 0.
        return selected

    def trace(self):
        result = super().trace()
        result.update(format='sparktts-position-group-cdf-v2',
                      seed_rule='HMAC-SHA256(key, absolute_semantic_index_uint64_be)',
                      selected_groups=torch.tensor(self.selected_groups, dtype=torch.long),
                      generation_mode='ordinary' if self.record_only else 'watermarked')
        result.pop('sample_id')
        return result


def replay_groups(trace, key):
    if trace['format'] != 'sparktts-position-group-cdf-v2':
        raise ValueError('Require position-only direct group-CDF v2 trace')
    return torch.tensor([inverse_cdf(masses(p, trace['groups']), position_uniform(key, t))
                         for t, p in enumerate(trace['p0'])], dtype=torch.long)
