"""Random vocabulary partitions and distribution-preserving group sampling."""

import numpy as np
from scipy import sparse


def random_groups(vocab_size, num_groups, seed):
    """Balanced random assignment; every group is nonempty."""
    if not 1 <= num_groups <= vocab_size:
        raise ValueError("Require 1 <= num_groups <= vocab_size")
    return np.random.default_rng(seed).permutation(vocab_size) % num_groups


def probabilities(values):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Expected a nonnegative finite vector with positive mass")
    return p / p.sum()


def group_probabilities(p, groups):
    p = probabilities(p)
    return np.bincount(groups, weights=p, minlength=int(np.max(groups)) + 1)


def grouped_sample(p, groups, uniform, rng):
    """Use a keyed uniform for the group and independent randomness within it.

    The caller must supply fresh, domain-separated keyed uniforms by position.
    Averaged over the keyed uniform, the token marginal is exactly p.
    """
    p = probabilities(p)
    if not 0 < uniform < 1:
        raise ValueError("uniform must be in (0, 1)")
    mass = group_probabilities(p, groups)
    cdf = np.cumsum(mass)
    cdf[-1] = 1.
    group = int(np.searchsorted(cdf, uniform, side="right"))
    members = np.flatnonzero(groups == group)
    token = int(rng.choice(members, p=probabilities(p[members])))
    return token, group


def effective_channel(counts, groups, beta=1e-4):
    """Pool observed pairs, then smooth at group level.

    Source-token mixture within a group is its observed training frequency.
    This is a stationary approximation, not a context-independent codec law.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    counts = sparse.coo_matrix(counts)
    groups = np.asarray(groups)
    if counts.shape != (len(groups), len(groups)):
        raise ValueError("Counts and vocabulary mapping must agree")
    k = int(groups.max()) + 1
    pooled = np.zeros((k, k), dtype=float)
    np.add.at(pooled, (groups[counts.row], groups[counts.col]), counts.data)
    channel = pooled + beta
    channel /= channel.sum(axis=1, keepdims=True)
    return pooled, channel


def log_e_path(expected, observed, group_mass, channel):
    """Channel-model factors C[S,y] / sum_g mass[g] C[g,y].

    Valid conditional-mean-one factors when S is fresh with law mass and
    independent of null observations and the past; channel is fixed in advance.
    Model misspecification affects power under this random-key null. Claims
    conditional on a fixed secret key require additional assumptions.
    """
    mass = probabilities(group_mass)
    channel = np.asarray(channel, dtype=float)
    if channel.shape != (len(mass), len(mass)) or not np.isfinite(channel).all() or (channel <= 0).any():
        raise ValueError("Expected a positive finite square channel")
    if not np.allclose(channel.sum(axis=1), 1):
        raise ValueError("Channel rows must sum to one")
    expected, observed = np.asarray(expected), np.asarray(observed)
    if expected.shape != observed.shape:
        raise ValueError("Expected and observed sequences must align")
    return np.cumsum(np.log(channel[expected, observed]) - np.log((mass @ channel)[observed]))
