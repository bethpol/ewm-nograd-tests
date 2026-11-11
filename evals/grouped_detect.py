"""Score grouped Spark-TTS traces against an explicitly supplied group channel."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evals.keyed_sparktts import read_key
from watermark.grouped import group_probabilities
from watermark.keyed_sampling import replay_tokens


def score(trace, observed, key, archive, alpha):
    if trace["format"] != "sparktts-keyed-grouped-v1":
        raise ValueError("Require a grouped generation trace")
    groups = trace["groups"].numpy()
    if not np.array_equal(groups, archive["groups"]):
        raise ValueError("Channel and generation must use exactly the same partition")
    channel = archive["channel"]
    k = int(groups.max()) + 1
    if (channel.shape != (k, k) or not np.isfinite(channel).all()
            or (channel <= 0).any() or not np.allclose(channel.sum(axis=1), 1)):
        raise ValueError("Require a positive row-stochastic group channel")
    observed = np.asarray(observed, dtype=np.int64).reshape(-1)
    if ((observed < 0) | (observed >= len(groups))).any():
        raise ValueError("Observation outside semantic vocabulary")
    expected = replay_tokens(trace, key).numpy()
    n = min(len(expected), len(observed))
    factors = []
    for t in range(n):
        mass = group_probabilities(trace["p0"][t].numpy(), groups)
        column = channel[:, groups[observed[t]]]
        factors.append(np.log(column[groups[expected[t]]]) - np.log(mass @ column))
    path = np.cumsum(factors)
    hits = np.flatnonzero(path >= np.log(1 / alpha))
    return {"compared_tokens": n, "missing_tokens": max(0, len(expected) - n),
            "extra_tokens": max(0, len(observed) - n),
            "first_detection": int(hits[0] + 1) if len(hits) else None,
            "final_log_e": float(path[-1]) if n else 0., "alpha": alpha,
            "scope": "saved-generation-distribution replay; stationary fitted group channel; aligned indices"}, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--channel", required=True, help="channel.npz from grouped_watermark; exact same partition")
    parser.add_argument("--key_file", default=".cache/keyed_sparktts/secret.key")
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("Require 0 < alpha < 1")
    trace = torch.load(args.trace, map_location="cpu", weights_only=True)
    with np.load(args.channel) as archive:
        result, path = score(trace, np.loadtxt(args.tokens, dtype=np.int64, ndmin=1),
                             read_key(args.key_file), archive, args.alpha)
    result.update(trace=str(Path(args.trace).resolve()), channel=str(Path(args.channel).resolve()),
                  tokens=str(Path(args.tokens).resolve()))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    np.save(destination.with_suffix(".log_e.npy"), path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
