"""Position-preserving audio windows and cumulative grouped evidence."""
import time
import numpy as np

SR = 16000
HOP = 320


def windows(audio_samples, chunk, left=100, look=0):
    if chunk <= 0 or left < 0 or look < 0 or audio_samples < 400:
        raise ValueError('Invalid window configuration or audio shorter than encoder receptive field')
    # Grid positions are nominal speech frames; insufficient convolution support is
    # represented by a missing observation, never by compacting subsequent indices.
    total = (audio_samples + HOP - 1) // HOP
    for a in range(0, total, chunk):
        b = min(a + chunk, total)
        start = max(0, a-left)
        stop = min(audio_samples, (b+look)*HOP)
        yield a, b, start, stop


def encode_windows(audio, encode, chunk, left=100, look=0):
    tokens, positions, arrivals, batches, records = [], [], [], [], []
    elapsed = 0.
    if chunk == 0:
        started = time.perf_counter()
        values = np.asarray(encode(audio), dtype=np.int64).reshape(-1)
        elapsed = time.perf_counter()-started
        return dict(tokens=values, positions=np.arange(len(values)),
                    arrivals=np.full(len(values), len(audio)/SR),
                    batches=np.zeros(len(values), dtype=np.int64),
                    records=[dict(batch=0, arrival=len(audio)/SR, seconds=elapsed,
                                  requested=len(audio)//HOP, returned=len(values))])
    for batch, (a, b, start, stop) in enumerate(windows(len(audio), chunk, left, look)):
        started = time.perf_counter()
        values = (np.asarray(encode(audio[start*HOP:stop]), dtype=np.int64).reshape(-1)
                  if stop-start*HOP >= 400 else np.empty(0, dtype=np.int64))
        seconds = time.perf_counter()-started
        offset = a-start
        count = min(b-a, max(0, len(values)-offset))
        tokens.extend(values[offset:offset+count])
        positions.extend(range(a, a+count))
        arrivals.extend([stop/SR]*count)
        batches.extend([batch]*count)
        records.append(dict(batch=batch, arrival=stop/SR, seconds=seconds,
                            requested=b-a, returned=count, start_position=a))
    return dict(tokens=np.asarray(tokens, dtype=np.int64),
                positions=np.asarray(positions, dtype=np.int64),
                arrivals=np.asarray(arrivals, dtype=float),
                batches=np.asarray(batches, dtype=np.int64), records=records)


class Evidence:
    def __init__(self, channel, alpha=.01):
        self.channel = np.asarray(channel, dtype=float)
        if (self.channel.ndim != 2 or self.channel.shape[0] != self.channel.shape[1]
                or not np.isfinite(self.channel).all() or (self.channel <= 0).any()
                or not np.allclose(self.channel.sum(1), 1)):
            raise ValueError('Require positive finite row-stochastic square channel')
        if not 0 < alpha < 1:
            raise ValueError('Require alpha in (0,1)')
        self.threshold = np.log(1/alpha)
        self.wealth = 0.
        self.last_position = -1
        self.first_detection = None
        self.path = []

    def update(self, positions, observed, expected, probabilities, arrival):
        positions = np.asarray(positions, dtype=np.int64)
        if (len(positions) and (positions[0] <= self.last_position
                              or (np.diff(positions) <= 0).any())):
            raise ValueError('Positions must increase; repeated evidence is forbidden')
        observed, expected = np.asarray(observed), np.asarray(expected)
        probabilities = np.asarray(probabilities, dtype=float)
        if (len(observed) != len(positions) or len(expected) != len(positions)
                or probabilities.shape != (len(positions), len(self.channel))):
            raise ValueError('Mismatched batch shapes')
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(1), 1)):
            raise ValueError('Require normalized nonnegative group masses')
        if ((observed < 0).any() or (observed >= len(self.channel)).any()
                or (expected < 0).any() or (expected >= len(self.channel)).any()):
            raise ValueError('Group index out of range')
        columns = self.channel[:, observed].T
        factors = np.log(self.channel[expected, observed]) - np.log((probabilities*columns).sum(1))
        for position, factor in zip(positions, factors):
            self.wealth += float(factor)
            self.path.append((int(position), self.wealth, float(arrival)))
        if len(positions):
            self.last_position = int(positions[-1])
        # Decisions are made once per received chunk, after all its factors.
        if self.first_detection is None and self.wealth >= self.threshold:
            self.first_detection = float(arrival)
        return self.wealth


def policies(chunks=(25, 50, 100), looks=(0, 25)):
    return [('full', 0, 0)] + [(f'c{c}_r{r}', c, r) for c in chunks for r in looks]
